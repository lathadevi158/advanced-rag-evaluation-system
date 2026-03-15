# 🚀 Deployment Guide

This guide covers multiple deployment options for the Production RAG System.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
   - [AWS EC2](#aws-ec2)
   - [Google Cloud Run](#google-cloud-run)
   - [Hugging Face Spaces](#hugging-face-spaces)
4. [Environment Configuration](#environment-configuration)
5. [Monitoring & Observability](#monitoring--observability)

---

## Local Development

### Prerequisites
- Python 3.12+
- Virtual environment

### Steps

1. **Clone and setup:**
```bash
git clone https://github.com/yourusername/production-rag-system.git
cd production-rag-system
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Run the server:**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Access the API:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Docker Deployment

### Build and Run

**Using Docker:**
```bash
docker build -t production-rag-system .
docker run -p 8000:8000 --env-file .env production-rag-system
```

**Using Docker Compose:**
```bash
# With Pinecone
docker-compose up -d

# With local Weaviate
docker-compose --profile weaviate up -d
```

**Stop services:**
```bash
docker-compose down
```

---

## Cloud Deployment

### AWS EC2

1. **Launch EC2 Instance:**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t3.medium (minimum)
   - Security Group: Open port 8000

2. **SSH into instance:**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install Docker:**
```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
```

4. **Deploy application:**
```bash
git clone https://github.com/yourusername/production-rag-system.git
cd production-rag-system

# Create .env file
nano .env
# Add your API keys

# Run with Docker Compose
docker-compose up -d
```

5. **Setup NGINX (optional):**
```bash
sudo apt install nginx
sudo nano /etc/nginx/sites-available/rag-system
```

NGINX config:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

### Google Cloud Run

1. **Install Google Cloud SDK:**
```bash
gcloud init
```

2. **Build and push Docker image:**
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/production-rag-system

# Deploy to Cloud Run
gcloud run deploy production-rag-system \
  --image gcr.io/YOUR_PROJECT_ID/production-rag-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key,PINECONE_API_KEY=your_key
```

3. **Set environment variables via console:**
   - Go to Cloud Run console
   - Select your service
   - Edit & Deploy → Variables & Secrets
   - Add all required environment variables

---

### Hugging Face Spaces

1. **Create new Space:**
   - Go to https://huggingface.co/spaces
   - New Space → Docker

2. **Add files:**
```
production-rag-system/
├── app.py  # Create entrypoint
├── requirements.txt
├── Dockerfile
└── src/
```

**app.py:**
```python
import uvicorn
from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

**Update Dockerfile:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

3. **Add secrets in Space settings:**
   - OPENAI_API_KEY
   - PINECONE_API_KEY
   - etc.

---

## Environment Configuration

### Required Variables

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  # Optional

# Vector Database (choose one)
VECTOR_DB=pinecone  # or weaviate

# Pinecone
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=production-rag

# Weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=...  # Optional

# Models
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4-turbo-preview

# RAG Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=10
RERANK_TOP_K=3

# Safety
ENABLE_GUARDRAILS=true
TOXICITY_THRESHOLD=0.7

# Observability (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=production-rag-system
```

---

## Monitoring & Observability

### LangSmith Integration

1. **Enable tracing:**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key
export LANGCHAIN_PROJECT=production-rag-system
```

2. **View traces:**
   - Go to https://smith.langchain.com
   - Select your project
   - View traces, latency, costs

### Health Checks

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "vector_db": "pinecone",
  "guardrails_enabled": true
}
```

### Logs

**Docker logs:**
```bash
docker-compose logs -f rag-api
```

**System logs:**
```bash
tail -f /var/log/rag-system.log
```

---

## Performance Optimization

### 1. Use Production Server

Replace `uvicorn` with `gunicorn` for production:

```bash
pip install gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 2. Enable Caching

Add Redis for caching embeddings:

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### 3. Load Balancing

Use NGINX or cloud load balancer for multiple instances.

---

## Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Use HTTPS** - Setup SSL certificates
3. **Rate limiting** - Implement API rate limiting
4. **Input validation** - Already implemented via Pydantic
5. **CORS** - Configure allowed origins in production
6. **Authentication** - Add API key authentication for production

---

## Troubleshooting

### Common Issues

**1. "Module not found" errors:**
```bash
# Ensure you're in the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. Vector DB connection failed:**
- Check API keys
- Verify index/collection exists
- Check network connectivity

**3. Out of memory:**
- Reduce batch size
- Increase instance memory
- Use smaller embedding model

**4. Slow response times:**
- Enable caching
- Reduce top_k
- Use faster embedding model

---

## Scaling

### Horizontal Scaling

1. **Deploy multiple instances**
2. **Add load balancer**
3. **Share vector database**
4. **Use distributed cache (Redis)**

### Vertical Scaling

1. **Increase instance size**
2. **Add GPU for faster inference**
3. **Optimize batch processing**

---

## Cost Optimization

### Estimated Monthly Costs

| Component | Cost (USD/month) |
|-----------|------------------|
| OpenAI API (10K queries) | $30-50 |
| Pinecone (Starter) | $70 |
| AWS EC2 t3.medium | $30 |
| **Total** | **$130-150** |

### Cost Reduction Tips

1. **Use smaller models** - text-embedding-3-small vs ada-002
2. **Cache embeddings** - Avoid re-computing
3. **Batch requests** - Reduce API calls
4. **Use serverless** - Pay per use (Cloud Run, Lambda)

---

**Need help? Open an issue on GitHub!**
