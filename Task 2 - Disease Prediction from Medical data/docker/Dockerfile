FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for postgres and building tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Run FastAPI backend by default
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
