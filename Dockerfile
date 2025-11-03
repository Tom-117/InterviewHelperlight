FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p uploads models

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/models
ENV TORCH_HOME=/app/models
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Port
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]