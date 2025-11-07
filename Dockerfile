FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        bash \
        curl \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip setuptools wheel


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models \
    TORCH_HOME=/app/models \
    MODEL_CACHE_DIR=/app/models


EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["web"]