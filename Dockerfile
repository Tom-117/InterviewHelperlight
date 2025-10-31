FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime


WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade-strategy eager -r requirements.txt

COPY main.py .
COPY model_loader.py .
COPY cv.txt .

RUN mkdir -p models
ENV TRANSFORMERS_CACHE=/app/models
ENV TORCH_HOME=/app/models

CMD ["python", "main.py"]