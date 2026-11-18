FROM python:3.11-slim

WORKDIR /app

# Dependencias para compilación en arquitectura ARM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

# Copiamos código y definimos paquetes
COPY . .
RUN touch src/__init__.py src/app/__init__.py

EXPOSE 8080

CMD ["uvicorn", "src.app.main_v2:app", "--host", "0.0.0.0", "--port", "8080"]
