FROM python:3.13-slim

# WeasyPrint dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libffi8 \
    libjpeg62-turbo \
    libxrender1 \
    libxext6 \
    libssl3 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000

CMD ["uvicorn", "app:asgi_app", "--host", "0.0.0.0", "--port", "${PORT}"]


