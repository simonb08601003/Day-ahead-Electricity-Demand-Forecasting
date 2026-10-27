# Serving only needs LightGBM, so no torch or mlflow in the image.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
ENV PYTHONPATH=/srv/src:/srv/app ARTIFACT_DIR=/srv/artifacts PORT=8080
EXPOSE 8080
# Cloud Run sets $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
