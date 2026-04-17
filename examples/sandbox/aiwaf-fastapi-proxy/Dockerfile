FROM python:3.11-slim

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir -e . --no-deps \
  && pip install --no-cache-dir fastapi starlette "uvicorn[standard]" httpx numpy scikit-learn joblib aiwaf-rust

WORKDIR /workspace/examples/sandbox/aiwaf-fastapi-proxy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
