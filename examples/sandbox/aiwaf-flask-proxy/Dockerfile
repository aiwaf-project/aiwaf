FROM python:3.11-slim

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir -e . --no-deps \
  && pip install --no-cache-dir flask>=2.3 requests numpy scikit-learn skops aiwaf-rust

WORKDIR /workspace/examples/sandbox/aiwaf-flask-proxy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
