FROM python:3.11-slim

WORKDIR /workspace

# Copy repo so we can `pip install` the local package.
COPY . /workspace

# Install AIWAF from source without pulling heavyweight optional deps.
RUN pip install --no-cache-dir -e . --no-deps \
  && pip install --no-cache-dir Django>=3.2 requests aiwaf-rust

WORKDIR /workspace/examples/sandbox/aiwaf-django-proxy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python manage.py migrate --noinput --run-syncdb && python manage.py runserver 0.0.0.0:${PORT:-3009}"]
