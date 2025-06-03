FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install poetry as the app user
RUN pip install poetry

COPY . .

# Configure poetry
RUN poetry config virtualenvs.create false

# Install dependencies (remove --no-root to install project dependencies)
RUN poetry install --without dev --no-interaction --no-ansi --no-root

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8000/api/v1/health", "||", "exit", "1"]

# Expose the application port
EXPOSE 8000

WORKDIR /app/backend

# Run the application
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]