.PHONY: help build up down logs clean dev prod test

# Default target
help:
	@echo "🏪 ATM Monitoring System - Docker Commands"
	@echo "==========================================="
	@echo "Development Commands:"
	@echo "  make dev          - Start development environment (DB + API only)"
	@echo "  make full         - Start full system including simulator"
	@echo "  make build        - Build all Docker images"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo "  make logs         - View logs from all services"
	@echo "  make clean        - Clean up volumes and images"
	@echo ""
	@echo "Production Commands:"
	@echo "  make prod         - Start production environment"
	@echo "  make prod-build   - Build for production"
	@echo ""
	@echo "Testing & Utilities:"
	@echo "  make test         - Run API tests"
	@echo "  make shell-api    - Shell into FastAPI container"
	@echo "  make shell-db     - Shell into PostgreSQL container"
	@echo "  make db-migrate   - Run database migrations"

# Development environment (DB + API only)
dev:
	@echo "🚀 Starting development environment..."
	@cp .env.example .env 2>/dev/null || true
	docker-compose up -d postgres redis backend

# Full system with simulator and diagnostics
full:
	@echo "🚀 Starting full system..."
	@cp .env.example .env 2>/dev/null || true
	docker-compose --profile simulator --profile diagnostics up -d

# Build all images
build:
	@echo "🔨 Building Docker images..."
	docker-compose build

# Start all core services
up:
	@echo "🚀 Starting all services..."
	docker-compose up -d

# Stop all services
down:
	@echo "🛑 Stopping all services..."
	docker-compose --profile simulator --profile diagnostics down

# View logs
logs:
	docker-compose logs -f

# Clean up everything
clean:
	@echo "🧹 Cleaning up..."
	docker-compose --profile simulator --profile diagnostics down -v
	docker system prune -f
	docker volume prune -f

# Production deployment
prod:
	@echo "🌟 Starting production environment..."
	@if [ ! -f .env.production ]; then echo "❌ .env.production file not found!"; exit 1; fi
	docker-compose --env-file .env.production up -d postgres redis backend

prod-build:
	@echo "🔨 Building for production..."
	docker-compose --env-file .env.production build

# API tests
test:
	@echo "🧪 Running API tests..."
	docker-compose exec backend python -m pytest

# Shell access
shell-api:
	docker-compose exec backend /bin/bash

shell-db:
	docker-compose exec postgres psql -U admin -d atm_insights

# Database migrations (if using Alembic)
db-migrate:
	docker-compose exec backend alembic upgrade head

# Quick setup for new developers
setup:
	@echo "⚙️ Setting up development environment..."
	@cp .env.example .env
	@echo "📝 Created .env file - please review and modify as needed"
	@make build
	@make dev
	@echo "✅ Development environment ready!"
	@echo "📍 API available at: http://localhost:8000"
	@echo "📚 API docs at: http://localhost:8000/docs"