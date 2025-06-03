.PHONY: help build up down clean dev prod test

# Default target
help:
	@echo "🏪 ATM Monitoring System - Docker Commands"
	@echo "==========================================="
	@echo "Development Commands:"
	@echo "  make build        - Build all Docker images"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo "  make clean        - Clean up volumes and images"
	@echo ""
	@echo "Testing & Utilities:"
	@echo "  make db-migrate   - Run database migrations"

# Build all images
build:
	@echo "🔨 Building Docker images..."
	docker-compose build backend

# Start all core services
run:
	@echo "🚀 Starting all services..."
	docker-compose up backend -d

# Stop all services
down:
	@echo "🛑 Stopping all services..."
	docker-compose down --profile backend

# Clean up everything
clean:
	@echo "🧹 Cleaning up..."
	docker-compose --profile backend down -v
	docker system prune -f
	docker volume prune -f

# Database migrations (if using Alembic)
db-migrate:
	docker-compose exec backend poetry run alembic upgrade head