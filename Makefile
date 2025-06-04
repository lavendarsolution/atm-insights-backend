.PHONY: help build up down clean dev prod test monitoring

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
	@echo "Monitoring Commands:"
	@echo "  make monitoring   - Start with monitoring stack (Prometheus + Grafana)"
	@echo "  make monitoring-only - Start only monitoring services"
	@echo "  make monitoring-down - Stop monitoring services"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make db-migrate   - Run database migrations"
	@echo "  make test         - Run tests"

# Build all images
build:
	@echo "🔨 Building Docker images..."
	docker-compose build backend

# Start core services (without monitoring)
up:
	@echo "🚀 Starting core services..."
	docker-compose up backend postgres redis -d

# Start all services including monitoring
monitoring:
	@echo "🚀 Starting all services with monitoring stack..."
	@echo "📊 This includes: Backend, PostgreSQL, Redis, Prometheus, Grafana, and Exporters"
	docker-compose --profile monitoring up -d
	@echo ""
	@echo "🎉 Services started! Access points:"
	@echo "  📈 Grafana Dashboard: http://localhost:3000 (admin/admin)"
	@echo "  📊 Prometheus: http://localhost:9090"
	@echo "  🔧 API Documentation: http://localhost:8000/docs"
	@echo "  ❤️ Health Check: http://localhost:8000/health"
	@echo "  📋 Metrics Endpoint: http://localhost:8000/metrics"

# Start only monitoring services (assumes core services are running)
monitoring-only:
	@echo "📊 Starting monitoring stack only..."
	docker-compose --profile monitoring up prometheus grafana node-exporter redis-exporter postgres-exporter -d

# Stop monitoring services
monitoring-down:
	@echo "📊 Stopping monitoring services..."
	docker-compose --profile monitoring stop prometheus grafana node-exporter redis-exporter postgres-exporter

# Start with development tools
dev:
	@echo "🚀 Starting development environment..."
	docker-compose --profile tools up postgres redis pgadmin -d
	@echo ""
	@echo "🎉 Development environment started!"
	@echo "  🔧 API Documentation: http://localhost:8000/docs"
	@echo "  🗄️ PgAdmin: http://localhost:5050"
	@echo "  ❤️ Health Check: http://localhost:8000/health"

# Start production environment
prod:
	@echo "🚀 Starting production environment with monitoring..."
	ENVIRONMENT=production docker-compose --profile monitoring up -d

# Stop all services
down:
	@echo "🛑 Stopping all services..."
	docker-compose --profile monitoring --profile tools down

# Clean up everything
clean:
	@echo "🧹 Cleaning up everything..."
	docker-compose --profile monitoring --profile tools down -v
	docker system prune -f
	docker volume prune -f
	@echo "✅ Cleanup completed!"


# Database migrations
db-migrate:
	@echo "🗄️ Running database migrations..."
	docker-compose exec backend poetry run alembic upgrade head

# Run tests
test:
	@echo "🧪 Running tests..."
	docker-compose exec backend poetry run pytest

# Show service status
status:
	@echo "📊 Service Status:"
	docker-compose ps

# Model Training
model-training:
	@echo "🚀 Starting model training..."
	docker-compose exec backend poetry run python train_models.py --trials 100 --cv-folds 5 --timeout 3600 --workers 8