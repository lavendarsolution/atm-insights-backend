# 🏪 ATM Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB-latest-orange.svg)](https://www.timescale.com/)
[![Redis](https://img.shields.io/badge/Redis-7+-red.svg)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)

**Complete ATM Monitoring, Real-time Analytics & Predictive Maintenance System**

A comprehensive solution for real-time telemetry collection from 500+ ATMs, dashboard monitoring, anomaly detection, and failure prediction.

## 🎯 Key Features

- 🔄 **Real-time Telemetry Collection**: Data collection from 500+ ATMs per second
- 📊 **Live Dashboard**: REST API-based monitoring interface
- 🤖 **AI-powered Diagnostics**: Machine learning for anomaly detection and failure prediction
- ⚡ **High-performance Time-series DB**: Efficient data storage with TimescaleDB
- 🚨 **Real-time Alerts**: Redis Pub/Sub based event processing
- 🐳 **Complete Docker Environment**: One-click development/production setup

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Make (optional, for convenience)

### Step 1: Project Setup

```bash
# Clone repository
git clone <repository-url>
cd atm-insights-backend
```

### Step 2: Environment-specific Launch

#### Development Environment (DB + API only)

```bash
make dev
```

#### Individual Service Launch

```bash
# Core services only
docker-compose up -d postgres redis backend
```

### Step 3: System Verification

#### API Endpoint Testing

```bash
# Health check
curl http://localhost:8000/health
```

#### Web Interfaces

- 📚 **API Documentation**: http://localhost:8000/docs
- 🔍 **Schema Explorer**: http://localhost:8000/redoc
- ❤️ **Health Check**: http://localhost:8000/health

## 🛠️ System Architecture

### Component Overview

```mermaid
graph TB
    ATM[ATM Devices] --> SIM[ATM Simulator]
    SIM --> API[FastAPI Backend]
    API --> PG[(PostgreSQL + TimescaleDB)]
    API --> REDIS[(Redis)]
    API --> DASH[Dashboard API]
    REDIS --> DIAG[Diagnostics Engine]
    DIAG --> ML[ML Models]
    PG --> ML
```

### 1. **FastAPI Backend** (Port 8000)

- **Role**: Central API server for telemetry ingestion and dashboard data
- **Features**:
  - Async telemetry processing
  - Auto-generated API documentation
  - Health monitoring
  - Environment-based configuration

### 2. **Diagnostics Engine**

- **Role**: Rule-based diagnostics + ML-powered predictions
- **Features**:
  - Real-time anomaly detection
  - Failure prediction models
  - Comprehensive reporting
  - Maintenance recommendations

### 3. **Database Layer**

- **PostgreSQL + TimescaleDB**: Optimized time-series storage
- **Redis**: Real-time pub/sub and caching

## 🐳 Docker Commands Reference

### Quick Commands (via Makefile)

| Command      | Description                              |
| ------------ | ---------------------------------------- |
| `make dev`   | Start development environment (DB + API) |
| `make down`  | Stop all services                        |
| `make clean` | Complete cleanup (removes volumes)       |
| `make prod`  | Start production environment             |
| `make test`  | Run API tests                            |

### Direct Docker Commands

```bash
# Start core services
docker-compose up -d postgres redis backend

# View service status
docker-compose ps

# View logs
docker-compose logs -f [service_name]

# Stop services
docker-compose down

# Complete cleanup
docker-compose down -v
```

## 🔍 Monitoring & Diagnostics

### System Health Monitoring

```bash
# Check overall system health
curl http://localhost:8000/health

# Monitor service logs
make logs

# Check container status
docker-compose ps
```

### Performance Metrics

- **Telemetry Ingestion Rate**: ~500 messages/30 seconds
- **Database Storage**: TimescaleDB with compression
- **API Response Time**: < 100ms for dashboard queries
- **Memory Usage**: ~2GB total for full system

## 🧪 Testing

### Automated Testing

```bash
# Run all tests
make test

# Run specific test categories
docker-compose exec backend python -m pytest tests/
```

## 🌟 Production Deployment

### Pre-deployment Checklist

1. **Security Configuration**

   ```bash
   # Copy and secure production environment
   cp .env.production.example .env.production

   # Update with secure passwords
   vim .env.production
   ```

2. **Resource Planning**

   - **CPU**: 4+ cores recommended
   - **Memory**: 8GB+ for 1000 ATMs
   - **Storage**: 100GB+ for time-series data
   - **Network**: Stable internet for ATM connections

3. **Production Deployment**

   ```bash
   # Deploy production environment
   make prod

   # Verify deployment
   curl http://your-domain.com/health
   ```

### Production Optimizations

- **Database**: Automated backups and replication
- **Caching**: Redis cluster for high availability
- **Load Balancing**: Multiple FastAPI instances
- **SSL/TLS**: HTTPS termination at load balancer
- **Monitoring**: Prometheus + Grafana integration

## 🛡️ Security Considerations

### Authentication & Authorization

- **JWT-based authentication** (planned)
- **Role-based access control** (planned)
- **API rate limiting** (configurable)

### Data Security

- **Encrypted passwords** in environment variables
- **Database connection encryption**
- **Redis password protection**
- **Container security** with non-root users

### Network Security

- **Internal Docker network** isolation
- **CORS configuration** for frontend integration
- **Trusted host middleware** in production

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd atm-insights-backend
make setup

# Start development environment
make dev

# Make changes and test
make test
```

### Code Style

- **Python**: Follow PEP 8, use Black formatter
- **FastAPI**: Async/await patterns, type hints
- **Docker**: Multi-stage builds, security best practices

### Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Performance Tuning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** - Modern, fast web framework for building APIs
- **TimescaleDB** - Time-series database built on PostgreSQL
- **Redis** - In-memory data structure store for real-time features
- **Docker** - Containerization platform for consistent deployments

---

## 🎯 Getting Started Now

**Ready to monitor 500+ ATMs?**

```bash
git clone <repository-url>
cd atm-insights-backend
make dev
```

**🌐 Access Points:**

- Dashboard API: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

**Need help?** Check the [Troubleshooting](#-troubleshooting) section or open an issue!

---

**⭐ Star this repository if you found it helpful!**
