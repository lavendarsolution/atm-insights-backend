# 🏪 ATM Insights Backend API

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB-latest-orange.svg)](https://www.timescale.com/)
[![Redis](https://img.shields.io/badge/Redis-7+-red.svg)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)

**High-Performance FastAPI Backend for Real-time ATM Monitoring System**

A scalable, async Python backend serving real-time telemetry data from 500+ ATMs, featuring comprehensive REST APIs, WebSocket real-time updates, and advanced time-series data management.

## 🎯 Key Features

- ⚡ **High-Performance API**: FastAPI with async/await for maximum throughput
- 🔄 **Real-time WebSocket**: Live data streams for dashboard and ATM monitoring
- 📊 **Time-Series Optimized**: TimescaleDB for efficient telemetry data storage
- 🚀 **Scalable Architecture**: Microservice-ready with Redis caching
- 🛡️ **Enterprise Security**: JWT authentication, CORS, and input validation
- 📈 **Advanced Analytics**: Built-in aggregation and analytics endpoints
- 🔍 **Auto-Generated Docs**: OpenAPI/Swagger documentation
- 🐳 **Container Ready**: Docker deployment with production configs

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with TimescaleDB extension
- Redis 7+
- Docker & Docker Compose (optional but recommended)

### Installation & Setup

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd atm-insights

# Start all services
make dev  # or docker-compose up -d postgres redis backend
```

#### Option 2: Local Development

```bash
# Clone and navigate to backend
git clone <repository-url>
cd atm-insights/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Edit with your database credentials

# Run database migrations
alembic upgrade head

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Configuration

Create a `.env` file in the backend directory:

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/atm_insights
POSTGRES_DB=atm_insights
POSTGRES_USER=username
POSTGRES_PASSWORD=password

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password

# API Configuration
API_V1_STR=/api/v1
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment
ENVIRONMENT=development
DEBUG=true

# CORS Origins (comma-separated)
BACKEND_CORS_ORIGINS=http://localhost:8080,http://localhost:3000

# TimescaleDB Configuration
TIMESCALE_COMPRESSION_ENABLED=true
TELEMETRY_RETENTION_DAYS=90
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Alternative documentation
open http://localhost:8000/redoc
```

## 🛠️ Technology Stack

### Core Framework

- **FastAPI** - Modern, fast web framework for building APIs
- **Python 3.11+** - Latest Python with performance improvements
- **Uvicorn** - Lightning-fast ASGI server
- **Pydantic** - Data validation using Python type annotations

### Database & Storage

- **PostgreSQL 15+** - Robust relational database
- **TimescaleDB** - Time-series database extension for telemetry data
- **Redis** - In-memory data store for caching and real-time features
- **SQLAlchemy** - Python SQL toolkit and ORM
- **Alembic** - Database migration tool

### Real-time & Background Processing

- **WebSocket** - Real-time bidirectional communication
- **Redis Pub/Sub** - Message broadcasting for real-time updates
- **Background Tasks** - Async task processing
- **APScheduler** - Advanced Python Scheduler for periodic tasks

### Security & Validation

- **JWT (JSON Web Tokens)** - Secure authentication
- **Passlib** - Password hashing utilities
- **CORS Middleware** - Cross-Origin Resource Sharing
- **Pydantic Models** - Request/response validation

## 📁 Project Structure

```
backend/
├── alembic/                 # Database migrations
│   ├── versions/           # Migration files
│   └── env.py             # Alembic configuration
├── api/                    # API layer
│   ├── routes/            # API route definitions
│   │   ├── dashboard.py   # Dashboard statistics
│   │   ├── atms.py        # ATM management
│   │   ├── telemetry.py   # Telemetry data
│   │   ├── alerts.py      # Alert management
│   │   └── auth.py        # Authentication
│   └── __init__.py
├── database/               # Database configuration
│   ├── session.py         # Database session management
│   └── __init__.py
├── dependencies/           # Dependency injection
│   ├── auth.py            # Authentication dependencies
│   └── __init__.py
├── models/                 # SQLAlchemy models
│   ├── atm.py             # ATM model
│   ├── atm_telemetry.py   # Telemetry model
│   ├── alert.py           # Alert model
│   ├── user.py            # User model
│   └── __init__.py
├── schemas/                # Pydantic schemas
│   ├── atm.py             # ATM schemas
│   ├── telemetry.py       # Telemetry schemas
│   ├── dashboard.py       # Dashboard schemas
│   ├── auth.py            # Authentication schemas
│   └── common.py          # Common schemas
├── services/               # Business logic layer
│   ├── auth_service.py    # Authentication service
│   ├── telemetry_service.py # Telemetry processing
│   ├── dashboard_service.py # Dashboard data aggregation
│   ├── cache_service.py   # Redis caching
│   └── background_service.py # Background tasks
├── utils/                  # Utility functions
│   ├── atm_utils.py       # ATM-specific utilities
│   └── __init__.py
├── main.py                # FastAPI application entry point
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
└── alembic.ini            # Alembic configuration
```

## 🔌 API Reference

### Core Endpoints

#### Health & System Status

```http
GET /health
GET /api/v1/health
```

#### Dashboard Statistics

```http
GET /api/v1/dashboard/stats
```

**Response:**

```json
{
  "total_atms": 500,
  "online_atms": 485,
  "offline_atms": 10,
  "error_atms": 5,
  "total_transactions_today": 12450,
  "avg_cash_level": 68.5,
  "critical_alerts": 3,
  "last_updated": "2025-01-20T10:30:00Z"
}
```

#### ATM Management

```http
GET /api/v1/atms                    # List ATMs with filtering
GET /api/v1/atms/{atm_id}          # Get specific ATM
POST /api/v1/atms                   # Create new ATM
PUT /api/v1/atms/{atm_id}          # Update ATM
DELETE /api/v1/atms/{atm_id}       # Delete ATM
```

**ATM List with Filtering:**

```http
GET /api/v1/atms?page=0&limit=10&status=online&search=NYC&region=northeast
```

#### Telemetry Data

```http
POST /api/v1/telemetry              # Ingest telemetry data
GET /api/v1/atms/{atm_id}/telemetry # Get ATM telemetry history
GET /api/v1/telemetry/aggregated    # Get aggregated telemetry data
```

**Telemetry Ingestion:**

```http
POST /api/v1/telemetry
Content-Type: application/json

{
  "atm_id": "ATM-NYC-001",
  "timestamp": "2025-01-20T10:30:00Z",
  "status": "online",
  "temperature_celsius": 22.5,
  "cash_level_percent": 75.0,
  "transactions_count": 45,
  "failed_transactions": 2,
  "cpu_usage_percent": 25.3,
  "memory_usage_percent": 65.2,
  "network_status": "connected",
  "network_latency_ms": 45,
  "error_code": null,
  "error_message": null
}
```

#### Alert Management

```http
GET /api/v1/alerts                  # List alerts
POST /api/v1/alerts                 # Create alert
PUT /api/v1/alerts/{alert_id}       # Update alert
DELETE /api/v1/alerts/{alert_id}    # Delete alert
```

#### Authentication

```http
POST /api/v1/auth/login             # User login
POST /api/v1/auth/refresh           # Refresh token
POST /api/v1/auth/logout            # User logout
GET /api/v1/auth/me                 # Get current user
```

### WebSocket Endpoints

#### Real-time Dashboard Updates

```websocket
WS /ws/dashboard
```

**Message Format:**

```json
{
  "type": "stats_update",
  "data": {
    "total_atms": 500,
    "online_atms": 485,
    "timestamp": "2025-01-20T10:30:00Z"
  }
}
```

#### ATM Status Updates

```websocket
WS /ws/atms
```

**Message Format:**

```json
{
  "type": "status_update",
  "atm_id": "ATM-NYC-001",
  "data": {
    "status": "online",
    "last_seen": "2025-01-20T10:30:00Z"
  }
}
```

#### Individual ATM Telemetry

```websocket
WS /ws/atms/{atm_id}/telemetry
```

**Message Format:**

```json
{
  "type": "telemetry_update",
  "atm_id": "ATM-NYC-001",
  "data": {
    "temperature_celsius": 22.5,
    "cash_level_percent": 75.0,
    "cpu_usage_percent": 25.3,
    "timestamp": "2025-01-20T10:30:00Z"
  }
}
```

## 🔄 Real-time Architecture

### WebSocket Implementation

The backend provides comprehensive real-time capabilities through WebSocket connections:

```python
# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.atm_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, atm_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if atm_id:
            if atm_id not in self.atm_connections:
                self.atm_connections[atm_id] = []
            self.atm_connections[atm_id].append(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

    async def send_to_atm_subscribers(self, atm_id: str, message: dict):
        if atm_id in self.atm_connections:
            for connection in self.atm_connections[atm_id]:
                await connection.send_json(message)
```

### Redis Pub/Sub Integration

```python
# Redis Pub/Sub for distributed real-time updates
class RedisPubSub:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.pubsub = redis_client.pubsub()

    async def publish_atm_update(self, atm_id: str, data: dict):
        message = {
            "type": "atm_update",
            "atm_id": atm_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.redis.publish(f"atm_updates:{atm_id}", json.dumps(message))

    async def subscribe_to_updates(self):
        await self.pubsub.subscribe("atm_updates:*")
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                await self.handle_atm_update(data)
```

### Background Task Processing

```python
# Background tasks for data processing
from fastapi import BackgroundTasks

@app.post("/api/v1/telemetry")
async def ingest_telemetry(
    telemetry: TelemetryCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Store telemetry data
    db_telemetry = create_telemetry(db, telemetry)

    # Process in background
    background_tasks.add_task(process_telemetry_alerts, db_telemetry)
    background_tasks.add_task(broadcast_telemetry_update, db_telemetry)

    return {"status": "accepted", "id": db_telemetry.id}

async def process_telemetry_alerts(telemetry: TelemetryData):
    """Process telemetry data for alert generation"""
    if telemetry.temperature_celsius > 35:
        await create_alert("high_temperature", telemetry.atm_id)

    if telemetry.cash_level_percent < 20:
        await create_alert("low_cash", telemetry.atm_id)
```

## 💾 Database Design

### TimescaleDB Integration

The backend leverages TimescaleDB for efficient time-series data storage:

```sql
-- ATM Telemetry Hypertable
CREATE TABLE atm_telemetry (
    id SERIAL PRIMARY KEY,
    atm_id VARCHAR(255) NOT NULL,
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(50),
    temperature_celsius FLOAT,
    cash_level_percent FLOAT,
    transactions_count INTEGER,
    failed_transactions INTEGER,
    cpu_usage_percent FLOAT,
    memory_usage_percent FLOAT,
    network_status VARCHAR(50),
    network_latency_ms INTEGER,
    error_code VARCHAR(100),
    error_message TEXT,
    uptime_seconds BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('atm_telemetry', 'time');

-- Create indexes for efficient queries
CREATE INDEX idx_atm_telemetry_atm_id_time ON atm_telemetry (atm_id, time DESC);
CREATE INDEX idx_atm_telemetry_status ON atm_telemetry (status, time DESC);

-- Enable compression for older data
ALTER TABLE atm_telemetry SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'atm_id'
);

-- Automatic compression policy
SELECT add_compression_policy('atm_telemetry', INTERVAL '7 days');
```

### Database Models

```python
# SQLAlchemy Models
class ATM(Base):
    __tablename__ = "atms"

    atm_id = Column(String(255), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    location_address = Column(Text)
    location_latitude = Column(Float)
    location_longitude = Column(Float)
    model = Column(String(255))
    manufacturer = Column(String(255))
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    telemetry = relationship("ATMTelemetry", back_populates="atm")
    alerts = relationship("Alert", back_populates="atm")

class ATMTelemetry(Base):
    __tablename__ = "atm_telemetry"

    id = Column(Integer, primary_key=True, index=True)
    atm_id = Column(String(255), ForeignKey("atms.atm_id"), nullable=False)
    time = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50))
    temperature_celsius = Column(Float)
    cash_level_percent = Column(Float)
    transactions_count = Column(Integer)
    failed_transactions = Column(Integer)
    cpu_usage_percent = Column(Float)
    memory_usage_percent = Column(Float)
    network_status = Column(String(50))
    network_latency_ms = Column(Integer)
    error_code = Column(String(100))
    error_message = Column(Text)
    uptime_seconds = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    atm = relationship("ATM", back_populates="telemetry")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    atm_id = Column(String(255), ForeignKey("atms.atm_id"), nullable=False)
    severity = Column(String(50), nullable=False)  # critical, high, medium, low
    message = Column(Text, nullable=False)
    alert_type = Column(String(100), nullable=False)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    atm = relationship("ATM", back_populates="alerts")
```

## 🔒 Security & Authentication

### JWT Authentication

```python
# JWT Token handling
from passlib.context import CryptContext
from jose import JWTError, jwt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user(db, username=username)
    if user is None:
        raise credentials_exception
    return user
```

### CORS Configuration

```python
# CORS middleware configuration
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Input Validation

```python
# Pydantic schemas for validation
class TelemetryCreate(BaseModel):
    atm_id: str = Field(..., min_length=1, max_length=255)
    timestamp: Optional[datetime] = None
    status: Optional[str] = Field(None, regex="^(online|offline|maintenance|error)$")
    temperature_celsius: Optional[float] = Field(None, ge=-50, le=100)
    cash_level_percent: Optional[float] = Field(None, ge=0, le=100)
    transactions_count: Optional[int] = Field(None, ge=0)
    failed_transactions: Optional[int] = Field(None, ge=0)
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    network_status: Optional[str] = Field(
        None, regex="^(connected|disconnected|unstable)$"
    )
    network_latency_ms: Optional[int] = Field(None, ge=0, le=10000)
    error_code: Optional[str] = Field(None, max_length=100)
    error_message: Optional[str] = Field(None, max_length=1000)
    uptime_seconds: Optional[int] = Field(None, ge=0)

    class Config:
        schema_extra = {
            "example": {
                "atm_id": "ATM-NYC-001",
                "timestamp": "2025-01-20T10:30:00Z",
                "status": "online",
                "temperature_celsius": 22.5,
                "cash_level_percent": 75.0,
                "transactions_count": 45,
                "failed_transactions": 2,
                "cpu_usage_percent": 25.3,
                "memory_usage_percent": 65.2,
                "network_status": "connected",
                "network_latency_ms": 45
            }
        }
```

## 📊 Analytics & Aggregation

### Dashboard Service

```python
# Dashboard data aggregation service
class DashboardService:
    def __init__(self, db: Session, redis_client):
        self.db = db
        self.redis = redis_client

    async def get_dashboard_stats(self) -> DashboardStats:
        # Check cache first
        cached_stats = await self.redis.get("dashboard_stats")
        if cached_stats:
            return DashboardStats.parse_raw(cached_stats)

        # Calculate fresh stats
        stats = self._calculate_dashboard_stats()

        # Cache for 30 seconds
        await self.redis.setex(
            "dashboard_stats",
            30,
            stats.json()
        )

        return stats

    def _calculate_dashboard_stats(self) -> DashboardStats:
        # Total ATMs
        total_atms = self.db.query(ATM).count()

        # Online ATMs (based on latest telemetry)
        online_atms = self.db.query(ATM).join(ATMTelemetry).filter(
            ATMTelemetry.status == "online",
            ATMTelemetry.time >= datetime.utcnow() - timedelta(minutes=5)
        ).distinct().count()

        # Error ATMs
        error_atms = self.db.query(ATM).join(ATMTelemetry).filter(
            ATMTelemetry.status == "error",
            ATMTelemetry.time >= datetime.utcnow() - timedelta(minutes=5)
        ).distinct().count()

        # Average cash level
        avg_cash_level = self.db.query(
            func.avg(ATMTelemetry.cash_level_percent)
        ).filter(
            ATMTelemetry.time >= datetime.utcnow() - timedelta(hours=1)
        ).scalar() or 0

        # Critical alerts
        critical_alerts = self.db.query(Alert).filter(
            Alert.severity == "critical",
            Alert.is_resolved == False
        ).count()

        # Transactions today
        today = datetime.utcnow().date()
        total_transactions = self.db.query(
            func.sum(ATMTelemetry.transactions_count)
        ).filter(
            func.date(ATMTelemetry.time) == today
        ).scalar() or 0

        return DashboardStats(
            total_atms=total_atms,
            online_atms=online_atms,
            offline_atms=total_atms - online_atms - error_atms,
            error_atms=error_atms,
            avg_cash_level=round(avg_cash_level, 1),
            critical_alerts=critical_alerts,
            total_transactions_today=total_transactions,
            last_updated=datetime.utcnow()
        )
```

### Telemetry Analytics

```python
# Advanced telemetry analytics
class TelemetryAnalytics:
    @staticmethod
    def get_atm_performance_metrics(
        db: Session,
        atm_id: str,
        hours: int = 24
    ) -> Dict:
        start_time = datetime.utcnow() - timedelta(hours=hours)

        # Query telemetry data
        telemetry_data = db.query(ATMTelemetry).filter(
            ATMTelemetry.atm_id == atm_id,
            ATMTelemetry.time >= start_time
        ).order_by(ATMTelemetry.time.desc()).all()

        if not telemetry_data:
            return {}

        # Calculate metrics
        uptime_percentage = len([t for t in telemetry_data if t.status == "online"]) / len(telemetry_data) * 100
        avg_temperature = statistics.mean([t.temperature_celsius for t in telemetry_data if t.temperature_celsius])
        avg_cpu_usage = statistics.mean([t.cpu_usage_percent for t in telemetry_data if t.cpu_usage_percent])
        avg_memory_usage = statistics.mean([t.memory_usage_percent for t in telemetry_data if t.memory_usage_percent])

        # Calculate trends
        cash_trend = TelemetryAnalytics._calculate_trend([
            t.cash_level_percent for t in telemetry_data[-10:] if t.cash_level_percent
        ])

        return {
            "uptime_percentage": round(uptime_percentage, 2),
            "avg_temperature": round(avg_temperature, 1),
            "avg_cpu_usage": round(avg_cpu_usage, 1),
            "avg_memory_usage": round(avg_memory_usage, 1),
            "cash_trend": cash_trend,
            "total_transactions": sum([t.transactions_count or 0 for t in telemetry_data]),
            "error_rate": len([t for t in telemetry_data if t.error_code]) / len(telemetry_data) * 100
        }

    @staticmethod
    def _calculate_trend(values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"

        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.05:
            return "increasing"
        elif second_avg < first_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
```

## 🔧 Development

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Show migration history
alembic history

# Show current revision
alembic current
```

### Development Commands

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html

# Code formatting
black .
isort .

# Linting
flake8
mypy .

# Generate OpenAPI schema
python -c "from main import app; import json; print(json.dumps(app.openapi()))" > openapi.json
```

### Docker Development

```bash
# Build development image
docker build -t atm-insights-backend .

# Run container
docker run -p 8000:8000 --env-file .env atm-insights-backend

# Development with volume mounting
docker run -p 8000:8000 -v $(pwd):/app --env-file .env atm-insights-backend

# Multi-stage build for production
docker build --target production -t atm-insights-backend:prod .
```

## 🚀 Production Deployment

### Environment Variables for Production

```bash
# Production .env
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=super-secure-secret-key-256-bits
DATABASE_URL=postgresql://user:pass@db-server:5432/atm_insights
REDIS_URL=redis://redis-server:6379/0

# Security
BACKEND_CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Performance
WORKERS=4
MAX_CONNECTIONS=100
POOL_SIZE=20

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
```

### Production Checklist

- [ ] **Environment Variables**: All production secrets configured
- [ ] **Database**: PostgreSQL with TimescaleDB extension installed
- [ ] **Redis**: Redis server configured and accessible
- [ ] **SSL/TLS**: HTTPS certificates configured
- [ ] **CORS**: Production origins whitelisted
- [ ] **Monitoring**: Error tracking and performance monitoring
- [ ] **Logging**: Structured logging with appropriate levels
- [ ] **Health Checks**: Load balancer health check endpoints
- [ ] **Database Migrations**: All migrations applied
- [ ] **Backup Strategy**: Database backup and recovery plan

### Performance Optimization

```python
# Database connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Redis connection pooling
redis_pool = redis.ConnectionPool.from_url(
    REDIS_URL,
    max_connections=20,
    retry_on_timeout=True
)

# Async database operations
async def get_telemetry_batch(atm_ids: List[str]) -> List[ATMTelemetry]:
    """Efficient batch telemetry retrieval"""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(ATMTelemetry)
            .where(ATMTelemetry.atm_id.in_(atm_ids))
            .order_by(ATMTelemetry.time.desc())
            .limit(1000)
        )
        return result.scalars().all()
```

## 🧪 Testing

### Test Structure

```bash
tests/
├── conftest.py              # Test configuration
├── test_api/               # API endpoint tests
│   ├── test_dashboard.py   # Dashboard API tests
│   ├── test_atms.py        # ATM API tests
│   └── test_telemetry.py   # Telemetry API tests
├── test_services/          # Service layer tests
│   ├── test_dashboard_service.py
│   └── test_telemetry_service.py
├── test_models/            # Model tests
└── test_utils/             # Utility function tests
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_api/test_dashboard.py

# Run tests with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_dashboard"
```

### Sample Test

```python
# tests/test_api/test_dashboard.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_dashboard_stats():
    """Test dashboard statistics endpoint"""
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200

    data = response.json()
    assert "total_atms" in data
    assert "online_atms" in data
    assert "critical_alerts" in data
    assert isinstance(data["total_atms"], int)

@pytest.mark.asyncio
async def test_telemetry_ingestion():
    """Test telemetry data ingestion"""
    telemetry_data = {
        "atm_id": "TEST-001",
        "status": "online",
        "temperature_celsius": 22.5,
        "cash_level_percent": 75.0,
        "transactions_count": 10
    }

    response = client.post("/api/v1/telemetry", json=telemetry_data)
    assert response.status_code == 200

    result = response.json()
    assert result["status"] == "accepted"
    assert "id" in result
```

## 📊 Monitoring & Observability

### Health Checks

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }

    # Database check
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    # Redis check
    try:
        redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    return health_status
```

### Logging Configuration

```python
import logging
import sys
from pythonjsonlogger import jsonlogger

# Configure structured logging
def setup_logging():
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    log_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO if not DEBUG else logging.DEBUG)

    return logger

# Usage in application
logger = setup_logging()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        "HTTP Request",
        extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "process_time": process_time,
            "client_ip": request.client.host
        }
    )

    return response
```

### Metrics Collection

```python
# Prometheus metrics integration
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

TELEMETRY_INGESTION_COUNT = Counter(
    'telemetry_messages_total',
    'Total telemetry messages ingested',
    ['atm_id', 'status']
)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# Middleware to collect metrics
@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(time.time() - start_time)

    return response
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd atm-insights/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Create feature branch
git checkout -b feature/amazing-feature
```

### Code Quality Standards

```bash
# Run all quality checks
./scripts/lint.sh

# Or run individually:
black --check .
isort --check-only .
flake8
mypy .
pytest
```

### API Design Guidelines

1. **RESTful Design**: Follow REST principles for resource endpoints
2. **Consistent Naming**: Use snake_case for JSON fields, kebab-case for URLs
3. **Error Handling**: Use appropriate HTTP status codes and error responses
4. **Validation**: Validate all inputs with Pydantic schemas
5. **Documentation**: Document all endpoints with docstrings and examples
6. **Versioning**: Use API versioning (e.g., `/api/v1/`)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI Team** - For the excellent web framework
- **TimescaleDB** - For superior time-series database capabilities
- **SQLAlchemy** - For robust ORM functionality
- **Pydantic** - For data validation and settings management
- **Redis** - For high-performance caching and messaging

---

## 🎯 Getting Started Now

**Ready to power your ATM monitoring system?**

```bash
git clone <repository-url>
cd atm-insights/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**🌐 Access Points:**

- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

**Need help?** Check the [Testing](#-testing) section or open an issue!

---

**⭐ Star this repository if you found it helpful!**
