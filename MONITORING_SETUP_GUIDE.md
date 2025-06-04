# 📊 ATM Insights - Monitoring Setup Guide

This guide will help you set up comprehensive monitoring for your ATM Insights system using Prometheus and Grafana.

## 📁 Directory Structure

First, create the required directory structure for monitoring configuration:

```bash
# Create monitoring directories
mkdir -p monitoring/prometheus/rules
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards
```

Your project structure should look like this:

```
atm-insights/
├── backend/
├── frontend/
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/
│   │       └── atm_insights_rules.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── datasources.yml
│       │   └── dashboards/
│       │       └── dashboards.yml
│       └── dashboards/
│           └── atm-insights-overview.json
├── docker-compose.yml
├── Makefile
└── .env
```

## 📄 Configuration Files

### 1. Save Grafana Dashboard JSON

**Location**: `monitoring/grafana/dashboards/atm-insights-overview.json`

Save the Grafana dashboard JSON configuration to this file. This dashboard will be automatically loaded when Grafana starts.

### 2. Prometheus Configuration

**Location**: `monitoring/prometheus/prometheus.yml`

This file contains the Prometheus scraping configuration for all services.

### 3. Prometheus Alert Rules

**Location**: `monitoring/prometheus/rules/atm_insights_rules.yml`

Contains alerting rules for various system conditions.

### 4. Grafana Datasource Configuration

**Location**: `monitoring/grafana/provisioning/datasources/datasources.yml`

Automatically configures Prometheus, PostgreSQL, and Redis datasources.

### 5. Grafana Dashboard Provisioning

**Location**: `monitoring/grafana/provisioning/dashboards/dashboards.yml`

Tells Grafana where to find dashboard JSON files.

## 🚀 Quick Setup

### Option 1: Automatic Setup (Recommended)

```bash
# Run the automated setup
make setup
```

This will:

1. Create all necessary directories
2. Build Docker images
3. Start all services with monitoring

### Option 2: Manual Setup

```bash
# 1. Create directories
make setup-monitoring

# 2. Create configuration files (save the artifacts provided above)

# 3. Build and start services
make build
make monitoring
```

## 🐳 Docker Commands

### Start Monitoring Stack

```bash
# Start everything including monitoring
make monitoring

# Start only monitoring services (assumes core services running)
make monitoring-only

# Start development environment
make dev

# Start production environment
make prod
```

### Stop Services

```bash
# Stop all services
make down

# Stop only monitoring services
make monitoring-down

# Clean everything (including volumes)
make clean
```

## 🌐 Access Points

After starting the monitoring stack, you can access:

| Service                | URL                           | Credentials             |
| ---------------------- | ----------------------------- | ----------------------- |
| **Grafana**            | http://localhost:3000         | admin/admin             |
| **Prometheus**         | http://localhost:9090         | None                    |
| **API Documentation**  | http://localhost:8000/docs    | None                    |
| **API Health Check**   | http://localhost:8000/health  | None                    |
| **Prometheus Metrics** | http://localhost:8000/metrics | None                    |
| **PgAdmin** (dev only) | http://localhost:5050         | admin@example.com/admin |

### Exporters (for advanced users)

| Exporter                | URL                   | Purpose          |
| ----------------------- | --------------------- | ---------------- |
| **Node Exporter**       | http://localhost:9100 | System metrics   |
| **Redis Exporter**      | http://localhost:9121 | Redis metrics    |
| **PostgreSQL Exporter** | http://localhost:9187 | Database metrics |

## 📊 Available Dashboards

### 1. ATM Insights - System Overview

**File**: `monitoring/grafana/dashboards/atm-insights-overview.json`

This dashboard includes:

- **HTTP Request Rate**: Real-time API request metrics
- **Total ATMs**: Current number of ATMs in the system
- **ATM Status Distribution**: Pie chart showing ATM statuses
- **Telemetry Messages Rate**: Rate of incoming telemetry data
- **Active WebSocket Connections**: Live connection count
- **HTTP Response Time**: 95th and 50th percentile response times
- **System Resources**: CPU and memory usage

### 2. Infrastructure Metrics

Automatically available through Prometheus:

- System CPU, memory, disk usage
- Database connection pools
- Redis performance metrics
- Network metrics

## 🔧 Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Monitoring Configuration
PROMETHEUS_ENABLED=true
METRICS_ENABLED=true
GRAFANA_USER=admin
GRAFANA_PASSWORD=your_secure_password

# Optional: Custom retention
PROMETHEUS_RETENTION_TIME=30d
PROMETHEUS_RETENTION_SIZE=10GB
```

### Backend Metrics Integration

The monitoring setup automatically includes:

- **HTTP Request Metrics**: Request count, duration, status codes
- **Telemetry Metrics**: Message processing rates and counts
- **ATM Status Metrics**: Real-time ATM status distribution
- **WebSocket Metrics**: Active connection counts
- **Database Metrics**: Connection pool usage
- **Cache Metrics**: Redis operation statistics
- **Alert Metrics**: Alert triggering rates

## 📈 Custom Metrics

### Adding Custom Metrics

To add custom metrics to your application:

```python
# In your service files
from services.metrics_service import metrics_service

# Record custom events
metrics_service.record_telemetry_message(atm_id, status)
metrics_service.record_alert(severity, rule_type)
metrics_service.update_atm_status_counts(status_counts)
```

### Available Metric Types

1. **Counters**: Always increasing values (requests, errors)
2. **Gauges**: Current values (connections, queue size)
3. **Histograms**: Distribution of values (response times)

## 🚨 Alerting

### Built-in Alert Rules

The system includes alerts for:

- **High Error Rate**: >5% error rate for 5 minutes
- **High Response Time**: >1 second 95th percentile
- **Service Down**: Any service unavailable for 1 minute
- **High Resource Usage**: CPU >80%, Memory >85%, Disk >85%
- **Database Issues**: Connection problems, too many connections
- **Redis Issues**: High memory usage, unavailability

### Adding Custom Alerts

Edit `monitoring/prometheus/rules/atm_insights_rules.yml`:

```yaml
- alert: CustomAlert
  expr: your_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Custom alert description"
    description: "Detailed description with {{ $value }}"
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Services Not Starting

```bash
# Check service status
make status

# View logs
make logs

# Check specific service
make logs-backend
make logs-prometheus
make logs-grafana
```

#### 2. Grafana Dashboard Not Loading

```bash
# Check if dashboard file exists
ls -la monitoring/grafana/dashboards/

# Check Grafana logs
make logs-grafana

# Restart Grafana
docker-compose restart grafana
```

#### 3. Prometheus Not Scraping Metrics

```bash
# Check Prometheus targets
open http://localhost:9090/targets

# Check if backend metrics endpoint is accessible
curl http://localhost:8000/metrics

# Verify Prometheus config
docker-compose exec prometheus cat /etc/prometheus/prometheus.yml
```

#### 4. Missing Metrics Data

```bash
# Check if metrics are enabled
curl http://localhost:8000/health

# Verify backend configuration
docker-compose exec backend env | grep PROMETHEUS
```

### Health Checks

```bash
# Run automated health checks
make health

# Check individual services
curl http://localhost:8000/health
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health
```

## 🔄 Maintenance

### Data Retention

- **Prometheus**: 30 days (configurable)
- **Grafana**: Dashboard configs persist in volumes
- **Database**: TimescaleDB automatic compression

### Backup

```bash
# Backup Grafana dashboards
docker-compose exec grafana tar -czf /tmp/dashboards.tar.gz /var/lib/grafana/dashboards
docker cp grafana:/tmp/dashboards.tar.gz ./backup/

# Backup Prometheus data
docker-compose exec prometheus tar -czf /tmp/prometheus.tar.gz /prometheus
docker cp prometheus:/tmp/prometheus.tar.gz ./backup/
```

### Updates

```bash
# Update monitoring stack
docker-compose --profile monitoring pull
docker-compose --profile monitoring up -d
```

## 📚 Advanced Configuration

### Custom Datasources

Add additional datasources to `monitoring/grafana/provisioning/datasources/datasources.yml`:

```yaml
- name: InfluxDB
  type: influxdb
  url: http://influxdb:8086
  database: atm_insights
```

### Dashboard Development

1. **Create in UI**: Design dashboards in Grafana UI
2. **Export JSON**: Dashboard Settings → JSON Model → Copy
3. **Save to File**: Save to `monitoring/grafana/dashboards/`
4. **Restart**: `docker-compose restart grafana`

### Prometheus Federation

For multi-cluster setups:

```yaml
# In prometheus.yml
- job_name: "federate"
  scrape_interval: 15s
  honor_labels: true
  metrics_path: "/federate"
  params:
    "match[]":
      - '{job="atm-insights-backend"}'
  static_configs:
    - targets:
        - "other-prometheus:9090"
```

## 🎯 Performance Optimization

### Prometheus

```yaml
# Reduce scrape intervals for less critical metrics
- job_name: "node-exporter"
  scrape_interval: 60s # Instead of 15s

# Use recording rules for expensive queries
- record: instance:cpu_usage:rate5m
  expr: 100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

### Grafana

- Use dashboard variables for filtering
- Limit time ranges for expensive queries
- Use query caching where possible
- Optimize panel refresh rates

## 🚀 Production Deployment

### Security Considerations

1. **Change Default Passwords**:

   ```env
   GRAFANA_PASSWORD=secure_password_here
   ```

2. **Enable Authentication**:

   ```yaml
   # In docker-compose.yml
   environment:
     - GF_AUTH_ANONYMOUS_ENABLED=false
     - GF_AUTH_BASIC_ENABLED=true
   ```

3. **Use HTTPS**:
   ```yaml
   # Add SSL certificates
   volumes:
     - ./ssl:/etc/ssl/certs
   ```

### Scaling

For high-traffic environments:

1. **Multiple Prometheus Instances**: Use federation
2. **Grafana Clustering**: Configure multiple Grafana instances
3. **Storage Optimization**: Use remote storage for Prometheus

## 📞 Support

### Getting Help

1. **Check Logs**: `make logs`
2. **Verify Config**: Use `make health`
3. **Monitor Status**: `make status`
4. **Access Endpoints**: `make endpoints`

### Useful Commands

```bash
# Quick status check
make health

# View all monitoring endpoints
make endpoints

# Complete system restart
make down && make monitoring

# Reset everything
make clean && make setup
```

---

🎉 **Congratulations!** Your ATM Insights system now has comprehensive monitoring with Prometheus and Grafana!
