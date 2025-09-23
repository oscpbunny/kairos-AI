#!/bin/bash
# Project Kairos: EC2 Instance Bootstrap Script
# Auto-configures new instances with the Kairos application stack

set -e

# Variables from Terraform
DB_HOST="${db_host}"
DB_NAME="${db_name}"
DB_USER="${db_user}"
DB_PASSWORD="${db_password}"
ENVIRONMENT="${environment}"

# System configuration
KAIROS_USER="kairos"
KAIROS_HOME="/opt/kairos"
LOG_FILE="/var/log/kairos-bootstrap.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "Starting Kairos instance bootstrap for environment: $ENVIRONMENT"

# Update system packages
log "Updating system packages..."
yum update -y

# Install essential packages
log "Installing essential packages..."
yum install -y \
    git \
    docker \
    python3 \
    python3-pip \
    postgresql15 \
    redis \
    awscli \
    cloudwatch-agent \
    htop \
    vim \
    curl \
    wget \
    unzip

# Install Docker Compose
log "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Start and enable services
log "Starting system services..."
systemctl start docker
systemctl enable docker
systemctl start postgresql
systemctl enable postgresql

# Create kairos user
log "Creating kairos user..."
useradd -m -s /bin/bash $KAIROS_USER
usermod -aG docker $KAIROS_USER

# Create application directory
log "Setting up application directories..."
mkdir -p $KAIROS_HOME/{app,logs,data,config}
chown -R $KAIROS_USER:$KAIROS_USER $KAIROS_HOME

# Install Python dependencies
log "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install \
    fastapi \
    uvicorn \
    asyncpg \
    redis \
    pydantic \
    prometheus_client \
    boto3 \
    psycopg2-binary \
    sqlalchemy \
    alembic

# Setup application configuration
log "Creating application configuration..."
cat > $KAIROS_HOME/config/app.env << EOF
# Kairos Application Configuration
ENVIRONMENT=$ENVIRONMENT
DB_HOST=$DB_HOST
DB_NAME=$DB_NAME  
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
DB_PORT=5432

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Application Settings
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

# AWS Settings
AWS_DEFAULT_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EOF

chown $KAIROS_USER:$KAIROS_USER $KAIROS_HOME/config/app.env

# Create systemd service for Kairos application
log "Creating systemd service..."
cat > /etc/systemd/system/kairos-app.service << EOF
[Unit]
Description=Kairos Application
After=network.target docker.service postgresql.service
Wants=docker.service postgresql.service

[Service]
Type=exec
User=$KAIROS_USER
Group=$KAIROS_USER
WorkingDirectory=$KAIROS_HOME/app
EnvironmentFile=$KAIROS_HOME/config/app.env
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=kairos-app

[Install]
WantedBy=multi-user.target
EOF

# Setup CloudWatch Agent
log "Configuring CloudWatch Agent..."
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "cwagent"
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/kairos-bootstrap.log",
                        "log_group_name": "/aws/ec2/kairos-system-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/bootstrap"
                    },
                    {
                        "file_path": "/var/log/messages",
                        "log_group_name": "/aws/ec2/kairos-system-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/system"
                    },
                    {
                        "file_path": "$KAIROS_HOME/logs/app.log",
                        "log_group_name": "/aws/ec2/kairos-app-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/application"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "Kairos/$ENVIRONMENT",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60,
                "totalcpu": false
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "netstat": {
                "measurement": [
                    "tcp_established",
                    "tcp_time_wait"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Start CloudWatch Agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# Setup health check endpoint
log "Creating health check script..."
cat > $KAIROS_HOME/app/health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Kairos Health Check Endpoint
Provides comprehensive health status for load balancer health checks
"""

import json
import sys
import time
import subprocess
import psycopg2
import redis
from datetime import datetime

def check_database():
    """Check database connectivity"""
    try:
        import os
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', 5432),
            connect_timeout=5
        )
        cursor = conn.cursor()
        cursor.execute('SELECT 1;')
        cursor.fetchone()
        conn.close()
        return True, "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

def check_redis():
    """Check Redis connectivity"""
    try:
        import os
        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            socket_connect_timeout=5
        )
        r.ping()
        return True, "Redis connection successful"
    except Exception as e:
        return False, f"Redis connection failed: {str(e)}"

def check_disk_space():
    """Check available disk space"""
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 5:
                usage_percent = parts[4].replace('%', '')
                if int(usage_percent) > 90:
                    return False, f"Disk usage too high: {usage_percent}%"
                return True, f"Disk usage OK: {usage_percent}%"
        return False, "Could not determine disk usage"
    except Exception as e:
        return False, f"Disk check failed: {str(e)}"

def main():
    """Main health check function"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {},
        "environment": os.getenv('ENVIRONMENT', 'unknown'),
        "instance_id": os.getenv('INSTANCE_ID', 'unknown')
    }
    
    overall_healthy = True
    
    # Database check
    db_healthy, db_message = check_database()
    health_status["checks"]["database"] = {
        "healthy": db_healthy,
        "message": db_message
    }
    if not db_healthy:
        overall_healthy = False
    
    # Redis check
    redis_healthy, redis_message = check_redis()
    health_status["checks"]["redis"] = {
        "healthy": redis_healthy,
        "message": redis_message
    }
    if not redis_healthy:
        overall_healthy = False
    
    # Disk space check
    disk_healthy, disk_message = check_disk_space()
    health_status["checks"]["disk"] = {
        "healthy": disk_healthy,
        "message": disk_message
    }
    if not disk_healthy:
        overall_healthy = False
    
    # Update overall status
    if not overall_healthy:
        health_status["status"] = "unhealthy"
    
    # Output JSON
    print("Content-Type: application/json")
    print()
    print(json.dumps(health_status, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if overall_healthy else 1)

if __name__ == "__main__":
    main()
EOF

chmod +x $KAIROS_HOME/app/health_check.py
chown $KAIROS_USER:$KAIROS_USER $KAIROS_HOME/app/health_check.py

# Create a simple FastAPI health endpoint
log "Creating FastAPI health endpoint..."
cat > $KAIROS_HOME/app/main.py << 'EOF'
"""
Kairos Application Main Entry Point
Simple FastAPI application for load balancer health checks
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
import subprocess
import sys
from datetime import datetime

app = FastAPI(
    title="Kairos Application",
    description="Kairos Autonomous Digital Organization",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Run the health check script
        result = subprocess.run([
            sys.executable, 
            "/opt/kairos/app/health_check.py"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Parse JSON output
            import json
            health_data = json.loads(result.stdout.split('\n')[-2])  # Get JSON line
            return JSONResponse(content=health_data, status_code=200)
        else:
            raise HTTPException(status_code=503, detail="Health check failed")
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=503, detail="Health check timeout")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Kairos Autonomous Digital Organization",
        "status": "operational",
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "instance_id": os.getenv("INSTANCE_ID", "unknown"),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/info")
async def info():
    """System information endpoint"""
    return {
        "application": "Kairos",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "instance_id": os.getenv("INSTANCE_ID", "unknown"),
        "python_version": sys.version,
        "timestamp": datetime.utcnow().isoformat()
    }
EOF

chown $KAIROS_USER:$KAIROS_USER $KAIROS_HOME/app/main.py

# Create log rotation configuration
log "Setting up log rotation..."
cat > /etc/logrotate.d/kairos << EOF
$KAIROS_HOME/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su $KAIROS_USER $KAIROS_USER
}
EOF

# Setup monitoring scripts
log "Creating monitoring scripts..."
cat > $KAIROS_HOME/scripts/monitor.sh << 'EOF'
#!/bin/bash
# Kairos Monitoring Script
# Collects and reports system metrics

METRICS_FILE="/tmp/kairos-metrics.txt"
TIMESTAMP=$(date +%s)

# CPU Usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

# Memory Usage  
MEM_TOTAL=$(free -m | awk 'NR==2{printf "%.2f", $3*100/$2}')

# Disk Usage
DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | cut -d'%' -f1)

# Load Average
LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | cut -d',' -f1)

# Output metrics
echo "kairos.cpu.usage $CPU_USAGE $TIMESTAMP" > $METRICS_FILE
echo "kairos.memory.usage $MEM_TOTAL $TIMESTAMP" >> $METRICS_FILE
echo "kairos.disk.usage $DISK_USAGE $TIMESTAMP" >> $METRICS_FILE
echo "kairos.system.load $LOAD_AVG $TIMESTAMP" >> $METRICS_FILE

# Send to CloudWatch (optional)
aws cloudwatch put-metric-data \
    --region $(curl -s http://169.254.169.254/latest/meta-data/placement/region) \
    --namespace "Kairos/Custom" \
    --metric-data file://$METRICS_FILE \
    2>/dev/null || true
EOF

chmod +x $KAIROS_HOME/scripts/monitor.sh
chown -R $KAIROS_USER:$KAIROS_USER $KAIROS_HOME/scripts

# Add monitoring to crontab
echo "*/5 * * * * $KAIROS_USER $KAIROS_HOME/scripts/monitor.sh" >> /etc/crontab

# Enable and start services
log "Enabling and starting services..."
systemctl daemon-reload
systemctl enable kairos-app
systemctl start kairos-app

# Wait for application to be ready
log "Waiting for application to be ready..."
sleep 30

# Test health endpoint
log "Testing health endpoint..."
for i in {1..10}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "Health endpoint responding successfully"
        break
    fi
    log "Health endpoint not ready, attempt $i/10"
    sleep 10
done

# Final system status
log "Bootstrap completed. System status:"
systemctl status kairos-app --no-pager -l

# Send completion notification to CloudWatch
aws logs create-log-stream \
    --log-group-name "/aws/ec2/kairos-system-$ENVIRONMENT" \
    --log-stream-name "$(curl -s http://169.254.169.254/latest/meta-data/instance-id)/bootstrap" \
    2>/dev/null || true

aws logs put-log-events \
    --log-group-name "/aws/ec2/kairos-system-$ENVIRONMENT" \
    --log-stream-name "$(curl -s http://169.254.169.254/latest/meta-data/instance-id)/bootstrap" \
    --log-events timestamp=$(date +%s000),message="Bootstrap completed successfully for $ENVIRONMENT environment" \
    2>/dev/null || true

log "Kairos instance bootstrap completed successfully!"
echo "Bootstrap completed at $(date)" >> /opt/kairos/BOOTSTRAP_COMPLETE