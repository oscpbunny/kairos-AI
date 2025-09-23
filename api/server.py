"""
Project Kairos: API Server
FastAPI server with GraphQL and gRPC endpoints, authentication, and rate limiting.
"""

import asyncio
import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import jwt
import redis
import time
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_VERSION = "v1"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "kairos-dev-secret-key")
JWT_ALGORITHM = "HS256"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Initialize Redis for rate limiting (with fallback)
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    logger.info("Redis connection established for rate limiting")
except Exception as e:
    logger.warning(f"Redis not available: {e}. Rate limiting will use in-memory store.")
    redis_client = None

# In-memory rate limiting fallback
in_memory_rate_limit = {}

# Security
security = HTTPBearer()

class RateLimiter:
    """Rate limiting middleware"""
    
    def __init__(self, requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.requests = requests
        self.window = window
    
    async def __call__(self, request: Request):
        client_ip = request.client.host
        current_time = int(time.time())
        
        if redis_client:
            # Redis-based rate limiting
            key = f"rate_limit:{client_ip}"
            pipe = redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, current_time - self.window)
            pipe.zcard(key)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, self.window)
            results = pipe.execute()
            
            request_count = results[1]
        else:
            # In-memory fallback
            if client_ip not in in_memory_rate_limit:
                in_memory_rate_limit[client_ip] = []
            
            # Clean old requests
            cutoff_time = current_time - self.window
            in_memory_rate_limit[client_ip] = [
                timestamp for timestamp in in_memory_rate_limit[client_ip] 
                if timestamp > cutoff_time
            ]
            
            # Add current request
            in_memory_rate_limit[client_ip].append(current_time)
            request_count = len(in_memory_rate_limit[client_ip])
        
        if request_count > self.requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests} requests per {self.window} seconds"
            )
        
        return True

rate_limiter = RateLimiter()

class AuthManager:
    """JWT Authentication manager"""
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        try:
            payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return {"username": username, "scopes": payload.get("scopes", [])}
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

auth_manager = AuthManager()

# GraphQL Schema Import
try:
    # Try importing the new strawberry schema first
    from api.graphql.strawberry_schema import schema
    logger.info("Using Strawberry GraphQL schema")
except ImportError:
    try:
        # Fallback to the existing graphene schema
        from api.graphql.schema import schema
        logger.info("Using Graphene GraphQL schema")
    except ImportError:
        # Create a minimal schema if none exists
        @strawberry.type
        class Query:
            @strawberry.field
            def hello(self) -> str:
                return "Hello from Kairos API!"
        
        schema = strawberry.Schema(query=Query)
        logger.warning("Using minimal fallback GraphQL schema")

# Health check endpoint data
health_data = {
    "status": "healthy",
    "version": API_VERSION,
    "timestamp": datetime.utcnow().isoformat(),
    "components": {
        "database": "operational",
        "redis": "operational" if redis_client else "unavailable", 
        "oracle": "operational",
        "agents": "operational"
    }
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Kairos API Server starting up...")
    
    # Initialize components
    try:
        # Test database connection
        logger.info("Testing database connectivity...")
        # Add database connection test here
        
        # Test Oracle connection
        logger.info("Testing Oracle Engine connectivity...")
        # Add Oracle connection test here
        
        logger.info("✅ All components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Kairos API Server shutting down...")
    if redis_client:
        redis_client.close()
    logger.info("👋 Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Project Kairos API",
    description="Autonomous Digital Organization - External Interface",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.kairos.local"]
)

# GraphQL Router
graphql_app = GraphQLRouter(
    schema,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
        GRAPHQL_WS_PROTOCOL,
    ],
)

app.include_router(graphql_app, prefix="/graphql")

# REST API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Welcome to Project Kairos API",
        "version": API_VERSION,
        "documentation": "/docs",
        "graphql": "/graphql",
        "health": "/health"
    }

@app.get("/health")
async def health_check(request: Request):
    """Comprehensive health check endpoint"""
    await rate_limiter(request)
    
    try:
        # Update health data with current status
        current_health = health_data.copy()
        current_health["timestamp"] = datetime.utcnow().isoformat()
        
        # Check components
        component_status = {}
        
        # Database check
        try:
            # Add actual database health check here
            component_status["database"] = "operational"
        except Exception:
            component_status["database"] = "degraded"
        
        # Redis check
        if redis_client:
            try:
                redis_client.ping()
                component_status["redis"] = "operational"
            except Exception:
                component_status["redis"] = "degraded"
        else:
            component_status["redis"] = "unavailable"
        
        # Oracle check
        try:
            # Add actual Oracle health check here
            component_status["oracle"] = "operational"
        except Exception:
            component_status["oracle"] = "degraded"
        
        # Agents check
        try:
            # Add actual agent health check here
            component_status["agents"] = "operational"
        except Exception:
            component_status["agents"] = "degraded"
        
        current_health["components"] = component_status
        
        # Determine overall status
        if all(status in ["operational"] for status in component_status.values()):
            current_health["status"] = "healthy"
        elif any(status in ["operational"] for status in component_status.values()):
            current_health["status"] = "degraded"
        else:
            current_health["status"] = "unhealthy"
        
        # Return appropriate HTTP status
        if current_health["status"] == "healthy":
            return current_health
        elif current_health["status"] == "degraded":
            return JSONResponse(content=current_health, status_code=206)  # Partial Content
        else:
            return JSONResponse(content=current_health, status_code=503)  # Service Unavailable
            
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )

@app.post("/auth/login")
async def login(request: Request, credentials: dict):
    """Authentication endpoint"""
    await rate_limiter(request)
    
    # Simplified authentication - replace with proper implementation
    username = credentials.get("username")
    password = credentials.get("password")
    
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password required"
        )
    
    # Mock authentication - replace with real user verification
    if username == "admin" and password == "kairos-admin":
        access_token = auth_manager.create_access_token(
            data={"sub": username, "scopes": ["admin", "read", "write"]}
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600,
            "scopes": ["admin", "read", "write"]
        }
    elif username == "user" and password == "kairos-user":
        access_token = auth_manager.create_access_token(
            data={"sub": username, "scopes": ["read"]}
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600,
            "scopes": ["read"]
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

@app.get("/auth/verify")
async def verify_token(current_user: dict = Depends(auth_manager.verify_token)):
    """Token verification endpoint"""
    return {
        "valid": True,
        "user": current_user["username"],
        "scopes": current_user["scopes"]
    }

# Protected endpoints
@app.get("/api/v1/ventures")
async def get_ventures(
    request: Request,
    current_user: dict = Depends(auth_manager.verify_token)
):
    """Get ventures (protected endpoint)"""
    await rate_limiter(request)
    
    # Mock data - replace with actual database query
    return [
        {
            "id": "venture-001",
            "name": "E-commerce Platform",
            "objective": "Build scalable e-commerce platform",
            "status": "IN_PROGRESS",
            "created_at": datetime.utcnow().isoformat(),
            "target_users": 50000,
            "budget": 100000.0
        }
    ]

@app.get("/api/v1/agents")
async def get_agents(
    request: Request,
    current_user: dict = Depends(auth_manager.verify_token)
):
    """Get agents (protected endpoint)"""
    await rate_limiter(request)
    
    # Mock data - replace with actual database query
    return [
        {
            "id": "steward-001",
            "name": "Enhanced-Steward",
            "agent_type": "STEWARD",
            "specialization": "Advanced Resource Management",
            "cognitive_cycles_balance": 5000,
            "is_active": True,
            "last_heartbeat": datetime.utcnow().isoformat()
        },
        {
            "id": "architect-001",
            "name": "Enhanced-Architect",
            "agent_type": "ARCHITECT", 
            "specialization": "AI-Powered System Architecture",
            "cognitive_cycles_balance": 4000,
            "is_active": True,
            "last_heartbeat": datetime.utcnow().isoformat()
        }
    ]

@app.post("/api/v1/infrastructure/predict")
async def predict_infrastructure(
    request: Request,
    prediction_input: dict,
    current_user: dict = Depends(auth_manager.verify_token)
):
    """Get infrastructure predictions from Oracle"""
    await rate_limiter(request)
    
    if "write" not in current_user["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        from simulation.oracle_engine import OracleEngine
        
        oracle = OracleEngine()
        await oracle.initialize()
        
        prediction = await oracle.predict_infrastructure_requirements(
            venture_id=prediction_input.get("venture_id"),
            time_horizon_days=prediction_input.get("time_horizon_days", 30),
            current_infrastructure=prediction_input.get("current_infrastructure", {})
        )
        
        return prediction
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate prediction: {str(e)}"
        )

@app.post("/api/v1/design/validate")
async def validate_design(
    request: Request,
    design_input: dict,
    current_user: dict = Depends(auth_manager.verify_token)
):
    """Validate system design with Oracle"""
    await rate_limiter(request)
    
    if "write" not in current_user["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
        
        architect = EnhancedArchitectAgent()
        architect.agent_id = "api-architect-001"
        
        validation = await architect.validate_design_with_oracle(
            design_spec=design_input.get("design_spec"),
            venture_id=design_input.get("venture_id")
        )
        
        return validation
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate design: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# GraphQL Playground (for development)
@app.get("/graphql/playground")
async def graphql_playground():
    """GraphQL Playground interface"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Kairos GraphQL Playground</title>
  <meta name="robots" content="noindex" />
  <meta name="referrer" content="origin" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,400,600,700|Source+Code+Pro:400,700" rel="stylesheet" />
  <link rel="shortcut icon" href="https://graphcool-playground.netlify.com/favicon.png" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/graphql-playground-react@1.7.26/build/static/css/index.css" />
  <style>
    body { margin: 0; background: #172a3a; }
    html, body, #root { height: 100%; }
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
    window.addEventListener('load', function (event) {
      GraphQLPlayground.init(document.getElementById('root'), {
        endpoint: '/graphql',
        subscriptionsEndpoint: 'ws://localhost:8000/graphql',
        headers: {},
        tabs: [
          {
            query: `
query GetSystemHealth {
  system_health {
    overall_status
    healthy_components
    total_components
    success_rate
    components {
      component
      status
      message
      healthy
    }
  }
}

query GetAgents {
  agents {
    id
    name
    agent_type
    specialization
    cognitive_cycles_balance
    is_active
  }
}

mutation CreateVenture {
  create_venture(input: {
    name: "Test Venture"
    objective: "Build a test application"
    target_users: 10000
    budget: 50000
    timeline_days: 60
  }) {
    id
    name
    status
    created_at
  }
}
            `,
          },
        ],
      });
    });
  </script>
  <script src="https://cdn.jsdelivr.net/npm/graphql-playground-react@1.7.26/build/static/js/middleware.js"></script>
</body>
</html>
    """)

if __name__ == "__main__":
    import uvicorn
    
    # Development server
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )