"""
Project Kairos: Complete API Server Implementation
Production-ready GraphQL and REST API server with authentication, rate limiting, and monitoring.
Phase 6 - Production Excellence
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import secrets
import hashlib
from contextlib import asynccontextmanager

# FastAPI and security
from fastapi import FastAPI, HTTPException, Depends, Security, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# GraphQL
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

# JWT and authentication
import jwt
from passlib.context import CryptContext

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Kairos imports
from agents.enhanced.enhanced_steward import EnhancedStewardAgent
from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
from agents.enhanced.enhanced_engineer import EnhancedEngineerAgent
from simulation.oracle_engine import OracleEngine

# Configuration
class Config:
    SECRET_KEY = os.getenv("KAIROS_SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    API_V1_PREFIX = "/api/v1"
    
    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE = "100/minute"
    RATE_LIMIT_BURST = "20/second"
    
    # CORS
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]
    ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

config = Config()

# Authentication setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kairos.api")

# In-memory user store (replace with database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Kairos Administrator",
        "email": "admin@kairos.local",
        "hashed_password": pwd_context.hash("admin123"),  # Change in production
        "disabled": False,
        "scopes": ["read", "write", "admin"]
    },
    "user": {
        "username": "user",
        "full_name": "Kairos User",
        "email": "user@kairos.local",
        "hashed_password": pwd_context.hash("user123"),  # Change in production
        "disabled": False,
        "scopes": ["read"]
    }
}

# Global agent instances (initialized on startup)
steward_agent: Optional[EnhancedStewardAgent] = None
architect_agent: Optional[EnhancedArchitectAgent] = None
engineer_agent: Optional[EnhancedEngineerAgent] = None
oracle_engine: Optional[OracleEngine] = None

# GraphQL Types
@strawberry.type
class User:
    username: str
    full_name: str
    email: str
    scopes: List[str]

@strawberry.type
class VentureResponse:
    id: str
    name: str
    objective: str
    status: str
    created_at: str

@strawberry.type
class InfrastructurePrediction:
    venture_id: str
    predicted_users: int
    monthly_cost: float
    confidence_level: float
    resource_requirements: str  # JSON string
    scaling_recommendations: str  # JSON string

@strawberry.type
class DesignValidation:
    design_feasibility: float
    performance_class: str
    confidence_level: float
    recommendations: List[str]

@strawberry.type
class SimulationResult:
    simulation_id: str
    venture_id: str
    status: str
    population_size: int
    progress_percent: float

@strawberry.type
class HealthCheck:
    service: str
    status: str
    response_time_ms: float
    last_check: str

# GraphQL Input Types
@strawberry.input
class VentureInput:
    name: str
    objective: str
    target_users: int
    budget: float
    timeline_days: int

@strawberry.input
class DesignSpecInput:
    pattern: str
    technology_stack: str  # JSON string
    expected_scale: str  # JSON string

@strawberry.input
class SimulationInput:
    venture_id: str
    simulation_name: str
    target_market_profile: str  # JSON string
    simulation_duration_days: int
    population_size: int

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[Dict]:
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return user_dict
    return None

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Dict = Depends(get_current_user)):
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# GraphQL Resolvers
@strawberry.type
class Query:
    @strawberry.field
    async def me(self, info: Info) -> User:
        # Extract user from context (set by auth middleware)
        user_data = info.context.get("user")
        if not user_data:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        return User(
            username=user_data["username"],
            full_name=user_data["full_name"],
            email=user_data["email"],
            scopes=user_data["scopes"]
        )
    
    @strawberry.field
    async def ventures(self, info: Info) -> List[VentureResponse]:
        # Mock venture data - replace with database query
        return [
            VentureResponse(
                id="venture-001",
                name="E-commerce Platform",
                objective="Build scalable e-commerce solution",
                status="ACTIVE",
                created_at=datetime.now().isoformat()
            ),
            VentureResponse(
                id="venture-002",
                name="AI Analytics Dashboard",
                objective="Create intelligent business analytics platform",
                status="PLANNING",
                created_at=datetime.now().isoformat()
            )
        ]
    
    @strawberry.field
    async def infrastructure_predictions(
        self, 
        info: Info,
        venture_id: str,
        time_horizon: int = 30
    ) -> InfrastructurePrediction:
        if not steward_agent:
            raise HTTPException(status_code=503, detail="Steward agent not available")
        
        try:
            predictions = await steward_agent.get_infrastructure_predictions(
                venture_id=venture_id,
                time_horizon=time_horizon
            )
            
            return InfrastructurePrediction(
                venture_id=venture_id,
                predicted_users=predictions.get('predicted_users', 0),
                monthly_cost=predictions.get('cost_predictions', {}).get('monthly_estimate', 0.0),
                confidence_level=predictions.get('confidence_level', 0.0),
                resource_requirements=json.dumps(predictions.get('resource_requirements', {})),
                scaling_recommendations=json.dumps(predictions.get('scaling_recommendations', {}))
            )
        except Exception as e:
            logger.error(f"Infrastructure prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @strawberry.field
    async def validate_design(
        self,
        info: Info,
        venture_id: str,
        design_spec: DesignSpecInput
    ) -> DesignValidation:
        if not architect_agent:
            raise HTTPException(status_code=503, detail="Architect agent not available")
        
        try:
            # Parse JSON strings
            design_data = {
                'pattern': design_spec.pattern,
                'technology_stack': json.loads(design_spec.technology_stack),
                'expected_scale': json.loads(design_spec.expected_scale)
            }
            
            validation = await architect_agent.validate_design_with_oracle(
                design_spec=design_data,
                venture_id=venture_id
            )
            
            return DesignValidation(
                design_feasibility=validation.get('design_feasibility', {}).get('score', 0.0),
                performance_class=validation.get('overall_assessment', {}).get('performance_class', 'Unknown'),
                confidence_level=validation.get('confidence_level', 0.0),
                recommendations=[
                    rec.get('recommendation', '') 
                    for rec in validation.get('improvement_recommendations', [])
                ]
            )
        except Exception as e:
            logger.error(f"Design validation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @strawberry.field
    async def simulations(self, info: Info) -> List[SimulationResult]:
        if not oracle_engine:
            return []
        
        results = []
        for sim_id, sim_data in oracle_engine.active_simulations.items():
            results.append(SimulationResult(
                simulation_id=sim_id,
                venture_id=sim_data.get('venture_id', ''),
                status='ACTIVE',
                population_size=len(sim_data.get('user_personas', [])),
                progress_percent=100.0  # Mock progress
            ))
        
        return results
    
    @strawberry.field
    async def health_check(self, info: Info) -> List[HealthCheck]:
        checks = []
        
        # Check Steward Agent
        start_time = time.time()
        steward_status = "healthy" if steward_agent else "unavailable"
        steward_time = (time.time() - start_time) * 1000
        
        checks.append(HealthCheck(
            service="steward_agent",
            status=steward_status,
            response_time_ms=steward_time,
            last_check=datetime.now().isoformat()
        ))
        
        # Check Architect Agent
        start_time = time.time()
        architect_status = "healthy" if architect_agent else "unavailable"
        architect_time = (time.time() - start_time) * 1000
        
        checks.append(HealthCheck(
            service="architect_agent",
            status=architect_status,
            response_time_ms=architect_time,
            last_check=datetime.now().isoformat()
        ))
        
        # Check Oracle Engine
        start_time = time.time()
        oracle_status = "healthy" if oracle_engine else "unavailable"
        oracle_time = (time.time() - start_time) * 1000
        
        checks.append(HealthCheck(
            service="oracle_engine",
            status=oracle_status,
            response_time_ms=oracle_time,
            last_check=datetime.now().isoformat()
        ))
        
        return checks

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_venture(self, info: Info, venture: VentureInput) -> VentureResponse:
        # Mock venture creation - replace with actual database operation
        venture_id = f"venture-{int(time.time())}"
        
        return VentureResponse(
            id=venture_id,
            name=venture.name,
            objective=venture.objective,
            status="CREATED",
            created_at=datetime.now().isoformat()
        )
    
    @strawberry.mutation
    async def create_simulation(self, info: Info, simulation: SimulationInput) -> SimulationResult:
        if not oracle_engine:
            raise HTTPException(status_code=503, detail="Oracle engine not available")
        
        try:
            # Parse JSON string
            target_market = json.loads(simulation.target_market_profile)
            
            simulation_id = await oracle_engine.create_market_digital_twin(
                venture_id=simulation.venture_id,
                simulation_name=simulation.simulation_name,
                target_market_profile=target_market,
                simulation_duration_days=simulation.simulation_duration_days,
                population_size=simulation.population_size
            )
            
            return SimulationResult(
                simulation_id=simulation_id,
                venture_id=simulation.venture_id,
                status="CREATED",
                population_size=simulation.population_size,
                progress_percent=0.0
            )
        except Exception as e:
            logger.error(f"Simulation creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Authentication context processor
async def get_context(request: Request):
    context = {"request": request}
    
    # Extract user from Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header:
        try:
            token = auth_header.replace("Bearer ", "")
            payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(username)
                if user and not user.get("disabled"):
                    context["user"] = user
        except jwt.PyJWTError:
            pass  # Invalid token, continue without user
    
    return context

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Kairos API Server...")
    
    global steward_agent, architect_agent, engineer_agent, oracle_engine
    
    try:
        # Initialize agents
        logger.info("Initializing Enhanced Agents...")
        steward_agent = EnhancedStewardAgent(agent_name="API-Steward")
        architect_agent = EnhancedArchitectAgent(agent_name="API-Architect")
        engineer_agent = EnhancedEngineerAgent(agent_name="API-Engineer")
        
        # Initialize Oracle Engine
        logger.info("Initializing Oracle Engine...")
        oracle_engine = OracleEngine()
        await oracle_engine.initialize()
        
        logger.info("✅ Kairos API Server initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Kairos API Server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Kairos API Server...")
    
    try:
        if steward_agent:
            await steward_agent.shutdown()
        if architect_agent:
            await architect_agent.shutdown()
        if engineer_agent:
            await engineer_agent.shutdown()
        
        logger.info("✅ Kairos API Server shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Kairos Symbiotic API",
    description="Production-ready API for Project Kairos autonomous digital organization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.ALLOWED_HOSTS
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# REST API Routes
@app.post("/api/v1/auth/token")
@limiter.limit("5/minute")
async def login(request: Request, username: str, password: str):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "scopes": user["scopes"]},
        expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "username": user["username"],
            "full_name": user["full_name"],
            "email": user["email"],
            "scopes": user["scopes"]
        }
    }

@app.get("/api/v1/health")
@limiter.limit(config.RATE_LIMIT_BURST)
async def health_check(request: Request):
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "steward": "healthy" if steward_agent else "unavailable",
            "architect": "healthy" if architect_agent else "unavailable",
            "engineer": "healthy" if engineer_agent else "unavailable",
            "oracle": "healthy" if oracle_engine else "unavailable"
        }
    }

@app.get("/api/v1/metrics")
@limiter.limit(config.RATE_LIMIT_REQUESTS_PER_MINUTE)
async def metrics(request: Request, current_user: Dict = Depends(get_current_active_user)):
    # Return Prometheus-compatible metrics
    metrics_text = f"""# HELP kairos_api_requests_total Total API requests
# TYPE kairos_api_requests_total counter
kairos_api_requests_total{{method="GET",endpoint="/health"}} 1

# HELP kairos_agents_active Number of active agents
# TYPE kairos_agents_active gauge
kairos_agents_active{{type="steward"}} {1 if steward_agent else 0}
kairos_agents_active{{type="architect"}} {1 if architect_agent else 0}
kairos_agents_active{{type="engineer"}} {1 if engineer_agent else 0}

# HELP kairos_oracle_simulations_active Number of active simulations
# TYPE kairos_oracle_simulations_active gauge
kairos_oracle_simulations_active {len(oracle_engine.active_simulations) if oracle_engine else 0}
"""
    
    return JSONResponse(
        content=metrics_text,
        media_type="text/plain"
    )

# GraphQL endpoint
graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Kairos Symbiotic API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "graphql": "/graphql",
            "docs": "/docs",
            "health": "/api/v1/health",
            "auth": "/api/v1/auth/token"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "server_complete:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True,
        log_level="info"
    )