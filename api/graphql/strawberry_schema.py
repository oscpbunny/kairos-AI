"""
Project Kairos: Strawberry GraphQL Schema
Modern async GraphQL schema using Strawberry for FastAPI integration.
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, AsyncGenerator
from enum import Enum

import strawberry
from strawberry.types import Info

# Import existing models and utilities
try:
    from core.models import Venture, Agent, Decision, Task
    from simulation.oracle_engine import OracleEngine
    from agents.enhanced.enhanced_steward import EnhancedStewardAgent
    from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
    from monitoring.health_checks import SystemHealthChecker
except ImportError as e:
    print(f"Warning: Some imports unavailable for GraphQL schema: {e}")

# Enums
@strawberry.enum
class AgentType(Enum):
    STEWARD = "STEWARD"
    ARCHITECT = "ARCHITECT"
    ENGINEER = "ENGINEER"
    ORACLE = "ORACLE"

@strawberry.enum
class VentureStatus(Enum):
    PLANNING = "PLANNING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"

@strawberry.enum
class TaskStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@strawberry.enum
class DecisionOutcome(Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"

@strawberry.enum
class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"

# Input Types
@strawberry.input
class VentureInput:
    name: str
    objective: str
    target_users: Optional[int] = 1000
    budget: Optional[float] = 10000.0
    timeline_days: Optional[int] = 30

@strawberry.input
class TaskInput:
    title: str
    description: str
    venture_id: str
    assigned_agent_id: Optional[str] = None
    priority: Optional[int] = 1
    estimated_hours: Optional[float] = 1.0

@strawberry.input
class DecisionInput:
    title: str
    description: str
    venture_id: str
    decision_data: str
    agent_id: str

@strawberry.input
class InfrastructurePredictionInput:
    venture_id: str
    time_horizon_days: Optional[int] = 30
    current_infrastructure: Optional[str] = "{}"

@strawberry.input
class DesignValidationInput:
    venture_id: str
    design_spec: str

# Output Types
@strawberry.type
class Agent:
    id: str
    name: str
    agent_type: AgentType
    specialization: str
    cognitive_cycles_balance: int
    is_active: bool
    created_at: datetime
    last_heartbeat: Optional[datetime] = None
    
    @strawberry.field
    async def tasks(self) -> List["Task"]:
        """Get tasks assigned to this agent"""
        # Mock implementation - replace with database query
        return []
    
    @strawberry.field
    async def decisions_made(self) -> List["Decision"]:
        """Get decisions made by this agent"""
        # Mock implementation - replace with database query
        return []

@strawberry.type
class Venture:
    id: str
    name: str
    objective: str
    status: VentureStatus
    target_users: int
    budget: float
    timeline_days: int
    created_at: datetime
    updated_at: datetime
    
    @strawberry.field
    async def tasks(self) -> List["Task"]:
        """Get tasks for this venture"""
        # Mock implementation - replace with database query
        return []
    
    @strawberry.field
    async def decisions(self) -> List["Decision"]:
        """Get decisions for this venture"""
        # Mock implementation - replace with database query
        return []
    
    @strawberry.field
    async def assigned_agents(self) -> List[Agent]:
        """Get agents assigned to this venture"""
        # Mock implementation - replace with database query
        return []

@strawberry.type
class Task:
    id: str
    title: str
    description: str
    status: TaskStatus
    venture_id: str
    assigned_agent_id: Optional[str] = None
    priority: int
    estimated_hours: float
    actual_hours: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    @strawberry.field
    async def venture(self) -> Optional[Venture]:
        """Get the venture this task belongs to"""
        # Mock implementation - replace with database query
        return None
    
    @strawberry.field
    async def assigned_agent(self) -> Optional[Agent]:
        """Get the agent assigned to this task"""
        # Mock implementation - replace with database query
        return None

@strawberry.type
class Decision:
    id: str
    title: str
    description: str
    decision_data: str
    outcome: DecisionOutcome
    venture_id: str
    agent_id: str
    cognitive_cycles_used: int
    created_at: datetime
    
    @strawberry.field
    async def venture(self) -> Optional[Venture]:
        """Get the venture this decision affects"""
        # Mock implementation - replace with database query
        return None
    
    @strawberry.field
    async def agent(self) -> Optional[Agent]:
        """Get the agent that made this decision"""
        # Mock implementation - replace with database query
        return None

@strawberry.type
class HealthComponent:
    component: str
    status: str
    message: str
    healthy: bool

@strawberry.type
class SystemHealth:
    overall_status: HealthStatus
    timestamp: datetime
    healthy_components: int
    total_components: int
    success_rate: float
    components: List[HealthComponent]

@strawberry.type
class InfrastructurePrediction:
    venture_id: str
    time_horizon_days: int
    monthly_estimate: float
    confidence: float
    recommendations: List[str]
    predicted_requirements: Dict[str, Any]
    timestamp: datetime

@strawberry.type
class DesignValidation:
    venture_id: str
    design_spec: str
    is_valid: bool
    confidence: float
    issues: List[str]
    recommendations: List[str]
    performance_metrics: Dict[str, Any]
    timestamp: datetime

# Subscription Types
@strawberry.type
class VentureUpdate:
    venture: Venture
    update_type: str
    timestamp: datetime

@strawberry.type
class TaskUpdate:
    task: Task
    update_type: str
    timestamp: datetime

@strawberry.type
class SystemEvent:
    event_type: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime

# Query Root
@strawberry.type
class Query:
    """GraphQL Query root"""
    
    @strawberry.field
    async def hello(self) -> str:
        """Simple health check"""
        return "Hello from Project Kairos GraphQL API!"
    
    @strawberry.field
    async def system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        try:
            health_checker = SystemHealthChecker()
            health_result = await health_checker.check_system_health()
            
            components = [
                HealthComponent(
                    component=comp["component"],
                    status=comp["status"],
                    message=comp["message"],
                    healthy=comp["healthy"]
                )
                for comp in health_result["components"]
            ]
            
            return SystemHealth(
                overall_status=HealthStatus(health_result["overall_status"]),
                timestamp=datetime.utcnow(),
                healthy_components=health_result["healthy_components"],
                total_components=health_result["total_components"],
                success_rate=health_result["success_rate"],
                components=components
            )
        except Exception as e:
            # Return degraded status if health check fails
            return SystemHealth(
                overall_status=HealthStatus.DEGRADED,
                timestamp=datetime.utcnow(),
                healthy_components=0,
                total_components=4,
                success_rate=0.0,
                components=[
                    HealthComponent(
                        component="health_checker",
                        status="error",
                        message=str(e),
                        healthy=False
                    )
                ]
            )
    
    @strawberry.field
    async def ventures(self, limit: Optional[int] = 10) -> List[Venture]:
        """Get all ventures"""
        # Mock data - replace with database query
        return [
            Venture(
                id="venture-001",
                name="E-commerce Platform",
                objective="Build scalable e-commerce platform",
                status=VentureStatus.IN_PROGRESS,
                target_users=50000,
                budget=100000.0,
                timeline_days=90,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]
    
    @strawberry.field
    async def venture(self, id: str) -> Optional[Venture]:
        """Get a specific venture by ID"""
        # Mock implementation - replace with database query
        if id == "venture-001":
            return Venture(
                id="venture-001",
                name="E-commerce Platform",
                objective="Build scalable e-commerce platform",
                status=VentureStatus.IN_PROGRESS,
                target_users=50000,
                budget=100000.0,
                timeline_days=90,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        return None
    
    @strawberry.field
    async def agents(self, active_only: Optional[bool] = True) -> List[Agent]:
        """Get all agents"""
        # Mock data - replace with database query
        return [
            Agent(
                id="steward-001",
                name="Enhanced-Steward",
                agent_type=AgentType.STEWARD,
                specialization="Advanced Resource Management",
                cognitive_cycles_balance=5000,
                is_active=True,
                created_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow()
            ),
            Agent(
                id="architect-001",
                name="Enhanced-Architect",
                agent_type=AgentType.ARCHITECT,
                specialization="AI-Powered System Architecture",
                cognitive_cycles_balance=4000,
                is_active=True,
                created_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow()
            )
        ]
    
    @strawberry.field
    async def agent(self, id: str) -> Optional[Agent]:
        """Get a specific agent by ID"""
        agents = await self.agents()
        return next((agent for agent in agents if agent.id == id), None)
    
    @strawberry.field
    async def tasks(self, venture_id: Optional[str] = None, agent_id: Optional[str] = None) -> List[Task]:
        """Get tasks, optionally filtered by venture or agent"""
        # Mock data - replace with database query
        return [
            Task(
                id="task-001",
                title="Database Schema Design",
                description="Design and implement database schema for user management",
                status=TaskStatus.IN_PROGRESS,
                venture_id="venture-001",
                assigned_agent_id="architect-001",
                priority=1,
                estimated_hours=8.0,
                actual_hours=5.5,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]
    
    @strawberry.field
    async def decisions(self, venture_id: Optional[str] = None) -> List[Decision]:
        """Get decisions, optionally filtered by venture"""
        # Mock data - replace with database query
        return [
            Decision(
                id="decision-001",
                title="Technology Stack Selection",
                description="Selected FastAPI + PostgreSQL + React stack",
                decision_data='{"stack": "FastAPI", "database": "PostgreSQL", "frontend": "React"}',
                outcome=DecisionOutcome.APPROVED,
                venture_id="venture-001",
                agent_id="architect-001",
                cognitive_cycles_used=150,
                created_at=datetime.utcnow()
            )
        ]
    
    @strawberry.field
    async def predict_infrastructure(self, input: InfrastructurePredictionInput) -> InfrastructurePrediction:
        """Get infrastructure predictions from Oracle"""
        try:
            oracle = OracleEngine()
            await oracle.initialize()
            
            current_infra = {}
            if input.current_infrastructure:
                import json
                current_infra = json.loads(input.current_infrastructure)
            
            prediction = await oracle.predict_infrastructure_requirements(
                venture_id=input.venture_id,
                time_horizon_days=input.time_horizon_days,
                current_infrastructure=current_infra
            )
            
            return InfrastructurePrediction(
                venture_id=input.venture_id,
                time_horizon_days=input.time_horizon_days,
                monthly_estimate=prediction.get("monthly_estimate", 0.0),
                confidence=prediction.get("confidence", 0.8),
                recommendations=prediction.get("recommendations", []),
                predicted_requirements=prediction.get("predicted_requirements", {}),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            # Return fallback prediction on error
            return InfrastructurePrediction(
                venture_id=input.venture_id,
                time_horizon_days=input.time_horizon_days,
                monthly_estimate=1000.0,
                confidence=0.1,
                recommendations=[f"Error generating prediction: {str(e)}"],
                predicted_requirements={"error": str(e)},
                timestamp=datetime.utcnow()
            )

# Mutation Root
@strawberry.type
class Mutation:
    """GraphQL Mutation root"""
    
    @strawberry.field
    async def create_venture(self, input: VentureInput) -> Venture:
        """Create a new venture"""
        # Mock implementation - replace with database insert
        venture_id = f"venture-{datetime.utcnow().timestamp():.0f}"
        
        venture = Venture(
            id=venture_id,
            name=input.name,
            objective=input.objective,
            status=VentureStatus.PLANNING,
            target_users=input.target_users or 1000,
            budget=input.budget or 10000.0,
            timeline_days=input.timeline_days or 30,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return venture
    
    @strawberry.field
    async def update_venture(self, id: str, input: VentureInput) -> Optional[Venture]:
        """Update an existing venture"""
        # Mock implementation - replace with database update
        venture = Venture(
            id=id,
            name=input.name,
            objective=input.objective,
            status=VentureStatus.IN_PROGRESS,
            target_users=input.target_users or 1000,
            budget=input.budget or 10000.0,
            timeline_days=input.timeline_days or 30,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return venture
    
    @strawberry.field
    async def create_task(self, input: TaskInput) -> Task:
        """Create a new task"""
        # Mock implementation - replace with database insert
        task_id = f"task-{datetime.utcnow().timestamp():.0f}"
        
        task = Task(
            id=task_id,
            title=input.title,
            description=input.description,
            status=TaskStatus.PENDING,
            venture_id=input.venture_id,
            assigned_agent_id=input.assigned_agent_id,
            priority=input.priority or 1,
            estimated_hours=input.estimated_hours or 1.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return task
    
    @strawberry.field
    async def update_task_status(self, id: str, status: TaskStatus) -> Optional[Task]:
        """Update task status"""
        # Mock implementation - replace with database update
        task = Task(
            id=id,
            title="Updated Task",
            description="Task description",
            status=status,
            venture_id="venture-001",
            assigned_agent_id="architect-001",
            priority=1,
            estimated_hours=4.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            completed_at=datetime.utcnow() if status == TaskStatus.COMPLETED else None
        )
        
        return task
    
    @strawberry.field
    async def create_decision(self, input: DecisionInput) -> Decision:
        """Create a new decision"""
        # Mock implementation - replace with database insert
        decision_id = f"decision-{datetime.utcnow().timestamp():.0f}"
        
        decision = Decision(
            id=decision_id,
            title=input.title,
            description=input.description,
            decision_data=input.decision_data,
            outcome=DecisionOutcome.PENDING,
            venture_id=input.venture_id,
            agent_id=input.agent_id,
            cognitive_cycles_used=100,
            created_at=datetime.utcnow()
        )
        
        return decision
    
    @strawberry.field
    async def validate_design(self, input: DesignValidationInput) -> DesignValidation:
        """Validate a system design with Oracle"""
        try:
            architect = EnhancedArchitectAgent()
            architect.agent_id = "api-architect-001"
            
            validation = await architect.validate_design_with_oracle(
                design_spec=input.design_spec,
                venture_id=input.venture_id
            )
            
            return DesignValidation(
                venture_id=input.venture_id,
                design_spec=input.design_spec,
                is_valid=validation.get("is_valid", False),
                confidence=validation.get("confidence", 0.5),
                issues=validation.get("issues", []),
                recommendations=validation.get("recommendations", []),
                performance_metrics=validation.get("performance_metrics", {}),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return DesignValidation(
                venture_id=input.venture_id,
                design_spec=input.design_spec,
                is_valid=False,
                confidence=0.0,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Fix validation service"],
                performance_metrics={"error": str(e)},
                timestamp=datetime.utcnow()
            )

# Subscription Root
@strawberry.type
class Subscription:
    """GraphQL Subscription root"""
    
    @strawberry.subscription
    async def venture_updates(self, venture_id: Optional[str] = None) -> AsyncGenerator[VentureUpdate, None]:
        """Subscribe to venture updates"""
        # Mock implementation - replace with real-time updates
        while True:
            await asyncio.sleep(30)  # Send update every 30 seconds
            
            venture = Venture(
                id="venture-001",
                name="E-commerce Platform",
                objective="Build scalable e-commerce platform",
                status=VentureStatus.IN_PROGRESS,
                target_users=50000,
                budget=100000.0,
                timeline_days=90,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            yield VentureUpdate(
                venture=venture,
                update_type="progress_update",
                timestamp=datetime.utcnow()
            )
    
    @strawberry.subscription
    async def task_updates(self) -> AsyncGenerator[TaskUpdate, None]:
        """Subscribe to task updates"""
        # Mock implementation - replace with real-time updates
        while True:
            await asyncio.sleep(45)  # Send update every 45 seconds
            
            task = Task(
                id="task-001",
                title="Database Schema Design",
                description="Design and implement database schema",
                status=TaskStatus.IN_PROGRESS,
                venture_id="venture-001",
                assigned_agent_id="architect-001",
                priority=1,
                estimated_hours=8.0,
                actual_hours=6.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            yield TaskUpdate(
                task=task,
                update_type="progress_update",
                timestamp=datetime.utcnow()
            )
    
    @strawberry.subscription
    async def system_events(self) -> AsyncGenerator[SystemEvent, None]:
        """Subscribe to system-wide events"""
        # Mock implementation - replace with real event stream
        while True:
            await asyncio.sleep(60)  # Send event every minute
            
            yield SystemEvent(
                event_type="health_check",
                message="System health check completed",
                data={"status": "healthy", "components": 4},
                timestamp=datetime.utcnow()
            )

# Create the schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)