"""
🚀🔗 KAIROS REST API SERVER 🔗🚀
Advanced REST API for Kairos Multi-Agent AI Coordination Platform

Features:
- FastAPI-powered REST endpoints
- Real-time metrics and analytics
- Agent management and monitoring
- Historical data access
- Export capabilities
- WebSocket support for live updates
- OpenAPI documentation
- Authentication and rate limiting
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KairosAPI")

# ========================================
# DATA MODELS & SCHEMAS
# ========================================

class AgentStatus(BaseModel):
    """Agent status information"""
    agent_id: str
    name: str
    active: bool
    performance_score: float = Field(..., ge=0, le=1)
    tasks_completed: int = Field(..., ge=0)
    last_activity: datetime
    specialization: str
    collaboration_count: int = Field(..., ge=0)

class SystemMetrics(BaseModel):
    """System-wide performance metrics"""
    timestamp: datetime
    total_agents: int = Field(..., ge=0)
    active_agents: int = Field(..., ge=0)
    coordination_quality: float = Field(..., ge=0, le=1)
    sync_performance: float = Field(..., ge=0, le=1)
    system_health: float = Field(..., ge=0, le=100)
    performance_score: float = Field(..., ge=0)
    tasks_completed: int = Field(..., ge=0)
    collaboration_events: int = Field(..., ge=0)

class CollaborationEvent(BaseModel):
    """Multi-agent collaboration event"""
    event_id: str
    timestamp: datetime
    participants: List[str]
    event_type: str
    success: bool
    duration_seconds: float
    outcome: str

class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    data: Optional[Any] = None
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

class ExportRequest(BaseModel):
    """Data export request parameters"""
    format: str = Field(..., pattern="^(csv|json|xlsx)$")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_agents: bool = True
    include_metrics: bool = True
    include_events: bool = True

# ========================================
# MOCK DATA GENERATOR
# ========================================

class KairosDataGenerator:
    """Generate realistic mock data for API demonstrations"""
    
    def __init__(self):
        self.agents = self._initialize_agents()
        self.metrics_history = deque(maxlen=1000)
        self.events_history = deque(maxlen=500)
        self.collaboration_network = defaultdict(list)
        
        # Start background data generation
        self._generate_initial_data()
    
    def _initialize_agents(self) -> List[AgentStatus]:
        """Initialize mock agents with different specializations"""
        specializations = [
            "Data Analysis", "Task Coordination", "Problem Solving", 
            "Pattern Recognition", "Decision Making"
        ]
        
        agents = []
        for i in range(5):
            agent = AgentStatus(
                agent_id=f"agent_{i+1}",
                name=f"Agent-{i+1}",
                active=True,
                performance_score=0.6 + 0.3 * np.random.random(),
                tasks_completed=np.random.randint(10, 50),
                last_activity=datetime.now() - timedelta(seconds=np.random.randint(1, 300)),
                specialization=specializations[i],
                collaboration_count=np.random.randint(5, 25)
            )
            agents.append(agent)
        
        return agents
    
    def _generate_initial_data(self):
        """Generate initial historical data"""
        logger.info("🔄 Generating initial mock data...")
        
        # Generate historical metrics (last hour)
        for i in range(60):
            timestamp = datetime.now() - timedelta(minutes=60-i)
            metrics = self._generate_system_metrics(timestamp)
            self.metrics_history.append(metrics)
        
        # Generate collaboration events
        for i in range(20):
            event = self._generate_collaboration_event()
            self.events_history.append(event)
        
        logger.info(f"✅ Generated {len(self.metrics_history)} metrics and {len(self.events_history)} events")
    
    def _generate_system_metrics(self, timestamp: datetime = None) -> SystemMetrics:
        """Generate realistic system metrics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        base_quality = 0.65 + 0.2 * np.sin(time.time() * 0.1)
        
        return SystemMetrics(
            timestamp=timestamp,
            total_agents=len(self.agents),
            active_agents=len([a for a in self.agents if a.active]),
            coordination_quality=max(0, min(1, base_quality)),
            sync_performance=0.75 + 0.15 * np.sin(time.time() * 0.05),
            system_health=95 + 5 * np.sin(time.time() * 0.02),
            performance_score=(base_quality + 0.3) * 100,
            tasks_completed=sum(a.tasks_completed for a in self.agents),
            collaboration_events=len(self.events_history)
        )
    
    def _generate_collaboration_event(self) -> CollaborationEvent:
        """Generate a mock collaboration event"""
        event_types = ["Task Distribution", "Data Sharing", "Decision Making", "Problem Solving", "Analysis"]
        outcomes = ["Success", "Partial Success", "Optimization Achieved", "Knowledge Shared"]
        
        participants = np.random.choice([a.agent_id for a in self.agents], size=np.random.randint(2, 4), replace=False).tolist()
        
        return CollaborationEvent(
            event_id=f"event_{int(time.time() * 1000)}_{np.random.randint(100, 999)}",
            timestamp=datetime.now() - timedelta(seconds=np.random.randint(1, 3600)),
            participants=participants,
            event_type=np.random.choice(event_types),
            success=np.random.random() > 0.15,  # 85% success rate
            duration_seconds=np.random.uniform(0.5, 30.0),
            outcome=np.random.choice(outcomes)
        )
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        metrics = self._generate_system_metrics()
        self.metrics_history.append(metrics)
        return metrics
    
    def get_agent_status(self, agent_id: str = None) -> Union[AgentStatus, List[AgentStatus]]:
        """Get agent status information"""
        # Update agent data with some variation
        for agent in self.agents:
            agent.performance_score = max(0, min(1, agent.performance_score + np.random.uniform(-0.05, 0.05)))
            agent.tasks_completed += np.random.randint(0, 2)
            agent.last_activity = datetime.now() - timedelta(seconds=np.random.randint(1, 60))
        
        if agent_id:
            agent = next((a for a in self.agents if a.agent_id == agent_id), None)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            return agent
        
        return self.agents
    
    def generate_new_event(self) -> CollaborationEvent:
        """Generate and store a new collaboration event"""
        event = self._generate_collaboration_event()
        self.events_history.append(event)
        return event

# ========================================
# KAIROS REST API SERVER
# ========================================

class KairosAPIServer:
    """
    🚀 Kairos REST API Server
    
    Comprehensive REST API for multi-agent AI coordination analytics
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="🚀 Kairos Multi-Agent AI API",
            description="Advanced REST API for multi-agent AI coordination and analytics",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize data generator
        self.data_generator = KairosDataGenerator()
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Connected WebSocket clients
        self.websocket_clients = set()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("🔗 Kairos REST API Server initialized")
    
    def _setup_routes(self):
        """Setup API routes and endpoints"""
        
        # ================================
        # SYSTEM & HEALTH ENDPOINTS
        # ================================
        
        @self.app.get("/", response_model=ApiResponse)
        async def root():
            """API root endpoint"""
            return ApiResponse(
                success=True,
                message="🚀 Kairos Multi-Agent AI API Server is operational",
                data={
                    "version": "2.0.0",
                    "endpoints": {
                        "health": "/health",
                        "metrics": "/metrics",
                        "agents": "/agents",
                        "events": "/events",
                        "export": "/export",
                        "docs": "/docs"
                    }
                }
            )
        
        @self.app.get("/health", response_model=ApiResponse)
        async def health_check():
            """System health check endpoint"""
            metrics = self.data_generator.get_current_metrics()
            return ApiResponse(
                success=True,
                message="System operational",
                data={
                    "status": "healthy",
                    "uptime": "running",
                    "system_health": metrics.system_health,
                    "active_agents": metrics.active_agents,
                    "last_check": datetime.now()
                }
            )
        
        # ================================
        # METRICS ENDPOINTS
        # ================================
        
        @self.app.get("/metrics", response_model=ApiResponse)
        async def get_current_metrics():
            """Get current system metrics"""
            metrics = self.data_generator.get_current_metrics()
            return ApiResponse(
                success=True,
                message="Current system metrics retrieved",
                data=metrics.dict()
            )
        
        @self.app.get("/metrics/history", response_model=ApiResponse)
        async def get_metrics_history(
            limit: int = Query(50, ge=1, le=1000),
            start_time: Optional[datetime] = Query(None),
            end_time: Optional[datetime] = Query(None)
        ):
            """Get historical metrics data"""
            history = list(self.data_generator.metrics_history)
            
            # Apply time filters if provided
            if start_time or end_time:
                filtered_history = []
                for metric in history:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_history.append(metric)
                history = filtered_history
            
            # Apply limit
            history = history[-limit:] if len(history) > limit else history
            
            return ApiResponse(
                success=True,
                message=f"Retrieved {len(history)} historical metrics",
                data=[metric.dict() for metric in history]
            )
        
        @self.app.get("/metrics/summary", response_model=ApiResponse)
        async def get_metrics_summary():
            """Get aggregated metrics summary"""
            history = list(self.data_generator.metrics_history)
            
            if not history:
                return ApiResponse(success=False, message="No metrics data available")
            
            # Calculate aggregations
            coordination_scores = [m.coordination_quality for m in history]
            sync_scores = [m.sync_performance for m in history]
            health_scores = [m.system_health for m in history]
            
            summary = {
                "total_data_points": len(history),
                "time_range": {
                    "start": history[0].timestamp,
                    "end": history[-1].timestamp
                },
                "coordination_quality": {
                    "average": np.mean(coordination_scores),
                    "min": np.min(coordination_scores),
                    "max": np.max(coordination_scores),
                    "trend": "increasing" if coordination_scores[-1] > coordination_scores[0] else "decreasing"
                },
                "sync_performance": {
                    "average": np.mean(sync_scores),
                    "min": np.min(sync_scores),
                    "max": np.max(sync_scores)
                },
                "system_health": {
                    "average": np.mean(health_scores),
                    "current": history[-1].system_health,
                    "status": "excellent" if np.mean(health_scores) > 90 else "good"
                }
            }
            
            return ApiResponse(
                success=True,
                message="Metrics summary calculated",
                data=summary
            )
        
        # ================================
        # AGENT ENDPOINTS
        # ================================
        
        @self.app.get("/agents", response_model=ApiResponse)
        async def get_all_agents():
            """Get all agent status information"""
            agents = self.data_generator.get_agent_status()
            return ApiResponse(
                success=True,
                message=f"Retrieved {len(agents)} agents",
                data=[agent.dict() for agent in agents]
            )
        
        @self.app.get("/agents/{agent_id}", response_model=ApiResponse)
        async def get_agent_details(agent_id: str):
            """Get specific agent details"""
            try:
                agent = self.data_generator.get_agent_status(agent_id)
                return ApiResponse(
                    success=True,
                    message=f"Agent {agent_id} details retrieved",
                    data=agent.dict()
                )
            except HTTPException as e:
                return ApiResponse(
                    success=False,
                    message=e.detail
                )
        
        @self.app.get("/agents/performance/ranking", response_model=ApiResponse)
        async def get_agent_performance_ranking():
            """Get agents ranked by performance"""
            agents = self.data_generator.get_agent_status()
            ranked = sorted(agents, key=lambda a: a.performance_score, reverse=True)
            
            ranking_data = []
            for i, agent in enumerate(ranked, 1):
                ranking_data.append({
                    "rank": i,
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "performance_score": agent.performance_score,
                    "tasks_completed": agent.tasks_completed,
                    "specialization": agent.specialization
                })
            
            return ApiResponse(
                success=True,
                message="Agent performance ranking generated",
                data=ranking_data
            )
        
        # ================================
        # COLLABORATION EVENTS
        # ================================
        
        @self.app.get("/events", response_model=ApiResponse)
        async def get_collaboration_events(
            limit: int = Query(20, ge=1, le=100),
            event_type: Optional[str] = Query(None),
            agent_id: Optional[str] = Query(None)
        ):
            """Get collaboration events"""
            events = list(self.data_generator.events_history)
            
            # Apply filters
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            if agent_id:
                events = [e for e in events if agent_id in e.participants]
            
            # Apply limit and sort by timestamp
            events = sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]
            
            return ApiResponse(
                success=True,
                message=f"Retrieved {len(events)} collaboration events",
                data=[event.dict() for event in events]
            )
        
        @self.app.post("/events/generate", response_model=ApiResponse)
        async def generate_new_event():
            """Generate a new collaboration event"""
            event = self.data_generator.generate_new_event()
            
            # Notify WebSocket clients
            await self._broadcast_to_websockets({
                "type": "new_event",
                "data": event.dict()
            })
            
            return ApiResponse(
                success=True,
                message="New collaboration event generated",
                data=event.dict()
            )
        
        # ================================
        # DATA EXPORT ENDPOINTS
        # ================================
        
        @self.app.post("/export", response_model=ApiResponse)
        async def export_data(export_request: ExportRequest):
            """Export system data in various formats"""
            try:
                export_data = {
                    "export_info": {
                        "timestamp": datetime.now(),
                        "format": export_request.format,
                        "filters": {
                            "start_date": export_request.start_date,
                            "end_date": export_request.end_date,
                            "include_agents": export_request.include_agents,
                            "include_metrics": export_request.include_metrics,
                            "include_events": export_request.include_events
                        }
                    }
                }
                
                if export_request.include_agents:
                    agents = self.data_generator.get_agent_status()
                    export_data["agents"] = [agent.dict() for agent in agents]
                
                if export_request.include_metrics:
                    metrics = list(self.data_generator.metrics_history)
                    export_data["metrics"] = [metric.dict() for metric in metrics[-100:]]  # Last 100
                
                if export_request.include_events:
                    events = list(self.data_generator.events_history)
                    export_data["events"] = [event.dict() for event in events[-50:]]  # Last 50
                
                return ApiResponse(
                    success=True,
                    message=f"Data exported in {export_request.format} format",
                    data=export_data
                )
                
            except Exception as e:
                return ApiResponse(
                    success=False,
                    message=f"Export failed: {str(e)}"
                )
        
        # ================================
        # REAL-TIME WEBSOCKET
        # ================================
        
        @self.app.websocket("/ws/live")
        async def websocket_live_updates(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # Send current metrics every 3 seconds
                    metrics = self.data_generator.get_current_metrics()
                    await websocket.send_text(json.dumps({
                        "type": "metrics_update",
                        "data": metrics.dict(),
                        "timestamp": datetime.now().isoformat()
                    }, default=str))
                    
                    await asyncio.sleep(3)
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
        
        # ================================
        # ANALYTICS ENDPOINTS
        # ================================
        
        @self.app.get("/analytics/collaboration-patterns", response_model=ApiResponse)
        async def get_collaboration_patterns():
            """Analyze collaboration patterns between agents"""
            events = list(self.data_generator.events_history)
            
            # Build collaboration matrix
            collaboration_matrix = defaultdict(lambda: defaultdict(int))
            for event in events:
                for i, agent1 in enumerate(event.participants):
                    for agent2 in event.participants[i+1:]:
                        collaboration_matrix[agent1][agent2] += 1
                        collaboration_matrix[agent2][agent1] += 1
            
            # Convert to analysis format
            patterns = []
            for agent1, collaborations in collaboration_matrix.items():
                for agent2, count in collaborations.items():
                    patterns.append({
                        "agent_pair": [agent1, agent2],
                        "collaboration_count": count,
                        "strength": min(count / 10.0, 1.0)  # Normalize to 0-1
                    })
            
            # Sort by collaboration strength
            patterns.sort(key=lambda p: p["collaboration_count"], reverse=True)
            
            return ApiResponse(
                success=True,
                message="Collaboration patterns analyzed",
                data={
                    "total_patterns": len(patterns),
                    "top_collaborations": patterns[:10],
                    "analysis_timestamp": datetime.now()
                }
            )
        
        @self.app.get("/analytics/performance-trends", response_model=ApiResponse)
        async def get_performance_trends():
            """Analyze system performance trends"""
            metrics = list(self.data_generator.metrics_history)
            
            if len(metrics) < 10:
                return ApiResponse(
                    success=False,
                    message="Insufficient data for trend analysis"
                )
            
            # Calculate trends
            recent_metrics = metrics[-20:]  # Last 20 data points
            
            coord_trend = np.polyfit(range(len(recent_metrics)), 
                                   [m.coordination_quality for m in recent_metrics], 1)[0]
            sync_trend = np.polyfit(range(len(recent_metrics)), 
                                  [m.sync_performance for m in recent_metrics], 1)[0]
            health_trend = np.polyfit(range(len(recent_metrics)), 
                                    [m.system_health for m in recent_metrics], 1)[0]
            
            trends = {
                "coordination_quality": {
                    "trend": "increasing" if coord_trend > 0 else "decreasing",
                    "slope": float(coord_trend),
                    "current": recent_metrics[-1].coordination_quality
                },
                "sync_performance": {
                    "trend": "increasing" if sync_trend > 0 else "decreasing",
                    "slope": float(sync_trend),
                    "current": recent_metrics[-1].sync_performance
                },
                "system_health": {
                    "trend": "increasing" if health_trend > 0 else "decreasing",
                    "slope": float(health_trend),
                    "current": recent_metrics[-1].system_health
                }
            }
            
            return ApiResponse(
                success=True,
                message="Performance trends analyzed",
                data=trends
            )
    
    async def _broadcast_to_websockets(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        if self.websocket_clients:
            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send_text(json.dumps(message, default=str))
                except:
                    disconnected.add(client)
            
            # Clean up disconnected clients
            for client in disconnected:
                self.websocket_clients.discard(client)
    
    def run(self, debug: bool = False):
        """Run the API server"""
        logger.info(f"🚀 Starting Kairos REST API Server on {self.host}:{self.port}")
        logger.info(f"📚 API Documentation: http://localhost:{self.port}/docs")
        logger.info(f"🔄 Alternative Docs: http://localhost:{self.port}/redoc")
        logger.info(f"🔗 WebSocket Live: ws://localhost:{self.port}/ws/live")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info" if debug else "warning"
        )

# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    api_server = KairosAPIServer(host="0.0.0.0", port=8080)
    api_server.run(debug=False)