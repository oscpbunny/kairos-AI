"""
Project Kairos: Enhanced Agent Swarm Base Class
The cognitive substrate foundation that all Kairos agents inherit from.

This base class provides:
- Cognitive Cycles economy integration
- gRPC communication capabilities
- Decision recording in causal ledger
- Performance monitoring and optimization
- Collaborative task management
- Self-optimization through market feedback
"""

import os
import json
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum

import grpc
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    
try:
    from prometheus_client import Counter, Histogram, Gauge
except ImportError:
    Counter = lambda *args, **kwargs: type('MockCounter', (), {'inc': lambda self, *a: None})()
    Histogram = lambda *args, **kwargs: type('MockHistogram', (), {'observe': lambda self, *a: None})()
    Gauge = lambda *args, **kwargs: type('MockGauge', (), {'set': lambda self, *a: None})()
import numpy as np

# Import our gRPC stubs (would be generated from kairos.proto)
# from api.grpc import kairos_pb2, kairos_pb2_grpc

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KairosAgentBase')

class AgentType(Enum):
    STEWARD = "STEWARD"
    ARCHITECT = "ARCHITECT"
    ENGINEER = "ENGINEER"
    STRATEGIST = "STRATEGIST"
    EMPATHY = "EMPATHY"
    ETHICIST = "ETHICIST"
    QA = "QA"

class TaskType(Enum):
    ANALYSIS = "ANALYSIS"
    DEVELOPMENT = "DEVELOPMENT"
    TESTING = "TESTING"
    DEPLOYMENT = "DEPLOYMENT"
    RESEARCH = "RESEARCH"
    OPTIMIZATION = "OPTIMIZATION"
    MONITORING = "MONITORING"

class DecisionType(Enum):
    STRATEGIC = "STRATEGIC"
    OPERATIONAL = "OPERATIONAL"
    REACTIVE = "REACTIVE"
    PREDICTIVE = "PREDICTIVE"

@dataclass
class AgentCapability:
    """Represents a specific capability an agent possesses"""
    name: str
    proficiency_level: float  # 0.0 to 1.0
    experience_points: int
    last_used: datetime
    success_rate: float

@dataclass
class TaskBid:
    """Represents a bid on a task"""
    task_id: str
    bid_amount_cc: int
    estimated_completion_time: int  # minutes
    proposed_approach: str
    confidence_score: float
    risk_factors: List[str]

@dataclass
class PerformanceMetrics:
    """Agent performance tracking"""
    tasks_completed: int = 0
    success_rate: float = 1.0
    average_completion_time: float = 0.0
    quality_score_avg: float = 0.8
    cc_earned_total: int = 0
    efficiency_trend: float = 1.0
    collaboration_score: float = 0.5

class KairosAgentBase(ABC):
    """
    Base class for all Kairos agents providing cognitive substrate integration
    """
    
    def __init__(
        self, 
        agent_name: str, 
        agent_type: AgentType, 
        specialization: str,
        initial_cc_balance: int = 1000
    ):
        self.agent_id: Optional[str] = None
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.specialization = specialization
        self.cognitive_cycles_balance = initial_cc_balance
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.capabilities: Dict[str, AgentCapability] = {}
        self.active_tasks: Dict[str, Dict] = {}
        self.pending_bids: Dict[str, TaskBid] = {}
        
        # Communication and collaboration
        self.message_handlers: Dict[str, Callable] = {}
        self.collaboration_partners: Set[str] = set()
        
        # Database and external connections
        self.db_config = self._load_db_config()
        self.grpc_channels: Dict[str, grpc.Channel] = {}
        
        # Internal state management
        self.is_active = True
        self.last_heartbeat = datetime.now()
        self.decision_history: List[str] = []
        
        # Metrics
        self.metrics = self._setup_metrics()
        
        # Economic intelligence
        self.market_awareness = {
            'current_avg_bounty': 0,
            'competition_level': 0.5,
            'specialization_premium': 0.0,
            'demand_trends': {}
        }
    
    def _load_db_config(self) -> Dict[str, str]:
        """Load database configuration"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'kairos_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup Prometheus metrics"""
        agent_label = f"{self.agent_type.value.lower()}_{self.agent_name.lower()}"
        
        return {
            'tasks_completed': Counter(f'kairos_agent_tasks_completed_{agent_label}', 
                                     'Tasks completed by agent'),
            'cc_earned': Counter(f'kairos_agent_cc_earned_{agent_label}', 
                               'Cognitive Cycles earned'),
            'decision_time': Histogram(f'kairos_agent_decision_time_seconds_{agent_label}',
                                     'Time taken to make decisions'),
            'current_cc_balance': Gauge(f'kairos_agent_cc_balance_{agent_label}',
                                      'Current CC balance'),
            'active_tasks': Gauge(f'kairos_agent_active_tasks_{agent_label}',
                                'Number of active tasks')
        }
    
    async def get_db_connection(self):
        """Get database connection"""
        if not psycopg2:
            return None
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        )
    
    async def initialize_agent(self) -> bool:
        """Initialize the agent in the system"""
        try:
            conn = await self.get_db_connection()
            
            async with conn.cursor() as cur:
                # Register or update agent in database
                await cur.execute(
                    """
                    INSERT INTO Agents (name, specialization, agent_type, cognitive_cycles_balance, 
                                      capabilities, is_active, last_heartbeat)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (name) 
                    DO UPDATE SET 
                        is_active = %s,
                        last_heartbeat = CURRENT_TIMESTAMP,
                        cognitive_cycles_balance = EXCLUDED.cognitive_cycles_balance
                    RETURNING id;
                    """,
                    (
                        self.agent_name, self.specialization, self.agent_type.value,
                        self.cognitive_cycles_balance, json.dumps(self._serialize_capabilities()),
                        True, True
                    )
                )
                
                result = await cur.fetchone()
                self.agent_id = result['id']
            
            conn.commit()
            conn.close()
            
            # Initialize gRPC connections
            await self._setup_grpc_connections()
            
            # Update metrics
            self.metrics['current_cc_balance'].set(self.cognitive_cycles_balance)
            
            logger.info(f"Agent {self.agent_name} initialized with ID: {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_name}: {e}")
            return False
    
    async def _setup_grpc_connections(self):
        """Setup gRPC connections to system services"""
        try:
            grpc_host = os.getenv('GRPC_HOST', 'localhost')
            grpc_port = os.getenv('GRPC_PORT', '50051')
            
            # Create channels for different services
            channel = grpc.aio.insecure_channel(f'{grpc_host}:{grpc_port}')
            
            self.grpc_channels['main'] = channel
            
            # Initialize service stubs (would use generated code)
            # self.agent_service = kairos_pb2_grpc.AgentServiceStub(channel)
            # self.task_service = kairos_pb2_grpc.TaskServiceStub(channel)
            # self.communication_service = kairos_pb2_grpc.CommunicationServiceStub(channel)
            # self.resource_service = kairos_pb2_grpc.ResourceServiceStub(channel)
            # self.decision_service = kairos_pb2_grpc.DecisionServiceStub(channel)
            
        except Exception as e:
            logger.warning(f"Failed to setup gRPC connections: {e}")
    
    def _serialize_capabilities(self) -> Dict[str, Any]:
        """Serialize capabilities for storage"""
        return {
            name: {
                'proficiency_level': cap.proficiency_level,
                'experience_points': cap.experience_points,
                'last_used': cap.last_used.isoformat(),
                'success_rate': cap.success_rate
            }
            for name, cap in self.capabilities.items()
        }
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific task - must be implemented by each agent type"""
        pass
    
    @abstractmethod
    async def evaluate_task_fit(self, task: Dict[str, Any]) -> float:
        """Evaluate how well this agent fits a task (0.0 to 1.0)"""
        pass
    
    @abstractmethod
    async def generate_task_bid(self, task: Dict[str, Any]) -> Optional[TaskBid]:
        """Generate a bid for a task based on agent's capabilities and market conditions"""
        pass
    
    async def update_heartbeat(self):
        """Update agent heartbeat and status in the system"""
        try:
            self.last_heartbeat = datetime.now()
            
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE Agents 
                    SET last_heartbeat = CURRENT_TIMESTAMP,
                        cognitive_cycles_balance = %s,
                        performance_metrics = %s
                    WHERE id = %s;
                    """,
                    (
                        self.cognitive_cycles_balance,
                        json.dumps(asdict(self.performance_metrics)),
                        self.agent_id
                    )
                )
            
            conn.commit()
            conn.close()
            
            # Update metrics
            self.metrics['current_cc_balance'].set(self.cognitive_cycles_balance)
            self.metrics['active_tasks'].set(len(self.active_tasks))
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {self.agent_name}: {e}")
    
    async def record_decision(
        self,
        venture_id: str,
        decision_type: DecisionType,
        triggered_by_event: str,
        rationale: str,
        confidence_level: float = 0.8,
        consulted_data_sources: Optional[Dict] = None,
        expected_outcomes: Optional[Dict] = None,
        cc_invested: int = 0
    ) -> str:
        """Record a decision in the causal ledger"""
        try:
            decision_start = datetime.now()
            decision_id = str(uuid.uuid4())
            
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO Decisions 
                    (id, venture_id, agent_id, decision_type, triggered_by_event, rationale,
                     confidence_level, consulted_data_sources, expected_outcomes, 
                     cognitive_cycles_invested, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP);
                    """,
                    (
                        decision_id, venture_id, self.agent_id, decision_type.value,
                        triggered_by_event, rationale, confidence_level,
                        json.dumps(consulted_data_sources or {}),
                        json.dumps(expected_outcomes or {}), cc_invested
                    )
                )
            
            conn.commit()
            conn.close()
            
            # Update decision history and metrics
            self.decision_history.append(decision_id)
            decision_time = (datetime.now() - decision_start).total_seconds()
            self.metrics['decision_time'].observe(decision_time)
            
            logger.info(f"Decision {decision_id} recorded by {self.agent_name}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Failed to record decision: {e}")
            return ""
    
    async def scan_available_tasks(self) -> List[Dict[str, Any]]:
        """Scan for available tasks that match agent's capabilities"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                # Get tasks that match agent's capabilities
                await cur.execute(
                    """
                    SELECT t.*, v.name as venture_name
                    FROM Tasks t
                    JOIN Ventures v ON t.venture_id = v.id
                    WHERE t.status IN ('BOUNTY_POSTED', 'BIDDING')
                    AND t.bidding_ends_at > CURRENT_TIMESTAMP
                    AND t.id NOT IN (
                        SELECT task_id FROM Task_Bids WHERE agent_id = %s
                    )
                    ORDER BY t.cc_bounty DESC, t.created_at ASC
                    LIMIT 20;
                    """,
                    (self.agent_id,)
                )
                
                tasks = await cur.fetchall()
            
            conn.close()
            
            # Evaluate task fit and filter
            suitable_tasks = []
            for task in tasks:
                fit_score = await self.evaluate_task_fit(task)
                if fit_score > 0.3:  # Only consider tasks with >30% fit
                    task_dict = dict(task)
                    task_dict['fit_score'] = fit_score
                    suitable_tasks.append(task_dict)
            
            # Sort by combination of bounty and fit
            suitable_tasks.sort(
                key=lambda t: (t['cc_bounty'] * t['fit_score']), 
                reverse=True
            )
            
            return suitable_tasks[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Failed to scan available tasks: {e}")
            return []
    
    async def participate_in_task_market(self):
        """Actively participate in the task marketplace by bidding on suitable tasks"""
        try:
            available_tasks = await self.scan_available_tasks()
            
            for task in available_tasks:
                # Skip if we don't have enough CC for deposit
                required_deposit = int(task['cc_bounty'] * 0.1)  # 10% deposit
                if self.cognitive_cycles_balance < required_deposit:
                    continue
                
                # Generate bid
                bid = await self.generate_task_bid(task)
                if bid and bid.bid_amount_cc <= self.cognitive_cycles_balance:
                    await self._place_bid(bid)
                    
                    # Limit concurrent bids to manage risk
                    if len(self.pending_bids) >= 5:
                        break
            
        except Exception as e:
            logger.error(f"Error participating in task market: {e}")
    
    async def _place_bid(self, bid: TaskBid) -> bool:
        """Place a bid on a task"""
        try:
            # Record bid in database
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO Task_Bids 
                    (task_id, agent_id, bid_amount_cc, estimated_completion_time, 
                     proposed_approach, confidence_score, risk_factors)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        bid.task_id, self.agent_id, bid.bid_amount_cc,
                        bid.estimated_completion_time, bid.proposed_approach,
                        bid.confidence_score, json.dumps(bid.risk_factors)
                    )
                )
                
                # Deduct deposit
                deposit = int(bid.bid_amount_cc * 0.1)
                await cur.execute(
                    """
                    UPDATE Agents 
                    SET cognitive_cycles_balance = cognitive_cycles_balance - %s
                    WHERE id = %s;
                    """,
                    (deposit, self.agent_id)
                )
                
                self.cognitive_cycles_balance -= deposit
            
            conn.commit()
            conn.close()
            
            # Track pending bid
            self.pending_bids[bid.task_id] = bid
            
            logger.info(f"Agent {self.agent_name} placed bid on task {bid.task_id}: {bid.bid_amount_cc} CC")
            return True
            
        except Exception as e:
            logger.error(f"Failed to place bid: {e}")
            return False
    
    async def check_task_assignments(self):
        """Check for new task assignments and start working on them"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                # Check for newly assigned tasks
                await cur.execute(
                    """
                    SELECT t.*, b.bid_amount_cc
                    FROM Tasks t
                    JOIN Task_Bids b ON t.id = b.task_id AND b.agent_id = %s AND b.status = 'ACCEPTED'
                    WHERE t.assigned_to_agent_id = %s 
                    AND t.status = 'ASSIGNED'
                    AND t.id NOT IN (%s);
                    """,
                    (self.agent_id, self.agent_id, 
                     ','.join([f"'{tid}'" for tid in self.active_tasks.keys()]) or "'dummy'")
                )
                
                new_assignments = await cur.fetchall()
            
            conn.close()
            
            # Start working on new assignments
            for task in new_assignments:
                await self._start_task(dict(task))
            
        except Exception as e:
            logger.error(f"Failed to check task assignments: {e}")
    
    async def _start_task(self, task: Dict[str, Any]):
        """Start working on an assigned task"""
        try:
            task_id = task['id']
            
            # Update task status to IN_PROGRESS
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE Tasks 
                    SET status = 'IN_PROGRESS', started_at = CURRENT_TIMESTAMP
                    WHERE id = %s;
                    """,
                    (task_id,)
                )
            
            conn.commit()
            conn.close()
            
            # Add to active tasks
            self.active_tasks[task_id] = task
            
            # Create async task to process it
            asyncio.create_task(self._execute_task(task))
            
            logger.info(f"Agent {self.agent_name} started task {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to start task: {e}")
    
    async def _execute_task(self, task: Dict[str, Any]):
        """Execute a task and handle completion"""
        task_id = task['id']
        start_time = datetime.now()
        
        try:
            # Process the task (implemented by subclass)
            result = await self.process_task(task)
            
            # Calculate performance metrics
            completion_time = datetime.now() - start_time
            quality_score = result.get('quality_score', 0.8)
            
            # Update task completion
            await self._complete_task(task_id, result, quality_score, completion_time)
            
            # Update agent performance
            self._update_performance_metrics(task, quality_score, completion_time)
            
        except Exception as e:
            logger.error(f"Task execution failed for {task_id}: {e}")
            await self._fail_task(task_id, str(e))
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task_id, None)
    
    async def _complete_task(
        self, 
        task_id: str, 
        result: Dict[str, Any], 
        quality_score: float, 
        completion_time: timedelta
    ):
        """Mark task as completed and claim payment"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                # Update task status
                await cur.execute(
                    """
                    UPDATE Tasks 
                    SET status = 'COMPLETED',
                        completed_at = CURRENT_TIMESTAMP,
                        actual_duration = %s,
                        quality_score = %s,
                        actual_deliverables = %s
                    WHERE id = %s;
                    """,
                    (
                        completion_time,
                        quality_score,
                        json.dumps(result.get('deliverables', {})),
                        task_id
                    )
                )
                
                # Get bounty amount and any bonuses
                await cur.execute(
                    """
                    SELECT t.cc_bounty, t.bonus_cc, b.bid_amount_cc
                    FROM Tasks t
                    JOIN Task_Bids b ON t.id = b.task_id AND b.agent_id = %s
                    WHERE t.id = %s;
                    """,
                    (self.agent_id, task_id)
                )
                
                payment_info = await cur.fetchone()
                if payment_info:
                    total_payment = payment_info['cc_bounty'] + (payment_info['bonus_cc'] or 0)
                    
                    # Pay the agent
                    await cur.execute(
                        """
                        UPDATE Agents 
                        SET cognitive_cycles_balance = cognitive_cycles_balance + %s
                        WHERE id = %s;
                        """,
                        (total_payment, self.agent_id)
                    )
                    
                    self.cognitive_cycles_balance += total_payment
                    self.metrics['cc_earned'].inc(total_payment)
            
            conn.commit()
            conn.close()
            
            # Remove from pending bids
            self.pending_bids.pop(task_id, None)
            
            logger.info(f"Agent {self.agent_name} completed task {task_id} with quality {quality_score:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
    
    async def _fail_task(self, task_id: str, error_reason: str):
        """Mark task as failed"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE Tasks 
                    SET status = 'FAILED',
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s;
                    """,
                    (task_id,)
                )
            
            conn.commit()
            conn.close()
            
            # Record failure in performance metrics
            self.performance_metrics.tasks_completed += 1
            # Success rate will be recalculated
            
            logger.warning(f"Agent {self.agent_name} failed task {task_id}: {error_reason}")
            
        except Exception as e:
            logger.error(f"Failed to mark task as failed: {e}")
    
    def _update_performance_metrics(self, task: Dict, quality_score: float, completion_time: timedelta):
        """Update agent's performance metrics"""
        self.performance_metrics.tasks_completed += 1
        
        # Update success rate (rolling average)
        current_successes = self.performance_metrics.success_rate * (self.performance_metrics.tasks_completed - 1)
        self.performance_metrics.success_rate = (current_successes + 1) / self.performance_metrics.tasks_completed
        
        # Update average completion time
        if self.performance_metrics.average_completion_time == 0:
            self.performance_metrics.average_completion_time = completion_time.total_seconds() / 3600
        else:
            self.performance_metrics.average_completion_time = (
                (self.performance_metrics.average_completion_time * (self.performance_metrics.tasks_completed - 1) + 
                 completion_time.total_seconds() / 3600) / self.performance_metrics.tasks_completed
            )
        
        # Update quality score average
        if self.performance_metrics.quality_score_avg == 0:
            self.performance_metrics.quality_score_avg = quality_score
        else:
            self.performance_metrics.quality_score_avg = (
                (self.performance_metrics.quality_score_avg * (self.performance_metrics.tasks_completed - 1) + 
                 quality_score) / self.performance_metrics.tasks_completed
            )
        
        # Update metrics
        self.metrics['tasks_completed'].inc()
    
    async def collaborate_with_agent(self, agent_id: str, message: str, message_type: str = "COLLABORATION"):
        """Send a collaboration message to another agent"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO Agent_Communications 
                    (from_agent_id, to_agent_id, message_type, content, created_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP);
                    """,
                    (self.agent_id, agent_id, message_type, message)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Agent {self.agent_name} sent {message_type} to {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to send collaboration message: {e}")
    
    async def check_messages(self):
        """Check for new messages from other agents"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT ac.*, a.name as from_agent_name
                    FROM Agent_Communications ac
                    JOIN Agents a ON ac.from_agent_id = a.id
                    WHERE ac.to_agent_id = %s AND ac.read_at IS NULL
                    ORDER BY ac.created_at ASC;
                    """,
                    (self.agent_id,)
                )
                
                messages = await cur.fetchall()
            
            conn.close()
            
            # Process messages
            for message in messages:
                await self._handle_message(dict(message))
            
        except Exception as e:
            logger.error(f"Failed to check messages: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from another agent"""
        message_type = message['message_type']
        content = message['content']
        from_agent = message['from_agent_name']
        
        # Mark as read
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE Agent_Communications SET read_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (message['id'],)
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to mark message as read: {e}")
        
        # Handle different message types
        if message_type == "COLLABORATION":
            await self._handle_collaboration_request(message)
        elif message_type == "RESOURCE_REQUEST":
            await self._handle_resource_request(message)
        elif message_type == "STATUS_UPDATE":
            await self._handle_status_update(message)
        
        logger.info(f"Agent {self.agent_name} processed {message_type} from {from_agent}")
    
    async def _handle_collaboration_request(self, message: Dict[str, Any]):
        """Handle collaboration request from another agent"""
        # Default implementation - can be overridden by subclasses
        from_agent_id = message['from_agent_id']
        self.collaboration_partners.add(from_agent_id)
    
    async def _handle_resource_request(self, message: Dict[str, Any]):
        """Handle resource request from another agent"""
        # Default implementation - can be overridden by subclasses
        pass
    
    async def _handle_status_update(self, message: Dict[str, Any]):
        """Handle status update from another agent"""
        # Default implementation - can be overridden by subclasses
        pass
    
    async def run_agent_loop(self):
        """Main agent operational loop"""
        logger.info(f"Starting agent loop for {self.agent_name}")
        
        while self.is_active:
            try:
                # Update heartbeat
                await self.update_heartbeat()
                
                # Participate in task market
                await self.participate_in_task_market()
                
                # Check for task assignments
                await self.check_task_assignments()
                
                # Check for messages
                await self.check_messages()
                
                # Agent-specific processing
                await self.agent_specific_processing()
                
                # Sleep between cycles
                await asyncio.sleep(30)  # 30 second cycles
                
            except Exception as e:
                logger.error(f"Error in agent loop for {self.agent_name}: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    @abstractmethod
    async def agent_specific_processing(self):
        """Agent-specific processing that runs each cycle"""
        pass
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        logger.info(f"Shutting down agent {self.agent_name}")
        
        self.is_active = False
        
        # Close gRPC connections
        for channel in self.grpc_channels.values():
            await channel.close()
        
        # Mark as inactive in database
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE Agents SET is_active = false WHERE id = %s",
                    (self.agent_id,)
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to mark agent as inactive: {e}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        if hasattr(self, 'is_active') and self.is_active:
            asyncio.create_task(self.shutdown())