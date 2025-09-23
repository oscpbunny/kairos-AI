"""
Project Kairos: Cross-Venture Collaboration Protocol  
Phase 7 - Advanced Intelligence

This module implements sophisticated inter-swarm communication and coordination
protocols that enable multiple Kairos instances to collaborate autonomously:

- Distributed Task Coordination
- Knowledge Sharing and Synchronization  
- Consensus Mechanisms
- Resource Allocation and Load Balancing
- Fault Tolerance and Recovery
- Emergent Swarm Behaviors
"""

import asyncio
import logging
import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import websockets
import aiohttp
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kairos.collaboration.swarm")

class SwarmRole(Enum):
    """Roles that Kairos instances can take in the swarm"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    OBSERVER = "observer"
    RELAY = "relay"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1

class MessageType(Enum):
    """Types of inter-swarm messages"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    KNOWLEDGE_SHARE = "knowledge_share"
    STATUS_UPDATE = "status_update"
    CONSENSUS_VOTE = "consensus_vote"
    RESOURCE_REQUEST = "resource_request"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"

@dataclass
class SwarmNode:
    """Represents a Kairos instance in the swarm"""
    node_id: str
    name: str
    role: SwarmRole
    capabilities: List[str]
    endpoint: str  # WebSocket endpoint
    status: str = "active"
    load: float = 0.0
    reputation: float = 1.0
    last_seen: datetime = field(default_factory=datetime.now)
    version: str = "7.0.0"
    specializations: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)

@dataclass
class SwarmTask:
    """Represents a collaborative task in the swarm"""
    task_id: str
    name: str
    description: str
    priority: TaskPriority
    requirements: Dict[str, Any]
    assigned_nodes: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmMessage:
    """Inter-swarm communication message"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: int = 300  # Time to live in seconds
    priority: int = 3
    requires_ack: bool = False
    correlation_id: Optional[str] = None

@dataclass
class ConsensusProposal:
    """Proposal for swarm consensus"""
    proposal_id: str
    proposer_id: str
    proposal_type: str
    description: str
    data: Dict[str, Any]
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)
    abstentions: Set[str] = field(default_factory=set)
    status: str = "voting"  # voting, approved, rejected, expired
    created_at: datetime = field(default_factory=datetime.now)
    deadline: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=5))

class SwarmIntelligence:
    """Core swarm intelligence coordination system"""
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.config = config
        
        # Node information
        self.my_node = SwarmNode(
            node_id=node_id,
            name=config.get('name', f'Kairos-{node_id[:8]}'),
            role=SwarmRole(config.get('role', 'worker')),
            capabilities=config.get('capabilities', []),
            endpoint=config.get('endpoint', f'ws://localhost:8080/swarm/{node_id}'),
            specializations=config.get('specializations', [])
        )
        
        # Swarm state
        self.swarm_nodes: Dict[str, SwarmNode] = {node_id: self.my_node}
        self.active_tasks: Dict[str, SwarmTask] = {}
        self.message_history: List[SwarmMessage] = []
        self.consensus_proposals: Dict[str, ConsensusProposal] = {}
        
        # Communication
        self.websocket_server = None
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Coordination algorithms
        self.task_allocator = TaskAllocator(self)
        self.consensus_engine = ConsensusEngine(self)
        self.knowledge_sync = KnowledgeSync(self)
        self.load_balancer = LoadBalancer(self)
        
        # Metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'tasks_completed': 0,
            'consensus_decisions': 0,
            'swarm_efficiency': 0.0,
            'network_latency': 0.0
        }
        
        # Setup message handlers
        self._setup_message_handlers()
        
        logger.info(f"🦾 SwarmIntelligence initialized for {self.my_node.name} ({node_id})")
    
    async def initialize(self) -> bool:
        """Initialize the swarm intelligence system"""
        try:
            logger.info(f"🚀 Initializing SwarmIntelligence for {self.my_node.name}...")
            
            # Start WebSocket server
            await self._start_websocket_server()
            
            # Connect to existing swarm nodes
            await self._connect_to_swarm()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._consensus_monitor())
            asyncio.create_task(self._task_monitor())
            asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"✅ SwarmIntelligence initialized for {self.my_node.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SwarmIntelligence: {e}")
            return False
    
    async def _start_websocket_server(self):
        """Start WebSocket server for inter-swarm communication"""
        try:
            port = int(self.my_node.endpoint.split(':')[-1].split('/')[0])
            
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                "0.0.0.0",
                port,
                ping_interval=20,
                ping_timeout=10
            )
            
            logger.info(f"🌐 WebSocket server started on {self.my_node.endpoint}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            raise
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        try:
            # Extract node ID from path
            node_id = path.split('/')[-1] if '/' in path else 'unknown'
            
            logger.info(f"🔗 New connection from node: {node_id}")
            self.connections[node_id] = websocket
            
            try:
                async for raw_message in websocket:
                    message_data = json.loads(raw_message)
                    message = SwarmMessage(**message_data)
                    await self._process_message(message)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"🔌 Connection closed for node: {node_id}")
            except Exception as e:
                logger.error(f"❌ Error handling message from {node_id}: {e}")
                
        finally:
            if node_id in self.connections:
                del self.connections[node_id]
    
    async def _connect_to_swarm(self):
        """Connect to existing swarm nodes"""
        bootstrap_nodes = self.config.get('bootstrap_nodes', [])
        
        for node_endpoint in bootstrap_nodes:
            try:
                async with aiohttp.ClientSession() as session:
                    # Discover other nodes
                    discovery_url = node_endpoint.replace('ws://', 'http://').replace('wss://', 'https://') + '/discover'
                    
                    async with session.get(discovery_url) as response:
                        if response.status == 200:
                            nodes_data = await response.json()
                            
                            for node_data in nodes_data.get('nodes', []):
                                node = SwarmNode(**node_data)
                                if node.node_id != self.node_id:
                                    self.swarm_nodes[node.node_id] = node
                                    
                            logger.info(f"🔍 Discovered {len(nodes_data.get('nodes', []))} swarm nodes")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to connect to bootstrap node {node_endpoint}: {e}")
        
        # Announce our presence
        await self._announce_presence()
    
    async def _announce_presence(self):
        """Announce this node's presence to the swarm"""
        announcement = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.STATUS_UPDATE,
            payload={
                'action': 'node_join',
                'node': asdict(self.my_node),
                'capabilities': self.my_node.capabilities,
                'specializations': self.my_node.specializations
            }
        )
        
        await self._broadcast_message(announcement)
        logger.info(f"📢 Announced presence to swarm")
    
    def _setup_message_handlers(self):
        """Setup message handlers for different message types"""
        self.message_handlers = {
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.TASK_RESPONSE: self._handle_task_response,
            MessageType.TASK_ASSIGNMENT: self._handle_task_assignment,
            MessageType.TASK_COMPLETION: self._handle_task_completion,
            MessageType.KNOWLEDGE_SHARE: self._handle_knowledge_share,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.CONSENSUS_VOTE: self._handle_consensus_vote,
            MessageType.RESOURCE_REQUEST: self._handle_resource_request,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.COORDINATION: self._handle_coordination,
            MessageType.EMERGENCY: self._handle_emergency
        }
    
    async def _process_message(self, message: SwarmMessage):
        """Process incoming swarm message"""
        try:
            self.metrics['messages_received'] += 1
            self.message_history.append(message)
            
            # Check TTL
            if (datetime.now() - message.timestamp).total_seconds() > message.ttl:
                logger.warning(f"⏰ Message {message.message_id} expired, ignoring")
                return
            
            # Route to appropriate handler
            if message.message_type in self.message_handlers:
                handler = self.message_handlers[message.message_type]
                await handler(message)
            else:
                logger.warning(f"❓ Unknown message type: {message.message_type}")
            
            # Send acknowledgment if required
            if message.requires_ack:
                await self._send_acknowledgment(message)
                
        except Exception as e:
            logger.error(f"❌ Error processing message {message.message_id}: {e}")
    
    async def _broadcast_message(self, message: SwarmMessage):
        """Broadcast message to all connected nodes"""
        message_data = json.dumps(asdict(message), default=str)
        disconnected_nodes = []
        
        for node_id, connection in self.connections.items():
            try:
                await connection.send(message_data)
                logger.debug(f"📤 Sent message {message.message_id} to {node_id}")
                
            except websockets.exceptions.ConnectionClosed:
                disconnected_nodes.append(node_id)
            except Exception as e:
                logger.error(f"❌ Failed to send message to {node_id}: {e}")
        
        # Clean up disconnected nodes
        for node_id in disconnected_nodes:
            if node_id in self.connections:
                del self.connections[node_id]
            if node_id in self.swarm_nodes:
                self.swarm_nodes[node_id].status = "disconnected"
        
        self.metrics['messages_sent'] += len(self.connections) - len(disconnected_nodes)
    
    async def _send_message(self, message: SwarmMessage):
        """Send message to specific recipient or broadcast"""
        if message.recipient_id is None:
            await self._broadcast_message(message)
        else:
            if message.recipient_id in self.connections:
                try:
                    message_data = json.dumps(asdict(message), default=str)
                    await self.connections[message.recipient_id].send(message_data)
                    self.metrics['messages_sent'] += 1
                    logger.debug(f"📤 Sent message {message.message_id} to {message.recipient_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to send message to {message.recipient_id}: {e}")
            else:
                logger.warning(f"⚠️ Recipient {message.recipient_id} not connected")
    
    # Message Handlers
    async def _handle_task_request(self, message: SwarmMessage):
        """Handle task request from another node"""
        task_data = message.payload.get('task')
        
        if task_data and self._can_handle_task(task_data):
            # Evaluate our capability to handle this task
            capability_score = self._evaluate_task_capability(task_data)
            
            response = SwarmMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.node_id,
                recipient_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                payload={
                    'task_id': task_data.get('task_id'),
                    'capability_score': capability_score,
                    'estimated_time': self._estimate_task_time(task_data),
                    'current_load': self.my_node.load,
                    'available_resources': self.my_node.resources
                },
                correlation_id=message.message_id
            )
            
            await self._send_message(response)
            logger.info(f"🎯 Responded to task request {task_data.get('task_id')} with score {capability_score}")
    
    async def _handle_task_response(self, message: SwarmMessage):
        """Handle task response from another node"""
        task_id = message.payload.get('task_id')
        capability_score = message.payload.get('capability_score', 0.0)
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            # Store the response for task allocation decision
            if 'responses' not in task.metadata:
                task.metadata['responses'] = []
            
            task.metadata['responses'].append({
                'node_id': message.sender_id,
                'capability_score': capability_score,
                'estimated_time': message.payload.get('estimated_time'),
                'current_load': message.payload.get('current_load'),
                'timestamp': datetime.now()
            })
            
            logger.info(f"📥 Received task response for {task_id} from {message.sender_id}")
    
    async def _handle_task_assignment(self, message: SwarmMessage):
        """Handle task assignment from coordinator"""
        task_data = message.payload.get('task')
        
        if task_data:
            task = SwarmTask(**task_data)
            self.active_tasks[task.task_id] = task
            
            logger.info(f"📋 Assigned task: {task.name} ({task.task_id})")
            
            # Start working on the task
            asyncio.create_task(self._execute_task(task))
    
    async def _handle_task_completion(self, message: SwarmMessage):
        """Handle task completion notification"""
        task_id = message.payload.get('task_id')
        result = message.payload.get('result')
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "completed"
            task.result = result
            task.progress = 1.0
            
            self.metrics['tasks_completed'] += 1
            logger.info(f"✅ Task completed: {task.name} ({task_id})")
    
    async def _handle_knowledge_share(self, message: SwarmMessage):
        """Handle knowledge sharing from another node"""
        knowledge_data = message.payload.get('knowledge')
        knowledge_type = message.payload.get('type')
        
        if knowledge_data:
            await self.knowledge_sync.process_shared_knowledge(knowledge_type, knowledge_data, message.sender_id)
            logger.info(f"🧠 Received knowledge share of type {knowledge_type} from {message.sender_id}")
    
    async def _handle_status_update(self, message: SwarmMessage):
        """Handle status update from another node"""
        action = message.payload.get('action')
        
        if action == 'node_join':
            node_data = message.payload.get('node')
            if node_data:
                node = SwarmNode(**node_data)
                self.swarm_nodes[node.node_id] = node
                logger.info(f"🤝 Node joined swarm: {node.name} ({node.node_id})")
        
        elif action == 'node_leave':
            node_id = message.payload.get('node_id')
            if node_id in self.swarm_nodes:
                self.swarm_nodes[node_id].status = "disconnected"
                logger.info(f"👋 Node left swarm: {node_id}")
        
        elif action == 'status_update':
            node_id = message.sender_id
            if node_id in self.swarm_nodes:
                node = self.swarm_nodes[node_id]
                node.load = message.payload.get('load', node.load)
                node.last_seen = datetime.now()
                node.status = message.payload.get('status', 'active')
    
    async def _handle_consensus_vote(self, message: SwarmMessage):
        """Handle consensus vote"""
        await self.consensus_engine.process_vote(message)
    
    async def _handle_resource_request(self, message: SwarmMessage):
        """Handle resource request from another node"""
        resource_type = message.payload.get('resource_type')
        amount = message.payload.get('amount')
        
        if resource_type in self.my_node.resources:
            available = self.my_node.resources[resource_type]
            if available >= amount:
                # Grant resource request
                response = SwarmMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.node_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.COORDINATION,
                    payload={
                        'action': 'resource_grant',
                        'resource_type': resource_type,
                        'amount': amount,
                        'expires_at': (datetime.now() + timedelta(minutes=30)).isoformat()
                    }
                )
                
                await self._send_message(response)
                self.my_node.resources[resource_type] -= amount
                
                logger.info(f"💰 Granted {amount} {resource_type} to {message.sender_id}")
    
    async def _handle_heartbeat(self, message: SwarmMessage):
        """Handle heartbeat from another node"""
        node_id = message.sender_id
        if node_id in self.swarm_nodes:
            self.swarm_nodes[node_id].last_seen = datetime.now()
            self.swarm_nodes[node_id].status = "active"
    
    async def _handle_coordination(self, message: SwarmMessage):
        """Handle coordination message"""
        action = message.payload.get('action')
        
        if action == 'resource_grant':
            resource_type = message.payload.get('resource_type')
            amount = message.payload.get('amount')
            
            if resource_type not in self.my_node.resources:
                self.my_node.resources[resource_type] = 0
            
            self.my_node.resources[resource_type] += amount
            logger.info(f"💰 Received {amount} {resource_type} from {message.sender_id}")
    
    async def _handle_emergency(self, message: SwarmMessage):
        """Handle emergency message"""
        emergency_type = message.payload.get('emergency_type')
        severity = message.payload.get('severity', 'medium')
        
        logger.warning(f"🚨 EMERGENCY: {emergency_type} (severity: {severity}) from {message.sender_id}")
        
        # Implement emergency response protocols
        if emergency_type == 'node_failure':
            await self._handle_node_failure(message.payload)
        elif emergency_type == 'task_failure':
            await self._handle_task_failure(message.payload)
        elif emergency_type == 'resource_exhaustion':
            await self._handle_resource_exhaustion(message.payload)
    
    # Core Methods
    async def submit_task(self, task: SwarmTask) -> str:
        """Submit a task to the swarm for collaborative execution"""
        task.task_id = str(uuid.uuid4())
        self.active_tasks[task.task_id] = task
        
        # Request task execution from swarm
        request = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.TASK_REQUEST,
            payload={'task': asdict(task)},
            requires_ack=False
        )
        
        await self._broadcast_message(request)
        logger.info(f"📤 Submitted task to swarm: {task.name} ({task.task_id})")
        
        return task.task_id
    
    async def propose_consensus(self, proposal_type: str, description: str, data: Dict[str, Any]) -> str:
        """Propose a decision for swarm consensus"""
        proposal_id = str(uuid.uuid4())
        
        proposal = ConsensusProposal(
            proposal_id=proposal_id,
            proposer_id=self.node_id,
            proposal_type=proposal_type,
            description=description,
            data=data
        )
        
        self.consensus_proposals[proposal_id] = proposal
        
        # Broadcast proposal to swarm
        message = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            recipient_id=None,
            message_type=MessageType.CONSENSUS_VOTE,
            payload={
                'action': 'proposal',
                'proposal': asdict(proposal)
            }
        )
        
        await self._broadcast_message(message)
        logger.info(f"🗳️ Proposed consensus: {description} ({proposal_id})")
        
        return proposal_id
    
    async def share_knowledge(self, knowledge_type: str, knowledge_data: Dict[str, Any]):
        """Share knowledge with the swarm"""
        message = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            recipient_id=None,
            message_type=MessageType.KNOWLEDGE_SHARE,
            payload={
                'type': knowledge_type,
                'knowledge': knowledge_data,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        await self._broadcast_message(message)
        logger.info(f"🧠 Shared knowledge of type: {knowledge_type}")
    
    # Background Tasks
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain swarm connectivity"""
        while True:
            try:
                heartbeat = SwarmMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.node_id,
                    recipient_id=None,
                    message_type=MessageType.HEARTBEAT,
                    payload={
                        'load': self.my_node.load,
                        'status': self.my_node.status,
                        'active_tasks': len(self.active_tasks),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                await self._broadcast_message(heartbeat)
                
                # Update our own last_seen
                self.my_node.last_seen = datetime.now()
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Heartbeat error: {e}")
                await asyncio.sleep(30)
    
    async def _consensus_monitor(self):
        """Monitor consensus proposals and handle voting"""
        while True:
            try:
                current_time = datetime.now()
                expired_proposals = []
                
                for proposal_id, proposal in self.consensus_proposals.items():
                    if proposal.status == "voting" and current_time > proposal.deadline:
                        # Calculate consensus result
                        total_votes = len(proposal.votes_for) + len(proposal.votes_against)
                        if total_votes > 0:
                            approval_rate = len(proposal.votes_for) / total_votes
                            if approval_rate > 0.5:
                                proposal.status = "approved"
                                await self._execute_consensus_decision(proposal)
                            else:
                                proposal.status = "rejected"
                        else:
                            proposal.status = "expired"
                        
                        expired_proposals.append(proposal_id)
                
                # Clean up expired proposals
                for proposal_id in expired_proposals:
                    if self.consensus_proposals[proposal_id].status in ["approved", "rejected", "expired"]:
                        self.metrics['consensus_decisions'] += 1
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Consensus monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _task_monitor(self):
        """Monitor active tasks and handle timeouts"""
        while True:
            try:
                current_time = datetime.now()
                completed_tasks = []
                
                for task_id, task in self.active_tasks.items():
                    # Check for task timeouts
                    if task.deadline and current_time > task.deadline and task.status != "completed":
                        task.status = "timeout"
                        logger.warning(f"⏰ Task timeout: {task.name} ({task_id})")
                        
                        # Notify swarm of task failure
                        failure_message = SwarmMessage(
                            message_id=str(uuid.uuid4()),
                            sender_id=self.node_id,
                            recipient_id=None,
                            message_type=MessageType.EMERGENCY,
                            payload={
                                'emergency_type': 'task_failure',
                                'task_id': task_id,
                                'reason': 'timeout'
                            }
                        )
                        await self._broadcast_message(failure_message)
                    
                    # Clean up completed tasks
                    elif task.status == "completed":
                        completed_tasks.append(task_id)
                
                # Remove completed tasks after some delay
                for task_id in completed_tasks:
                    task = self.active_tasks[task_id]
                    if (current_time - task.created_at).total_seconds() > 300:  # 5 minutes
                        del self.active_tasks[task_id]
                
                await asyncio.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                logger.error(f"❌ Task monitor error: {e}")
                await asyncio.sleep(15)
    
    async def _cleanup_loop(self):
        """Clean up old messages and disconnected nodes"""
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up old messages
                cutoff_time = current_time - timedelta(hours=1)
                self.message_history = [
                    msg for msg in self.message_history 
                    if msg.timestamp > cutoff_time
                ]
                
                # Update node statuses based on last seen
                for node_id, node in self.swarm_nodes.items():
                    if node_id != self.node_id:  # Don't check ourselves
                        time_since_seen = (current_time - node.last_seen).total_seconds()
                        if time_since_seen > 120:  # 2 minutes
                            node.status = "unreachable"
                        elif time_since_seen > 300:  # 5 minutes
                            node.status = "disconnected"
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")
                await asyncio.sleep(60)
    
    # Utility Methods
    def _can_handle_task(self, task_data: Dict[str, Any]) -> bool:
        """Check if this node can handle the given task"""
        required_capabilities = task_data.get('requirements', {}).get('capabilities', [])
        return all(cap in self.my_node.capabilities for cap in required_capabilities)
    
    def _evaluate_task_capability(self, task_data: Dict[str, Any]) -> float:
        """Evaluate how well this node can handle a task (0.0 - 1.0)"""
        if not self._can_handle_task(task_data):
            return 0.0
        
        # Base score from capability match
        required_capabilities = task_data.get('requirements', {}).get('capabilities', [])
        capability_score = len(required_capabilities) / len(self.my_node.capabilities) if self.my_node.capabilities else 0.0
        
        # Adjust based on current load
        load_penalty = self.my_node.load * 0.5
        
        # Adjust based on specialization match
        specialization_bonus = 0.0
        task_type = task_data.get('requirements', {}).get('type')
        if task_type in self.my_node.specializations:
            specialization_bonus = 0.3
        
        final_score = min(1.0, capability_score - load_penalty + specialization_bonus)
        return max(0.0, final_score)
    
    def _estimate_task_time(self, task_data: Dict[str, Any]) -> float:
        """Estimate time to complete task in seconds"""
        complexity = task_data.get('requirements', {}).get('complexity', 'medium')
        
        base_times = {
            'simple': 60,
            'medium': 300,
            'complex': 900,
            'expert': 1800
        }
        
        base_time = base_times.get(complexity, 300)
        
        # Adjust based on current load
        load_multiplier = 1.0 + (self.my_node.load * 0.5)
        
        return base_time * load_multiplier
    
    async def _execute_task(self, task: SwarmTask):
        """Execute an assigned task"""
        try:
            logger.info(f"🔄 Starting task execution: {task.name}")
            task.status = "running"
            
            # Simulate task execution
            estimated_time = self._estimate_task_time(asdict(task))
            
            # Update progress periodically
            for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
                await asyncio.sleep(estimated_time / 5)
                task.progress = progress
                
                # Send progress update
                progress_message = SwarmMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.node_id,
                    recipient_id=None,
                    message_type=MessageType.STATUS_UPDATE,
                    payload={
                        'action': 'task_progress',
                        'task_id': task.task_id,
                        'progress': progress
                    }
                )
                await self._broadcast_message(progress_message)
            
            # Complete task
            task.status = "completed"
            task.result = {
                'success': True,
                'output': f'Task {task.name} completed successfully',
                'execution_time': estimated_time,
                'completed_by': self.node_id
            }
            
            # Notify completion
            completion_message = SwarmMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.node_id,
                recipient_id=None,
                message_type=MessageType.TASK_COMPLETION,
                payload={
                    'task_id': task.task_id,
                    'result': task.result
                }
            )
            await self._broadcast_message(completion_message)
            
            logger.info(f"✅ Task completed: {task.name}")
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            task.status = "failed"
            task.result = {'success': False, 'error': str(e)}
    
    async def _send_acknowledgment(self, original_message: SwarmMessage):
        """Send acknowledgment for a message"""
        ack_message = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.STATUS_UPDATE,
            payload={'action': 'acknowledgment'},
            correlation_id=original_message.message_id
        )
        
        await self._send_message(ack_message)
    
    # Emergency Handlers
    async def _handle_node_failure(self, payload: Dict[str, Any]):
        """Handle node failure emergency"""
        failed_node_id = payload.get('node_id')
        if failed_node_id in self.swarm_nodes:
            self.swarm_nodes[failed_node_id].status = "failed"
            
            # Reassign tasks from failed node
            reassigned_tasks = []
            for task_id, task in self.active_tasks.items():
                if failed_node_id in task.assigned_nodes:
                    task.assigned_nodes.remove(failed_node_id)
                    reassigned_tasks.append(task_id)
            
            logger.warning(f"🚨 Node failure handled: {failed_node_id}, reassigned {len(reassigned_tasks)} tasks")
    
    async def _handle_task_failure(self, payload: Dict[str, Any]):
        """Handle task failure emergency"""
        task_id = payload.get('task_id')
        reason = payload.get('reason', 'unknown')
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "failed"
            
            logger.warning(f"🚨 Task failure handled: {task.name} - {reason}")
    
    async def _handle_resource_exhaustion(self, payload: Dict[str, Any]):
        """Handle resource exhaustion emergency"""
        resource_type = payload.get('resource_type')
        requesting_node = payload.get('node_id')
        
        # Try to provide resources if available
        if resource_type in self.my_node.resources and self.my_node.resources[resource_type] > 0:
            available = min(self.my_node.resources[resource_type] * 0.2, 100)  # Share up to 20%
            
            resource_message = SwarmMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.node_id,
                recipient_id=requesting_node,
                message_type=MessageType.COORDINATION,
                payload={
                    'action': 'resource_grant',
                    'resource_type': resource_type,
                    'amount': available
                }
            )
            
            await self._send_message(resource_message)
            self.my_node.resources[resource_type] -= available
            
            logger.info(f"💰 Emergency resource aid: {available} {resource_type} to {requesting_node}")
    
    async def _execute_consensus_decision(self, proposal: ConsensusProposal):
        """Execute a consensus decision that was approved"""
        logger.info(f"✅ Executing consensus decision: {proposal.description}")
        
        if proposal.proposal_type == 'role_change':
            # Handle role change consensus
            node_id = proposal.data.get('node_id')
            new_role = proposal.data.get('new_role')
            
            if node_id in self.swarm_nodes:
                self.swarm_nodes[node_id].role = SwarmRole(new_role)
                logger.info(f"👑 Role changed: {node_id} -> {new_role}")
        
        elif proposal.proposal_type == 'task_priority':
            # Handle task priority change
            task_id = proposal.data.get('task_id')
            new_priority = proposal.data.get('new_priority')
            
            if task_id in self.active_tasks:
                self.active_tasks[task_id].priority = TaskPriority(new_priority)
                logger.info(f"🎯 Task priority changed: {task_id} -> {new_priority}")
        
        # Add more consensus decision types as needed
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        active_nodes = sum(1 for node in self.swarm_nodes.values() if node.status == "active")
        
        return {
            'node_id': self.node_id,
            'node_name': self.my_node.name,
            'role': self.my_node.role.value,
            'swarm_size': len(self.swarm_nodes),
            'active_nodes': active_nodes,
            'active_tasks': len(self.active_tasks),
            'connections': len(self.connections),
            'metrics': self.metrics,
            'status': self.my_node.status
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'swarm_nodes': len(self.swarm_nodes),
            'active_connections': len(self.connections),
            'active_tasks': len(self.active_tasks),
            'consensus_proposals': len(self.consensus_proposals)
        }
    
    async def shutdown(self):
        """Shutdown the swarm intelligence system"""
        logger.info("🔄 Shutting down SwarmIntelligence...")
        
        # Announce departure
        departure_message = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.node_id,
            recipient_id=None,
            message_type=MessageType.STATUS_UPDATE,
            payload={
                'action': 'node_leave',
                'node_id': self.node_id
            }
        )
        await self._broadcast_message(departure_message)
        
        # Close connections
        for connection in self.connections.values():
            try:
                await connection.close()
            except:
                pass
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        logger.info("✅ SwarmIntelligence shutdown complete")


# Supporting Classes (simplified for demo)
class TaskAllocator:
    def __init__(self, swarm: SwarmIntelligence):
        self.swarm = swarm

class ConsensusEngine:
    def __init__(self, swarm: SwarmIntelligence):
        self.swarm = swarm
    
    async def process_vote(self, message: SwarmMessage):
        # Simplified vote processing
        pass

class KnowledgeSync:
    def __init__(self, swarm: SwarmIntelligence):
        self.swarm = swarm
    
    async def process_shared_knowledge(self, knowledge_type: str, knowledge_data: Dict[str, Any], sender_id: str):
        # Simplified knowledge synchronization
        pass

class LoadBalancer:
    def __init__(self, swarm: SwarmIntelligence):
        self.swarm = swarm


# Factory function
def create_swarm_intelligence(node_id: str, config: Dict[str, Any]) -> SwarmIntelligence:
    """Create and return a configured swarm intelligence system"""
    return SwarmIntelligence(node_id, config)