#!/usr/bin/env python3
"""
gRPC Server - Project Kairos Symbiotic Interface
High-performance RPC server for internal agent communication.

This server provides:
1. Fast bi-directional communication between agents
2. Streaming interfaces for real-time coordination
3. Task bidding and assignment protocols
4. Oracle prediction request/response
5. Economic transaction processing
6. Causal ledger updates

Author: Kairos Development Team
Version: 2.0
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncIterator

import grpc
from grpc import aio
from concurrent import futures
import asyncpg
import redis.asyncio as redis

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import generated protobuf classes (would be generated from .proto files)
# For now, we'll define the service classes directly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# gRPC Service Implementations
class AgentCommunicationService:
    """Core agent-to-agent communication service"""
    
    def __init__(self, db_pool, redis_client, oracle_engine, economy_engine):
        self.db_pool = db_pool
        self.redis = redis_client
        self.oracle = oracle_engine
        self.economy = economy_engine
        self.connected_agents = {}  # Track connected agents
    
    async def RegisterAgent(self, request, context):
        """Register an agent with the communication service"""
        try:
            agent_id = request.agent_id
            agent_info = {
                'id': agent_id,
                'name': request.agent_name,
                'role': request.agent_role,
                'capabilities': list(request.capabilities),
                'connected_at': datetime.now().isoformat(),
                'last_heartbeat': datetime.now().isoformat()
            }
            
            self.connected_agents[agent_id] = agent_info
            
            # Update agent status in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE Agents 
                    SET status = 'ACTIVE', last_activity = NOW()
                    WHERE id = $1
                """, agent_id)
            
            logger.info(f"Agent {agent_id} registered successfully")
            
            return {
                'success': True,
                'message': f'Agent {agent_id} registered',
                'assigned_id': agent_id
            }
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return {
                'success': False,
                'message': str(e),
                'assigned_id': ''
            }
    
    async def SendMessage(self, request, context):
        """Send message from one agent to another"""
        try:
            sender_id = request.sender_id
            recipient_id = request.recipient_id
            message_type = request.message_type
            content = request.content
            
            message = {
                'id': str(uuid.uuid4()),
                'sender_id': sender_id,
                'recipient_id': recipient_id,
                'message_type': message_type,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store message in database for audit trail
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO agent_messages (id, sender_id, recipient_id, message_type, content, timestamp)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                """, message['id'], sender_id, recipient_id, message_type, content)
            
            # Publish to Redis for real-time delivery
            channel = f"agent_messages:{recipient_id}"
            await self.redis.publish(channel, json.dumps(message))
            
            logger.info(f"Message sent from {sender_id} to {recipient_id}")
            
            return {
                'success': True,
                'message_id': message['id'],
                'delivered_at': message['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return {
                'success': False,
                'message_id': '',
                'error': str(e)
            }
    
    async def StreamMessages(self, request, context):
        """Stream incoming messages to an agent"""
        agent_id = request.agent_id
        
        try:
            # Subscribe to agent's message channel
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(f"agent_messages:{agent_id}")
            
            logger.info(f"Started message stream for agent {agent_id}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    message_data = json.loads(message['data'])
                    
                    # Yield message to agent
                    yield {
                        'message_id': message_data['id'],
                        'sender_id': message_data['sender_id'],
                        'message_type': message_data['message_type'],
                        'content': message_data['content'],
                        'timestamp': message_data['timestamp']
                    }
                    
        except asyncio.CancelledError:
            await pubsub.unsubscribe(f"agent_messages:{agent_id}")
            await pubsub.close()
            logger.info(f"Message stream closed for agent {agent_id}")
    
    async def RequestTaskBid(self, request, context):
        """Request bids for a specific task"""
        try:
            task_id = request.task_id
            requesting_agent = request.requesting_agent_id
            
            # Get task details
            async with self.db_pool.acquire() as conn:
                task = await conn.fetchrow("""
                    SELECT * FROM Tasks WHERE id = $1
                """, task_id)
                
                if not task:
                    return {
                        'success': False,
                        'error': 'Task not found'
                    }
            
            # Broadcast bid request to relevant agents
            bid_request = {
                'request_id': str(uuid.uuid4()),
                'task_id': task_id,
                'task_description': task['description'],
                'cc_bounty': float(task['cc_bounty']),
                'estimated_hours': float(task.get('estimated_hours', 0) or 0),
                'requesting_agent': requesting_agent,
                'deadline': (datetime.now() + timedelta(hours=2)).isoformat()
            }
            
            # Publish to all agents capable of handling this task type
            await self.redis.publish("task_bid_requests", json.dumps(bid_request))
            
            return {
                'success': True,
                'request_id': bid_request['request_id'],
                'broadcast_count': len(self.connected_agents)
            }
            
        except Exception as e:
            logger.error(f"Failed to request task bid: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def SubmitBid(self, request, context):
        """Submit a bid for a task"""
        try:
            bid_data = {
                'id': str(uuid.uuid4()),
                'task_id': request.task_id,
                'agent_id': request.agent_id,
                'cc_amount': request.cc_amount,
                'estimated_hours': request.estimated_hours,
                'proposal': request.proposal,
                'confidence_score': request.confidence_score,
                'status': 'ACTIVE'
            }
            
            # Validate agent has sufficient CC balance
            async with self.db_pool.acquire() as conn:
                agent_balance = await conn.fetchval("""
                    SELECT cc_balance FROM Agents WHERE id = $1
                """, request.agent_id)
                
                if agent_balance < request.cc_amount:
                    return {
                        'success': False,
                        'bid_id': '',
                        'error': 'Insufficient CC balance'
                    }
                
                # Insert bid
                await conn.execute("""
                    INSERT INTO Bids (id, task_id, agent_id, cc_amount, estimated_hours, 
                                     proposal, confidence_score, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                """, bid_data['id'], bid_data['task_id'], bid_data['agent_id'],
                bid_data['cc_amount'], bid_data['estimated_hours'], bid_data['proposal'],
                bid_data['confidence_score'], bid_data['status'])
            
            # Notify task owner about new bid
            notification = {
                'type': 'new_bid',
                'bid_id': bid_data['id'],
                'task_id': bid_data['task_id'],
                'agent_id': bid_data['agent_id'],
                'cc_amount': bid_data['cc_amount']
            }
            
            await self.redis.publish("bid_notifications", json.dumps(notification))
            
            logger.info(f"Bid {bid_data['id']} submitted by agent {request.agent_id}")
            
            return {
                'success': True,
                'bid_id': bid_data['id'],
                'submitted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to submit bid: {e}")
            return {
                'success': False,
                'bid_id': '',
                'error': str(e)
            }

class OracleService:
    """Service for Oracle prediction requests"""
    
    def __init__(self, db_pool, redis_client, oracle_engine):
        self.db_pool = db_pool
        self.redis = redis_client
        self.oracle = oracle_engine
    
    async def RequestPrediction(self, request, context):
        """Request a prediction from the Oracle"""
        try:
            prediction_request = {
                'agent_id': request.agent_id,
                'scenario_type': request.scenario_type,
                'parameters': json.loads(request.parameters),
                'prediction_horizon': timedelta(hours=request.prediction_horizon_hours),
                'confidence_threshold': request.confidence_threshold
            }
            
            # Generate prediction using Oracle engine
            prediction = await self.oracle.generate_prediction(prediction_request)
            
            # Store prediction in database
            simulation_id = str(uuid.uuid4())
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO Simulations (id, scenario_type, parameters, results, 
                                           confidence_score, created_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                """, simulation_id, request.scenario_type, 
                json.dumps(prediction_request['parameters']),
                json.dumps(prediction),
                prediction.get('confidence_scores', {}).get('overall', 0.8))
            
            logger.info(f"Oracle prediction {simulation_id} generated for agent {request.agent_id}")
            
            return {
                'success': True,
                'prediction_id': simulation_id,
                'confidence_score': prediction.get('confidence_scores', {}).get('overall', 0.8),
                'predictions': json.dumps(prediction.get('scenarios', {})),
                'recommendations': json.dumps(prediction.get('recommendations', {})),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate Oracle prediction: {e}")
            return {
                'success': False,
                'prediction_id': '',
                'error': str(e)
            }
    
    async def StreamPredictions(self, request, context):
        """Stream Oracle predictions to subscribed agents"""
        agent_id = request.agent_id
        scenario_types = list(request.scenario_types) if request.scenario_types else ['all']
        
        try:
            # Subscribe to Oracle prediction updates
            channels = [f"oracle_predictions:{st}" for st in scenario_types]
            if 'all' in scenario_types:
                channels.append("oracle_predictions:all")
            
            pubsub = self.redis.pubsub()
            for channel in channels:
                await pubsub.subscribe(channel)
            
            logger.info(f"Started Oracle prediction stream for agent {agent_id}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    prediction_data = json.loads(message['data'])
                    
                    yield {
                        'prediction_id': prediction_data['id'],
                        'scenario_type': prediction_data['scenario_type'],
                        'confidence_score': prediction_data['confidence_score'],
                        'predictions': json.dumps(prediction_data['predictions']),
                        'recommendations': json.dumps(prediction_data['recommendations']),
                        'timestamp': prediction_data['timestamp']
                    }
                    
        except asyncio.CancelledError:
            for channel in channels:
                await pubsub.unsubscribe(channel)
            await pubsub.close()
            logger.info(f"Oracle prediction stream closed for agent {agent_id}")

class EconomyService:
    """Service for economic transactions and CC management"""
    
    def __init__(self, db_pool, redis_client, economy_engine):
        self.db_pool = db_pool
        self.redis = redis_client
        self.economy = economy_engine
    
    async def GetBalance(self, request, context):
        """Get agent's current CC balance"""
        try:
            async with self.db_pool.acquire() as conn:
                balance = await conn.fetchval("""
                    SELECT cc_balance FROM Agents WHERE id = $1
                """, request.agent_id)
                
                if balance is None:
                    return {
                        'success': False,
                        'balance': 0.0,
                        'error': 'Agent not found'
                    }
                
                return {
                    'success': True,
                    'balance': float(balance),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get balance for agent {request.agent_id}: {e}")
            return {
                'success': False,
                'balance': 0.0,
                'error': str(e)
            }
    
    async def TransferCC(self, request, context):
        """Transfer CC between agents"""
        try:
            # Validate and process transfer using economy engine
            transfer_result = await self.economy.transfer_cc(
                from_agent=request.from_agent_id,
                to_agent=request.to_agent_id,
                amount=request.amount,
                reason=request.reason,
                metadata={'grpc_transfer': True}
            )
            
            if transfer_result['success']:
                # Notify both agents
                notification = {
                    'type': 'cc_transfer',
                    'transaction_id': transfer_result['transaction_id'],
                    'from_agent': request.from_agent_id,
                    'to_agent': request.to_agent_id,
                    'amount': request.amount,
                    'reason': request.reason
                }
                
                await self.redis.publish(f"agent_notifications:{request.from_agent_id}", 
                                       json.dumps(notification))
                await self.redis.publish(f"agent_notifications:{request.to_agent_id}", 
                                       json.dumps(notification))
                
                logger.info(f"CC transfer completed: {request.amount} CC from {request.from_agent_id} to {request.to_agent_id}")
                
                return {
                    'success': True,
                    'transaction_id': transfer_result['transaction_id'],
                    'new_balance_sender': transfer_result['new_balance_sender'],
                    'new_balance_recipient': transfer_result['new_balance_recipient'],
                    'processed_at': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'transaction_id': '',
                    'error': transfer_result.get('error', 'Transfer failed')
                }
                
        except Exception as e:
            logger.error(f"Failed to transfer CC: {e}")
            return {
                'success': False,
                'transaction_id': '',
                'error': str(e)
            }
    
    async def GetTransactionHistory(self, request, context):
        """Get transaction history for an agent"""
        try:
            agent_id = request.agent_id
            limit = min(request.limit, 100)  # Cap at 100
            
            # Get transaction history from economy engine
            transactions = await self.economy.get_transaction_history(
                agent_id=agent_id,
                limit=limit
            )
            
            for transaction in transactions:
                yield {
                    'transaction_id': transaction['id'],
                    'type': transaction['type'],
                    'amount': transaction['amount'],
                    'counterparty': transaction.get('counterparty', ''),
                    'reason': transaction.get('reason', ''),
                    'timestamp': transaction['timestamp'],
                    'balance_after': transaction.get('balance_after', 0.0)
                }
                
        except Exception as e:
            logger.error(f"Failed to get transaction history: {e}")
            # Return empty stream on error

class CausalLedgerService:
    """Service for causal ledger operations"""
    
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis = redis_client
    
    async def RecordDecision(self, request, context):
        """Record a decision in the causal ledger"""
        try:
            decision_data = {
                'id': str(uuid.uuid4()),
                'agent_id': request.agent_id,
                'decision_type': request.decision_type,
                'decision_data': json.loads(request.decision_data),
                'context': json.loads(request.context) if request.context else None,
                'causal_chain_id': request.causal_chain_id if request.causal_chain_id else None
            }
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO Decisions (id, agent_id, decision_type, decision_data, 
                                         context, causal_chain_id, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW())
                """, decision_data['id'], decision_data['agent_id'], 
                decision_data['decision_type'], json.dumps(decision_data['decision_data']),
                json.dumps(decision_data['context']) if decision_data['context'] else None,
                decision_data['causal_chain_id'])
            
            # Publish to decision stream
            notification = {
                'decision_id': decision_data['id'],
                'agent_id': decision_data['agent_id'],
                'decision_type': decision_data['decision_type'],
                'timestamp': datetime.now().isoformat()
            }
            
            await self.redis.publish("decisions:all", json.dumps(notification))
            await self.redis.publish(f"decisions:{decision_data['agent_id']}", 
                                   json.dumps(notification))
            
            logger.info(f"Decision {decision_data['id']} recorded for agent {request.agent_id}")
            
            return {
                'success': True,
                'decision_id': decision_data['id'],
                'recorded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to record decision: {e}")
            return {
                'success': False,
                'decision_id': '',
                'error': str(e)
            }
    
    async def StreamDecisions(self, request, context):
        """Stream causal ledger updates"""
        agent_id = request.agent_id if hasattr(request, 'agent_id') else None
        
        try:
            # Subscribe to decision stream
            channel = f"decisions:{agent_id}" if agent_id else "decisions:all"
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel)
            
            logger.info(f"Started decision stream for {agent_id or 'all agents'}")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    decision_data = json.loads(message['data'])
                    
                    yield {
                        'decision_id': decision_data['decision_id'],
                        'agent_id': decision_data['agent_id'],
                        'decision_type': decision_data['decision_type'],
                        'timestamp': decision_data['timestamp']
                    }
                    
        except asyncio.CancelledError:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            logger.info(f"Decision stream closed")

class HealthService:
    """Health check and system status service"""
    
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis = redis_client
    
    async def CheckHealth(self, request, context):
        """Check overall system health"""
        try:
            # Check database connectivity
            db_healthy = False
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                db_healthy = True
            except Exception as e:
                logger.warning(f"Database health check failed: {e}")
            
            # Check Redis connectivity
            redis_healthy = False
            try:
                await self.redis.ping()
                redis_healthy = True
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
            
            # Get system metrics
            async with self.db_pool.acquire() as conn:
                metrics = await conn.fetchrow("""
                    SELECT 
                        (SELECT COUNT(*) FROM Agents WHERE status = 'ACTIVE') as active_agents,
                        (SELECT COUNT(*) FROM Tasks WHERE status IN ('BOUNTY_POSTED', 'BIDDING')) as pending_tasks,
                        (SELECT COUNT(*) FROM Ventures WHERE status = 'ACTIVE') as active_ventures
                """)
            
            overall_healthy = db_healthy and redis_healthy
            
            return {
                'healthy': overall_healthy,
                'database_healthy': db_healthy,
                'redis_healthy': redis_healthy,
                'active_agents': metrics['active_agents'] or 0,
                'pending_tasks': metrics['pending_tasks'] or 0,
                'active_ventures': metrics['active_ventures'] or 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# gRPC Server Setup
class KairosGRPCServer:
    """Main gRPC server for Kairos agent communication"""
    
    def __init__(self):
        self.server = None
        self.db_pool = None
        self.redis_client = None
        self.oracle_engine = None
        self.economy_engine = None
        
        # Services
        self.agent_service = None
        self.oracle_service = None
        self.economy_service = None
        self.ledger_service = None
        self.health_service = None
    
    async def initialize(self):
        """Initialize database connections and services"""
        try:
            # Database configuration
            db_config = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', 5432)),
                'database': os.getenv('POSTGRES_DB', 'kairos'),
                'user': os.getenv('POSTGRES_USER', 'kairos'),
                'password': os.getenv('POSTGRES_PASSWORD', 'kairos_password')
            }
            
            # Create database pool
            self.db_pool = await asyncpg.create_pool(**db_config)
            logger.info("Database pool created")
            
            # Create Redis client
            self.redis_client = await redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0))
            )
            logger.info("Redis client connected")
            
            # Import and initialize Oracle and Economy engines (would be actual imports)
            # For now, using mock objects
            self.oracle_engine = type('MockOracle', (), {
                'generate_prediction': lambda self, request: {
                    'prediction_id': str(uuid.uuid4()),
                    'scenarios': {'scenario1': 'result1'},
                    'recommendations': {'rec1': 'value1'},
                    'confidence_scores': {'overall': 0.85}
                }
            })()
            
            self.economy_engine = type('MockEconomy', (), {
                'transfer_cc': lambda self, **kwargs: {
                    'success': True,
                    'transaction_id': str(uuid.uuid4()),
                    'new_balance_sender': 1000.0,
                    'new_balance_recipient': 500.0
                },
                'get_transaction_history': lambda self, **kwargs: []
            })()
            
            # Initialize services
            self.agent_service = AgentCommunicationService(
                self.db_pool, self.redis_client, self.oracle_engine, self.economy_engine
            )
            self.oracle_service = OracleService(
                self.db_pool, self.redis_client, self.oracle_engine
            )
            self.economy_service = EconomyService(
                self.db_pool, self.redis_client, self.economy_engine
            )
            self.ledger_service = CausalLedgerService(
                self.db_pool, self.redis_client
            )
            self.health_service = HealthService(
                self.db_pool, self.redis_client
            )
            
            logger.info("All services initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize gRPC server: {e}")
            raise
    
    async def start_server(self, port=50051):
        """Start the gRPC server"""
        try:
            self.server = aio.server(
                futures.ThreadPoolExecutor(max_workers=100),
                options=[
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000)
                ]
            )
            
            # Add services to server (would use actual protobuf service definitions)
            # For now, this is conceptual
            logger.info("Services added to gRPC server")
            
            listen_addr = f'[::]:{port}'
            self.server.add_insecure_port(listen_addr)
            
            await self.server.start()
            logger.info(f"🚀 Kairos gRPC server started on {listen_addr}")
            
            # Wait for termination
            await self.server.wait_for_termination()
            
        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the gRPC server"""
        if self.server:
            await self.server.stop(5)  # 5 second grace period
            logger.info("gRPC server stopped")
        
        # Close connections
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("All connections closed")

# Main function
async def main():
    """Main function to run the gRPC server"""
    server = KairosGRPCServer()
    
    try:
        await server.initialize()
        await server.start_server(port=50051)
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop_server()
    except Exception as e:
        logger.error(f"Server error: {e}")
        await server.stop_server()

if __name__ == "__main__":
    asyncio.run(main())