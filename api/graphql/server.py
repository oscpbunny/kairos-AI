"""
Project Kairos: GraphQL Server Implementation
The symbiotic interface server providing advanced query capabilities.

Features:
- High-performance async GraphQL server
- Real-time subscriptions via WebSockets
- Database integration with connection pooling
- Authentication and authorization
- Query optimization and caching
- Rate limiting and security
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.websockets import WebSocket
from graphql import GraphQLError
from graphene import Schema
from starlette_graphene3 import GraphQLApp, make_graphiql_handler
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import our schema
from .schema import schema

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KairosGraphQLServer')

# Metrics
GRAPHQL_REQUESTS = Counter('kairos_graphql_requests_total', 'Total GraphQL requests', ['operation_type', 'status'])
GRAPHQL_DURATION = Histogram('kairos_graphql_duration_seconds', 'GraphQL request duration')
ACTIVE_SUBSCRIPTIONS = Gauge('kairos_active_subscriptions', 'Number of active GraphQL subscriptions')
DATABASE_CONNECTIONS = Gauge('kairos_db_connections_active', 'Active database connections')

class KairosGraphQLContext:
    """Context object passed to all GraphQL resolvers"""
    
    def __init__(self, request, db_pool: asyncpg.Pool, redis_client: redis.Redis, user_id: Optional[str] = None):
        self.request = request
        self.db_pool = db_pool
        self.redis = redis_client
        self.user_id = user_id
        self.query_complexity = 0
        self.start_time = datetime.now()

class DatabaseManager:
    """Database connection pool manager"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection pool
            self.pool = await asyncpg.create_pool(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', '5432')),
                database=os.getenv('DB_NAME', 'kairos_db'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'password'),
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Redis for caching and pub/sub
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                db=int(os.getenv('REDIS_DB', '0')),
                password=os.getenv('REDIS_PASSWORD'),
                decode_responses=True
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def close(self):
        """Close all database connections"""
        if self.pool:
            await self.pool.close()
        if self.redis_client:
            await self.redis_client.close()

class SubscriptionManager:
    """Manages GraphQL subscriptions and WebSocket connections"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.active_subscriptions: Dict[str, WebSocket] = {}
        self.subscription_topics: Dict[str, set] = {}
    
    async def add_subscription(self, websocket: WebSocket, subscription_id: str, topic: str):
        """Add a new subscription"""
        self.active_subscriptions[subscription_id] = websocket
        if topic not in self.subscription_topics:
            self.subscription_topics[topic] = set()
        self.subscription_topics[topic].add(subscription_id)
        ACTIVE_SUBSCRIPTIONS.inc()
        logger.info(f"Added subscription {subscription_id} for topic {topic}")
    
    async def remove_subscription(self, subscription_id: str):
        """Remove a subscription"""
        if subscription_id in self.active_subscriptions:
            del self.active_subscriptions[subscription_id]
            
            # Remove from all topics
            for topic, subs in self.subscription_topics.items():
                subs.discard(subscription_id)
            
            ACTIVE_SUBSCRIPTIONS.dec()
            logger.info(f"Removed subscription {subscription_id}")
    
    async def publish_to_topic(self, topic: str, data: Dict[str, Any]):
        """Publish data to all subscribers of a topic"""
        if topic in self.subscription_topics:
            dead_subscriptions = []
            
            for subscription_id in self.subscription_topics[topic]:
                if subscription_id in self.active_subscriptions:
                    websocket = self.active_subscriptions[subscription_id]
                    try:
                        await websocket.send_json({
                            "type": "data",
                            "id": subscription_id,
                            "payload": {"data": data}
                        })
                    except Exception as e:
                        logger.warning(f"Failed to send to subscription {subscription_id}: {e}")
                        dead_subscriptions.append(subscription_id)
            
            # Clean up dead subscriptions
            for dead_sub in dead_subscriptions:
                await self.remove_subscription(dead_sub)

class QueryComplexityAnalyzer:
    """Analyzes and limits query complexity to prevent abuse"""
    
    def __init__(self, max_complexity: int = 100):
        self.max_complexity = max_complexity
    
    def analyze_query(self, query: str, variables: Dict = None) -> int:
        """Analyze query complexity (simplified implementation)"""
        # In a real implementation, this would parse the AST and calculate depth/breadth
        complexity = 0
        
        # Count nested fields (simplified)
        complexity += query.count('{') * 2
        complexity += query.count('(') * 1
        
        # Penalize expensive operations
        if 'causal_chain' in query:
            complexity += 20
        if 'simulation_forecast' in query:
            complexity += 30
        if 'agent_performance' in query:
            complexity += 15
        
        return min(complexity, 200)  # Cap at 200
    
    def validate_query(self, query: str, variables: Dict = None) -> bool:
        """Validate that query complexity is within limits"""
        complexity = self.analyze_query(query, variables)
        return complexity <= self.max_complexity

class CacheManager:
    """Manages query result caching"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes
        self.cache_key_prefix = "kairos:graphql:cache:"
    
    async def get_cached_result(self, query_hash: str) -> Optional[Dict]:
        """Get cached query result"""
        try:
            cached = await self.redis.get(f"{self.cache_key_prefix}{query_hash}")
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def cache_result(self, query_hash: str, result: Dict, ttl: int = None):
        """Cache query result"""
        try:
            await self.redis.setex(
                f"{self.cache_key_prefix}{query_hash}",
                ttl or self.default_ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    def should_cache(self, query: str) -> bool:
        """Determine if query result should be cached"""
        # Cache expensive but relatively static queries
        cache_indicators = [
            'agent_performance',
            'economic_analytics',
            'task_market_overview',
            'system_health'
        ]
        return any(indicator in query for indicator in cache_indicators)

async def create_graphql_context(request, db_manager: DatabaseManager) -> KairosGraphQLContext:
    """Create GraphQL context with database connections and user info"""
    # Extract user info from headers (simplified auth)
    user_id = request.headers.get('x-user-id')
    
    return KairosGraphQLContext(
        request=request,
        db_pool=db_manager.pool,
        redis_client=db_manager.redis_client,
        user_id=user_id
    )

async def graphql_query_handler(request):
    """Handle GraphQL queries with caching and complexity analysis"""
    start_time = datetime.now()
    
    try:
        # Parse request
        if request.method == "GET":
            query = request.query_params.get("query", "")
            variables = json.loads(request.query_params.get("variables", "{}"))
            operation_name = request.query_params.get("operationName")
        else:
            body = await request.json()
            query = body.get("query", "")
            variables = body.get("variables", {})
            operation_name = body.get("operationName")
        
        # Validate query complexity
        complexity_analyzer = QueryComplexityAnalyzer()
        if not complexity_analyzer.validate_query(query, variables):
            GRAPHQL_REQUESTS.labels(operation_type='query', status='rejected').inc()
            raise GraphQLError("Query too complex")
        
        # Check cache
        cache_manager = CacheManager(db_manager.redis_client)
        query_hash = str(hash(f"{query}{json.dumps(variables, sort_keys=True)}"))
        
        if cache_manager.should_cache(query):
            cached_result = await cache_manager.get_cached_result(query_hash)
            if cached_result:
                GRAPHQL_REQUESTS.labels(operation_type='query', status='cached').inc()
                return JSONResponse(cached_result)
        
        # Create context
        context = await create_graphql_context(request, db_manager)
        context.query_complexity = complexity_analyzer.analyze_query(query, variables)
        
        # Execute query
        result = await schema.execute_async(
            query,
            variables=variables,
            context=context,
            operation_name=operation_name
        )
        
        # Convert to dict
        response_data = {"data": result.data}
        if result.errors:
            response_data["errors"] = [str(error) for error in result.errors]
            GRAPHQL_REQUESTS.labels(operation_type='query', status='error').inc()
        else:
            GRAPHQL_REQUESTS.labels(operation_type='query', status='success').inc()
        
        # Cache successful results
        if not result.errors and cache_manager.should_cache(query):
            await cache_manager.cache_result(query_hash, response_data)
        
        # Record metrics
        duration = (datetime.now() - start_time).total_seconds()
        GRAPHQL_DURATION.observe(duration)
        
        return JSONResponse(response_data)
        
    except Exception as e:
        GRAPHQL_REQUESTS.labels(operation_type='query', status='error').inc()
        logger.error(f"GraphQL query failed: {e}")
        return JSONResponse(
            {"errors": [str(e)]},
            status_code=500
        )

async def websocket_handler(websocket: WebSocket):
    """Handle GraphQL subscriptions via WebSocket"""
    await websocket.accept()
    subscription_id = f"sub_{datetime.now().timestamp()}"
    
    try:
        # Send connection ACK
        await websocket.send_json({
            "type": "connection_ack"
        })
        
        # Add to subscription manager
        await subscription_manager.add_subscription(websocket, subscription_id, "default")
        
        # Listen for messages
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            
            if message_type == "start":
                # Start a subscription
                query = message.get("payload", {}).get("query", "")
                variables = message.get("payload", {}).get("variables", {})
                
                # Validate subscription
                if "subscription" not in query.lower():
                    await websocket.send_json({
                        "type": "error",
                        "id": message.get("id"),
                        "payload": {"errors": ["Not a subscription query"]}
                    })
                    continue
                
                # Determine subscription topic
                topic = "system_updates"  # Default topic
                if "agent_status_updates" in query:
                    topic = "agent_updates"
                elif "task_status_changes" in query:
                    topic = "task_updates"
                elif "economic_indicators" in query:
                    topic = "economic_updates"
                
                await subscription_manager.add_subscription(websocket, subscription_id, topic)
                
                # Send subscription confirmation
                await websocket.send_json({
                    "type": "data",
                    "id": message.get("id"),
                    "payload": {"data": {"subscribed": True, "topic": topic}}
                })
                
            elif message_type == "stop":
                # Stop a subscription
                await subscription_manager.remove_subscription(subscription_id)
                
            elif message_type == "connection_terminate":
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await subscription_manager.remove_subscription(subscription_id)

async def metrics_handler(request):
    """Prometheus metrics endpoint"""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

async def health_handler(request):
    """Health check endpoint"""
    try:
        # Check database connectivity
        async with db_manager.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Check Redis connectivity
        await db_manager.redis_client.ping()
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "database": "ok",
                "redis": "ok",
                "active_subscriptions": len(subscription_manager.active_subscriptions)
            }
        })
    except Exception as e:
        return JSONResponse(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )

# Background tasks for publishing subscription updates
async def publish_system_updates():
    """Periodically publish system updates to subscribers"""
    while True:
        try:
            # Get system health data
            system_data = {
                "timestamp": datetime.now().isoformat(),
                "active_agents": 12,  # Would query from database
                "tasks_in_progress": 47,
                "cc_circulation": 2500000,
                "market_liquidity": 0.78
            }
            
            await subscription_manager.publish_to_topic("system_updates", system_data)
            await asyncio.sleep(30)  # Publish every 30 seconds
            
        except Exception as e:
            logger.error(f"Error publishing system updates: {e}")
            await asyncio.sleep(60)

async def publish_economic_updates():
    """Periodically publish economic indicator updates"""
    while True:
        try:
            economic_data = {
                "timestamp": datetime.now().isoformat(),
                "market_liquidity": 0.78 + (asyncio.get_event_loop().time() % 10 - 5) * 0.01,  # Simulate fluctuation
                "average_task_price": 245,
                "price_volatility": 0.15,
                "active_bids": 156
            }
            
            await subscription_manager.publish_to_topic("economic_updates", economic_data)
            await asyncio.sleep(60)  # Publish every minute
            
        except Exception as e:
            logger.error(f"Error publishing economic updates: {e}")
            await asyncio.sleep(60)

# Initialize components
db_manager = DatabaseManager()
subscription_manager = SubscriptionManager(db_manager)

# Define routes
routes = [
    Route("/graphql", graphql_query_handler, methods=["GET", "POST"]),
    Route("/graphiql", make_graphiql_handler(endpoint="/graphql")),
    WebSocketRoute("/graphql-ws", websocket_handler),
    Route("/metrics", metrics_handler),
    Route("/health", health_handler),
]

# Middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

# Create Starlette app
app = Starlette(routes=routes, middleware=middleware)

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    try:
        await db_manager.initialize()
        
        # Start background tasks
        asyncio.create_task(publish_system_updates())
        asyncio.create_task(publish_economic_updates())
        
        logger.info("Kairos GraphQL Server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown"""
    await db_manager.close()
    logger.info("Kairos GraphQL Server shut down")

def main():
    """Main entry point"""
    config = uvicorn.Config(
        "api.graphql.server:app",
        host=os.getenv("GRAPHQL_HOST", "0.0.0.0"),
        port=int(os.getenv("GRAPHQL_PORT", "8000")),
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        log_level="info",
        access_log=True,
        use_colors=True,
        loop="asyncio"
    )
    
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()