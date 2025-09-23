#!/usr/bin/env python3
"""
Kairos Symbiotic Interface Launcher
Unified launcher for GraphQL and gRPC API servers.

This launcher provides:
1. Coordinated startup of GraphQL and gRPC servers
2. Shared connection pooling and resource management
3. Health monitoring and service discovery
4. Graceful shutdown and cleanup
5. Configuration management
6. Logging coordination

Author: Kairos Development Team
Version: 2.0
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json
import yaml

import uvicorn
from fastapi import FastAPI
import asyncpg
import redis.asyncio as redis

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.graphql.enhanced_schema import schema
from api.grpc.server import KairosGRPCServer
from simulation.oracle_engine import OracleEngine
from economy.cognitive_cycles_engine import CognitiveCyclesEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('E:\\kairos\\logs\\api_launcher.log', mode='a')
    ]
)
logger = logging.getLogger('KairosAPILauncher')

class KairosSymbioticInterface:
    """Main launcher and coordinator for Kairos API services"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the symbiotic interface"""
        self.config = self._load_configuration(config_path)
        self.services = {}
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.db_pool = None
        self.redis_client = None
        self.oracle_engine = None
        self.economy_engine = None
        
        # API servers
        self.graphql_app = None
        self.grpc_server = None
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.service_registry = {}
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        # Default configuration
        config = {
            'database': {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', 5432)),
                'database': os.getenv('POSTGRES_DB', 'kairos'),
                'user': os.getenv('POSTGRES_USER', 'kairos'),
                'password': os.getenv('POSTGRES_PASSWORD', 'kairos_password'),
                'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
                'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', 20))
            },
            'redis': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
                'db': int(os.getenv('REDIS_DB', 0)),
                'password': os.getenv('REDIS_PASSWORD', None)
            },
            'graphql': {
                'host': os.getenv('GRAPHQL_HOST', '0.0.0.0'),
                'port': int(os.getenv('GRAPHQL_PORT', 8000)),
                'debug': os.getenv('GRAPHQL_DEBUG', 'false').lower() == 'true',
                'cors_origins': os.getenv('CORS_ORIGINS', '*').split(',')
            },
            'grpc': {
                'host': os.getenv('GRPC_HOST', '0.0.0.0'),
                'port': int(os.getenv('GRPC_PORT', 50051)),
                'max_workers': int(os.getenv('GRPC_MAX_WORKERS', 100))
            },
            'security': {
                'api_key': os.getenv('KAIROS_API_KEY', 'kairos_default_key'),
                'jwt_secret': os.getenv('JWT_SECRET', 'kairos_jwt_secret'),
                'rate_limit': int(os.getenv('API_RATE_LIMIT', 1000))
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'file': os.getenv('LOG_FILE', 'E:\\kairos\\logs\\symbiotic_interface.log')
            }
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                
                # Merge configurations (file overrides defaults)
                config.update(file_config)
                logger.info(f"Configuration loaded from {config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        
        return config
    
    async def initialize_core_components(self):
        """Initialize database, Redis, Oracle, and Economy engines"""
        try:
            logger.info("Initializing core components...")
            
            # Database connection pool
            db_config = self.config['database'].copy()
            # Convert to asyncpg parameter names
            min_size = db_config.pop('min_connections', 5)
            max_size = db_config.pop('max_connections', 20)
            
            self.db_pool = await asyncpg.create_pool(
                **db_config,
                min_size=min_size,
                max_size=max_size,
                command_timeout=60
            )
            logger.info("✅ Database pool created")
            
            # Redis client
            redis_config = self.config['redis'].copy()
            if redis_config['password'] is None:
                redis_config.pop('password')
            
            self.redis_client = redis.Redis(**redis_config, decode_responses=True)
            await self.redis_client.ping()  # Test connection
            logger.info("✅ Redis client connected")
            
            # Oracle engine
            self.oracle_engine = OracleEngine(self.config['database'])
            await self.oracle_engine.initialize()
            logger.info("✅ Oracle engine initialized")
            
            # Economy engine
            self.economy_engine = CognitiveCyclesEngine(self.config['database'])
            await self.economy_engine.initialize()
            logger.info("✅ Economy engine initialized")
            
            # Update service registry
            self.service_registry.update({
                'database': {'status': 'healthy', 'last_check': datetime.now()},
                'redis': {'status': 'healthy', 'last_check': datetime.now()},
                'oracle': {'status': 'healthy', 'last_check': datetime.now()},
                'economy': {'status': 'healthy', 'last_check': datetime.now()}
            })
            
            logger.info("🎯 All core components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize core components: {e}")
            raise
    
    async def setup_graphql_server(self):
        """Setup and configure GraphQL server"""
        try:
            logger.info("Setting up GraphQL server...")
            
            from strawberry.fastapi import GraphQLRouter
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.middleware.gzip import GZipMiddleware
            
            # Create FastAPI app
            self.graphql_app = FastAPI(
                title="Kairos Symbiotic Interface",
                description="GraphQL API for the Kairos Autonomous Digital Organization",
                version="2.0",
                docs_url="/docs",
                redoc_url="/redoc"
            )
            
            # Add middleware
            self.graphql_app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config['graphql']['cors_origins'],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self.graphql_app.add_middleware(GZipMiddleware, minimum_size=1000)
            
            # Create GraphQL router with context
            async def get_context():
                return {
                    'db_pool': self.db_pool,
                    'redis': self.redis_client,
                    'oracle_engine': self.oracle_engine,
                    'economy_engine': self.economy_engine
                }
            
            graphql_router = GraphQLRouter(schema, context_getter=get_context)
            self.graphql_app.include_router(graphql_router, prefix="/graphql")
            
            # Health check endpoint
            @self.graphql_app.get("/health")
            async def health_check():
                return await self._comprehensive_health_check()
            
            # System status endpoint
            @self.graphql_app.get("/status")
            async def system_status():
                return {
                    "service": "Kairos Symbiotic Interface",
                    "version": "2.0",
                    "timestamp": datetime.now().isoformat(),
                    "services": self.service_registry,
                    "uptime": (datetime.now() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0
                }
            
            # Metrics endpoint
            @self.graphql_app.get("/metrics")
            async def metrics():
                return await self._collect_metrics()
            
            logger.info("✅ GraphQL server configured")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup GraphQL server: {e}")
            raise
    
    async def setup_grpc_server(self):
        """Setup and configure gRPC server"""
        try:
            logger.info("Setting up gRPC server...")
            
            self.grpc_server = KairosGRPCServer()
            await self.grpc_server.initialize()
            
            logger.info("✅ gRPC server configured")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup gRPC server: {e}")
            raise
    
    async def start_servers(self):
        """Start both GraphQL and gRPC servers concurrently"""
        try:
            logger.info("🚀 Starting API servers...")
            self.start_time = datetime.now()
            
            # Create tasks for both servers
            tasks = []
            
            # GraphQL server task
            if self.graphql_app:
                graphql_config = uvicorn.Config(
                    self.graphql_app,
                    host=self.config['graphql']['host'],
                    port=self.config['graphql']['port'],
                    log_level="info",
                    access_log=True
                )
                graphql_server = uvicorn.Server(graphql_config)
                tasks.append(asyncio.create_task(graphql_server.serve()))
                
                self.service_registry['graphql'] = {
                    'status': 'running',
                    'host': self.config['graphql']['host'],
                    'port': self.config['graphql']['port'],
                    'last_check': datetime.now()
                }
                
                logger.info(f"✅ GraphQL server starting on {self.config['graphql']['host']}:{self.config['graphql']['port']}")
            
            # gRPC server task
            if self.grpc_server:
                tasks.append(asyncio.create_task(
                    self.grpc_server.start_server(self.config['grpc']['port'])
                ))
                
                self.service_registry['grpc'] = {
                    'status': 'running',
                    'host': self.config['grpc']['host'],
                    'port': self.config['grpc']['port'],
                    'last_check': datetime.now()
                }
                
                logger.info(f"✅ gRPC server starting on {self.config['grpc']['host']}:{self.config['grpc']['port']}")
            
            # Health monitoring task
            tasks.append(asyncio.create_task(self._health_monitor()))
            
            # Service discovery announcement
            await self._announce_services()
            
            logger.info("🎯 All servers started successfully!")
            logger.info("=" * 60)
            logger.info("🌟 KAIROS SYMBIOTIC INTERFACE ONLINE 🌟")
            logger.info("=" * 60)
            logger.info(f"GraphQL Playground: http://{self.config['graphql']['host']}:{self.config['graphql']['port']}/graphql")
            logger.info(f"GraphQL Docs: http://{self.config['graphql']['host']}:{self.config['graphql']['port']}/docs")
            logger.info(f"gRPC Server: {self.config['grpc']['host']}:{self.config['grpc']['port']}")
            logger.info("=" * 60)
            
            # Wait for shutdown signal or server completion
            done, pending = await asyncio.wait(
                tasks + [asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            logger.error(f"❌ Failed to start servers: {e}")
            raise
    
    async def _health_monitor(self):
        """Background health monitoring task"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check database health
                try:
                    async with self.db_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    self.service_registry['database']['status'] = 'healthy'
                except Exception as e:
                    self.service_registry['database']['status'] = 'unhealthy'
                    logger.warning(f"Database health check failed: {e}")
                
                # Check Redis health
                try:
                    await self.redis_client.ping()
                    self.service_registry['redis']['status'] = 'healthy'
                except Exception as e:
                    self.service_registry['redis']['status'] = 'unhealthy'
                    logger.warning(f"Redis health check failed: {e}")
                
                # Update last check timestamps
                for service in self.service_registry.values():
                    service['last_check'] = datetime.now()
                
                # Log overall health status
                healthy_services = sum(1 for s in self.service_registry.values() if s['status'] == 'healthy')
                total_services = len(self.service_registry)
                
                if healthy_services == total_services:
                    logger.debug(f"💚 All {total_services} services healthy")
                else:
                    logger.warning(f"⚠️  {healthy_services}/{total_services} services healthy")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "system": {}
        }
        
        # Check each service
        for service_name, service_info in self.service_registry.items():
            health_status["services"][service_name] = {
                "status": service_info.get("status", "unknown"),
                "last_check": service_info.get("last_check", datetime.now()).isoformat()
            }
            
            if service_info.get("status") != "healthy":
                health_status["healthy"] = False
        
        # System metrics
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    system_metrics = await conn.fetchrow("""
                        SELECT 
                            (SELECT COUNT(*) FROM Agents WHERE status = 'ACTIVE') as active_agents,
                            (SELECT COUNT(*) FROM Tasks WHERE status IN ('BOUNTY_POSTED', 'BIDDING')) as pending_tasks,
                            (SELECT COUNT(*) FROM Ventures WHERE status = 'ACTIVE') as active_ventures,
                            (SELECT COUNT(*) FROM Decisions WHERE timestamp >= NOW() - INTERVAL '1 hour') as recent_decisions
                    """)
                    
                    health_status["system"] = {
                        "active_agents": system_metrics["active_agents"] or 0,
                        "pending_tasks": system_metrics["pending_tasks"] or 0,
                        "active_ventures": system_metrics["active_ventures"] or 0,
                        "recent_decisions": system_metrics["recent_decisions"] or 0
                    }
        
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            health_status["system"]["error"] = str(e)
        
        return health_status
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0,
            "services": self.service_registry,
            "database": {},
            "economic": {},
            "oracle": {}
        }
        
        try:
            if self.db_pool:
                # Database metrics
                metrics["database"]["pool_size"] = self.db_pool.get_size()
                metrics["database"]["pool_idle"] = self.db_pool.get_idle_size()
                
                # System-wide metrics
                async with self.db_pool.acquire() as conn:
                    db_metrics = await conn.fetchrow("""
                        SELECT 
                            (SELECT COUNT(*) FROM Agents) as total_agents,
                            (SELECT COUNT(*) FROM Ventures) as total_ventures,
                            (SELECT COUNT(*) FROM Tasks) as total_tasks,
                            (SELECT COUNT(*) FROM Decisions) as total_decisions,
                            (SELECT COUNT(*) FROM Bids) as total_bids,
                            (SELECT COUNT(*) FROM Simulations) as total_simulations
                    """)
                    
                    metrics["database"].update({
                        "total_agents": db_metrics["total_agents"] or 0,
                        "total_ventures": db_metrics["total_ventures"] or 0,
                        "total_tasks": db_metrics["total_tasks"] or 0,
                        "total_decisions": db_metrics["total_decisions"] or 0,
                        "total_bids": db_metrics["total_bids"] or 0,
                        "total_simulations": db_metrics["total_simulations"] or 0
                    })
                
                # Economic metrics
                if self.economy_engine:
                    economic_data = await conn.fetchrow("""
                        SELECT 
                            SUM(cc_balance) as total_cc,
                            AVG(cc_balance) as avg_cc_balance,
                            COUNT(*) as agents_with_balance
                        FROM Agents 
                        WHERE cc_balance > 0
                    """)
                    
                    metrics["economic"] = {
                        "total_cc_circulation": float(economic_data["total_cc"] or 0),
                        "average_cc_balance": float(economic_data["avg_cc_balance"] or 0),
                        "agents_with_balance": economic_data["agents_with_balance"] or 0
                    }
                
                # Oracle metrics
                if self.oracle_engine:
                    oracle_data = await conn.fetchrow("""
                        SELECT 
                            COUNT(*) as predictions_today,
                            AVG(confidence_score) as avg_confidence
                        FROM Simulations 
                        WHERE created_at >= CURRENT_DATE
                    """)
                    
                    metrics["oracle"] = {
                        "predictions_today": oracle_data["predictions_today"] or 0,
                        "average_confidence": float(oracle_data["avg_confidence"] or 0)
                    }
        
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def _announce_services(self):
        """Announce services to Redis for service discovery"""
        try:
            service_announcement = {
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "graphql": {
                        "type": "graphql",
                        "host": self.config['graphql']['host'],
                        "port": self.config['graphql']['port'],
                        "endpoints": {
                            "graphql": "/graphql",
                            "playground": "/graphql",
                            "docs": "/docs",
                            "health": "/health",
                            "status": "/status",
                            "metrics": "/metrics"
                        }
                    },
                    "grpc": {
                        "type": "grpc",
                        "host": self.config['grpc']['host'],
                        "port": self.config['grpc']['port'],
                        "services": [
                            "AgentCommunicationService",
                            "OracleService", 
                            "EconomyService",
                            "CausalLedgerService",
                            "HealthService"
                        ]
                    }
                }
            }
            
            await self.redis_client.setex(
                "kairos:services:symbiotic_interface", 
                300,  # 5 minutes TTL
                json.dumps(service_announcement)
            )
            
            logger.info("📡 Services announced for discovery")
            
        except Exception as e:
            logger.warning(f"Failed to announce services: {e}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown of all services"""
        logger.info("🔄 Initiating graceful shutdown...")
        
        self.shutdown_event.set()
        
        try:
            # Stop gRPC server
            if self.grpc_server:
                await self.grpc_server.stop_server()
                logger.info("✅ gRPC server stopped")
            
            # Close Oracle and Economy engines
            if self.oracle_engine:
                await self.oracle_engine.cleanup()
                logger.info("✅ Oracle engine cleaned up")
            
            if self.economy_engine:
                await self.economy_engine.cleanup()
                logger.info("✅ Economy engine cleaned up")
            
            # Close database pool
            if self.db_pool:
                await self.db_pool.close()
                logger.info("✅ Database pool closed")
            
            # Close Redis client
            if self.redis_client:
                await self.redis_client.close()
                logger.info("✅ Redis client closed")
            
            logger.info("🏁 Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    async def run(self):
        """Main run method"""
        try:
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Initialize core components
            await self.initialize_core_components()
            
            # Setup servers
            await self.setup_graphql_server()
            await self.setup_grpc_server()
            
            # Start servers
            await self.start_servers()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            await self.shutdown()

async def main():
    """Main entry point"""
    # Configuration file path (optional)
    config_path = os.getenv('KAIROS_CONFIG_PATH')
    
    # Create and run the symbiotic interface
    interface = KairosSymbioticInterface(config_path)
    
    try:
        await interface.run()
    except Exception as e:
        logger.error(f"Failed to start Kairos Symbiotic Interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs('E:\\kairos\\logs', exist_ok=True)
    
    # Run the interface
    asyncio.run(main())