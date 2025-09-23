"""
Project Kairos: API Server Startup Script
Starts both GraphQL (FastAPI) and gRPC servers concurrently.
"""

import asyncio
import logging
import signal
import sys
import os
from typing import List

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerManager:
    """Manages both GraphQL and gRPC servers"""
    
    def __init__(self):
        self.tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
    
    async def start_graphql_server(self):
        """Start the FastAPI GraphQL server"""
        try:
            import uvicorn
            from api.server import app
            
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                reload=False  # Disable reload for production
            )
            server = uvicorn.Server(config)
            
            logger.info("🚀 Starting GraphQL server on http://0.0.0.0:8000")
            logger.info("📊 GraphQL Playground: http://localhost:8000/graphql/playground")
            logger.info("📖 API Documentation: http://localhost:8000/docs")
            
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start GraphQL server: {e}")
            self.shutdown_event.set()
    
    async def start_grpc_server(self):
        """Start the gRPC server"""
        try:
            from api.grpc.server import serve
            
            logger.info("🚀 Starting gRPC server on port 50051")
            await serve()
            
        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            self.shutdown_event.set()
    
    async def health_monitor(self):
        """Monitor server health and log status periodically"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.shutdown_event.is_set():
                    logger.info("💓 Servers health check: Both GraphQL and gRPC servers running")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health monitor error: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start_all_servers(self):
        """Start all servers concurrently"""
        try:
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Create server tasks
            graphql_task = asyncio.create_task(
                self.start_graphql_server(), 
                name="graphql-server"
            )
            grpc_task = asyncio.create_task(
                self.start_grpc_server(), 
                name="grpc-server"
            )
            health_task = asyncio.create_task(
                self.health_monitor(),
                name="health-monitor"
            )
            
            self.tasks = [graphql_task, grpc_task, health_task]
            
            logger.info("✅ All servers starting...")
            logger.info("📡 GraphQL API: http://localhost:8000/graphql")
            logger.info("🔗 gRPC API: localhost:50051")
            logger.info("💊 Health Check: http://localhost:8000/health")
            logger.info("🎮 Playground: http://localhost:8000/graphql/playground")
            
            # Wait for shutdown signal or task completion
            done, pending = await asyncio.wait(
                self.tasks + [asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            logger.info("🔄 Initiating server shutdown...")
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Wait for done tasks to complete
            for task in done:
                if not task.cancelled():
                    try:
                        await task
                    except Exception as e:
                        logger.error(f"Task completion error: {e}")
            
            logger.info("👋 All servers stopped gracefully")
            
        except Exception as e:
            logger.error(f"Server manager error: {e}")
            raise

async def test_server_connectivity():
    """Test server connectivity after startup"""
    import aiohttp
    import time
    
    # Wait for servers to start
    await asyncio.sleep(3)
    
    logger.info("🔍 Testing server connectivity...")
    
    # Test GraphQL server
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as resp:
                if resp.status == 200:
                    logger.info("✅ GraphQL server: Healthy")
                else:
                    logger.warning(f"⚠️ GraphQL server: Status {resp.status}")
    except Exception as e:
        logger.error(f"❌ GraphQL server test failed: {e}")
    
    # Test gRPC server (basic connection test)
    try:
        import grpc
        channel = grpc.aio.insecure_channel('localhost:50051')
        # Try to get channel state
        state = channel.get_state()
        logger.info(f"✅ gRPC server: Channel state {state}")
        await channel.close()
    except Exception as e:
        logger.error(f"❌ gRPC server test failed: {e}")

def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = [
        'fastapi',
        'uvicorn',
        'strawberry',
        'grpc',
        'pydantic',
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"❌ Missing required dependencies: {missing_modules}")
        logger.info("📦 Install dependencies: pip install -r api/requirements.txt")
        return False
    
    return True

async def main():
    """Main entry point"""
    logger.info("🌟 Project Kairos API Server Manager")
    logger.info("=====================================")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        manager = ServerManager()
        
        # Start connectivity test in background
        asyncio.create_task(test_server_connectivity())
        
        # Start all servers
        await manager.start_all_servers()
        
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("🏁 Server manager exiting")

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())