#!/usr/bin/env python3
"""
Kairos Native System Launcher
Runs the Kairos ecosystem natively on Windows with containerized infrastructure.
"""

import os
import sys
import time
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path

# Set environment variables for native execution
os.environ.update({
    'POSTGRES_HOST': 'localhost',
    'POSTGRES_USER': 'kairos',
    'POSTGRES_PASSWORD': 'kairos_password',
    'POSTGRES_DB': 'kairos',
    'REDIS_HOST': 'localhost',
    'KAIROS_ENV': 'native'
})

def print_banner():
    """Print the Kairos system banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    🌟 PROJECT KAIROS 🌟                      ║
║              Autonomous Digital Organization                  ║
║                     Native Launch Mode                       ║
╚═══════════════════════════════════════════════════════════════╝

🏗️  Architecture: Enhanced ADO with Cognitive Substrate
💰 Economy: Cognitive Cycles (CC) Internal Currency  
🔮 Oracle: Pre-Cognitive Simulation Engine
🌐 Interface: GraphQL + gRPC Symbiotic APIs
🚀 Agents: Steward, Architect, Engineer + Swarm

⚡ Running natively on Windows with containerized infrastructure
📊 Database: PostgreSQL (containerized)
📦 Cache: Redis (containerized)

Starting system components...
"""
    print(banner)

def check_docker_services():
    """Check if Docker services are running."""
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=kairos'], 
                              capture_output=True, text=True)
        if 'kairos_postgres' in result.stdout and 'kairos_redis' in result.stdout:
            print("✅ Docker infrastructure services are running")
            return True
        else:
            print("❌ Docker infrastructure services not found")
            print("Please run: docker-compose -f docker-compose.minimal.yml up -d")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker services: {e}")
        return False

def test_connections():
    """Test database and Redis connections."""
    try:
        # Test PostgreSQL
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            user='kairos',
            password='kairos_password',
            database='kairos',
            port=5432
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
        
        # Test Redis
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✅ Redis connection successful")
        
        return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

class KairosSystemMonitor:
    """Simple system monitor for native execution."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.status = {
            'database': '🟢 Connected',
            'redis': '🟢 Connected',
            'api_server': '🟡 Starting...',
            'vision_board': '🟡 Starting...',
            'agents': '🟡 Initializing...'
        }
    
    def print_status(self):
        """Print current system status."""
        uptime = datetime.now() - self.start_time
        print(f"\n📊 KAIROS SYSTEM STATUS - Uptime: {str(uptime).split('.')[0]}")
        print("=" * 60)
        for component, status in self.status.items():
            print(f"{component.ljust(15)}: {status}")
        print("=" * 60)
        print(f"🌐 GraphQL Endpoint: http://localhost:8000/graphql")
        print(f"⚡ gRPC Endpoint: localhost:50051")
        print(f"📈 Database: postgresql://kairos:***@localhost:5432/kairos")
        print(f"🔄 Redis: redis://localhost:6379")

def main():
    """Main launcher function."""
    print_banner()
    
    # Check Docker services
    if not check_docker_services():
        return False
    
    # Test connections
    if not test_connections():
        return False
    
    # Initialize system monitor
    monitor = KairosSystemMonitor()
    
    print("\n🚀 Kairos Native System Started Successfully!")
    print("\n📋 Available Components:")
    print("   • Database & Redis (containerized)")
    print("   • Native Python environment")
    print("   • GraphQL/gRPC APIs ready")
    print("   • Agent swarm capability")
    print("   • Vision Board TUI")
    
    print("\n🎯 Quick Actions:")
    print("   1. Run API Server: python api/launcher.py")  
    print("   2. Run Vision Board: python tui/vision_board.py")
    print("   3. Run Agent Swarm: python agents/enhanced/swarm_launcher.py")
    print("   4. View logs: docker-compose -f docker-compose.minimal.yml logs")
    print("   5. Stop system: docker-compose -f docker-compose.minimal.yml down")
    
    print("\n⚡ System ready for native execution!")
    
    # Keep alive and show periodic status
    try:
        while True:
            monitor.print_status()
            time.sleep(30)  # Show status every 30 seconds
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Kairos system...")
        print("Database and Redis containers will remain running.")
        print("To stop all services: docker-compose -f docker-compose.minimal.yml down")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)