#!/usr/bin/env python3
"""
Kairos API Server Launcher
Fixes import path and hostname resolution issues for Docker container execution
"""

import sys
import os
import asyncio
import logging

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('APILauncher')

def main():
    """Launch the Kairos API Server"""
    try:
        logger.info("Starting Kairos API Server...")
        
        # Set database connection environment variables to use container names
        os.environ['DB_HOST'] = os.getenv('POSTGRES_HOST', 'postgres')
        os.environ['DB_NAME'] = os.getenv('POSTGRES_DB', 'kairos')  
        os.environ['DB_USER'] = os.getenv('POSTGRES_USER', 'kairos')
        os.environ['DB_PASSWORD'] = os.getenv('POSTGRES_PASSWORD', 'kairos_password')
        os.environ['DB_PORT'] = os.getenv('POSTGRES_PORT', '5432')
        
        # Set Redis connection environment variables
        os.environ['REDIS_HOST'] = os.getenv('REDIS_HOST', 'redis')
        os.environ['REDIS_PORT'] = os.getenv('REDIS_PORT', '6379')
        
        logger.info(f"Database: {os.environ['DB_HOST']}:{os.environ['DB_PORT']}")
        logger.info(f"Redis: {os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}")
        
        # Import and start the API server
        from api.launcher import main as api_main
        
        logger.info("Launching API server...")
        return api_main()
        
    except Exception as e:
        logger.error(f"Failed to start API Server: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)