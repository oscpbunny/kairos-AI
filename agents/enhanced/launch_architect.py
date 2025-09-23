#!/usr/bin/env python3
"""
Kairos Enhanced Architect Agent Launcher
Fixes import path issues for Docker container execution
"""

import sys
import os
import asyncio
import logging

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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

logger = logging.getLogger('ArchitectLauncher')

async def main():
    """Launch the Enhanced Architect Agent"""
    try:
        logger.info("Starting Enhanced Architect Agent...")
        
        # Import after path is set
        from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
        
        # Create and initialize the agent
        architect = EnhancedArchitectAgent()
        
        logger.info("Initializing Architect Agent...")
        success = await architect.initialize_agent()
        
        if not success:
            logger.error("Failed to initialize Architect Agent")
            return 1
            
        logger.info("Architect Agent initialized successfully")
        
        # Start the agent's main loop
        logger.info("Starting Architect main processing loop...")
        await architect.start_agent_loop()
        
    except Exception as e:
        logger.error(f"Failed to start Architect Agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)