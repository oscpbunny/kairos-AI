"""
Project Kairos: Enhanced Agent Swarm Launcher
The orchestration system that launches and manages the complete ecosystem of enhanced Kairos agents.

This launcher manages:
- Enhanced Steward (Resource Broker)
- Enhanced Architect (System Designer)
- Enhanced Engineer (Development Automation)
- Multi-agent coordination and communication
- System health monitoring and self-healing
- Economic balancing and CC distribution
"""

import asyncio
import logging
import signal
import sys
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import argparse

# Import enhanced agents
from .enhanced_steward import EnhancedStewardAgent
from .enhanced_architect import EnhancedArchitectAgent
from .enhanced_engineer import EnhancedEngineerAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kairos_swarm.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('KairosSwarmLauncher')

class SwarmOrchestrator:
    """
    Orchestrates the enhanced agent swarm with intelligent coordination
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_configuration(config_path)
        self.agents: Dict[str, Any] = {}
        self.running = False
        self.health_monitor_task = None
        self.coordination_task = None
        self.start_time = None
        
        # System statistics
        self.system_stats = {
            'total_tasks_processed': 0,
            'total_cc_circulated': 0,
            'agent_uptime': {},
            'collaboration_events': 0,
            'system_health_score': 100.0
        }
        
        # Initialize signal handlers
        self._setup_signal_handlers()
    
    def _load_configuration(self, config_path: str) -> Dict[str, Any]:
        """Load swarm configuration"""
        default_config = {
            'agents': {
                'steward': {
                    'enabled': True,
                    'name': 'Enhanced-Steward',
                    'initial_cc_balance': 5000,
                    'max_concurrent_tasks': 3
                },
                'architect': {
                    'enabled': True,
                    'name': 'Enhanced-Architect',
                    'initial_cc_balance': 4000,
                    'max_concurrent_tasks': 2
                },
                'engineer': {
                    'enabled': True,
                    'name': 'Enhanced-Engineer',
                    'initial_cc_balance': 3500,
                    'max_concurrent_tasks': 4
                }
            },
            'system': {
                'health_check_interval': 30,  # seconds
                'coordination_interval': 60,   # seconds
                'cc_redistribution_threshold': 1000,
                'max_agent_restart_attempts': 3,
                'enable_auto_scaling': True
            },
            'database': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'kairos_db'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password')
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    self._merge_configs(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _merge_configs(self, base: Dict, overlay: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize_agents(self):
        """Initialize all configured agents"""
        logger.info("Initializing enhanced agent swarm...")
        
        initialization_tasks = []
        
        # Initialize Steward (Resource Broker)
        if self.config['agents']['steward']['enabled']:
            steward_config = self.config['agents']['steward']
            steward = EnhancedStewardAgent(
                agent_name=steward_config['name'],
                initial_cc_balance=steward_config['initial_cc_balance']
            )
            initialization_tasks.append(self._initialize_agent('steward', steward))
        
        # Initialize Architect (System Designer)
        if self.config['agents']['architect']['enabled']:
            architect_config = self.config['agents']['architect']
            architect = EnhancedArchitectAgent(
                agent_name=architect_config['name'],
                initial_cc_balance=architect_config['initial_cc_balance']
            )
            initialization_tasks.append(self._initialize_agent('architect', architect))
        
        # Initialize Engineer (Development Automation)
        if self.config['agents']['engineer']['enabled']:
            engineer_config = self.config['agents']['engineer']
            engineer = EnhancedEngineerAgent(
                agent_name=engineer_config['name'],
                initial_cc_balance=engineer_config['initial_cc_balance']
            )
            initialization_tasks.append(self._initialize_agent('engineer', engineer))
        
        # Wait for all agents to initialize
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        successful_inits = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent initialization failed: {result}")
            else:
                successful_inits += 1
        
        logger.info(f"Successfully initialized {successful_inits}/{len(initialization_tasks)} agents")
        
        if successful_inits == 0:
            raise RuntimeError("No agents could be initialized")
        
        return successful_inits > 0
    
    async def _initialize_agent(self, agent_type: str, agent_instance) -> bool:
        """Initialize a single agent"""
        try:
            success = await agent_instance.initialize_agent()
            if success:
                self.agents[agent_type] = {
                    'instance': agent_instance,
                    'task': None,
                    'restart_count': 0,
                    'last_health_check': datetime.now(),
                    'status': 'initialized'
                }
                logger.info(f"Successfully initialized {agent_type} agent: {agent_instance.agent_name}")
                return True
            else:
                logger.error(f"Failed to initialize {agent_type} agent")
                return False
        except Exception as e:
            logger.error(f"Exception during {agent_type} agent initialization: {e}")
            return False
    
    async def start_swarm(self):
        """Start the complete agent swarm"""
        logger.info("Starting Kairos Enhanced Agent Swarm...")
        self.running = True
        self.start_time = datetime.now()
        
        # Start all agent loops
        agent_tasks = []
        for agent_type, agent_data in self.agents.items():
            task = asyncio.create_task(
                self._run_agent_with_monitoring(agent_type, agent_data['instance'])
            )
            agent_data['task'] = task
            agent_data['status'] = 'running'
            agent_tasks.append(task)
            
            logger.info(f"Started {agent_type} agent: {agent_data['instance'].agent_name}")
        
        # Start system monitoring tasks
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.coordination_task = asyncio.create_task(self._coordination_loop())
        
        logger.info(f"Kairos swarm fully operational with {len(self.agents)} agents")
        
        try:
            # Wait for all tasks to complete or for shutdown signal
            await asyncio.gather(
                *agent_tasks,
                self.health_monitor_task,
                self.coordination_task,
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Swarm execution error: {e}")
        finally:
            logger.info("Swarm execution completed")
    
    async def _run_agent_with_monitoring(self, agent_type: str, agent_instance):
        """Run agent with monitoring and auto-restart capability"""
        agent_data = self.agents[agent_type]
        
        while self.running:
            try:
                logger.info(f"Starting agent loop for {agent_type}")
                await agent_instance.run_agent_loop()
            except Exception as e:
                logger.error(f"Agent {agent_type} crashed: {e}")
                agent_data['restart_count'] += 1
                
                max_restarts = self.config['system']['max_agent_restart_attempts']
                if agent_data['restart_count'] >= max_restarts:
                    logger.error(f"Agent {agent_type} exceeded max restart attempts ({max_restarts})")
                    agent_data['status'] = 'failed'
                    break
                
                logger.info(f"Restarting agent {agent_type} (attempt {agent_data['restart_count']})")
                await asyncio.sleep(5)  # Wait before restart
                
                # Reinitialize agent
                try:
                    await agent_instance.initialize_agent()
                    agent_data['status'] = 'running'
                    logger.info(f"Successfully restarted agent {agent_type}")
                except Exception as restart_e:
                    logger.error(f"Failed to restart agent {agent_type}: {restart_e}")
                    await asyncio.sleep(10)  # Wait longer before next attempt
    
    async def _health_monitor_loop(self):
        """Monitor system and agent health"""
        interval = self.config['system']['health_check_interval']
        
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(interval)
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks"""
        current_time = datetime.now()
        total_health_score = 0
        active_agents = 0
        
        for agent_type, agent_data in self.agents.items():
            agent_instance = agent_data['instance']
            
            # Check agent heartbeat
            if hasattr(agent_instance, 'last_heartbeat'):
                heartbeat_age = (current_time - agent_instance.last_heartbeat).total_seconds()
                if heartbeat_age > 120:  # 2 minutes
                    logger.warning(f"Agent {agent_type} heartbeat is stale ({heartbeat_age:.1f}s)")
                    agent_health = 50
                else:
                    agent_health = 100
            else:
                agent_health = 75  # Assume healthy if no heartbeat info
            
            # Check agent CC balance
            if hasattr(agent_instance, 'cognitive_cycles_balance'):
                if agent_instance.cognitive_cycles_balance < 100:
                    logger.warning(f"Agent {agent_type} has low CC balance: {agent_instance.cognitive_cycles_balance}")
                    agent_health = min(agent_health, 60)
            
            # Check active tasks
            if hasattr(agent_instance, 'active_tasks'):
                max_concurrent = self.config['agents'][agent_type].get('max_concurrent_tasks', 5)
                if len(agent_instance.active_tasks) >= max_concurrent:
                    logger.info(f"Agent {agent_type} at max capacity: {len(agent_instance.active_tasks)} tasks")
            
            total_health_score += agent_health
            active_agents += 1
            agent_data['last_health_check'] = current_time
            
            # Update system statistics
            uptime = (current_time - self.start_time).total_seconds() if self.start_time else 0
            self.system_stats['agent_uptime'][agent_type] = uptime
        
        # Calculate overall system health
        if active_agents > 0:
            self.system_stats['system_health_score'] = total_health_score / active_agents
        
        # Log health summary
        if current_time.minute % 5 == 0:  # Every 5 minutes
            logger.info(f"System health: {self.system_stats['system_health_score']:.1f}% "
                       f"({active_agents} agents active)")
    
    async def _coordination_loop(self):
        """Coordinate agent activities and manage system resources"""
        interval = self.config['system']['coordination_interval']
        
        while self.running:
            try:
                await self._coordinate_agent_activities()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Coordination error: {e}")
                await asyncio.sleep(interval)
    
    async def _coordinate_agent_activities(self):
        """Coordinate activities between agents"""
        try:
            # Check for CC redistribution needs
            await self._balance_cognitive_cycles()
            
            # Facilitate collaboration on complex tasks
            await self._facilitate_collaboration()
            
            # Update system statistics
            await self._update_system_statistics()
            
        except Exception as e:
            logger.error(f"Failed to coordinate agent activities: {e}")
    
    async def _balance_cognitive_cycles(self):
        """Balance Cognitive Cycles between agents based on performance and needs"""
        threshold = self.config['system']['cc_redistribution_threshold']
        balances = {}
        
        # Collect current balances
        for agent_type, agent_data in self.agents.items():
            agent_instance = agent_data['instance']
            if hasattr(agent_instance, 'cognitive_cycles_balance'):
                balances[agent_type] = agent_instance.cognitive_cycles_balance
        
        if not balances:
            return
        
        # Find agents that need CC redistribution
        low_balance_agents = {k: v for k, v in balances.items() if v < threshold}
        high_balance_agents = {k: v for k, v in balances.items() if v > threshold * 3}
        
        if low_balance_agents and high_balance_agents:
            logger.info(f"CC redistribution needed - Low: {low_balance_agents}, High: {high_balance_agents}")
            
            # Simple redistribution logic
            for low_agent in low_balance_agents:
                for high_agent in high_balance_agents:
                    if balances[high_agent] > threshold * 2:
                        transfer_amount = min(500, balances[high_agent] // 4)
                        
                        # Update balances
                        self.agents[low_agent]['instance'].cognitive_cycles_balance += transfer_amount
                        self.agents[high_agent]['instance'].cognitive_cycles_balance -= transfer_amount
                        
                        self.system_stats['total_cc_circulated'] += transfer_amount
                        
                        logger.info(f"Transferred {transfer_amount} CC from {high_agent} to {low_agent}")
                        break
    
    async def _facilitate_collaboration(self):
        """Facilitate collaboration between agents on complex tasks"""
        try:
            # This would implement sophisticated collaboration logic
            # For now, just log collaboration opportunities
            
            collaboration_opportunities = 0
            
            for agent_type, agent_data in self.agents.items():
                agent_instance = agent_data['instance']
                if hasattr(agent_instance, 'active_tasks') and len(agent_instance.active_tasks) > 0:
                    # Check for tasks that could benefit from collaboration
                    for task_id, task_data in agent_instance.active_tasks.items():
                        if task_data.get('cc_bounty', 0) > 1500:  # High-value task
                            collaboration_opportunities += 1
            
            if collaboration_opportunities > 0:
                self.system_stats['collaboration_events'] += collaboration_opportunities
                logger.info(f"Identified {collaboration_opportunities} collaboration opportunities")
                
        except Exception as e:
            logger.error(f"Failed to facilitate collaboration: {e}")
    
    async def _update_system_statistics(self):
        """Update comprehensive system statistics"""
        current_time = datetime.now()
        
        # Update total tasks processed
        total_tasks = 0
        for agent_type, agent_data in self.agents.items():
            agent_instance = agent_data['instance']
            if hasattr(agent_instance, 'performance_metrics'):
                total_tasks += agent_instance.performance_metrics.tasks_completed
        
        self.system_stats['total_tasks_processed'] = total_tasks
        
        # Log periodic summary
        if current_time.minute % 10 == 0:  # Every 10 minutes
            uptime = (current_time - self.start_time).total_seconds() if self.start_time else 0
            logger.info(f"System summary - Uptime: {uptime:.0f}s, Tasks: {total_tasks}, "
                       f"Health: {self.system_stats['system_health_score']:.1f}%")
    
    async def shutdown(self):
        """Gracefully shutdown the entire swarm"""
        logger.info("Initiating swarm shutdown...")
        self.running = False
        
        # Cancel monitoring tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        if self.coordination_task:
            self.coordination_task.cancel()
        
        # Shutdown all agents
        shutdown_tasks = []
        for agent_type, agent_data in self.agents.items():
            if agent_data['task']:
                agent_data['task'].cancel()
            
            # Graceful agent shutdown
            agent_instance = agent_data['instance']
            shutdown_tasks.append(agent_instance.shutdown())
        
        if shutdown_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*shutdown_tasks), timeout=30)
                logger.info("All agents shut down successfully")
            except asyncio.TimeoutError:
                logger.warning("Agent shutdown timeout - forcing termination")
            except Exception as e:
                logger.error(f"Error during agent shutdown: {e}")
        
        # Final system statistics
        total_uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        logger.info(f"Swarm shutdown complete - Total uptime: {total_uptime:.1f}s, "
                   f"Tasks processed: {self.system_stats['total_tasks_processed']}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds() if self.start_time else 0
        
        status = {
            'running': self.running,
            'uptime_seconds': uptime,
            'agent_count': len(self.agents),
            'system_health_score': self.system_stats['system_health_score'],
            'total_tasks_processed': self.system_stats['total_tasks_processed'],
            'agents': {}
        }
        
        for agent_type, agent_data in self.agents.items():
            agent_instance = agent_data['instance']
            agent_status = {
                'name': agent_instance.agent_name,
                'status': agent_data['status'],
                'restart_count': agent_data['restart_count'],
                'cc_balance': getattr(agent_instance, 'cognitive_cycles_balance', 0),
                'active_tasks': len(getattr(agent_instance, 'active_tasks', {})),
                'last_health_check': agent_data['last_health_check'].isoformat()
            }
            status['agents'][agent_type] = agent_status
        
        return status

# Main execution functions
async def main():
    """Main entry point for the swarm launcher"""
    parser = argparse.ArgumentParser(description='Kairos Enhanced Agent Swarm Launcher')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--status', action='store_true', help='Show status and exit')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create orchestrator
    orchestrator = SwarmOrchestrator(config_path=args.config)
    
    try:
        # Initialize agents
        success = await orchestrator.initialize_agents()
        if not success:
            logger.error("Failed to initialize agent swarm")
            sys.exit(1)
        
        if args.status:
            status = orchestrator.get_status()
            print(json.dumps(status, indent=2, default=str))
            return
        
        # Start the swarm
        await orchestrator.start_swarm()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Swarm execution failed: {e}")
        sys.exit(1)
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Swarm launcher interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)