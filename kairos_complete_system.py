"""
🚀🎯 KAIROS COMPLETE SYSTEM INTEGRATION 🎯🚀
Comprehensive Multi-Agent AI Coordination Platform

This script orchestrates all Kairos components:
- Enhanced Analytics Dashboard (Port 8051)
- REST API Server (Port 8080)
- ML Analytics Engine
- Real-time data synchronization
- Unified monitoring and control

Features:
✨ Professional web dashboard with real-time charts
✨ Comprehensive REST API with OpenAPI docs
✨ Advanced ML analytics with clustering, anomaly detection, and predictions
✨ Data export capabilities (CSV/JSON)
✨ WebSocket live updates
✨ Performance optimization recommendations
"""

import asyncio
import threading
import time
import signal
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Import our custom modules
sys.path.append('E:/kairos')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('KairosSystem')

class KairosSystemOrchestrator:
    """
    🎯 Kairos System Orchestrator
    
    Coordinates all system components and provides unified control
    """
    
    def __init__(self):
        self.components = {}
        self.running = False
        self.data_sync_task = None
        
        # System status
        self.system_stats = {
            'start_time': None,
            'components_active': 0,
            'total_requests': 0,
            'dashboard_views': 0,
            'api_calls': 0,
            'ml_analyses': 0
        }
        
        logger.info("🚀 Kairos System Orchestrator initialized")
    
    def start_enhanced_dashboard(self):
        """Start the enhanced analytics dashboard"""
        try:
            logger.info("🎨 Starting Enhanced Analytics Dashboard...")
            
            # Import and start dashboard
            from monitoring.enhanced_dashboard import KairosEnhancedDashboard
            
            dashboard = KairosEnhancedDashboard(port=8051)
            
            # Run dashboard in separate thread
            def run_dashboard():
                dashboard.run(debug=False)
            
            dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            dashboard_thread.start()
            
            self.components['dashboard'] = {
                'instance': dashboard,
                'thread': dashboard_thread,
                'port': 8051,
                'status': 'running'
            }
            
            logger.info("✅ Enhanced Dashboard started on http://localhost:8051")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Enhanced Dashboard: {e}")
            return False
    
    def start_rest_api_server(self):
        """Start the REST API server"""
        try:
            logger.info("🔗 Starting REST API Server...")
            
            # Import and start API server
            from api.rest.kairos_api_server import KairosAPIServer
            
            api_server = KairosAPIServer(host="0.0.0.0", port=8080)
            
            # Run API server in separate thread
            def run_api():
                api_server.run(debug=False)
            
            api_thread = threading.Thread(target=run_api, daemon=True)
            api_thread.start()
            
            self.components['api'] = {
                'instance': api_server,
                'thread': api_thread,
                'port': 8080,
                'status': 'running'
            }
            
            logger.info("✅ REST API Server started on http://localhost:8080")
            logger.info("📚 API Documentation: http://localhost:8080/docs")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start REST API Server: {e}")
            return False
    
    def start_ml_analytics(self):
        """Start the ML analytics engine"""
        try:
            logger.info("🧠 Starting ML Analytics Engine...")
            
            # Import and initialize ML analytics
            from analytics.ml_engine_demo import SimplifiedMLAnalytics, generate_mock_data
            
            ml_engine = SimplifiedMLAnalytics()
            
            # Load initial mock data
            system_metrics, agents_data, events_data = generate_mock_data()
            ml_engine.add_system_metrics(system_metrics)
            ml_engine.add_agent_data(agents_data)
            ml_engine.add_collaboration_events(events_data)
            
            self.components['ml_analytics'] = {
                'instance': ml_engine,
                'status': 'running',
                'last_analysis': None
            }
            
            logger.info("✅ ML Analytics Engine started with initial data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start ML Analytics Engine: {e}")
            return False
    
    async def run_ml_analysis_cycle(self):
        """Run periodic ML analysis"""
        while self.running:
            try:
                if 'ml_analytics' in self.components:
                    ml_engine = self.components['ml_analytics']['instance']
                    
                    logger.info("🔬 Running ML analysis cycle...")
                    
                    # Run comprehensive analysis
                    results = {}
                    
                    # Performance clustering
                    results['clustering'] = ml_engine.analyze_performance_clusters()
                    
                    # Anomaly detection
                    results['anomaly_detection'] = ml_engine.detect_anomalies()
                    
                    # Performance prediction
                    results['predictions'] = ml_engine.predict_performance()
                    
                    # Collaboration network analysis
                    results['collaboration_network'] = ml_engine.analyze_collaboration_network()
                    
                    # Update stats
                    self.system_stats['ml_analyses'] += 1
                    self.components['ml_analytics']['last_analysis'] = datetime.now()
                    
                    # Log key insights
                    for analysis_type, result in results.items():
                        if 'insights' in result:
                            logger.info(f"📊 {analysis_type}: {len(result['insights'])} insights generated")
                
                # Wait 5 minutes before next analysis
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ ML analysis cycle failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    def start_data_sync_task(self):
        """Start background data synchronization"""
        async def data_sync_loop():
            while self.running:
                try:
                    # Generate new mock data periodically
                    if 'ml_analytics' in self.components:
                        ml_engine = self.components['ml_analytics']['instance']
                        
                        # Generate new system metrics
                        import numpy as np
                        new_metrics = [{
                            'timestamp': datetime.now().isoformat(),
                            'coordination_quality': max(0, min(1, 0.7 + 0.2 * np.random.normal())),
                            'sync_performance': max(0, min(1, 0.75 + 0.15 * np.random.normal())),
                            'system_health': max(0, min(100, 92 + 5 * np.random.normal())),
                            'performance_score': max(0, min(100, 80 + 10 * np.random.normal())),
                            'total_agents': 5,
                            'active_agents': 5,
                            'tasks_completed': np.random.randint(1, 10)
                        }]\n                        \n                        ml_engine.add_system_metrics(new_metrics)\n                        \n                        # Update system stats\n                        self.system_stats['total_requests'] += 1\n                    \n                    # Wait 30 seconds before next update\n                    await asyncio.sleep(30)\n                    \n                except Exception as e:\n                    logger.error(f\"❌ Data sync failed: {e}\")\n                    await asyncio.sleep(60)\n        \n        # Start the data sync task\n        self.data_sync_task = asyncio.create_task(data_sync_loop())\n    \n    def get_system_status(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive system status\"\"\"\n        status = {\n            'system_info': {\n                'running': self.running,\n                'start_time': self.system_stats['start_time'],\n                'uptime': str(datetime.now() - self.system_stats['start_time']) if self.system_stats['start_time'] else None,\n                'components_active': len([c for c in self.components.values() if c.get('status') == 'running'])\n            },\n            'components': {},\n            'statistics': self.system_stats,\n            'endpoints': {\n                'dashboard': 'http://localhost:8051',\n                'api': 'http://localhost:8080',\n                'api_docs': 'http://localhost:8080/docs',\n                'api_redoc': 'http://localhost:8080/redoc'\n            }\n        }\n        \n        # Add component details\n        for name, component in self.components.items():\n            status['components'][name] = {\n                'status': component.get('status', 'unknown'),\n                'port': component.get('port'),\n                'last_analysis': component.get('last_analysis')\n            }\n        \n        return status\n    \n    def print_system_banner(self):\n        \"\"\"Print system startup banner\"\"\"\n        banner = f\"\"\"\n╔══════════════════════════════════════════════════════════════════════════════╗\n║                    🚀 KAIROS MULTI-AGENT AI PLATFORM 🚀                    ║\n║                         Complete System Integration                          ║\n╠══════════════════════════════════════════════════════════════════════════════╣\n║                                                                              ║\n║  🎨 Enhanced Analytics Dashboard → http://localhost:8051                    ║\n║  🔗 REST API Server            → http://localhost:8080                    ║\n║  📚 API Documentation          → http://localhost:8080/docs               ║\n║  🧠 ML Analytics Engine        → Running continuous analysis              ║\n║                                                                              ║\n║  📊 Features:                                                               ║\n║     • Real-time multi-agent coordination monitoring                         ║\n║     • Advanced ML analytics (clustering, anomaly detection, predictions)   ║\n║     • Professional web dashboard with export capabilities                   ║\n║     • Comprehensive REST API with OpenAPI documentation                     ║\n║     • WebSocket live updates and real-time notifications                   ║\n║     • Performance optimization recommendations                               ║\n║                                                                              ║\n║  🎯 Status: {len(self.components)} components active                                                ║\n║  ⏱️  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                           ║\n║                                                                              ║\n╚══════════════════════════════════════════════════════════════════════════════╝\n        \"\"\"\n        print(banner)\n    \n    def print_component_status(self):\n        \"\"\"Print detailed component status\"\"\"\n        print(\"\\n📊 COMPONENT STATUS:\")\n        print(\"─\" * 60)\n        \n        for name, component in self.components.items():\n            status = component.get('status', 'unknown')\n            port = component.get('port', 'N/A')\n            \n            status_icon = \"✅\" if status == 'running' else \"❌\"\n            print(f\"{status_icon} {name.upper():20} {status:10} Port: {port}\")\n        \n        print(\"─\" * 60)\n        print(f\"📈 Total Active Components: {len(self.components)}\")\n        print(f\"🔄 System Requests: {self.system_stats['total_requests']}\")\n        print(f\"🧠 ML Analyses: {self.system_stats['ml_analyses']}\")\n    \n    async def start_system(self):\n        \"\"\"Start the complete Kairos system\"\"\"\n        logger.info(\"🚀 Starting Kairos Complete System...\")\n        self.system_stats['start_time'] = datetime.now()\n        self.running = True\n        \n        # Start all components\n        components_started = 0\n        \n        # Start Enhanced Dashboard\n        if self.start_enhanced_dashboard():\n            components_started += 1\n            await asyncio.sleep(2)  # Wait for component to initialize\n        \n        # Start REST API Server\n        if self.start_rest_api_server():\n            components_started += 1\n            await asyncio.sleep(2)\n        \n        # Start ML Analytics Engine\n        if self.start_ml_analytics():\n            components_started += 1\n            await asyncio.sleep(1)\n        \n        self.system_stats['components_active'] = components_started\n        \n        # Print system banner\n        self.print_system_banner()\n        self.print_component_status()\n        \n        if components_started > 0:\n            logger.info(f\"✅ Kairos System started successfully with {components_started} active components\")\n            \n            # Start background tasks\n            self.start_data_sync_task()\n            \n            # Start ML analysis cycle\n            ml_analysis_task = asyncio.create_task(self.run_ml_analysis_cycle())\n            \n            print(\"\\n🎯 SYSTEM READY - Press Ctrl+C to stop\")\n            print(\"\\n💡 Quick Access URLs:\")\n            print(\"   🎨 Dashboard:  http://localhost:8051\")\n            print(\"   🔗 API:        http://localhost:8080\")\n            print(\"   📚 API Docs:   http://localhost:8080/docs\")\n            print(\"   🔄 Live Data:  ws://localhost:8080/ws/live\")\n            \n            return True\n        else:\n            logger.error(\"❌ Failed to start Kairos System - no components running\")\n            return False\n    \n    def stop_system(self):\n        \"\"\"Stop the complete Kairos system\"\"\"\n        logger.info(\"🛑 Stopping Kairos System...\")\n        self.running = False\n        \n        # Cancel background tasks\n        if self.data_sync_task:\n            self.data_sync_task.cancel()\n        \n        # Stop components (threads will terminate when main process exits)\n        self.components.clear()\n        \n        uptime = datetime.now() - self.system_stats['start_time'] if self.system_stats['start_time'] else timedelta(0)\n        \n        print(\"\\n\" + \"=\"*80)\n        print(\"🏁 KAIROS SYSTEM SHUTDOWN COMPLETE\")\n        print(f\"⏱️  Total Uptime: {uptime}\")\n        print(f\"📊 System Requests Processed: {self.system_stats['total_requests']}\")\n        print(f\"🧠 ML Analyses Completed: {self.system_stats['ml_analyses']}\")\n        print(\"Thank you for using Kairos Multi-Agent AI Platform! 🚀\")\n        print(\"=\"*80)\n        \n        logger.info(\"✅ Kairos System stopped successfully\")\n\ndef signal_handler(sig, frame):\n    \"\"\"Handle system interrupt signals\"\"\"\n    print(\"\\n🛑 Received interrupt signal, shutting down gracefully...\")\n    if 'orchestrator' in globals():\n        orchestrator.stop_system()\n    sys.exit(0)\n\nasync def main():\n    \"\"\"Main system entry point\"\"\"\n    global orchestrator\n    \n    # Register signal handlers for graceful shutdown\n    signal.signal(signal.SIGINT, signal_handler)\n    signal.signal(signal.SIGTERM, signal_handler)\n    \n    # Create and start orchestrator\n    orchestrator = KairosSystemOrchestrator()\n    \n    try:\n        success = await orchestrator.start_system()\n        \n        if success:\n            # Keep the system running\n            while orchestrator.running:\n                await asyncio.sleep(1)\n        \n    except KeyboardInterrupt:\n        orchestrator.stop_system()\n    except Exception as e:\n        logger.error(f\"❌ System error: {e}\")\n        orchestrator.stop_system()\n    \nif __name__ == \"__main__\":\n    print(\"🚀 Initializing Kairos Complete System...\")\n    \n    # Check if we're in the right directory\n    import os\n    if not os.path.exists('E:/kairos'):\n        print(\"❌ Error: Kairos directory not found. Please ensure you're running from the correct location.\")\n        sys.exit(1)\n    \n    try:\n        asyncio.run(main())\n    except KeyboardInterrupt:\n        print(\"\\n🛑 System interrupted by user\")\n    except Exception as e:\n        print(f\"\\n❌ System startup failed: {e}\")\n        sys.exit(1)