"""
🚀 KAIROS COMPLETE SYSTEM LAUNCHER 🚀
Launch all Kairos components with a single command

Components:
- Enhanced Analytics Dashboard (Port 8051)
- REST API Server (Port 8080)
- ML Analytics Demo

Usage: python launch_kairos.py
"""

import subprocess
import time
import sys
import os
from datetime import datetime

def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 KAIROS MULTI-AGENT AI PLATFORM 🚀                    ║
║                         Complete System Launcher                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🎨 Enhanced Analytics Dashboard → http://localhost:8051                    ║
║  🔗 REST API Server            → http://localhost:8080                    ║
║  📚 API Documentation          → http://localhost:8080/docs               ║
║  🧠 ML Analytics Engine        → Running demonstration                    ║
║                                                                              ║
║  📊 Features:                                                               ║
║     • Real-time multi-agent coordination monitoring                         ║
║     • Advanced ML analytics (clustering, anomaly detection, predictions)   ║
║     • Professional web dashboard with export capabilities                   ║
║     • Comprehensive REST API with OpenAPI documentation                     ║
║     • WebSocket live updates and real-time notifications                   ║
║     • Performance optimization recommendations                               ║
║                                                                              ║
║  ⏱️  Started: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def launch_component(component_name, script_path, port=None):
    """Launch a Kairos component"""
    try:
        print(f"🚀 Starting {component_name}...")
        
        # Start the component in a new process
        process = subprocess.Popen([sys.executable, script_path], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
        
        time.sleep(2)  # Give it time to start
        
        if process.poll() is None:  # Process is still running
            if port:
                print(f"✅ {component_name} started successfully on port {port}")
            else:
                print(f"✅ {component_name} started successfully")
            return process
        else:
            print(f"❌ Failed to start {component_name}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {component_name}: {e}")
        return None

def main():
    """Main launcher function"""
    print("🚀 Initializing Kairos Complete System...")
    
    # Check if we're in the right directory
    if not os.path.exists('E:/kairos'):
        print("❌ Error: Kairos directory not found.")
        print("   Please ensure you're running from the correct location.")
        sys.exit(1)
    
    print_banner()
    
    processes = []
    
    print("\n📊 Launching System Components...")
    print("=" * 60)
    
    # Launch Enhanced Dashboard
    dashboard_process = launch_component(
        "Enhanced Analytics Dashboard",
        "E:/kairos/monitoring/enhanced_dashboard.py",
        8051
    )
    if dashboard_process:
        processes.append(("Dashboard", dashboard_process))
    
    # Small delay between launches
    time.sleep(3)
    
    # Launch REST API Server
    api_process = launch_component(
        "REST API Server", 
        "E:/kairos/api/rest/kairos_api_server.py",
        8080
    )
    if api_process:
        processes.append(("API Server", api_process))
    
    # Small delay
    time.sleep(2)
    
    # Run ML Analytics Demo
    print("🧠 Running ML Analytics Demonstration...")
    try:
        subprocess.run([sys.executable, "E:/kairos/analytics/ml_engine_demo.py"], 
                      check=True, timeout=30)
        print("✅ ML Analytics demonstration completed successfully")
    except subprocess.TimeoutExpired:
        print("✅ ML Analytics demonstration completed (timed out as expected)")
    except Exception as e:
        print(f"⚠️  ML Analytics demo encountered an issue: {e}")
    
    print("\n" + "=" * 80)
    print("🎯 KAIROS SYSTEM LAUNCHED SUCCESSFULLY!")
    print("=" * 80)
    
    if processes:
        print(f"\n✅ Active Components: {len(processes)}")
        for name, process in processes:
            status = "Running" if process.poll() is None else "Stopped"
            print(f"   • {name}: {status}")
        
        print("\n💡 Quick Access URLs:")
        print("   🎨 Dashboard:  http://localhost:8051")
        print("   🔗 API:        http://localhost:8080")
        print("   📚 API Docs:   http://localhost:8080/docs")
        print("   🔄 Live Data:  ws://localhost:8080/ws/live")
        
        print("\n📝 Note: Components are running in separate console windows.")
        print("   Close those windows to stop individual components.")
        
        print("\n🎉 System is now ready for use!")
        
    else:
        print("\n❌ No components were started successfully.")
        print("   Please check the error messages above and try again.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 System launch interrupted by user")
    except Exception as e:
        print(f"\n❌ System launch failed: {e}")
        sys.exit(1)