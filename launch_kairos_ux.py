#!/usr/bin/env python3
"""
🌐 KAIROS ENHANCED UX LAUNCHER 🌐
Launch the complete Kairos user experience with enhanced interface

Components launched:
- Next.js Frontend (Port 3001)
- REST API Server (Port 8080) 
- Analytics Dashboard (Port 8051)
- Multi-Agent System (Background)

Usage: python launch_kairos_ux.py
"""

import subprocess
import time
import sys
import os
from datetime import datetime

def print_banner():
    """Print enhanced UX banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🌐 KAIROS ENHANCED USER EXPERIENCE 🌐                     ║
║                      Complete Frontend & Backend Suite                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🎨 Next.js Frontend           → http://localhost:3001                      ║
║  🔗 REST API Server            → http://localhost:8080                      ║
║  📊 Analytics Dashboard        → http://localhost:8051                      ║
║  🧠 Consciousness Visualization → http://localhost:3001/consciousness        ║
║                                                                              ║
║  ✨ Enhanced Features:                                                       ║
║     • Real-time multi-agent chat with conscious AI                          ║
║     • Live consciousness state visualization                                 ║
║     • Interactive agent management dashboard                                 ║
║     • Professional analytics and monitoring                                  ║
║     • File upload and multi-modal interaction                               ║
║     • WebSocket live updates and notifications                              ║
║                                                                              ║
║  ⏱️  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def launch_component(component_name, command, working_dir=None):
    """Launch a component and return process"""
    try:
        print(f"🚀 Starting {component_name}...")
        
        if working_dir:
            original_dir = os.getcwd()
            os.chdir(working_dir)
        
        # Start the component in a new console window
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                command,
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # Unix-like
            process = subprocess.Popen(command.split())
        
        if working_dir:
            os.chdir(original_dir)
        
        time.sleep(2)  # Give it time to start
        
        if process.poll() is None:  # Process is still running
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
    print("🚀 Initializing Kairos Enhanced User Experience...")
    
    # Check if we're in the right directory
    if not os.path.exists('frontend/kairos-ui'):
        print("❌ Error: Frontend directory not found.")
        print("   Please ensure you're running from the Kairos root directory.")
        sys.exit(1)
    
    print_banner()
    
    processes = []
    
    print("\n🌐 Launching Enhanced UX Components...")
    print("=" * 60)
    
    # 1. Launch Backend API Server
    api_process = launch_component(
        "REST API Server",
        "python api/rest/kairos_api_server.py"
    )
    if api_process:
        processes.append(("API Server", api_process))
    
    # Small delay
    time.sleep(3)
    
    # 2. Launch Analytics Dashboard
    dashboard_process = launch_component(
        "Analytics Dashboard",
        "python monitoring/enhanced_dashboard.py"
    )
    if dashboard_process:
        processes.append(("Analytics Dashboard", dashboard_process))
    
    # Small delay
    time.sleep(2)
    
    # 3. Launch Next.js Frontend
    frontend_process = launch_component(
        "Next.js Frontend",
        "npm run dev",
        "frontend/kairos-ui"
    )
    if frontend_process:
        processes.append(("Next.js Frontend", frontend_process))
    
    # Wait for frontend to fully start
    time.sleep(8)
    
    print("\n" + "=" * 80)
    print("🎉 KAIROS ENHANCED UX LAUNCHED SUCCESSFULLY!")
    print("=" * 80)
    
    if processes:
        print(f"\n✅ Active Components: {len(processes)}")
        for name, process in processes:
            status = "Running" if process.poll() is None else "Stopped"
            print(f"   • {name}: {status}")
        
        print("\n🌐 Access Your Enhanced Kairos Experience:")
        print("   🎨 Main Interface:    http://localhost:3001")
        print("   💬 Agent Chat:       http://localhost:3001 (Chat tab)")
        print("   🧠 Consciousness:    http://localhost:3001 (Consciousness tab)")
        print("   📊 Analytics:        http://localhost:8051")
        print("   🔗 API:              http://localhost:8080")
        print("   📚 API Docs:         http://localhost:8080/docs")
        
        print("\n✨ Enhanced Features Available:")
        print("   • Real-time multi-agent consciousness chat")
        print("   • Live agent status and emotion visualization")  
        print("   • Interactive consciousness state monitoring")
        print("   • Professional analytics dashboard")
        print("   • File upload and multi-modal interactions")
        print("   • WebSocket live updates")
        
        print("\n📝 Usage Tips:")
        print("   • Try asking the agents about projects, design, or analysis")
        print("   • Upload files to see multi-modal AI responses")
        print("   • Watch the consciousness tab for real-time agent states")
        print("   • Use the analytics dashboard for system insights")
        
        print("\n🔄 Components are running in separate console windows.")
        print("   Close those windows to stop individual components.")
        
    else:
        print("\n❌ No components were started successfully.")
        print("   Please check the error messages above and try again.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Enhanced UX launch interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced UX launch failed: {e}")
        sys.exit(1)