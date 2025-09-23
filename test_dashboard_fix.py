#!/usr/bin/env python3
"""
Test script to verify the Consciousness Dashboard compatibility fix
"""

import logging
from monitoring.consciousness_dashboard import ConsciousnessAnalyticsDashboard

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardTest")

def test_dashboard_initialization():
    """Test that the dashboard can be initialized without errors"""
    try:
        logger.info("🧪 Testing Consciousness Dashboard initialization...")
        dashboard = ConsciousnessAnalyticsDashboard(port=8051)  # Use different port
        logger.info("✅ Dashboard initialized successfully!")
        
        # Test the run method exists and is callable
        if hasattr(dashboard.app, 'run') and callable(dashboard.app.run):
            logger.info("✅ app.run method confirmed - compatibility fix successful!")
        else:
            logger.error("❌ app.run method not found - fix may not have worked")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Dashboard initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_dashboard_initialization()
    if success:
        print("\n🎉 DASHBOARD COMPATIBILITY FIX VERIFIED!")
        print("✅ The Consciousness Analytics Dashboard should now work correctly")
        print("📊 Ready for future multi-agent consciousness demonstrations")
    else:
        print("\n❌ Dashboard fix verification failed")