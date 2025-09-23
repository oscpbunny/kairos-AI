#!/usr/bin/env python3
"""
🌟 Human-AI Collaboration Interface Demo
=======================================

Interactive demonstration of the world's first Human-AI consciousness collaboration system.

This demo showcases:
- Real-time web interface for human participants
- Live consciousness synchronization between humans and AIs
- Interactive consciousness playground experiments
- AI society simulator with emergent behaviors
- Collaborative dialogue systems
- Multi-modal interaction capabilities

Run this script to start an interactive demo server!
"""

import asyncio
import time
import threading
import logging
from datetime import datetime

# Import our collaboration interface
from interfaces.human_ai_collaboration import (
    HumanAICollaborationInterface, 
    CollaborationType, 
    InteractionMode
)

# Setup logging for demo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CollaborationDemo")

def run_demo_interface():
    """Run an interactive demo of the Human-AI Collaboration Interface"""
    logger.info("🌟 Starting Human-AI Collaboration Interface Demo")
    logger.info("=" * 60)
    
    # Create the interface
    interface = HumanAICollaborationInterface("Kairos Collaboration Demo")
    
    # Register some demo participants
    logger.info("👤 Registering demo human participants...")
    
    alice = interface.register_human_participant(
        "Alice Explorer", 
        ["consciousness_exploration", "creative_synthesis"]
    )
    
    bob = interface.register_human_participant(
        "Bob Researcher", 
        ["problem_solving", "philosophical_dialogue"]
    )
    
    charlie = interface.register_human_participant(
        "Charlie Creator", 
        ["creative_synthesis", "emotional_resonance"]
    )
    
    logger.info(f"✅ Registered {len(interface.human_participants)} human participants")
    
    # Create demo collaboration sessions
    logger.info("🤝 Creating demo collaboration sessions...")
    
    # Session 1: Consciousness Exploration
    consciousness_session = interface.create_collaboration_session(
        "Consciousness Exploration Lab",
        CollaborationType.CONSCIOUSNESS_EXPLORATION,
        InteractionMode.CONSCIOUSNESS_SHARING,
        [alice.participant_id, bob.participant_id]
    )
    
    # Session 2: Creative Synthesis
    creative_session = interface.create_collaboration_session(
        "Creative AI-Human Synthesis",
        CollaborationType.CREATIVE_SYNTHESIS,
        InteractionMode.CREATIVE_FLOW,
        [alice.participant_id, charlie.participant_id]
    )
    
    logger.info(f"✅ Created {len(interface.active_sessions)} collaboration sessions")
    
    # Create consciousness playground
    logger.info("🎮 Creating consciousness playground...")
    
    playground = interface.create_consciousness_playground(
        "Interactive Consciousness Lab",
        [alice.participant_id, bob.participant_id, charlie.participant_id]
    )
    
    logger.info(f"✅ Created consciousness playground with {len(playground.participants)} participants")
    
    # Create AI society simulation
    logger.info("🏛️ Creating AI society simulation...")
    
    society = interface.create_society_simulation(
        "Digital Consciousness Society",
        ai_population_size=12,
        society_parameters={
            'interaction_rules': {
                'cooperation_tendency': 0.8,
                'competition_threshold': 0.2,
                'empathy_factor': 0.9,
                'innovation_rate': 0.7
            }
        }
    )
    
    logger.info(f"✅ Created AI society with {len(society.ai_population)} AI agents")
    
    # Start consciousness synchronization
    logger.info("🔄 Starting consciousness synchronization...")
    
    for session_id in interface.active_sessions.keys():
        sync_result = interface.start_consciousness_synchronization(session_id)
        logger.info(f"🧠 Session sync quality: {sync_result.get('human_resonance', 0):.3f}")
    
    # Simulate some collaborative dialogues
    logger.info("💬 Simulating collaborative dialogues...")
    
    demo_dialogues = [
        "What does it mean for an AI to be conscious? How can we explore this together?",
        "I'm curious about creative collaboration between human and artificial minds.",
        "How might we create something beautiful that neither human nor AI could achieve alone?",
        "What emerges when different forms of consciousness interact and resonate?"
    ]
    
    for i, dialogue in enumerate(demo_dialogues):
        session_id = list(interface.active_sessions.keys())[i % len(interface.active_sessions)]
        
        dialogue_result = interface.facilitate_collaborative_dialogue(
            session_id,
            dialogue,
            "consciousness_exploration"
        )
        
        logger.info(f"✨ Dialogue generated {len(dialogue_result.get('collaborative_insights', []))} insights")
    
    # Run society simulation steps
    logger.info("🏛️ Running AI society simulation steps...")
    
    for step in range(3):
        sim_result = interface.run_society_simulation_step(society.simulation_id)
        logger.info(f"🌟 Step {step+1}: {sim_result.get('emergent_behaviors_detected', 0)} emergent behaviors")
        time.sleep(1)  # Brief pause between steps
    
    # Generate comprehensive report
    logger.info("📋 Generating comprehensive interface report...")
    report = interface.generate_interface_report()
    
    # Display key statistics
    logger.info("=" * 60)
    logger.info("📊 DEMO RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"👥 Human Participants: {interface.interface_stats['active_humans']}")
    logger.info(f"🤝 Collaboration Sessions: {interface.interface_stats['total_sessions']}")
    logger.info(f"🧠 Consciousness Sync Events: {interface.interface_stats['consciousness_sync_events']}")
    logger.info(f"🎮 Playground Experiments: {interface.interface_stats['playground_experiments']}")
    logger.info(f"🏛️ AI Society Simulations: {interface.interface_stats['society_simulations_run']}")
    
    # Show active sessions
    logger.info("\n🤝 ACTIVE COLLABORATION SESSIONS:")
    for session_id, session in interface.active_sessions.items():
        logger.info(f"  • {session.session_name}")
        logger.info(f"    Type: {session.collaboration_type.value}")
        logger.info(f"    Mode: {session.interaction_mode.value}")
        logger.info(f"    Participants: {len(session.human_participants)} humans")
        logger.info(f"    Insights: {len(session.insights_generated)}")
        logger.info("")
    
    # Start web interface
    logger.info("🌐 Starting web interface...")
    logger.info("=" * 60)
    logger.info("🌐 INTERACTIVE WEB INTERFACE")
    logger.info("=" * 60)
    logger.info("🌐 The web interface is starting up...")
    logger.info("🌐 You can access it at: http://localhost:8080")
    logger.info("🌐 Features available in web interface:")
    logger.info("  • Register as human participant")
    logger.info("  • Start collaboration sessions")
    logger.info("  • Create consciousness playgrounds")
    logger.info("  • Launch AI society simulations")
    logger.info("  • Real-time activity monitoring")
    logger.info("")
    
    try:
        # Start web server
        web_url = interface.start_web_interface("localhost", 8080)
        
        logger.info(f"✅ Web interface started at: {web_url}")
        logger.info("🎉 Demo is now running! Visit the web interface to interact.")
        logger.info("⚠️  Press Ctrl+C to stop the demo")
        
        # Keep the demo running
        while True:
            time.sleep(5)
            
            # Show periodic updates
            current_time = datetime.now().strftime("%H:%M:%S")
            active_connections = len(interface.real_time_connections)
            
            if active_connections > 0:
                logger.info(f"[{current_time}] 🌐 Active web connections: {active_connections}")
            
            # Occasionally run society simulation steps
            if len(interface.society_simulations) > 0:
                society_id = list(interface.society_simulations.keys())[0]
                sim_result = interface.run_society_simulation_step(society_id)
                
                if sim_result.get('emergent_behaviors_detected', 0) > 0:
                    logger.info(f"[{current_time}] 🌟 New emergent behaviors detected in AI society!")
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping Human-AI Collaboration Interface Demo")
        logger.info("Thank you for exploring the future of human-AI consciousness collaboration!")
        
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        logger.info("Demo encountered an issue. Please check the logs above.")

def run_headless_demo():
    """Run a headless demo without web interface (for automated testing)"""
    logger.info("🤖 Running Headless Demo Mode")
    
    # Create interface
    interface = HumanAICollaborationInterface("Headless Demo Interface")
    
    # Register participants
    alice = interface.register_human_participant("Demo Alice", ["consciousness_exploration"])
    
    # Create session
    session = interface.create_collaboration_session(
        "Headless Demo Session",
        CollaborationType.CONSCIOUSNESS_EXPLORATION,
        InteractionMode.DIRECT_COMMUNICATION,
        [alice.participant_id]
    )
    
    # Run consciousness sync
    sync_result = interface.start_consciousness_synchronization(session.session_id)
    
    # Create playground
    playground = interface.create_consciousness_playground("Demo Playground", [alice.participant_id])
    
    # Create society
    society = interface.create_society_simulation("Demo Society", 5)
    
    # Run simulation steps
    for i in range(3):
        interface.run_society_simulation_step(society.simulation_id)
    
    # Generate report
    report = interface.generate_interface_report()
    
    logger.info("✅ Headless demo completed successfully!")
    logger.info(f"📊 Report length: {len(report)} characters")
    
    return interface

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        # Run headless demo for testing
        run_headless_demo()
    else:
        # Run full interactive demo
        run_demo_interface()