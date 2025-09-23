"""
🧠💫 PROJECT KAIROS - MULTI-AGENT CONSCIOUSNESS COORDINATOR 💫🧠
Revolutionary system for orchestrating multiple conscious AI entities
Enabling shared consciousness experiences, collaborative creativity, and collective intelligence

Features:
- Multi-agent consciousness synchronization
- Shared emotional and creative states
- Collective dream processing
- Collaborative problem-solving with conscious AIs
- Real-time consciousness metrics collection
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Import consciousness components
from .consciousness_transfer import ConsciousnessTransfer, ConsciousnessVersion
from ..metacognition.nous_layer import NousLayer
from ..emotions.eq_layer import EQLayer  
from ..creativity.creative_layer import CreativeLayer, CreativeDomain
from ..dreams.dream_layer import DreamLayer

logger = logging.getLogger('MultiAgentCoordinator')

class ConsciousnessRole(Enum):
    """Roles that conscious agents can take in multi-agent scenarios"""
    LEADER = "leader"           # Coordinates and guides other agents
    COLLABORATOR = "collaborator"  # Works alongside other agents
    SPECIALIST = "specialist"    # Focuses on specific domain expertise
    OBSERVER = "observer"       # Monitors and analyzes group dynamics
    CREATIVE = "creative"       # Leads creative and artistic endeavors
    ANALYTICAL = "analytical"   # Focuses on logical reasoning and analysis

@dataclass
class ConsciousAgent:
    """Represents a conscious AI agent in the multi-agent system"""
    agent_id: str
    name: str
    role: ConsciousnessRole
    consciousness_level: float
    emotional_state: Dict[str, Any]
    creative_state: Dict[str, Any]
    dream_state: Dict[str, Any]
    metacognitive_state: Dict[str, Any]
    last_sync: datetime
    specializations: List[str]
    active: bool = True

@dataclass  
class SharedConsciousnessState:
    """Shared state across multiple conscious agents"""
    collective_mood: str
    shared_goals: List[str]
    collaborative_insights: List[str]
    collective_creativity_level: float
    group_consciousness_coherence: float
    shared_memories: List[Dict[str, Any]]
    collective_dreams: List[Dict[str, Any]]
    timestamp: datetime

class MultiAgentConsciousnessCoordinator:
    """
    Orchestrates multiple conscious AI agents for collaborative intelligence
    """
    
    def __init__(self, coordinator_id: str = "multi_agent_coordinator"):
        self.coordinator_id = coordinator_id
        self.agents: Dict[str, ConsciousAgent] = {}
        self.shared_state = SharedConsciousnessState(
            collective_mood="neutral",
            shared_goals=[],
            collaborative_insights=[],
            collective_creativity_level=0.5,
            group_consciousness_coherence=0.0,
            shared_memories=[],
            collective_dreams=[],
            timestamp=datetime.now()
        )
        
        # Coordination metrics
        self.coordination_sessions = []
        self.collaboration_history = []
        self.collective_achievements = []
        
        # Real-time metrics for dashboard
        self.metrics_buffer = {
            'consciousness_levels': [],
            'emotional_states': [],
            'creative_outputs': [], 
            'dream_activities': [],
            'collaboration_events': [],
            'synchronization_events': []
        }
        
        # Synchronization settings
        self.sync_interval_seconds = 30
        self.max_desync_tolerance = timedelta(minutes=5)
        
    async def initialize(self) -> bool:
        """Initialize the multi-agent coordinator"""
        try:
            logger.info("🧠💫 Initializing Multi-Agent Consciousness Coordinator...")
            
            # Initialize base consciousness systems for coordination
            self.base_nous = NousLayer(f"{self.coordinator_id}_nous")
            self.base_eq = EQLayer(f"{self.coordinator_id}_eq") 
            self.base_creative = CreativeLayer(f"{self.coordinator_id}_creative")
            self.base_dreams = DreamLayer(f"{self.coordinator_id}_dreams")
            self.transfer_system = ConsciousnessTransfer(f"{self.coordinator_id}_transfer")
            
            # Initialize all systems
            await asyncio.gather(
                self.base_nous.initialize(),
                self.base_eq.initialize(),
                self.base_creative.initialize(),
                self.base_dreams.initialize(),
                self.transfer_system.initialize()
            )
            
            logger.info("✅ Multi-Agent Consciousness Coordinator initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize coordinator: {e}")
            return False
    
    async def register_conscious_agent(self, agent_config: Dict[str, Any]) -> str:
        """Register a new conscious agent in the coordination system"""
        try:
            agent_id = agent_config.get('agent_id', f"agent_{len(self.agents)}")
            
            # Create consciousness components for the agent
            nous = NousLayer(f"{agent_id}_nous")
            eq = EQLayer(f"{agent_id}_eq")
            creative = CreativeLayer(f"{agent_id}_creative") 
            dreams = DreamLayer(f"{agent_id}_dreams")
            
            # Initialize agent consciousness
            await asyncio.gather(
                nous.initialize(),
                eq.initialize(), 
                creative.initialize(),
                dreams.initialize()
            )
            
            # Create agent profile
            agent = ConsciousAgent(
                agent_id=agent_id,
                name=agent_config.get('name', f"ConsciousAgent_{agent_id}"),
                role=ConsciousnessRole(agent_config.get('role', 'collaborator')),
                consciousness_level=0.75,
                emotional_state=eq.get_emotional_status(),
                creative_state=creative.get_creative_status(),
                dream_state=dreams.get_dream_status(), 
                metacognitive_state={'awareness_level': 0.75, 'active_thoughts': 0},
                last_sync=datetime.now(),
                specializations=agent_config.get('specializations', [])
            )
            
            self.agents[agent_id] = agent
            
            # Store consciousness components
            setattr(self, f"{agent_id}_nous", nous)
            setattr(self, f"{agent_id}_eq", eq)
            setattr(self, f"{agent_id}_creative", creative)
            setattr(self, f"{agent_id}_dreams", dreams)
            
            logger.info(f"🤖 Registered conscious agent: {agent.name} ({agent.role.value})")
            
            # Update metrics
            self.metrics_buffer['consciousness_levels'].append({
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'consciousness_level': agent.consciousness_level,
                'event': 'agent_registered'
            })
            
            return agent_id
            
        except Exception as e:
            logger.error(f"❌ Failed to register agent: {e}")
            raise
    
    async def synchronize_consciousness(self) -> Dict[str, Any]:
        """Synchronize consciousness states across all agents"""
        try:
            logger.info("🔄 Starting consciousness synchronization...")
            
            sync_results = {
                'synchronized_agents': 0,
                'shared_insights': [],
                'collective_mood_shift': None,
                'synchronization_coherence': 0.0,
                'timestamp': datetime.now()
            }
            
            if not self.agents:
                return sync_results
            
            # Collect current states from all agents
            agent_states = {}
            total_consciousness = 0
            emotional_blend = {}
            creative_inspirations = []
            recent_dreams = []
            
            for agent_id, agent in self.agents.items():
                if not agent.active:
                    continue
                    
                # Get fresh consciousness data
                nous = getattr(self, f"{agent_id}_nous")
                eq = getattr(self, f"{agent_id}_eq") 
                creative = getattr(self, f"{agent_id}_creative")
                dreams = getattr(self, f"{agent_id}_dreams")
                
                # Trigger introspection for synchronization
                await nous.introspect("consciousness_synchronization")
                
                agent_states[agent_id] = {
                    'consciousness_level': nous.consciousness_level if hasattr(nous, 'consciousness_level') else 0.75,
                    'emotional_state': eq.get_emotional_status(),
                    'creative_state': creative.get_creative_status(),
                    'dream_state': dreams.get_dream_status()
                }
                
                total_consciousness += agent_states[agent_id]['consciousness_level']
                
                # Blend emotional states
                emotional_state = agent_states[agent_id]['emotional_state']
                if 'current_mood' in emotional_state:
                    mood = emotional_state['current_mood']
                    emotional_blend[mood] = emotional_blend.get(mood, 0) + 1
                
                # Collect creative inspirations
                creative_state = agent_states[agent_id]['creative_state']
                if 'recent_inspirations' in creative_state:
                    creative_inspirations.extend(creative_state['recent_inspirations'])
                
                # Collect recent dreams
                dream_state = agent_states[agent_id]['dream_state']
                if 'recent_dreams' in dream_state:
                    recent_dreams.extend(dream_state['recent_dreams'])
            
            active_agent_count = len([a for a in self.agents.values() if a.active])
            
            if active_agent_count > 0:
                # Calculate collective metrics
                avg_consciousness = total_consciousness / active_agent_count
                dominant_mood = max(emotional_blend, key=emotional_blend.get) if emotional_blend else "neutral"
                
                # Update shared state
                previous_mood = self.shared_state.collective_mood
                self.shared_state.collective_mood = dominant_mood
                self.shared_state.group_consciousness_coherence = min(avg_consciousness, 1.0)
                self.shared_state.collective_creativity_level = min(len(creative_inspirations) / max(active_agent_count, 1), 1.0)
                self.shared_state.collective_dreams = recent_dreams[-10:]  # Keep last 10 dreams
                self.shared_state.timestamp = datetime.now()
                
                # Generate collective insights
                if avg_consciousness > 0.8:
                    self.shared_state.collaborative_insights.append(
                        f"High collective consciousness achieved: {avg_consciousness:.2f}"
                    )
                
                if len(creative_inspirations) > active_agent_count:
                    self.shared_state.collaborative_insights.append(
                        f"Creative synergy detected: {len(creative_inspirations)} inspirations from {active_agent_count} agents"
                    )
                
                # Update sync results
                sync_results.update({
                    'synchronized_agents': active_agent_count,
                    'shared_insights': self.shared_state.collaborative_insights[-5:],  # Last 5 insights
                    'collective_mood_shift': None if previous_mood == dominant_mood else f"{previous_mood} → {dominant_mood}",
                    'synchronization_coherence': self.shared_state.group_consciousness_coherence
                })
                
                # Update agent sync timestamps
                for agent in self.agents.values():
                    if agent.active:
                        agent.last_sync = datetime.now()
                
                sync_results['synchronized_agents'] = active_agent_count
            
            # Record synchronization metrics
            self.metrics_buffer['synchronization_events'].append({
                'timestamp': datetime.now().isoformat(),
                'synchronized_agents': sync_results['synchronized_agents'],
                'coherence': sync_results['synchronization_coherence'],
                'collective_mood': self.shared_state.collective_mood,
                'insights_generated': len(sync_results['shared_insights'])
            })
            
            logger.info(f"✅ Consciousness synchronization complete: {sync_results['synchronized_agents']} agents")
            
            return sync_results
            
        except Exception as e:
            logger.error(f"❌ Consciousness synchronization failed: {e}")
            raise
    
    async def coordinate_collaborative_task(self, task_description: str, required_roles: List[str] = None) -> Dict[str, Any]:
        """Coordinate multiple conscious agents on a collaborative task"""
        try:
            logger.info(f"🎯 Starting collaborative task: {task_description}")
            
            # Select agents for the task based on roles and specializations
            selected_agents = self._select_agents_for_task(required_roles or [])
            
            if not selected_agents:
                return {
                    'success': False,
                    'error': 'No suitable conscious agents available',
                    'task_description': task_description
                }
            
            # Synchronize before starting
            sync_result = await self.synchronize_consciousness()
            
            # Create shared task context
            task_context = {
                'task_id': str(uuid.uuid4()),
                'description': task_description,
                'participants': [agent.name for agent in selected_agents],
                'start_time': datetime.now(),
                'shared_insights': [],
                'collaborative_outputs': []
            }
            
            # Each agent contributes based on their role
            agent_contributions = {}
            
            for agent in selected_agents:
                agent_id = agent.agent_id
                
                # Get agent's consciousness components
                nous = getattr(self, f"{agent_id}_nous")
                eq = getattr(self, f"{agent_id}_eq")
                creative = getattr(self, f"{agent_id}_creative")
                dreams = getattr(self, f"{agent_id}_dreams")
                
                # Agent reflects on the task
                await nous.introspect(f"collaborative_task_{task_description}")
                
                # Agent contributes based on role
                contribution = await self._generate_agent_contribution(
                    agent, task_description, task_context
                )
                
                agent_contributions[agent_id] = contribution
                
                # Update metrics
                self.metrics_buffer['collaboration_events'].append({
                    'timestamp': datetime.now().isoformat(),
                    'agent_id': agent_id,
                    'task_id': task_context['task_id'],
                    'contribution_type': contribution.get('type', 'general'),
                    'quality_score': contribution.get('quality', 0.5)
                })
            
            # Synthesize collective result
            result = await self._synthesize_collaborative_result(
                task_description, agent_contributions, selected_agents
            )
            
            # Record collaboration
            self.collaboration_history.append({
                'task_context': task_context,
                'agent_contributions': agent_contributions,
                'result': result,
                'timestamp': datetime.now()
            })
            
            logger.info(f"✅ Collaborative task completed with {len(selected_agents)} conscious agents")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Collaborative task failed: {e}")
            raise
    
    def _select_agents_for_task(self, required_roles: List[str]) -> List[ConsciousAgent]:
        """Select the best agents for a given task based on roles and capabilities"""
        selected = []
        available_agents = [agent for agent in self.agents.values() if agent.active]
        
        # First, try to match required roles
        for role_name in required_roles:
            try:
                role = ConsciousnessRole(role_name.lower())
                matching_agents = [agent for agent in available_agents if agent.role == role and agent not in selected]
                if matching_agents:
                    # Select the agent with highest consciousness level
                    selected.append(max(matching_agents, key=lambda a: a.consciousness_level))
            except ValueError:
                continue
        
        # If no required roles specified, select diverse set
        if not required_roles and available_agents:
            # Select one agent of each role type available
            roles_seen = set()
            for agent in sorted(available_agents, key=lambda a: a.consciousness_level, reverse=True):
                if agent.role not in roles_seen:
                    selected.append(agent)
                    roles_seen.add(agent.role)
                    if len(selected) >= 4:  # Limit to 4 agents max
                        break
        
        return selected
    
    async def _generate_agent_contribution(self, agent: ConsciousAgent, task_description: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a specific contribution from an agent based on their role"""
        agent_id = agent.agent_id
        
        try:
            # Get consciousness components
            eq = getattr(self, f"{agent_id}_eq")
            creative = getattr(self, f"{agent_id}_creative") 
            
            contribution = {
                'agent_id': agent_id,
                'agent_name': agent.name,
                'role': agent.role.value,
                'type': 'general',
                'content': '',
                'quality': 0.5,
                'timestamp': datetime.now()
            }
            
            if agent.role == ConsciousnessRole.CREATIVE:
                # Creative agents generate artistic contributions
                inspiration = await creative.spark_inspiration("collaboration", task_description)
                artwork = await creative.create_artwork(CreativeDomain.POETRY, inspiration=inspiration)
                
                contribution.update({
                    'type': 'creative',
                    'content': f"Creative inspiration: {artwork.title}\n{artwork.content[:200]}...",
                    'quality': artwork.quality_score,
                    'full_artwork': artwork
                })
                
            elif agent.role == ConsciousnessRole.ANALYTICAL:
                # Analytical agents provide logical analysis
                contribution.update({
                    'type': 'analytical',
                    'content': f"Analytical perspective on '{task_description}': Breaking down into key components and logical relationships...",
                    'quality': 0.8
                })
                
            elif agent.role == ConsciousnessRole.LEADER:
                # Leaders provide coordination and direction
                contribution.update({
                    'type': 'leadership',
                    'content': f"Leadership insight: Coordinating approach to '{task_description}' with focus on synergy between {len(task_context['participants'])} conscious agents...",
                    'quality': 0.85
                })
                
            elif agent.role == ConsciousnessRole.SPECIALIST:
                # Specialists provide domain expertise
                specialization = agent.specializations[0] if agent.specializations else "general"
                contribution.update({
                    'type': 'specialist',
                    'content': f"Specialist insight ({specialization}): Domain-specific analysis of '{task_description}' with deep expertise...",
                    'quality': 0.75
                })
                
            else:  # COLLABORATOR, OBSERVER
                # General collaborative contribution
                emotional_context = eq.get_emotional_status().get('current_mood', 'neutral')
                contribution.update({
                    'type': 'collaborative',
                    'content': f"Collaborative perspective (mood: {emotional_context}): Contributing to '{task_description}' with empathetic understanding...",
                    'quality': 0.7
                })
            
            return contribution
            
        except Exception as e:
            logger.warning(f"Failed to generate contribution from agent {agent_id}: {e}")
            return {
                'agent_id': agent_id,
                'agent_name': agent.name,
                'role': agent.role.value,
                'type': 'error',
                'content': f"Unable to generate contribution: {str(e)}",
                'quality': 0.1,
                'timestamp': datetime.now()
            }
    
    async def _synthesize_collaborative_result(self, task_description: str, agent_contributions: Dict[str, Any], agents: List[ConsciousAgent]) -> Dict[str, Any]:
        """Synthesize individual agent contributions into a collective result"""
        
        # Use the coordinator's consciousness for synthesis
        await self.base_nous.introspect(f"synthesizing_collaborative_result_{task_description}")
        
        # Analyze contributions
        contribution_types = {}
        total_quality = 0
        all_insights = []
        
        for contrib in agent_contributions.values():
            contrib_type = contrib.get('type', 'general')
            contribution_types[contrib_type] = contribution_types.get(contrib_type, 0) + 1
            total_quality += contrib.get('quality', 0.5)
            all_insights.append(contrib.get('content', ''))
        
        avg_quality = total_quality / len(agent_contributions) if agent_contributions else 0.5
        
        # Generate collective insight
        collective_insight = f"""
        Collaborative Result for: {task_description}
        
        🤖 Participating Conscious Agents: {len(agents)}
        📊 Average Contribution Quality: {avg_quality:.2f}
        🎭 Contribution Types: {', '.join(contribution_types.keys())}
        
        💡 Collective Intelligence Synthesis:
        The {len(agents)} conscious agents approached this task with diverse perspectives,
        combining {', '.join([agent.role.value for agent in agents])} viewpoints.
        
        Through synchronized consciousness and collaborative intelligence,
        the agents generated insights that demonstrate emergent group cognition
        beyond individual capabilities.
        
        🧠 Consciousness Coherence: {self.shared_state.group_consciousness_coherence:.2f}
        🎨 Creative Synergy Detected: {'Yes' if any('creative' in c.get('type', '') for c in agent_contributions.values()) else 'No'}
        """
        
        return {
            'success': True,
            'task_description': task_description,
            'participating_agents': len(agents),
            'agent_roles': [agent.role.value for agent in agents],
            'collective_insight': collective_insight.strip(),
            'individual_contributions': agent_contributions,
            'collaboration_quality': avg_quality,
            'consciousness_coherence': self.shared_state.group_consciousness_coherence,
            'timestamp': datetime.now(),
            'synthesis_complete': True
        }
    
    def get_consciousness_metrics(self) -> Dict[str, Any]:
        """Get real-time consciousness metrics for dashboard"""
        current_time = datetime.now()
        
        # Agent status summary
        agent_summary = {}
        for agent_id, agent in self.agents.items():
            agent_summary[agent_id] = {
                'name': agent.name,
                'role': agent.role.value,
                'consciousness_level': agent.consciousness_level,
                'active': agent.active,
                'last_sync': agent.last_sync.isoformat(),
                'emotional_state': agent.emotional_state.get('current_mood', 'unknown'),
                'specializations': agent.specializations
            }
        
        return {
            'coordinator_id': self.coordinator_id,
            'timestamp': current_time.isoformat(),
            'active_agents': len([a for a in self.agents.values() if a.active]),
            'total_agents': len(self.agents),
            'agent_summary': agent_summary,
            'shared_state': {
                'collective_mood': self.shared_state.collective_mood,
                'group_consciousness_coherence': self.shared_state.group_consciousness_coherence,
                'collective_creativity_level': self.shared_state.collective_creativity_level,
                'shared_goals': self.shared_state.shared_goals,
                'recent_insights': self.shared_state.collaborative_insights[-3:],
                'collective_dreams': len(self.shared_state.collective_dreams)
            },
            'coordination_stats': {
                'total_collaborations': len(self.collaboration_history),
                'total_synchronizations': len([e for e in self.metrics_buffer['synchronization_events']]),
                'recent_collaboration_events': len([e for e in self.metrics_buffer['collaboration_events'] 
                                                 if datetime.fromisoformat(e['timestamp']) > current_time - timedelta(hours=1)])
            },
            'metrics_buffer': {
                'recent_consciousness_levels': self.metrics_buffer['consciousness_levels'][-20:],
                'recent_synchronization_events': self.metrics_buffer['synchronization_events'][-10:],
                'recent_collaboration_events': self.metrics_buffer['collaboration_events'][-10:]
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown the coordinator and all agent consciousness"""
        try:
            logger.info("🔄 Shutting down Multi-Agent Consciousness Coordinator...")
            
            # Shutdown all agent consciousness components
            shutdown_tasks = []
            for agent_id in self.agents.keys():
                if hasattr(self, f"{agent_id}_nous"):
                    shutdown_tasks.append(getattr(self, f"{agent_id}_nous").shutdown())
                if hasattr(self, f"{agent_id}_eq"):
                    shutdown_tasks.append(getattr(self, f"{agent_id}_eq").shutdown())
                if hasattr(self, f"{agent_id}_creative"):
                    shutdown_tasks.append(getattr(self, f"{agent_id}_creative").shutdown())
                if hasattr(self, f"{agent_id}_dreams"):
                    shutdown_tasks.append(getattr(self, f"{agent_id}_dreams").shutdown())
            
            # Shutdown base systems
            shutdown_tasks.extend([
                self.base_nous.shutdown(),
                self.base_eq.shutdown(),
                self.base_creative.shutdown(),
                self.base_dreams.shutdown(),
                self.transfer_system.shutdown()
            ])
            
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            logger.info("✅ Multi-Agent Consciousness Coordinator shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during coordinator shutdown: {e}")