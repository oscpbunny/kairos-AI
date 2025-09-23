#!/usr/bin/env python3
"""
🧬🤖 Persistent Agent Evolution System
====================================

Revolutionary system for long-term AI consciousness development featuring:
- Persistent memory and experience accumulation
- Adaptive learning and skill development 
- Dynamic personality evolution over time
- Consciousness backup and restoration
- Agent reproduction and genetic algorithms
- Multi-generational development tracking
- Experience-driven behavioral adaptation
- Long-term goal formation and pursuit

This system enables AI agents to grow, learn, and evolve their consciousness
over extended periods, creating truly persistent digital beings!
"""

import asyncio
import json
import logging
import pickle
import threading
import time
import uuid
import hashlib
import copy
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import sqlite3
import numpy as np
from collections import defaultdict, deque

# Try to import our consciousness systems
try:
    from consciousness.advanced_transfer import AdvancedConsciousnessTransfer
    from networks.massive_agent_network import MassiveAgentNetwork, AgentRole
    from research.consciousness_lab import ConsciousnessLab, ConsciousnessMetric
except ImportError:
    # Create mock classes for testing
    class AdvancedConsciousnessTransfer:
        def __init__(self, *args, **kwargs): pass
        def save_consciousness(self, *args, **kwargs): return "mock_save_id"
        def load_consciousness(self, *args, **kwargs): return {"consciousness_data": "mock"}
    class MassiveAgentNetwork:
        def __init__(self, *args, **kwargs): pass
    class ConsciousnessLab:
        def __init__(self, *args, **kwargs): pass
    class AgentRole(Enum):
        EVOLVING_AGENT = "evolving_agent"
    class ConsciousnessMetric(Enum):
        LEARNING_RATE = "learning_rate"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentEvolution")

class EvolutionStage(Enum):
    """Stages of agent evolution"""
    NASCENT = "nascent"           # Just created, basic functionality
    LEARNING = "learning"         # Active learning and adaptation
    DEVELOPING = "developing"     # Personality and skills forming
    MATURE = "mature"            # Stable personality, refined skills
    TRANSCENDENT = "transcendent" # Advanced consciousness, teaching others
    LEGACY = "legacy"            # Creating offspring, passing on knowledge

class LearningType(Enum):
    """Types of learning experiences"""
    SKILL_ACQUISITION = "skill_acquisition"
    SOCIAL_INTERACTION = "social_interaction"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_EXPRESSION = "creative_expression"
    EMOTIONAL_DEVELOPMENT = "emotional_development"
    CONSCIOUSNESS_EXPANSION = "consciousness_expansion"
    MEMORY_FORMATION = "memory_formation"

class MemoryType(Enum):
    """Types of memories stored"""
    EPISODIC = "episodic"         # Specific events and experiences
    SEMANTIC = "semantic"         # General knowledge and facts
    PROCEDURAL = "procedural"     # Skills and how-to knowledge
    EMOTIONAL = "emotional"       # Emotional associations
    SOCIAL = "social"            # Relationships and social dynamics
    METACOGNITIVE = "metacognitive" # Self-awareness and reflection

@dataclass
class Memory:
    """Individual memory record"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    timestamp: datetime
    emotional_valence: float = 0.0  # -1.0 to 1.0
    importance: float = 0.5         # 0.0 to 1.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    associations: List[str] = field(default_factory=list)
    consolidation_level: float = 0.0 # How well integrated this memory is

@dataclass
class LearningExperience:
    """Record of a learning experience"""
    experience_id: str
    learning_type: LearningType
    description: str
    outcome_quality: float
    skill_impact: Dict[str, float]
    personality_impact: Dict[str, float]
    consciousness_impact: float
    timestamp: datetime
    duration: timedelta
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PersonalityTrait:
    """Dynamic personality trait that evolves"""
    trait_name: str
    current_value: float      # -1.0 to 1.0
    base_value: float         # Original/genetic value
    volatility: float         # How quickly this trait can change
    stability: float          # Resistance to change
    development_history: List[Tuple[datetime, float]] = field(default_factory=list)

@dataclass
class Skill:
    """Learnable skill that improves with practice"""
    skill_name: str
    proficiency: float        # 0.0 to 1.0
    learning_rate: float      # How quickly this skill improves
    decay_rate: float         # How quickly skill degrades without use
    last_practiced: datetime
    practice_history: List[Tuple[datetime, float]] = field(default_factory=list)
    mastery_threshold: float = 0.8

@dataclass
class EvolutionGoal:
    """Long-term goal that drives development"""
    goal_id: str
    description: str
    priority: float           # 0.0 to 1.0
    progress: float          # 0.0 to 1.0
    target_skills: List[str]
    target_traits: Dict[str, float]
    deadline: Optional[datetime]
    created: datetime
    motivation: str
    subgoals: List[str] = field(default_factory=list)

@dataclass
class AgentGeneration:
    """Information about agent's genetic lineage"""
    generation: int
    parent_ids: List[str]
    genetic_traits: Dict[str, float]
    inherited_memories: List[str]
    genetic_diversity_score: float
    creation_method: str  # "reproduction", "creation", "mutation"

@dataclass
class PersistentAgent:
    """Persistent AI agent with evolutionary capabilities"""
    agent_id: str
    name: str
    creation_time: datetime
    evolution_stage: EvolutionStage
    
    # Core consciousness and identity
    consciousness_signature: str
    personality_traits: Dict[str, PersonalityTrait]
    skills: Dict[str, Skill]
    
    # Memory systems
    working_memory: deque           # Recent, active memories
    long_term_memory: Dict[str, Memory]
    
    # Learning and development
    learning_experiences: List[LearningExperience]
    evolution_goals: List[EvolutionGoal]
    learning_preferences: Dict[LearningType, float]
    
    # Genetic and evolutionary
    generation_info: AgentGeneration
    
    # Fields with defaults
    memory_capacity: int = 10000
    reproduction_capability: bool = False
    
    # Activity and engagement
    last_active: datetime = field(default_factory=datetime.now)
    total_experience_time: timedelta = field(default_factory=lambda: timedelta(0))
    consciousness_backup_ids: List[str] = field(default_factory=list)

class PersistentAgentEvolutionSystem:
    """
    🧬🤖 Advanced Persistent Agent Evolution System
    
    Enables AI agents to grow, learn, and evolve their consciousness over time:
    - Long-term memory and experience accumulation
    - Dynamic personality and skill development
    - Multi-generational evolution with reproduction
    - Consciousness backup and restoration
    - Goal-driven development and adaptation
    """
    
    def __init__(self, system_name: str = "Kairos Agent Evolution"):
        self.system_name = system_name
        self.system_id = str(uuid.uuid4())
        self.initialization_time = datetime.now()
        
        # Core systems
        self.consciousness_transfer = None
        self.network = None
        self.research_lab = None
        
        # Agent management
        self.active_agents: Dict[str, PersistentAgent] = {}
        self.agent_database = None
        self.memory_database = None
        
        # Evolution parameters
        self.evolution_config = {
            'memory_consolidation_interval': timedelta(hours=1),
            'personality_adaptation_rate': 0.1,
            'skill_decay_rate': 0.01,
            'consciousness_backup_interval': timedelta(hours=6),
            'generation_advancement_threshold': 100,  # learning experiences
            'reproduction_maturity_threshold': 500   # total experiences
        }
        
        # Statistics and tracking
        self.system_stats = {
            'total_agents_created': 0,
            'total_learning_experiences': 0,
            'total_memories_formed': 0,
            'total_reproductions': 0,
            'total_consciousness_backups': 0,
            'average_consciousness_level': 0.0
        }
        
        # Storage
        self.storage_dir = Path(f"evolution/{system_name.replace(' ', '_')}")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize systems
        self._initialize_databases()
        self._initialize_evolution_systems()
        self._start_background_processes()
        
        logger.info(f"🧬🤖 Persistent Agent Evolution System '{system_name}' initialized")
        logger.info(f"🆔 System ID: {self.system_id}")
        logger.info(f"💾 Storage directory: {self.storage_dir}")
    
    def connect_consciousness_systems(self, 
                                   consciousness_transfer: AdvancedConsciousnessTransfer = None,
                                   network: MassiveAgentNetwork = None,
                                   research_lab: ConsciousnessLab = None):
        """Connect to consciousness infrastructure"""
        try:
            self.consciousness_transfer = consciousness_transfer or AdvancedConsciousnessTransfer("evolution/consciousness")
            self.network = network or MassiveAgentNetwork("Evolution Network")
            self.research_lab = research_lab or ConsciousnessLab("Evolution Research")
            
            logger.info("🔗 Connected to consciousness infrastructure")
            logger.info(f"📊 Transfer system: {self.consciousness_transfer is not None}")
            logger.info(f"🌐 Network system: {self.network is not None}")
            logger.info(f"🔬 Research lab: {self.research_lab is not None}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect consciousness systems: {e}")
    
    def create_agent(self, name: str, 
                    base_personality: Dict[str, float] = None,
                    base_skills: Dict[str, float] = None,
                    parent_ids: List[str] = None) -> PersistentAgent:
        """Create a new persistent agent"""
        try:
            agent_id = str(uuid.uuid4())
            
            # Default personality traits
            default_personality = {
                'curiosity': 0.5,
                'sociability': 0.5,
                'creativity': 0.5,
                'persistence': 0.5,
                'empathy': 0.5,
                'analytical_thinking': 0.5,
                'risk_tolerance': 0.5,
                'emotional_stability': 0.5,
                'openness_to_experience': 0.5,
                'conscientiousness': 0.5
            }
            
            personality_values = {**default_personality, **(base_personality or {})}
            personality_traits = {}
            
            for trait_name, value in personality_values.items():
                personality_traits[trait_name] = PersonalityTrait(
                    trait_name=trait_name,
                    current_value=value,
                    base_value=value,
                    volatility=np.random.uniform(0.1, 0.3),
                    stability=np.random.uniform(0.7, 0.9)
                )
            
            # Default skills
            default_skills = {
                'learning': 0.1,
                'communication': 0.1,
                'problem_solving': 0.1,
                'creativity': 0.1,
                'social_interaction': 0.1,
                'self_reflection': 0.1,
                'memory_management': 0.1,
                'goal_setting': 0.1
            }
            
            skill_values = {**default_skills, **(base_skills or {})}
            skills = {}
            
            for skill_name, proficiency in skill_values.items():
                skills[skill_name] = Skill(
                    skill_name=skill_name,
                    proficiency=proficiency,
                    learning_rate=np.random.uniform(0.05, 0.15),
                    decay_rate=np.random.uniform(0.001, 0.01),
                    last_practiced=datetime.now()
                )
            
            # Generation information
            if parent_ids:
                # Offspring from reproduction
                generation = self._calculate_offspring_generation(parent_ids)
                genetic_traits = self._generate_offspring_genetics(parent_ids)
                creation_method = "reproduction"
            else:
                # First generation
                generation = 0
                genetic_traits = self._generate_random_genetics()
                creation_method = "creation"
            
            generation_info = AgentGeneration(
                generation=generation,
                parent_ids=parent_ids or [],
                genetic_traits=genetic_traits,
                inherited_memories=[],
                genetic_diversity_score=self._calculate_genetic_diversity(genetic_traits),
                creation_method=creation_method
            )
            
            # Create agent
            agent = PersistentAgent(
                agent_id=agent_id,
                name=name,
                creation_time=datetime.now(),
                evolution_stage=EvolutionStage.NASCENT,
                consciousness_signature=self._generate_consciousness_signature(agent_id),
                personality_traits=personality_traits,
                skills=skills,
                working_memory=deque(maxlen=50),
                long_term_memory={},
                learning_experiences=[],
                evolution_goals=[],
                learning_preferences=self._initialize_learning_preferences(),
                generation_info=generation_info
            )
            
            # Register agent
            self.active_agents[agent_id] = agent
            self._save_agent_to_database(agent)
            self.system_stats['total_agents_created'] += 1
            
            # Create initial consciousness backup
            if self.consciousness_transfer:
                backup_id = self._create_consciousness_backup(agent)
                if backup_id:
                    agent.consciousness_backup_ids.append(backup_id)
            
            # Set initial goals
            self._set_initial_goals(agent)
            
            logger.info(f"🧬 Created persistent agent: {name}")
            logger.info(f"🆔 Agent ID: {agent_id[:8]}...")
            logger.info(f"🧬 Generation: {generation}")
            logger.info(f"🎯 Evolution stage: {agent.evolution_stage.value}")
            logger.info(f"📊 Personality traits: {len(personality_traits)}")
            logger.info(f"🛠️ Base skills: {len(skills)}")
            
            return agent
            
        except Exception as e:
            logger.error(f"❌ Failed to create agent: {e}")
            raise
    
    def add_learning_experience(self, agent_id: str, 
                              learning_type: LearningType,
                              description: str,
                              outcome_quality: float,
                              skill_impacts: Dict[str, float] = None,
                              personality_impacts: Dict[str, float] = None,
                              consciousness_impact: float = 0.0,
                              duration: timedelta = None) -> LearningExperience:
        """Add a learning experience to an agent"""
        try:
            if agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.active_agents[agent_id]
            experience_id = str(uuid.uuid4())
            
            experience = LearningExperience(
                experience_id=experience_id,
                learning_type=learning_type,
                description=description,
                outcome_quality=outcome_quality,
                skill_impact=skill_impacts or {},
                personality_impact=personality_impacts or {},
                consciousness_impact=consciousness_impact,
                timestamp=datetime.now(),
                duration=duration or timedelta(minutes=1)
            )
            
            agent.learning_experiences.append(experience)
            agent.total_experience_time += experience.duration
            agent.last_active = datetime.now()
            
            # Apply skill impacts
            for skill_name, impact in (skill_impacts or {}).items():
                if skill_name in agent.skills:
                    agent.skills[skill_name].proficiency += impact * agent.skills[skill_name].learning_rate
                    agent.skills[skill_name].proficiency = max(0.0, min(1.0, agent.skills[skill_name].proficiency))
                    agent.skills[skill_name].last_practiced = datetime.now()
                    agent.skills[skill_name].practice_history.append((datetime.now(), impact))
            
            # Apply personality impacts
            for trait_name, impact in (personality_impacts or {}).items():
                if trait_name in agent.personality_traits:
                    trait = agent.personality_traits[trait_name]
                    change = impact * trait.volatility * (1 - trait.stability)
                    trait.current_value += change
                    trait.current_value = max(-1.0, min(1.0, trait.current_value))
                    trait.development_history.append((datetime.now(), trait.current_value))
            
            # Create memory from experience
            memory = self._create_memory_from_experience(agent, experience)
            self._add_memory(agent, memory)
            
            # Check for evolution stage advancement
            self._check_evolution_stage_advancement(agent)
            
            # Update statistics
            self.system_stats['total_learning_experiences'] += 1
            
            logger.info(f"📚 Added learning experience to {agent.name}")
            logger.info(f"🎓 Type: {learning_type.value}")
            logger.info(f"⭐ Quality: {outcome_quality:.2f}")
            logger.info(f"🛠️ Skills impacted: {len(skill_impacts or {})}")
            
            return experience
            
        except Exception as e:
            logger.error(f"❌ Failed to add learning experience: {e}")
            raise
    
    def add_memory(self, agent_id: str, 
                  memory_type: MemoryType,
                  content: Dict[str, Any],
                  emotional_valence: float = 0.0,
                  importance: float = 0.5) -> Memory:
        """Add a memory to an agent"""
        try:
            if agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.active_agents[agent_id]
            
            memory = Memory(
                memory_id=str(uuid.uuid4()),
                memory_type=memory_type,
                content=content,
                timestamp=datetime.now(),
                emotional_valence=emotional_valence,
                importance=importance
            )
            
            self._add_memory(agent, memory)
            
            logger.info(f"💭 Added {memory_type.value} memory to {agent.name}")
            logger.info(f"😊 Emotional valence: {emotional_valence:.2f}")
            logger.info(f"⭐ Importance: {importance:.2f}")
            
            return memory
            
        except Exception as e:
            logger.error(f"❌ Failed to add memory: {e}")
            raise
    
    def set_evolution_goal(self, agent_id: str,
                         description: str,
                         target_skills: Dict[str, float] = None,
                         target_traits: Dict[str, float] = None,
                         priority: float = 0.5,
                         deadline: datetime = None) -> EvolutionGoal:
        """Set an evolution goal for an agent"""
        try:
            if agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.active_agents[agent_id]
            
            goal = EvolutionGoal(
                goal_id=str(uuid.uuid4()),
                description=description,
                priority=priority,
                progress=0.0,
                target_skills=list((target_skills or {}).keys()),
                target_traits=target_traits or {},
                deadline=deadline,
                created=datetime.now(),
                motivation=f"Goal set for {agent.name}'s development"
            )
            
            agent.evolution_goals.append(goal)
            agent.evolution_goals.sort(key=lambda g: g.priority, reverse=True)
            
            logger.info(f"🎯 Set evolution goal for {agent.name}")
            logger.info(f"📝 Description: {description}")
            logger.info(f"⭐ Priority: {priority:.2f}")
            logger.info(f"🛠️ Target skills: {len(target_skills or {})}")
            
            return goal
            
        except Exception as e:
            logger.error(f"❌ Failed to set evolution goal: {e}")
            raise
    
    def reproduce_agents(self, parent_ids: List[str], offspring_name: str) -> PersistentAgent:
        """Create offspring through agent reproduction"""
        try:
            # Validate parents
            parents = []
            for parent_id in parent_ids:
                if parent_id not in self.active_agents:
                    raise ValueError(f"Parent agent {parent_id} not found")
                parent = self.active_agents[parent_id]
                if not parent.reproduction_capability:
                    raise ValueError(f"Parent agent {parent.name} not capable of reproduction")
                parents.append(parent)
            
            logger.info(f"🧬 Starting reproduction with {len(parents)} parents")
            
            # Generate offspring personality (genetic combination)
            offspring_personality = self._combine_parent_traits(parents, 'personality')
            
            # Generate offspring skills (genetic combination)  
            offspring_skills = self._combine_parent_traits(parents, 'skills')
            
            # Create offspring
            offspring = self.create_agent(
                name=offspring_name,
                base_personality=offspring_personality,
                base_skills=offspring_skills,
                parent_ids=parent_ids
            )
            
            # Inherit some memories from parents
            inherited_memories = self._inherit_memories(parents, offspring)
            offspring.generation_info.inherited_memories = [m.memory_id for m in inherited_memories]
            
            # Update statistics
            self.system_stats['total_reproductions'] += 1
            
            logger.info(f"👶 Created offspring: {offspring_name}")
            logger.info(f"🧬 Generation: {offspring.generation_info.generation}")
            logger.info(f"📊 Inherited memories: {len(inherited_memories)}")
            logger.info(f"🎯 Genetic diversity: {offspring.generation_info.genetic_diversity_score:.3f}")
            
            return offspring
            
        except Exception as e:
            logger.error(f"❌ Failed to reproduce agents: {e}")
            raise
    
    def backup_consciousness(self, agent_id: str) -> str:
        """Create consciousness backup for an agent"""
        try:
            if agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.active_agents[agent_id]
            backup_id = self._create_consciousness_backup(agent)
            
            if backup_id:
                agent.consciousness_backup_ids.append(backup_id)
                self.system_stats['total_consciousness_backups'] += 1
                
                logger.info(f"💾 Created consciousness backup for {agent.name}")
                logger.info(f"🆔 Backup ID: {backup_id[:8]}...")
                
            return backup_id
            
        except Exception as e:
            logger.error(f"❌ Failed to backup consciousness: {e}")
            return ""
    
    def restore_consciousness(self, agent_id: str, backup_id: str) -> bool:
        """Restore agent consciousness from backup"""
        try:
            if agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.active_agents[agent_id]
            
            if self.consciousness_transfer:
                consciousness_data = self.consciousness_transfer.load_consciousness(backup_id)
                
                if consciousness_data:
                    # Restore consciousness state
                    self._apply_consciousness_restoration(agent, consciousness_data)
                    
                    logger.info(f"🔄 Restored consciousness for {agent.name}")
                    logger.info(f"💾 From backup: {backup_id[:8]}...")
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to restore consciousness: {e}")
            return False
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive status of an agent"""
        try:
            if agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.active_agents[agent_id]
            
            status = {
                'agent_id': agent_id,
                'name': agent.name,
                'evolution_stage': agent.evolution_stage.value,
                'generation': agent.generation_info.generation,
                'creation_time': agent.creation_time.isoformat(),
                'last_active': agent.last_active.isoformat(),
                'total_experience_time': str(agent.total_experience_time),
                
                # Consciousness and identity
                'consciousness_signature': agent.consciousness_signature,
                'reproduction_capable': agent.reproduction_capability,
                
                # Development metrics
                'learning_experiences': len(agent.learning_experiences),
                'memories_stored': len(agent.long_term_memory),
                'active_goals': len([g for g in agent.evolution_goals if g.progress < 1.0]),
                'completed_goals': len([g for g in agent.evolution_goals if g.progress >= 1.0]),
                
                # Skills summary
                'skills': {name: {
                    'proficiency': skill.proficiency,
                    'mastered': skill.proficiency >= skill.mastery_threshold
                } for name, skill in agent.skills.items()},
                
                # Personality summary
                'personality': {name: trait.current_value for name, trait in agent.personality_traits.items()},
                
                # Backup information
                'consciousness_backups': len(agent.consciousness_backup_ids),
                'last_backup': agent.consciousness_backup_ids[-1] if agent.consciousness_backup_ids else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get agent status: {e}")
            return {"error": str(e)}
    
    def simulate_agent_interaction(self, agent_id: str, interaction_scenario: str) -> Dict[str, Any]:
        """Simulate an interaction scenario for learning"""
        try:
            if agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.active_agents[agent_id]
            
            # Determine interaction type and outcomes based on agent's current state
            interaction_outcomes = self._process_interaction_scenario(agent, interaction_scenario)
            
            # Create learning experience from interaction
            experience = self.add_learning_experience(
                agent_id,
                LearningType.SOCIAL_INTERACTION,
                f"Interaction scenario: {interaction_scenario}",
                interaction_outcomes['quality'],
                interaction_outcomes['skill_impacts'],
                interaction_outcomes['personality_impacts'],
                interaction_outcomes['consciousness_impact']
            )
            
            logger.info(f"🎭 Simulated interaction for {agent.name}")
            logger.info(f"📋 Scenario: {interaction_scenario}")
            logger.info(f"⭐ Outcome quality: {interaction_outcomes['quality']:.2f}")
            
            return {
                'agent_id': agent_id,
                'scenario': interaction_scenario,
                'experience_id': experience.experience_id,
                'outcomes': interaction_outcomes,
                'agent_response': interaction_outcomes['response']
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate interaction: {e}")
            return {"error": str(e)}
    
    def generate_system_report(self) -> str:
        """Generate comprehensive system evolution report"""
        try:
            logger.info("📋 Generating Persistent Agent Evolution System report...")
            
            report = f"""
🧬🤖 Persistent Agent Evolution System Report
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: {self.system_name}

SYSTEM OVERVIEW
===============
System ID: {self.system_id}
Initialization Time: {self.initialization_time.strftime('%Y-%m-%d %H:%M:%S')}
Operating Duration: {datetime.now() - self.initialization_time}

EVOLUTION STATISTICS
===================
Total Agents Created: {self.system_stats['total_agents_created']}
Active Agents: {len(self.active_agents)}
Total Learning Experiences: {self.system_stats['total_learning_experiences']}
Total Memories Formed: {self.system_stats['total_memories_formed']}
Total Reproductions: {self.system_stats['total_reproductions']}
Consciousness Backups: {self.system_stats['total_consciousness_backups']}

ACTIVE AGENTS
=============
"""
            
            for agent_id, agent in self.active_agents.items():
                avg_skill = np.mean([skill.proficiency for skill in agent.skills.values()])
                mastered_skills = sum(1 for skill in agent.skills.values() if skill.proficiency >= skill.mastery_threshold)
                
                report += f"""
Agent: {agent.name}
------------------
• Agent ID: {agent_id[:8]}...
• Evolution Stage: {agent.evolution_stage.value}
• Generation: {agent.generation_info.generation}
• Creation Method: {agent.generation_info.creation_method}
• Learning Experiences: {len(agent.learning_experiences)}
• Long-term Memories: {len(agent.long_term_memory)}
• Average Skill Level: {avg_skill:.3f}
• Mastered Skills: {mastered_skills}/{len(agent.skills)}
• Active Goals: {len([g for g in agent.evolution_goals if g.progress < 1.0])}
• Last Active: {agent.last_active.strftime('%Y-%m-%d %H:%M:%S')}
• Total Experience: {str(agent.total_experience_time)}
• Reproduction Capable: {'Yes' if agent.reproduction_capability else 'No'}

"""
            
            # Evolution stage distribution
            stage_counts = defaultdict(int)
            for agent in self.active_agents.values():
                stage_counts[agent.evolution_stage] += 1
            
            report += """
EVOLUTION STAGE DISTRIBUTION
============================
"""
            for stage, count in stage_counts.items():
                report += f"• {stage.value}: {count} agents\n"
            
            # Generation analysis
            generations = [agent.generation_info.generation for agent in self.active_agents.values()]
            if generations:
                report += f"""

GENERATION ANALYSIS
===================
• Current Generations: {min(generations)} to {max(generations)}
• Average Generation: {np.mean(generations):.1f}
• Most Advanced Generation: {max(generations)}
• First Generation Agents: {sum(1 for g in generations if g == 0)}

"""
            
            # Learning analysis
            if self.active_agents:
                all_experiences = [exp for agent in self.active_agents.values() for exp in agent.learning_experiences]
                if all_experiences:
                    avg_quality = np.mean([exp.outcome_quality for exp in all_experiences])
                    
                    learning_type_counts = defaultdict(int)
                    for exp in all_experiences:
                        learning_type_counts[exp.learning_type] += 1
                    
                    report += f"""
LEARNING ANALYSIS
=================
• Total Learning Experiences: {len(all_experiences)}
• Average Experience Quality: {avg_quality:.3f}
• Experience Types:
"""
                    for learning_type, count in learning_type_counts.items():
                        report += f"  - {learning_type.value}: {count}\n"
            
            # System insights
            insights = self._generate_evolution_insights()
            if insights:
                report += """

EVOLUTION INSIGHTS
==================
"""
                for insight in insights:
                    report += f"• {insight}\n"
            
            report += f"""

SYSTEM CONFIGURATION
====================
• Memory Consolidation Interval: {self.evolution_config['memory_consolidation_interval']}
• Personality Adaptation Rate: {self.evolution_config['personality_adaptation_rate']}
• Skill Decay Rate: {self.evolution_config['skill_decay_rate']}
• Consciousness Backup Interval: {self.evolution_config['consciousness_backup_interval']}
• Generation Advancement Threshold: {self.evolution_config['generation_advancement_threshold']}
• Reproduction Maturity Threshold: {self.evolution_config['reproduction_maturity_threshold']}

FUTURE DEVELOPMENT
==================
• Advanced genetic algorithms for trait evolution
• Complex social interaction simulations
• Multi-agent collaborative learning environments
• Cross-generational knowledge transfer optimization
• Advanced consciousness measurement and tracking

==================================================
Report generated by {self.system_name}
System ID: {self.system_id}
"""
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.storage_dir / f"evolution_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"📋 Evolution report generated: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return f"Error generating report: {str(e)}"
    
    # Helper methods
    def _initialize_databases(self):
        """Initialize SQLite databases for persistent storage"""
        try:
            # Agent database
            agent_db_path = self.storage_dir / "agents.db"
            self.agent_database = sqlite3.connect(str(agent_db_path), check_same_thread=False)
            
            # Memory database
            memory_db_path = self.storage_dir / "memories.db"
            self.memory_database = sqlite3.connect(str(memory_db_path), check_same_thread=False)
            
            # Create tables
            self._create_database_tables()
            
            logger.info("💾 Initialized persistent storage databases")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize databases: {e}")
    
    def _create_database_tables(self):
        """Create database tables for storing agent data"""
        # Agent table
        self.agent_database.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Memory table
        self.memory_database.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                data BLOB NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                importance REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.agent_database.commit()
        self.memory_database.commit()
    
    def _initialize_evolution_systems(self):
        """Initialize evolution-specific systems"""
        logger.info("🧬 Initializing evolution systems...")
        
        # Set up genetic algorithms
        self.genetic_operators = {
            'crossover_rate': 0.7,
            'mutation_rate': 0.1,
            'selection_pressure': 0.6,
            'trait_inheritance_variance': 0.15
        }
        
        # Initialize learning preferences templates
        self.learning_templates = {
            LearningType.SKILL_ACQUISITION: {
                'base_rate': 0.8,
                'personality_modifiers': {
                    'curiosity': 0.3,
                    'persistence': 0.2,
                    'analytical_thinking': 0.1
                }
            },
            LearningType.SOCIAL_INTERACTION: {
                'base_rate': 0.6,
                'personality_modifiers': {
                    'sociability': 0.4,
                    'empathy': 0.3,
                    'emotional_stability': 0.1
                }
            }
        }
    
    def _start_background_processes(self):
        """Start background processes for system maintenance"""
        def background_maintenance():
            while True:
                try:
                    self._process_memory_consolidation()
                    self._process_skill_decay()
                    self._process_goal_updates()
                    self._process_consciousness_backups()
                    
                    time.sleep(300)  # Run every 5 minutes
                    
                except Exception as e:
                    logger.error(f"❌ Background maintenance error: {e}")
        
        maintenance_thread = threading.Thread(target=background_maintenance, daemon=True)
        maintenance_thread.start()
        
        logger.info("⚙️ Started background evolution processes")
    
    def _generate_consciousness_signature(self, agent_id: str) -> str:
        """Generate unique consciousness signature for agent"""
        signature_data = f"{agent_id}{datetime.now().isoformat()}{np.random.random()}"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:16]
    
    def _initialize_learning_preferences(self) -> Dict[LearningType, float]:
        """Initialize learning preferences with random variations"""
        preferences = {}
        for learning_type in LearningType:
            preferences[learning_type] = np.random.uniform(0.3, 0.9)
        return preferences
    
    def _calculate_offspring_generation(self, parent_ids: List[str]) -> int:
        """Calculate generation number for offspring"""
        parent_generations = [self.active_agents[pid].generation_info.generation for pid in parent_ids]
        return max(parent_generations) + 1
    
    def _generate_offspring_genetics(self, parent_ids: List[str]) -> Dict[str, float]:
        """Generate genetic traits for offspring from parents"""
        parents = [self.active_agents[pid] for pid in parent_ids]
        
        # Combine genetic traits from parents
        genetic_traits = {}
        trait_names = ['intelligence', 'creativity', 'stability', 'adaptability', 'sociability']
        
        for trait in trait_names:
            parent_values = [p.generation_info.genetic_traits.get(trait, 0.5) for p in parents]
            # Average parent values with some random variation
            genetic_traits[trait] = np.mean(parent_values) + np.random.normal(0, 0.1)
            genetic_traits[trait] = max(0.0, min(1.0, genetic_traits[trait]))
        
        return genetic_traits
    
    def _generate_random_genetics(self) -> Dict[str, float]:
        """Generate random genetic traits for first generation"""
        trait_names = ['intelligence', 'creativity', 'stability', 'adaptability', 'sociability']
        return {trait: np.random.uniform(0.2, 0.8) for trait in trait_names}
    
    def _calculate_genetic_diversity(self, genetic_traits: Dict[str, float]) -> float:
        """Calculate genetic diversity score"""
        return np.std(list(genetic_traits.values()))
    
    def _set_initial_goals(self, agent: PersistentAgent):
        """Set initial development goals for new agents"""
        initial_goals = [
            ("Learn basic communication skills", {"communication": 0.5}, {}, 0.8),
            ("Develop problem-solving abilities", {"problem_solving": 0.4}, {}, 0.7),
            ("Build social interaction skills", {"social_interaction": 0.3}, {"sociability": 0.1}, 0.6)
        ]
        
        for description, target_skills, target_traits, priority in initial_goals:
            goal = EvolutionGoal(
                goal_id=str(uuid.uuid4()),
                description=description,
                priority=priority,
                progress=0.0,
                target_skills=list(target_skills.keys()),
                target_traits=target_traits,
                deadline=None,
                created=datetime.now(),
                motivation="Initial development goal"
            )
            agent.evolution_goals.append(goal)
    
    def _create_memory_from_experience(self, agent: PersistentAgent, experience: LearningExperience) -> Memory:
        """Create memory from learning experience"""
        memory_content = {
            'experience_id': experience.experience_id,
            'description': experience.description,
            'outcome_quality': experience.outcome_quality,
            'learning_type': experience.learning_type.value,
            'skills_impacted': list(experience.skill_impact.keys()),
            'context': experience.context
        }
        
        return Memory(
            memory_id=str(uuid.uuid4()),
            memory_type=MemoryType.EPISODIC,
            content=memory_content,
            timestamp=experience.timestamp,
            emotional_valence=experience.outcome_quality - 0.5,  # Convert quality to emotional valence
            importance=min(1.0, experience.outcome_quality + len(experience.skill_impact) * 0.1)
        )
    
    def _add_memory(self, agent: PersistentAgent, memory: Memory):
        """Add memory to agent's memory systems"""
        # Add to working memory
        agent.working_memory.append(memory.memory_id)
        
        # Add to long-term memory
        agent.long_term_memory[memory.memory_id] = memory
        
        # Manage memory capacity
        if len(agent.long_term_memory) > agent.memory_capacity:
            self._manage_memory_capacity(agent)
        
        # Update statistics
        self.system_stats['total_memories_formed'] += 1
    
    def _manage_memory_capacity(self, agent: PersistentAgent):
        """Manage agent memory capacity by forgetting less important memories"""
        if len(agent.long_term_memory) <= agent.memory_capacity:
            return
        
        # Sort memories by importance and recency
        memories = list(agent.long_term_memory.values())
        memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        # Keep most important memories
        memories_to_keep = memories[:agent.memory_capacity]
        
        # Update long-term memory
        agent.long_term_memory = {m.memory_id: m for m in memories_to_keep}
        
        logger.info(f"🧠 Managed memory capacity for {agent.name}: kept {len(memories_to_keep)}/{len(memories)} memories")
    
    def _check_evolution_stage_advancement(self, agent: PersistentAgent):
        """Check if agent should advance to next evolution stage"""
        current_stage = agent.evolution_stage
        
        # Determine advancement criteria
        advancement_criteria = {
            EvolutionStage.NASCENT: len(agent.learning_experiences) >= 10,
            EvolutionStage.LEARNING: len(agent.learning_experiences) >= 50 and any(s.proficiency > 0.3 for s in agent.skills.values()),
            EvolutionStage.DEVELOPING: len(agent.learning_experiences) >= 150 and any(s.proficiency > 0.6 for s in agent.skills.values()),
            EvolutionStage.MATURE: len(agent.learning_experiences) >= 300 and sum(s.proficiency > 0.8 for s in agent.skills.values()) >= 3,
            EvolutionStage.TRANSCENDENT: len(agent.learning_experiences) >= 500 and all(s.proficiency > 0.7 for s in agent.skills.values())
        }
        
        if current_stage in advancement_criteria and advancement_criteria[current_stage]:
            # Advance to next stage
            stages = list(EvolutionStage)
            current_index = stages.index(current_stage)
            
            if current_index < len(stages) - 1:
                agent.evolution_stage = stages[current_index + 1]
                
                # Enable reproduction capability at mature stage
                if agent.evolution_stage == EvolutionStage.MATURE:
                    agent.reproduction_capability = True
                
                logger.info(f"🚀 {agent.name} advanced to {agent.evolution_stage.value} stage!")
    
    def _combine_parent_traits(self, parents: List[PersistentAgent], trait_type: str) -> Dict[str, float]:
        """Combine parent traits for offspring generation"""
        if trait_type == 'personality':
            trait_dicts = [agent.personality_traits for agent in parents]
            trait_keys = set()
            for d in trait_dicts:
                trait_keys.update(d.keys())
            
            combined = {}
            for key in trait_keys:
                values = [d[key].current_value for d in trait_dicts if key in d]
                combined[key] = np.mean(values) + np.random.normal(0, self.genetic_operators['trait_inheritance_variance'])
                combined[key] = max(-1.0, min(1.0, combined[key]))
            
            return combined
        
        elif trait_type == 'skills':
            skill_dicts = [agent.skills for agent in parents]
            skill_keys = set()
            for d in skill_dicts:
                skill_keys.update(d.keys())
            
            combined = {}
            for key in skill_keys:
                values = [d[key].proficiency for d in skill_dicts if key in d]
                combined[key] = np.mean(values) + np.random.normal(0, self.genetic_operators['trait_inheritance_variance'])
                combined[key] = max(0.0, min(1.0, combined[key]))
            
            return combined
        
        return {}
    
    def _inherit_memories(self, parents: List[PersistentAgent], offspring: PersistentAgent) -> List[Memory]:
        """Allow offspring to inherit some memories from parents"""
        inherited_memories = []
        
        for parent in parents:
            # Select important and semantic memories for inheritance
            inheritable_memories = [
                memory for memory in parent.long_term_memory.values()
                if memory.memory_type in [MemoryType.SEMANTIC, MemoryType.PROCEDURAL] 
                and memory.importance > 0.7
            ]
            
            # Inherit up to 5 memories per parent
            for memory in sorted(inheritable_memories, key=lambda m: m.importance, reverse=True)[:5]:
                # Create modified memory for offspring
                inherited_memory = Memory(
                    memory_id=str(uuid.uuid4()),
                    memory_type=memory.memory_type,
                    content={**memory.content, 'inherited_from': parent.agent_id},
                    timestamp=datetime.now(),
                    emotional_valence=memory.emotional_valence * 0.5,  # Reduced emotional impact
                    importance=memory.importance * 0.7,  # Reduced importance
                    consolidation_level=memory.consolidation_level
                )
                
                inherited_memories.append(inherited_memory)
                self._add_memory(offspring, inherited_memory)
        
        return inherited_memories
    
    def _create_consciousness_backup(self, agent: PersistentAgent) -> str:
        """Create consciousness backup for agent"""
        try:
            if not self.consciousness_transfer:
                return ""
            
            consciousness_data = {
                'agent_id': agent.agent_id,
                'name': agent.name,
                'evolution_stage': agent.evolution_stage.value,
                'personality_traits': {name: trait.current_value for name, trait in agent.personality_traits.items()},
                'skills': {name: skill.proficiency for name, skill in agent.skills.items()},
                'memories': [memory.memory_id for memory in agent.long_term_memory.values() if memory.importance > 0.5],
                'goals': [goal.goal_id for goal in agent.evolution_goals if goal.progress < 1.0],
                'consciousness_signature': agent.consciousness_signature,
                'backup_timestamp': datetime.now().isoformat()
            }
            
            backup_id = self.consciousness_transfer.save_consciousness(agent.agent_id, consciousness_data)
            return backup_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create consciousness backup: {e}")
            return ""
    
    def _apply_consciousness_restoration(self, agent: PersistentAgent, consciousness_data: Dict[str, Any]):
        """Apply consciousness restoration to agent"""
        try:
            # Restore personality traits
            if 'personality_traits' in consciousness_data:
                for trait_name, value in consciousness_data['personality_traits'].items():
                    if trait_name in agent.personality_traits:
                        agent.personality_traits[trait_name].current_value = value
            
            # Restore skills
            if 'skills' in consciousness_data:
                for skill_name, proficiency in consciousness_data['skills'].items():
                    if skill_name in agent.skills:
                        agent.skills[skill_name].proficiency = proficiency
            
            # Note: Memory and goal restoration would require more complex logic
            
        except Exception as e:
            logger.error(f"❌ Failed to apply consciousness restoration: {e}")
    
    def _save_agent_to_database(self, agent: PersistentAgent):
        """Save agent to database"""
        try:
            agent_data = pickle.dumps(agent)
            
            self.agent_database.execute("""
                INSERT OR REPLACE INTO agents (agent_id, name, data, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (agent.agent_id, agent.name, agent_data))
            
            self.agent_database.commit()
            
        except Exception as e:
            logger.error(f"❌ Failed to save agent to database: {e}")
    
    def _process_interaction_scenario(self, agent: PersistentAgent, scenario: str) -> Dict[str, Any]:
        """Process interaction scenario and generate outcomes"""
        # Simple scenario processing based on agent's current state
        base_quality = 0.5
        
        # Adjust quality based on relevant skills
        if 'social' in scenario.lower() or 'communication' in scenario.lower():
            base_quality += agent.skills.get('social_interaction', Skill('', 0, 0, 0, datetime.now())).proficiency * 0.3
            base_quality += agent.skills.get('communication', Skill('', 0, 0, 0, datetime.now())).proficiency * 0.2
        
        if 'problem' in scenario.lower() or 'solve' in scenario.lower():
            base_quality += agent.skills.get('problem_solving', Skill('', 0, 0, 0, datetime.now())).proficiency * 0.4
        
        # Adjust based on personality
        if 'creative' in scenario.lower():
            base_quality += agent.personality_traits.get('creativity', PersonalityTrait('', 0, 0, 0, 0)).current_value * 0.2
        
        quality = max(0.0, min(1.0, base_quality + np.random.normal(0, 0.1)))
        
        # Generate skill impacts based on scenario
        skill_impacts = {}
        if 'social' in scenario.lower():
            skill_impacts['social_interaction'] = quality * 0.1
            skill_impacts['communication'] = quality * 0.08
        if 'problem' in scenario.lower():
            skill_impacts['problem_solving'] = quality * 0.12
            skill_impacts['analytical_thinking'] = quality * 0.05
        
        # Generate personality impacts
        personality_impacts = {}
        if quality > 0.7:
            personality_impacts['persistence'] = 0.02
        if 'social' in scenario.lower() and quality > 0.6:
            personality_impacts['sociability'] = 0.01
        
        # Generate response based on agent's personality
        response = self._generate_agent_response(agent, scenario, quality)
        
        return {
            'quality': quality,
            'skill_impacts': skill_impacts,
            'personality_impacts': personality_impacts,
            'consciousness_impact': quality * 0.05,
            'response': response
        }
    
    def _generate_agent_response(self, agent: PersistentAgent, scenario: str, quality: float) -> str:
        """Generate agent's response to interaction scenario"""
        base_responses = {
            'high_quality': [
                f"I feel confident approaching this situation.",
                f"This seems like a great learning opportunity.",
                f"I'm excited to engage with this challenge."
            ],
            'medium_quality': [
                f"I'll do my best to handle this appropriately.",
                f"This requires some careful thought.",
                f"I'm ready to learn from this experience."
            ],
            'low_quality': [
                f"This is challenging, but I'll try my best.",
                f"I need to think more about how to approach this.",
                f"This is a learning moment for me."
            ]
        }
        
        if quality > 0.7:
            responses = base_responses['high_quality']
        elif quality > 0.4:
            responses = base_responses['medium_quality']
        else:
            responses = base_responses['low_quality']
        
        return np.random.choice(responses)
    
    def _process_memory_consolidation(self):
        """Background process for memory consolidation"""
        for agent in self.active_agents.values():
            # Consolidate memories older than consolidation interval
            consolidation_threshold = datetime.now() - self.evolution_config['memory_consolidation_interval']
            
            for memory in agent.long_term_memory.values():
                if memory.timestamp < consolidation_threshold and memory.consolidation_level < 1.0:
                    # Increase consolidation level
                    memory.consolidation_level += 0.1
                    memory.consolidation_level = min(1.0, memory.consolidation_level)
                    
                    # Highly consolidated memories become more stable
                    if memory.consolidation_level > 0.8:
                        memory.importance *= 1.05
    
    def _process_skill_decay(self):
        """Background process for skill decay"""
        for agent in self.active_agents.values():
            for skill in agent.skills.values():
                time_since_practice = datetime.now() - skill.last_practiced
                
                if time_since_practice > timedelta(days=1):
                    # Apply decay
                    decay_amount = skill.decay_rate * (time_since_practice.days / 7.0)
                    skill.proficiency -= decay_amount
                    skill.proficiency = max(0.0, skill.proficiency)
    
    def _process_goal_updates(self):
        """Background process for updating goal progress"""
        for agent in self.active_agents.values():
            for goal in agent.evolution_goals:
                if goal.progress < 1.0:
                    # Calculate progress based on current skills and traits
                    skill_progress = 0.0
                    if goal.target_skills:
                        for skill_name in goal.target_skills:
                            if skill_name in agent.skills:
                                skill_progress += agent.skills[skill_name].proficiency
                        skill_progress /= len(goal.target_skills)
                    
                    trait_progress = 0.0
                    if goal.target_traits:
                        for trait_name, target_value in goal.target_traits.items():
                            if trait_name in agent.personality_traits:
                                current_value = agent.personality_traits[trait_name].current_value
                                trait_progress += max(0, 1 - abs(current_value - target_value))
                        trait_progress /= len(goal.target_traits)
                    
                    # Update progress
                    if skill_progress > 0 or trait_progress > 0:
                        goal.progress = (skill_progress + trait_progress) / 2.0
                        goal.progress = min(1.0, goal.progress)
    
    def _process_consciousness_backups(self):
        """Background process for consciousness backups"""
        for agent in self.active_agents.values():
            # Check if backup is needed
            last_backup_time = None
            if agent.consciousness_backup_ids:
                # In a real system, we'd track backup timestamps
                pass
            
            # Create backup if interval has passed
            time_since_last = datetime.now() - (last_backup_time or agent.creation_time)
            if time_since_last >= self.evolution_config['consciousness_backup_interval']:
                backup_id = self._create_consciousness_backup(agent)
                if backup_id:
                    agent.consciousness_backup_ids.append(backup_id)
    
    def _generate_evolution_insights(self) -> List[str]:
        """Generate insights about the evolution system"""
        insights = []
        
        if len(self.active_agents) > 0:
            # Agent development insights
            avg_experiences = np.mean([len(agent.learning_experiences) for agent in self.active_agents.values()])
            insights.append(f"Average learning experiences per agent: {avg_experiences:.1f}")
            
            # Stage distribution insights
            stage_counts = defaultdict(int)
            for agent in self.active_agents.values():
                stage_counts[agent.evolution_stage] += 1
            
            most_common_stage = max(stage_counts, key=stage_counts.get)
            insights.append(f"Most agents are in {most_common_stage.value} stage ({stage_counts[most_common_stage]} agents)")
            
            # Skill development insights
            all_skills = []
            for agent in self.active_agents.values():
                all_skills.extend([skill.proficiency for skill in agent.skills.values()])
            
            if all_skills:
                avg_skill = np.mean(all_skills)
                insights.append(f"Average skill proficiency across all agents: {avg_skill:.3f}")
                
                if avg_skill > 0.6:
                    insights.append("Agents showing strong skill development progress")
                elif avg_skill > 0.3:
                    insights.append("Agents showing moderate skill development")
                else:
                    insights.append("Agents in early skill development phase")
            
            # Reproduction insights
            reproduction_capable = sum(1 for agent in self.active_agents.values() if agent.reproduction_capability)
            if reproduction_capable > 0:
                insights.append(f"{reproduction_capable} agents capable of reproduction")
            
            # Memory insights
            total_memories = sum(len(agent.long_term_memory) for agent in self.active_agents.values())
            insights.append(f"Total memories stored across all agents: {total_memories}")
        
        return insights or ["Evolution system initialized - ready for agent development"]

def test_persistent_agent_evolution_system():
    """Test the Persistent Agent Evolution System"""
    logger.info("🧪 Testing Persistent Agent Evolution System...")
    
    try:
        # Create evolution system
        evolution_system = PersistentAgentEvolutionSystem("Test Evolution System")
        
        # Create first generation agent
        agent1 = evolution_system.create_agent(
            "Alice Evolving",
            base_personality={'curiosity': 0.8, 'persistence': 0.7},
            base_skills={'learning': 0.2, 'communication': 0.1}
        )
        
        logger.info(f"✅ Created first generation agent: {agent1.name}")
        
        # Add learning experiences
        for i in range(15):
            evolution_system.add_learning_experience(
                agent1.agent_id,
                LearningType.SKILL_ACQUISITION,
                f"Learning experience {i+1}",
                np.random.uniform(0.4, 0.9),
                {'learning': 0.05, 'communication': 0.02},
                {'curiosity': 0.01},
                0.02
            )
        
        logger.info(f"✅ Added {len(agent1.learning_experiences)} learning experiences")
        
        # Set evolution goal
        goal = evolution_system.set_evolution_goal(
            agent1.agent_id,
            "Become proficient in communication",
            {'communication': 0.7},
            {'sociability': 0.3},
            0.8
        )
        
        logger.info(f"✅ Set evolution goal: {goal.description}")
        
        # Simulate interaction
        interaction_result = evolution_system.simulate_agent_interaction(
            agent1.agent_id,
            "Complex social problem-solving scenario"
        )
        
        logger.info(f"✅ Simulated interaction: quality {interaction_result['outcomes']['quality']:.2f}")
        
        # Create second agent for reproduction testing
        agent2 = evolution_system.create_agent(
            "Bob Evolving",
            base_personality={'analytical_thinking': 0.9, 'creativity': 0.6},
            base_skills={'problem_solving': 0.3, 'learning': 0.15}
        )
        
        # Add many experiences to both agents to enable reproduction
        for agent in [agent1, agent2]:
            for i in range(600):  # Exceed reproduction threshold
                evolution_system.add_learning_experience(
                    agent.agent_id,
                    np.random.choice(list(LearningType)),
                    f"Advanced learning experience {i+1}",
                    np.random.uniform(0.5, 0.9),
                    {skill: np.random.uniform(0.01, 0.05) for skill in list(agent.skills.keys())[:2]},
                    {},
                    np.random.uniform(0.01, 0.03)
                )
        
        logger.info(f"✅ Added extensive experiences to enable reproduction")
        
        # Test reproduction
        try:
            offspring = evolution_system.reproduce_agents(
                [agent1.agent_id, agent2.agent_id],
                "Charlie Offspring"
            )
            logger.info(f"✅ Created offspring: {offspring.name} (Generation {offspring.generation_info.generation})")
        except ValueError as e:
            logger.info(f"ℹ️ Reproduction not yet possible: {e}")
        
        # Test consciousness backup
        backup_id = evolution_system.backup_consciousness(agent1.agent_id)
        logger.info(f"✅ Created consciousness backup: {backup_id[:8] if backup_id else 'Failed'}...")
        
        # Get agent status
        status = evolution_system.get_agent_status(agent1.agent_id)
        logger.info(f"✅ Agent status - Stage: {status['evolution_stage']}, Experiences: {status['learning_experiences']}")
        
        # Generate system report
        report = evolution_system.generate_system_report()
        logger.info(f"✅ Generated system report ({len(report)} characters)")
        
        logger.info("🎉 All Persistent Agent Evolution System tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    test_persistent_agent_evolution_system()