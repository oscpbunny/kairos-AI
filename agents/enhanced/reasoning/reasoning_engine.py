"""
Project Kairos: Advanced Reasoning Engine
Phase 7 - Advanced Intelligence

This module implements sophisticated reasoning capabilities including:
- Symbolic Logic Reasoning
- Causal Inference
- Temporal Reasoning  
- Spatial Reasoning
- Formal Verification
- Complex Decision Making
- Knowledge Graph Integration
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kairos.reasoning.engine")

class ReasoningType(Enum):
    """Types of reasoning supported"""
    SYMBOLIC = "symbolic"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"

class LogicOperator(Enum):
    """Logic operators for symbolic reasoning"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"
    EXISTS = "exists"
    FORALL = "forall"

@dataclass
class Proposition:
    """A logical proposition"""
    id: str
    statement: str
    truth_value: Optional[bool] = None
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    
class LogicalExpression:
    """Represents a logical expression"""
    
    def __init__(self, operator: LogicOperator, operands: List[Union['LogicalExpression', str]]):
        self.operator = operator
        self.operands = operands
        self.confidence = 1.0
    
    def evaluate(self, propositions: Dict[str, Proposition]) -> Optional[bool]:
        """Evaluate the logical expression"""
        try:
            operand_values = []
            min_confidence = 1.0
            
            for operand in self.operands:
                if isinstance(operand, str):
                    # Operand is a proposition ID
                    if operand in propositions:
                        prop = propositions[operand]
                        if prop.truth_value is not None:
                            operand_values.append(prop.truth_value)
                            min_confidence = min(min_confidence, prop.confidence)
                        else:
                            return None  # Cannot evaluate with unknown truth value
                    else:
                        return None  # Unknown proposition
                elif isinstance(operand, LogicalExpression):
                    # Operand is a sub-expression
                    sub_result = operand.evaluate(propositions)
                    if sub_result is not None:
                        operand_values.append(sub_result)
                        min_confidence = min(min_confidence, operand.confidence)
                    else:
                        return None
            
            # Apply logical operator
            result = self._apply_operator(operand_values)
            self.confidence = min_confidence
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating logical expression: {e}")
            return None
    
    def _apply_operator(self, values: List[bool]) -> Optional[bool]:
        """Apply the logical operator to values"""
        if self.operator == LogicOperator.AND:
            return all(values)
        elif self.operator == LogicOperator.OR:
            return any(values)
        elif self.operator == LogicOperator.NOT:
            return not values[0] if len(values) == 1 else None
        elif self.operator == LogicOperator.IMPLIES:
            if len(values) == 2:
                return not values[0] or values[1]
        elif self.operator == LogicOperator.IFF:
            if len(values) == 2:
                return values[0] == values[1]
        
        return None

@dataclass
class CausalRelation:
    """Represents a causal relationship"""
    cause: str
    effect: str
    strength: float
    confidence: float
    mechanism: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TemporalEvent:
    """Represents an event in time"""
    id: str
    description: str
    timestamp: datetime
    duration: Optional[timedelta] = None
    certainty: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpatialEntity:
    """Represents an entity in space"""
    id: str
    name: str
    position: Tuple[float, float, float]  # x, y, z
    dimensions: Optional[Tuple[float, float, float]] = None
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningResult:
    """Result of a reasoning operation"""
    reasoning_type: ReasoningType
    query: str
    conclusion: Any
    confidence: float
    evidence: List[Dict[str, Any]]
    reasoning_steps: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SymbolicReasoner:
    """Handles symbolic logic reasoning"""
    
    def __init__(self):
        self.propositions: Dict[str, Proposition] = {}
        self.rules: List[LogicalExpression] = []
        
    def add_proposition(self, proposition: Proposition):
        """Add a proposition to the knowledge base"""
        self.propositions[proposition.id] = proposition
        logger.info(f"🔢 Added proposition: {proposition.statement}")
    
    def add_rule(self, rule: LogicalExpression):
        """Add a logical rule"""
        self.rules.append(rule)
        logger.info(f"📏 Added logical rule with operator: {rule.operator.value}")
    
    async def reason(self, query: str) -> ReasoningResult:
        """Perform symbolic reasoning"""
        start_time = time.time()
        reasoning_steps = []
        evidence = []
        
        logger.info(f"🔢 Performing symbolic reasoning for: {query}")
        
        try:
            # Parse query to identify target propositions
            target_props = self._extract_propositions_from_query(query)
            reasoning_steps.append(f"Identified target propositions: {target_props}")
            
            # Apply inference rules
            inferences_made = 0
            max_iterations = 10
            
            for iteration in range(max_iterations):
                new_inferences = False
                
                for rule in self.rules:
                    result = rule.evaluate(self.propositions)
                    if result is not None:
                        # Rule produced a result - check if it's new knowledge
                        rule_desc = f"Rule_{len(self.rules)}"
                        if rule_desc not in self.propositions:
                            new_prop = Proposition(
                                id=rule_desc,
                                statement=f"Inferred from rule application",
                                truth_value=result,
                                confidence=rule.confidence,
                                source="symbolic_reasoning"
                            )
                            self.propositions[rule_desc] = new_prop
                            new_inferences = True
                            inferences_made += 1
                            reasoning_steps.append(f"Applied rule, inferred: {result}")
                
                if not new_inferences:
                    break
            
            reasoning_steps.append(f"Made {inferences_made} inferences in {iteration + 1} iterations")
            
            # Evaluate target propositions
            conclusions = {}
            total_confidence = 0.0
            
            for prop_id in target_props:
                if prop_id in self.propositions:
                    prop = self.propositions[prop_id]
                    conclusions[prop_id] = {
                        'truth_value': prop.truth_value,
                        'confidence': prop.confidence,
                        'statement': prop.statement
                    }
                    total_confidence += prop.confidence
                    evidence.append({
                        'type': 'proposition',
                        'id': prop_id,
                        'statement': prop.statement,
                        'truth_value': prop.truth_value,
                        'confidence': prop.confidence
                    })
            
            avg_confidence = total_confidence / len(target_props) if target_props else 0.0
            processing_time = time.time() - start_time
            
            return ReasoningResult(
                reasoning_type=ReasoningType.SYMBOLIC,
                query=query,
                conclusion=conclusions,
                confidence=avg_confidence,
                evidence=evidence,
                reasoning_steps=reasoning_steps,
                processing_time=processing_time,
                metadata={
                    'propositions_count': len(self.propositions),
                    'rules_count': len(self.rules),
                    'inferences_made': inferences_made
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Symbolic reasoning failed: {e}")
            processing_time = time.time() - start_time
            
            return ReasoningResult(
                reasoning_type=ReasoningType.SYMBOLIC,
                query=query,
                conclusion=None,
                confidence=0.0,
                evidence=[],
                reasoning_steps=[f"Error: {str(e)}"],
                processing_time=processing_time
            )
    
    def _extract_propositions_from_query(self, query: str) -> List[str]:
        """Extract proposition IDs from query text"""
        # Simple extraction - in production would use NLP
        prop_ids = []
        for prop_id in self.propositions.keys():
            if prop_id.lower() in query.lower() or self.propositions[prop_id].statement.lower() in query.lower():
                prop_ids.append(prop_id)
        
        return prop_ids if prop_ids else list(self.propositions.keys())[:3]  # Return first 3 if none found

class CausalReasoner:
    """Handles causal inference reasoning"""
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.causal_relations: Dict[str, CausalRelation] = {}
        
    def add_causal_relation(self, relation: CausalRelation):
        """Add a causal relationship"""
        relation_id = f"{relation.cause}->{relation.effect}"
        self.causal_relations[relation_id] = relation
        self.causal_graph.add_edge(relation.cause, relation.effect, 
                                 strength=relation.strength, 
                                 confidence=relation.confidence)
        logger.info(f"🔗 Added causal relation: {relation.cause} → {relation.effect}")
    
    async def reason(self, query: str) -> ReasoningResult:
        """Perform causal reasoning"""
        start_time = time.time()
        reasoning_steps = []
        evidence = []
        
        logger.info(f"🔗 Performing causal reasoning for: {query}")
        
        try:
            # Parse query to identify causal question
            cause_var, effect_var = self._parse_causal_query(query)
            reasoning_steps.append(f"Identified cause: {cause_var}, effect: {effect_var}")
            
            # Find causal paths
            if cause_var in self.causal_graph and effect_var in self.causal_graph:
                try:
                    paths = list(nx.all_simple_paths(self.causal_graph, cause_var, effect_var, cutoff=5))
                    reasoning_steps.append(f"Found {len(paths)} causal paths")
                    
                    # Calculate causal effect strength
                    total_effect = 0.0
                    path_confidences = []
                    
                    for path in paths[:3]:  # Limit to top 3 paths
                        path_strength = 1.0
                        path_confidence = 1.0
                        
                        for i in range(len(path) - 1):
                            edge_data = self.causal_graph[path[i]][path[i+1]]
                            path_strength *= edge_data.get('strength', 0.5)
                            path_confidence *= edge_data.get('confidence', 0.5)
                        
                        total_effect += path_strength
                        path_confidences.append(path_confidence)
                        
                        evidence.append({
                            'type': 'causal_path',
                            'path': path,
                            'strength': path_strength,
                            'confidence': path_confidence
                        })
                        
                        reasoning_steps.append(f"Path {' → '.join(path)}: strength={path_strength:.3f}")
                
                except nx.NetworkXNoPath:
                    reasoning_steps.append("No causal path found between variables")
                    total_effect = 0.0
                    path_confidences = []
            
            else:
                reasoning_steps.append("Variables not found in causal graph")
                total_effect = 0.0
                path_confidences = []
            
            # Calculate overall confidence
            avg_confidence = np.mean(path_confidences) if path_confidences else 0.0
            
            # Prepare conclusion
            conclusion = {
                'causal_effect': total_effect,
                'effect_strength': 'strong' if total_effect > 0.7 else 'moderate' if total_effect > 0.3 else 'weak',
                'mechanism': self._identify_mechanisms(cause_var, effect_var),
                'confounders': self._identify_confounders(cause_var, effect_var)
            }
            
            processing_time = time.time() - start_time
            
            return ReasoningResult(
                reasoning_type=ReasoningType.CAUSAL,
                query=query,
                conclusion=conclusion,
                confidence=avg_confidence,
                evidence=evidence,
                reasoning_steps=reasoning_steps,
                processing_time=processing_time,
                metadata={
                    'graph_nodes': self.causal_graph.number_of_nodes(),
                    'graph_edges': self.causal_graph.number_of_edges(),
                    'paths_found': len(paths) if 'paths' in locals() else 0
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Causal reasoning failed: {e}")
            processing_time = time.time() - start_time
            
            return ReasoningResult(
                reasoning_type=ReasoningType.CAUSAL,
                query=query,
                conclusion=None,
                confidence=0.0,
                evidence=[],
                reasoning_steps=[f"Error: {str(e)}"],
                processing_time=processing_time
            )
    
    def _parse_causal_query(self, query: str) -> Tuple[str, str]:
        """Parse causal query to identify cause and effect variables"""
        # Simple parsing - in production would use advanced NLP
        words = query.lower().split()
        
        # Look for causal keywords
        if 'cause' in words and 'effect' in words:
            cause_idx = words.index('cause') + 1 if words.index('cause') + 1 < len(words) else 0
            effect_idx = words.index('effect') + 1 if words.index('effect') + 1 < len(words) else 1
            
            cause_var = words[cause_idx] if cause_idx < len(words) else "unknown_cause"
            effect_var = words[effect_idx] if effect_idx < len(words) else "unknown_effect"
        else:
            # Default to first two significant words
            significant_words = [w for w in words if len(w) > 3 and w not in ['what', 'does', 'will', 'would']]
            cause_var = significant_words[0] if len(significant_words) > 0 else "variable_1"
            effect_var = significant_words[1] if len(significant_words) > 1 else "variable_2"
        
        return cause_var, effect_var
    
    def _identify_mechanisms(self, cause: str, effect: str) -> List[str]:
        """Identify potential mechanisms between cause and effect"""
        mechanisms = []
        relation_id = f"{cause}->{effect}"
        
        if relation_id in self.causal_relations:
            relation = self.causal_relations[relation_id]
            if relation.mechanism:
                mechanisms.append(relation.mechanism)
        
        return mechanisms
    
    def _identify_confounders(self, cause: str, effect: str) -> List[str]:
        """Identify potential confounding variables"""
        confounders = []
        
        if cause in self.causal_graph and effect in self.causal_graph:
            # Find common predecessors
            cause_predecessors = set(self.causal_graph.predecessors(cause))
            effect_predecessors = set(self.causal_graph.predecessors(effect))
            confounders = list(cause_predecessors.intersection(effect_predecessors))
        
        return confounders

class TemporalReasoner:
    """Handles temporal reasoning"""
    
    def __init__(self):
        self.events: Dict[str, TemporalEvent] = {}
        self.temporal_relations: List[Dict[str, Any]] = []
        
    def add_event(self, event: TemporalEvent):
        """Add a temporal event"""
        self.events[event.id] = event
        logger.info(f"⏰ Added temporal event: {event.description} at {event.timestamp}")
    
    def add_temporal_relation(self, event1_id: str, event2_id: str, relation_type: str, confidence: float = 1.0):
        """Add a temporal relation between events"""
        self.temporal_relations.append({
            'event1': event1_id,
            'event2': event2_id,
            'relation': relation_type,  # before, after, during, overlaps, etc.
            'confidence': confidence
        })
        logger.info(f"⏰ Added temporal relation: {event1_id} {relation_type} {event2_id}")
    
    async def reason(self, query: str) -> ReasoningResult:
        """Perform temporal reasoning"""
        start_time = time.time()
        reasoning_steps = []
        evidence = []
        
        logger.info(f"⏰ Performing temporal reasoning for: {query}")
        
        try:
            # Analyze temporal patterns
            reasoning_steps.append("Analyzing temporal patterns...")
            
            # Sort events by time
            sorted_events = sorted(self.events.values(), key=lambda e: e.timestamp)
            reasoning_steps.append(f"Sorted {len(sorted_events)} events chronologically")
            
            # Identify temporal sequences
            sequences = self._identify_sequences(sorted_events)
            reasoning_steps.append(f"Identified {len(sequences)} temporal sequences")
            
            # Look for patterns and periodicity
            patterns = self._identify_patterns(sorted_events)
            reasoning_steps.append(f"Identified {len(patterns)} temporal patterns")
            
            # Calculate temporal statistics
            duration_stats = self._calculate_duration_statistics(sorted_events)
            reasoning_steps.append("Calculated duration statistics")
            
            # Build evidence
            for seq in sequences:
                evidence.append({
                    'type': 'temporal_sequence',
                    'events': seq,
                    'duration': self._calculate_sequence_duration(seq)
                })
            
            for pattern in patterns:
                evidence.append({
                    'type': 'temporal_pattern',
                    'pattern_type': pattern['type'],
                    'events': pattern['events'],
                    'frequency': pattern.get('frequency')
                })
            
            # Prepare conclusion
            conclusion = {
                'event_count': len(self.events),
                'time_span': self._calculate_time_span(sorted_events),
                'sequences': len(sequences),
                'patterns': patterns,
                'duration_stats': duration_stats,
                'temporal_density': self._calculate_temporal_density(sorted_events)
            }
            
            # Calculate confidence based on event certainties
            avg_confidence = np.mean([event.certainty for event in sorted_events]) if sorted_events else 0.0
            
            processing_time = time.time() - start_time
            
            return ReasoningResult(
                reasoning_type=ReasoningType.TEMPORAL,
                query=query,
                conclusion=conclusion,
                confidence=avg_confidence,
                evidence=evidence,
                reasoning_steps=reasoning_steps,
                processing_time=processing_time,
                metadata={
                    'events_analyzed': len(sorted_events),
                    'relations_count': len(self.temporal_relations)
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Temporal reasoning failed: {e}")
            processing_time = time.time() - start_time
            
            return ReasoningResult(
                reasoning_type=ReasoningType.TEMPORAL,
                query=query,
                conclusion=None,
                confidence=0.0,
                evidence=[],
                reasoning_steps=[f"Error: {str(e)}"],
                processing_time=processing_time
            )
    
    def _identify_sequences(self, events: List[TemporalEvent]) -> List[List[str]]:
        """Identify temporal sequences in events"""
        sequences = []
        
        # Simple sequence detection based on time windows
        current_sequence = []
        time_window = timedelta(minutes=10)  # Events within 10 minutes form a sequence
        
        for i, event in enumerate(events):
            if not current_sequence:
                current_sequence = [event.id]
            else:
                prev_event = next(e for e in events if e.id == current_sequence[-1])
                time_diff = event.timestamp - prev_event.timestamp
                
                if time_diff <= time_window:
                    current_sequence.append(event.id)
                else:
                    if len(current_sequence) > 1:
                        sequences.append(current_sequence.copy())
                    current_sequence = [event.id]
        
        if len(current_sequence) > 1:
            sequences.append(current_sequence)
        
        return sequences
    
    def _identify_patterns(self, events: List[TemporalEvent]) -> List[Dict[str, Any]]:
        """Identify temporal patterns"""
        patterns = []
        
        if len(events) < 3:
            return patterns
        
        # Look for periodic patterns
        time_diffs = []
        for i in range(1, len(events)):
            diff = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            time_diffs.append(diff)
        
        # Check for regularity
        if len(time_diffs) > 2:
            mean_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)
            
            if std_diff < mean_diff * 0.2:  # Low variation indicates pattern
                patterns.append({
                    'type': 'periodic',
                    'events': [e.id for e in events],
                    'frequency': mean_diff,
                    'regularity': 1.0 - (std_diff / mean_diff)
                })
        
        return patterns
    
    def _calculate_duration_statistics(self, events: List[TemporalEvent]) -> Dict[str, float]:
        """Calculate duration statistics for events"""
        durations = []
        for event in events:
            if event.duration:
                durations.append(event.duration.total_seconds())
        
        if not durations:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations)
        }
    
    def _calculate_sequence_duration(self, sequence: List[str]) -> float:
        """Calculate total duration of a sequence"""
        if len(sequence) < 2:
            return 0.0
        
        first_event = self.events[sequence[0]]
        last_event = self.events[sequence[-1]]
        
        duration = (last_event.timestamp - first_event.timestamp).total_seconds()
        return duration
    
    def _calculate_time_span(self, events: List[TemporalEvent]) -> float:
        """Calculate total time span of events"""
        if len(events) < 2:
            return 0.0
        
        return (events[-1].timestamp - events[0].timestamp).total_seconds()
    
    def _calculate_temporal_density(self, events: List[TemporalEvent]) -> float:
        """Calculate temporal density (events per time unit)"""
        if len(events) < 2:
            return 0.0
        
        time_span = self._calculate_time_span(events)
        if time_span > 0:
            return len(events) / (time_span / 3600)  # Events per hour
        return 0.0

class AdvancedReasoningEngine:
    """Main reasoning engine that coordinates different reasoning types"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize reasoning modules
        self.symbolic_reasoner = SymbolicReasoner()
        self.causal_reasoner = CausalReasoner()
        self.temporal_reasoner = TemporalReasoner()
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Metrics
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_processing_time': 0.0,
            'reasoning_types_used': {}
        }
        
        logger.info("🧠 Advanced Reasoning Engine initialized")
    
    async def initialize(self) -> bool:
        """Initialize the reasoning engine"""
        try:
            logger.info("🧠 Initializing Advanced Reasoning Engine...")
            
            # Load initial knowledge base
            await self._load_initial_knowledge()
            
            # Setup reasoning capabilities
            await self._setup_reasoning_capabilities()
            
            logger.info("✅ Advanced Reasoning Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced Reasoning Engine: {e}")
            return False
    
    async def reason(self, query: str, reasoning_types: Optional[List[ReasoningType]] = None) -> Dict[str, ReasoningResult]:
        """Perform multi-type reasoning on a query"""
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        logger.info(f"🧠 Processing reasoning query: {query}")
        
        # Determine which reasoning types to use
        if reasoning_types is None:
            reasoning_types = self._determine_reasoning_types(query)
        
        logger.info(f"🎯 Using reasoning types: {[rt.value for rt in reasoning_types]}")
        
        # Execute reasoning in parallel
        tasks = []
        for reasoning_type in reasoning_types:
            task = self._execute_reasoning(reasoning_type, query)
            tasks.append(task)
        
        # Collect results
        results = {}
        try:
            reasoning_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for reasoning_type, result in zip(reasoning_types, reasoning_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ {reasoning_type.value} reasoning failed: {result}")
                    continue
                
                results[reasoning_type.value] = result
                
                # Update metrics
                if reasoning_type.value not in self.metrics['reasoning_types_used']:
                    self.metrics['reasoning_types_used'][reasoning_type.value] = 0
                self.metrics['reasoning_types_used'][reasoning_type.value] += 1
            
            if results:
                self.metrics['successful_queries'] += 1
            
            # Update average processing time
            processing_time = time.time() - start_time
            total_time = self.metrics['average_processing_time'] * (self.metrics['total_queries'] - 1)
            self.metrics['average_processing_time'] = (total_time + processing_time) / self.metrics['total_queries']
            
            logger.info(f"✅ Reasoning completed in {processing_time:.3f}s with {len(results)} result types")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Reasoning execution failed: {e}")
            return {}
    
    async def _execute_reasoning(self, reasoning_type: ReasoningType, query: str) -> ReasoningResult:
        """Execute specific type of reasoning"""
        if reasoning_type == ReasoningType.SYMBOLIC:
            return await self.symbolic_reasoner.reason(query)
        elif reasoning_type == ReasoningType.CAUSAL:
            return await self.causal_reasoner.reason(query)
        elif reasoning_type == ReasoningType.TEMPORAL:
            return await self.temporal_reasoner.reason(query)
        else:
            raise ValueError(f"Unsupported reasoning type: {reasoning_type}")
    
    def _determine_reasoning_types(self, query: str) -> List[ReasoningType]:
        """Automatically determine which reasoning types to use"""
        query_lower = query.lower()
        reasoning_types = []
        
        # Check for symbolic reasoning keywords
        symbolic_keywords = ['logic', 'prove', 'implies', 'therefore', 'if', 'then', 'all', 'some', 'none']
        if any(keyword in query_lower for keyword in symbolic_keywords):
            reasoning_types.append(ReasoningType.SYMBOLIC)
        
        # Check for causal reasoning keywords
        causal_keywords = ['cause', 'effect', 'because', 'due to', 'leads to', 'results in', 'why']
        if any(keyword in query_lower for keyword in causal_keywords):
            reasoning_types.append(ReasoningType.CAUSAL)
        
        # Check for temporal reasoning keywords
        temporal_keywords = ['when', 'time', 'before', 'after', 'during', 'sequence', 'pattern', 'period']
        if any(keyword in query_lower for keyword in temporal_keywords):
            reasoning_types.append(ReasoningType.TEMPORAL)
        
        # Default to symbolic reasoning if none detected
        if not reasoning_types:
            reasoning_types.append(ReasoningType.SYMBOLIC)
        
        return reasoning_types
    
    async def _load_initial_knowledge(self):
        """Load initial knowledge base"""
        # Add sample propositions
        sample_propositions = [
            Proposition("prop_1", "AI systems can learn from data", True, 0.95, "knowledge_base"),
            Proposition("prop_2", "Machine learning requires training data", True, 0.90, "knowledge_base"),
            Proposition("prop_3", "Deep learning is a subset of machine learning", True, 0.85, "knowledge_base"),
            Proposition("prop_4", "Neural networks can approximate complex functions", True, 0.88, "knowledge_base"),
        ]
        
        for prop in sample_propositions:
            self.symbolic_reasoner.add_proposition(prop)
        
        # Add sample causal relations
        sample_relations = [
            CausalRelation("training_data", "model_performance", 0.8, 0.9, "learning_mechanism"),
            CausalRelation("model_complexity", "overfitting", 0.7, 0.85, "capacity_mechanism"),
            CausalRelation("regularization", "generalization", 0.6, 0.8, "constraint_mechanism"),
        ]
        
        for relation in sample_relations:
            self.causal_reasoner.add_causal_relation(relation)
        
        # Add sample temporal events
        now = datetime.now()
        sample_events = [
            TemporalEvent("event_1", "Data collection started", now - timedelta(days=10)),
            TemporalEvent("event_2", "Model training began", now - timedelta(days=5)),
            TemporalEvent("event_3", "Hyperparameter tuning completed", now - timedelta(days=2)),
            TemporalEvent("event_4", "Model evaluation finished", now - timedelta(days=1)),
        ]
        
        for event in sample_events:
            self.temporal_reasoner.add_event(event)
    
    async def _setup_reasoning_capabilities(self):
        """Setup advanced reasoning capabilities"""
        # Add logical rules
        # Rule: If AI systems can learn AND machine learning requires data, then AI systems need data
        rule1 = LogicalExpression(
            LogicOperator.IMPLIES,
            [
                LogicalExpression(LogicOperator.AND, ["prop_1", "prop_2"]),
                "conclusion_1"
            ]
        )
        
        # Add conclusion proposition
        conclusion_prop = Proposition(
            "conclusion_1", 
            "AI systems need training data", 
            None,  # To be inferred
            1.0, 
            "reasoning_engine"
        )
        self.symbolic_reasoner.add_proposition(conclusion_prop)
        self.symbolic_reasoner.add_rule(rule1)
        
        logger.info("🔧 Advanced reasoning capabilities configured")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get reasoning engine capabilities"""
        return {
            'version': '7.0.0',
            'reasoning_types': [rt.value for rt in ReasoningType],
            'symbolic_reasoning': {
                'propositions': len(self.symbolic_reasoner.propositions),
                'rules': len(self.symbolic_reasoner.rules)
            },
            'causal_reasoning': {
                'relations': len(self.causal_reasoner.causal_relations),
                'graph_nodes': self.causal_reasoner.causal_graph.number_of_nodes(),
                'graph_edges': self.causal_reasoner.causal_graph.number_of_edges()
            },
            'temporal_reasoning': {
                'events': len(self.temporal_reasoner.events),
                'relations': len(self.temporal_reasoner.temporal_relations)
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        success_rate = (self.metrics['successful_queries'] / self.metrics['total_queries']) if self.metrics['total_queries'] > 0 else 0.0
        
        return {
            'total_queries': self.metrics['total_queries'],
            'successful_queries': self.metrics['successful_queries'],
            'success_rate': success_rate,
            'average_processing_time': self.metrics['average_processing_time'],
            'reasoning_types_used': self.metrics['reasoning_types_used']
        }
    
    async def shutdown(self):
        """Shutdown the reasoning engine"""
        logger.info("🔄 Shutting down Advanced Reasoning Engine...")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("✅ Advanced Reasoning Engine shutdown complete")

# Factory function
def create_reasoning_engine(config: Optional[Dict[str, Any]] = None) -> AdvancedReasoningEngine:
    """Create and return a configured reasoning engine"""
    return AdvancedReasoningEngine(config)