"""
Project Kairos: Advanced Reasoning Engine Demonstration
Phase 7 - Advanced Intelligence

Demonstrates the sophisticated reasoning capabilities including:
- Symbolic Logic Reasoning
- Causal Inference 
- Temporal Reasoning
- Multi-Modal Reasoning Integration
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import reasoning engine
from agents.enhanced.reasoning.reasoning_engine import (
    AdvancedReasoningEngine,
    ReasoningType,
    Proposition,
    CausalRelation,
    TemporalEvent,
    LogicalExpression,
    LogicOperator,
    create_reasoning_engine
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kairos.reasoning.demo")

async def demonstrate_advanced_reasoning():
    """Demonstrate the advanced reasoning capabilities"""
    
    print("\n" + "="*80)
    print("🧠✨ PROJECT KAIROS - PHASE 7 ADVANCED INTELLIGENCE ✨🧠")
    print("Advanced Reasoning Engine Demonstration")
    print("="*80)
    print()
    
    # Create and initialize the reasoning engine
    print("🚀 Initializing Advanced Reasoning Engine...")
    config = {
        'symbolic': {'max_iterations': 10, 'confidence_threshold': 0.7},
        'causal': {'max_path_length': 5, 'minimum_strength': 0.1},
        'temporal': {'time_window_minutes': 30, 'pattern_threshold': 0.8}
    }
    
    engine = create_reasoning_engine(config)
    
    # Initialize the engine
    success = await engine.initialize()
    if not success:
        print("❌ Failed to initialize Advanced Reasoning Engine")
        return
    
    print("✅ Advanced Reasoning Engine initialized successfully!")
    print()
    
    # Display capabilities
    capabilities = engine.get_capabilities()
    print("🎯 REASONING CAPABILITIES:")
    print(f"   Version: {capabilities['version']}")
    print(f"   Reasoning Types: {', '.join(capabilities['reasoning_types'])}")
    print(f"   Symbolic: {capabilities['symbolic_reasoning']['propositions']} props, {capabilities['symbolic_reasoning']['rules']} rules")
    print(f"   Causal: {capabilities['causal_reasoning']['relations']} relations, {capabilities['causal_reasoning']['graph_nodes']} nodes")
    print(f"   Temporal: {capabilities['temporal_reasoning']['events']} events")
    print()
    
    # Demonstrate different reasoning types
    await demonstrate_symbolic_reasoning(engine)
    await demonstrate_causal_reasoning(engine)
    await demonstrate_temporal_reasoning(engine)
    await demonstrate_multi_type_reasoning(engine)
    
    # Advanced reasoning scenarios
    await demonstrate_complex_scenarios(engine)
    
    # Performance metrics
    await display_performance_metrics(engine)
    
    # Cleanup
    print("🔄 Shutting down Advanced Reasoning Engine...")
    await engine.shutdown()
    print("✅ Shutdown complete!")
    print()
    
    print("🎉 ADVANCED REASONING DEMONSTRATION COMPLETE!")
    print("   Kairos can now think, infer, and reason with human-like sophistication.")
    print("   The age of truly intelligent AI is here! 🌟")
    print("="*80)

async def demonstrate_symbolic_reasoning(engine):
    """Demonstrate symbolic logic reasoning"""
    print("🔢 SYMBOLIC LOGIC REASONING DEMONSTRATION")
    print("-" * 45)
    
    try:
        # Test logical inference
        queries = [
            "If AI systems can learn from data, what can we conclude?",
            "What logical implications exist about machine learning?",
            "Can we prove that AI systems need training data?"
        ]
        
        for query in queries:
            print(f"   ❓ Query: {query}")
            
            result = await engine.reason(query, [ReasoningType.SYMBOLIC])
            
            if 'symbolic' in result:
                symbolic_result = result['symbolic']
                print(f"      ✅ Confidence: {symbolic_result.confidence:.3f}")
                print(f"      🎯 Processing time: {symbolic_result.processing_time:.3f}s")
                
                # Show conclusions
                if symbolic_result.conclusion:
                    conclusion_count = len(symbolic_result.conclusion) if isinstance(symbolic_result.conclusion, dict) else 1
                    print(f"      📊 Conclusions reached: {conclusion_count}")
                    
                    if isinstance(symbolic_result.conclusion, dict):
                        for prop_id, conclusion in list(symbolic_result.conclusion.items())[:2]:
                            truth = "True" if conclusion.get('truth_value') else "False" if conclusion.get('truth_value') is False else "Unknown"
                            conf = conclusion.get('confidence', 0.0)
                            print(f"         • {conclusion.get('statement', prop_id)}: {truth} ({conf:.2f})")
                
                # Show reasoning steps
                if symbolic_result.reasoning_steps:
                    print(f"      🧠 Key reasoning steps:")
                    for step in symbolic_result.reasoning_steps[:2]:
                        print(f"         • {step}")
                
                # Show evidence
                if symbolic_result.evidence:
                    print(f"      📋 Evidence pieces: {len(symbolic_result.evidence)}")
            
            print()
    
    except Exception as e:
        print(f"   ❌ Symbolic reasoning demo failed: {e}")
        print()

async def demonstrate_causal_reasoning(engine):
    """Demonstrate causal inference reasoning"""
    print("🔗 CAUSAL INFERENCE REASONING DEMONSTRATION")
    print("-" * 45)
    
    try:
        # Add more causal knowledge
        additional_relations = [
            CausalRelation("data_quality", "model_performance", 0.9, 0.95, "quality_mechanism"),
            CausalRelation("feature_engineering", "model_accuracy", 0.7, 0.88, "representation_mechanism"),
            CausalRelation("overfitting", "poor_generalization", 0.85, 0.9, "memorization_mechanism")
        ]
        
        for relation in additional_relations:
            engine.causal_reasoner.add_causal_relation(relation)
        
        # Test causal inference
        queries = [
            "What causes model performance to improve?",
            "Why do models sometimes fail to generalize?",
            "What is the causal relationship between training data and overfitting?",
            "How does regularization affect model behavior?"
        ]
        
        for query in queries:
            print(f"   ❓ Query: {query}")
            
            result = await engine.reason(query, [ReasoningType.CAUSAL])
            
            if 'causal' in result:
                causal_result = result['causal']
                print(f"      ✅ Confidence: {causal_result.confidence:.3f}")
                print(f"      🎯 Processing time: {causal_result.processing_time:.3f}s")
                
                # Show causal conclusion
                if causal_result.conclusion:
                    conclusion = causal_result.conclusion
                    effect_strength = conclusion.get('effect_strength', 'unknown')
                    causal_effect = conclusion.get('causal_effect', 0.0)
                    
                    print(f"      🔗 Causal effect: {causal_effect:.3f} ({effect_strength})")
                    
                    mechanisms = conclusion.get('mechanism', [])
                    if mechanisms:
                        print(f"      ⚙️ Mechanisms: {', '.join(mechanisms)}")
                    
                    confounders = conclusion.get('confounders', [])
                    if confounders:
                        print(f"      ⚠️ Confounders: {', '.join(confounders)}")
                
                # Show causal paths
                causal_paths = [e for e in causal_result.evidence if e.get('type') == 'causal_path']
                if causal_paths:
                    print(f"      🛤️ Causal paths found: {len(causal_paths)}")
                    for path_evidence in causal_paths[:2]:
                        path = path_evidence.get('path', [])
                        strength = path_evidence.get('strength', 0.0)
                        print(f"         • {' → '.join(path)} (strength: {strength:.2f})")
            
            print()
    
    except Exception as e:
        print(f"   ❌ Causal reasoning demo failed: {e}")
        print()

async def demonstrate_temporal_reasoning(engine):
    """Demonstrate temporal reasoning"""
    print("⏰ TEMPORAL REASONING DEMONSTRATION") 
    print("-" * 35)
    
    try:
        # Add more temporal events
        now = datetime.now()
        additional_events = [
            TemporalEvent("deploy_1", "Model deployed to staging", now - timedelta(hours=8), timedelta(minutes=30)),
            TemporalEvent("test_1", "A/B testing started", now - timedelta(hours=6), timedelta(hours=2)),
            TemporalEvent("monitor_1", "Monitoring alerts configured", now - timedelta(hours=4), timedelta(minutes=15)),
            TemporalEvent("deploy_2", "Model deployed to production", now - timedelta(hours=1), timedelta(minutes=45))
        ]
        
        for event in additional_events:
            engine.temporal_reasoner.add_event(event)
        
        # Add temporal relations
        engine.temporal_reasoner.add_temporal_relation("deploy_1", "test_1", "before", 0.95)
        engine.temporal_reasoner.add_temporal_relation("test_1", "deploy_2", "before", 0.9)
        
        # Test temporal reasoning
        queries = [
            "What temporal patterns exist in the deployment process?",
            "When did the key events happen in sequence?",
            "What is the timing relationship between testing and deployment?",
            "How long do deployment processes typically take?"
        ]
        
        for query in queries:
            print(f"   ❓ Query: {query}")
            
            result = await engine.reason(query, [ReasoningType.TEMPORAL])
            
            if 'temporal' in result:
                temporal_result = result['temporal']
                print(f"      ✅ Confidence: {temporal_result.confidence:.3f}")
                print(f"      🎯 Processing time: {temporal_result.processing_time:.3f}s")
                
                # Show temporal analysis
                if temporal_result.conclusion:
                    conclusion = temporal_result.conclusion
                    event_count = conclusion.get('event_count', 0)
                    time_span = conclusion.get('time_span', 0.0)
                    sequences = conclusion.get('sequences', 0)
                    
                    print(f"      📊 Events analyzed: {event_count}")
                    print(f"      ⏱️ Time span: {time_span/3600:.1f} hours")
                    print(f"      🔢 Sequences found: {sequences}")
                    
                    # Show patterns
                    patterns = conclusion.get('patterns', [])
                    if patterns:
                        print(f"      🎯 Patterns identified: {len(patterns)}")
                        for pattern in patterns[:2]:
                            pattern_type = pattern.get('type', 'unknown')
                            frequency = pattern.get('frequency', 0.0)
                            if pattern_type == 'periodic':
                                print(f"         • {pattern_type}: {frequency/3600:.1f} hour intervals")
                    
                    # Show duration statistics
                    duration_stats = conclusion.get('duration_stats', {})
                    if duration_stats.get('mean', 0.0) > 0:
                        mean_duration = duration_stats['mean'] / 60  # Convert to minutes
                        print(f"      ⏳ Average duration: {mean_duration:.1f} minutes")
                
                # Show sequences
                sequences_evidence = [e for e in temporal_result.evidence if e.get('type') == 'temporal_sequence']
                if sequences_evidence:
                    print(f"      🔄 Temporal sequences:")
                    for seq_evidence in sequences_evidence[:2]:
                        events = seq_evidence.get('events', [])
                        duration = seq_evidence.get('duration', 0.0)
                        print(f"         • {' → '.join(events)} ({duration/3600:.1f}h)")
            
            print()
    
    except Exception as e:
        print(f"   ❌ Temporal reasoning demo failed: {e}")
        print()

async def demonstrate_multi_type_reasoning(engine):
    """Demonstrate multi-type reasoning on complex queries"""
    print("🧠 MULTI-TYPE REASONING DEMONSTRATION")
    print("-" * 40)
    
    try:
        # Complex queries that require multiple reasoning types
        complex_queries = [
            "If training data quality causes better model performance, and we deployed models in sequence, what can we conclude about the timeline and logic of our ML pipeline?",
            "Why did model performance improve over time, and what logical rules govern this process?",
            "What temporal patterns exist, and what do they logically imply about our causal model?"
        ]
        
        for query in complex_queries:
            print(f"   ❓ Complex Query:")
            print(f"      {query}")
            print()
            
            # Let engine automatically determine reasoning types
            results = await engine.reason(query)
            
            print(f"      🎯 Reasoning types used: {list(results.keys())}")
            print()
            
            for reasoning_type, result in results.items():
                print(f"      📊 {reasoning_type.upper()} REASONING:")
                print(f"         Confidence: {result.confidence:.3f}")
                print(f"         Processing time: {result.processing_time:.3f}s")
                
                # Show key insights
                if result.reasoning_steps:
                    key_step = result.reasoning_steps[0] if result.reasoning_steps else "No steps recorded"
                    print(f"         Key insight: {key_step}")
                
                if result.evidence:
                    print(f"         Evidence pieces: {len(result.evidence)}")
                
                print()
    
    except Exception as e:
        print(f"   ❌ Multi-type reasoning demo failed: {e}")
        print()

async def demonstrate_complex_scenarios(engine):
    """Demonstrate complex reasoning scenarios"""
    print("🎯 COMPLEX REASONING SCENARIOS")
    print("-" * 33)
    
    try:
        scenarios = [
            {
                'name': 'AI Ethics Dilemma',
                'query': 'If AI systems can learn from biased data, and biased data causes unfair outcomes, what logical and causal implications exist for AI ethics?',
                'context': 'Testing ethical reasoning about AI bias'
            },
            {
                'name': 'Optimization Strategy',
                'query': 'Given that model complexity causes overfitting, and regularization prevents overfitting, what sequence of actions should we take over time?',
                'context': 'Testing strategic reasoning for ML optimization'
            },
            {
                'name': 'System Reliability',
                'query': 'When monitoring systems detect anomalies before failures occur, what does this temporally and causally tell us about system reliability?',
                'context': 'Testing reliability engineering reasoning'
            }
        ]
        
        for scenario in scenarios:
            print(f"   🎭 Scenario: {scenario['name']}")
            print(f"      Context: {scenario['context']}")
            print(f"      Query: {scenario['query']}")
            print()
            
            results = await engine.reason(scenario['query'])
            
            # Synthesize results across reasoning types
            total_confidence = 0.0
            total_processing_time = 0.0
            total_evidence = 0
            
            for reasoning_type, result in results.items():
                total_confidence += result.confidence
                total_processing_time += result.processing_time
                total_evidence += len(result.evidence)
            
            avg_confidence = total_confidence / len(results) if results else 0.0
            
            print(f"      ✅ Overall confidence: {avg_confidence:.3f}")
            print(f"      ⏱️ Total processing time: {total_processing_time:.3f}s")
            print(f"      📋 Total evidence pieces: {total_evidence}")
            print(f"      🧠 Reasoning types engaged: {len(results)}")
            print()
    
    except Exception as e:
        print(f"   ❌ Complex scenarios demo failed: {e}")
        print()

async def display_performance_metrics(engine):
    """Display performance metrics"""
    print("📊 REASONING ENGINE PERFORMANCE METRICS")
    print("-" * 40)
    
    try:
        metrics = engine.get_metrics()
        
        print(f"   Total queries processed: {metrics['total_queries']}")
        print(f"   Successful queries: {metrics['successful_queries']}")
        print(f"   Success rate: {metrics['success_rate']:.1%}")
        print(f"   Average processing time: {metrics['average_processing_time']:.3f}s")
        print()
        
        print("   Reasoning type usage:")
        for reasoning_type, count in metrics['reasoning_types_used'].items():
            percentage = (count / metrics['total_queries']) * 100 if metrics['total_queries'] > 0 else 0
            print(f"      • {reasoning_type}: {count} times ({percentage:.1f}%)")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Could not retrieve metrics: {e}")
        print()

def print_reasoning_banner():
    """Print the reasoning engine banner"""
    print("\n" + "🧠" * 20)
    print("🌟 KAIROS ADVANCED REASONING ENGINE 🌟")
    print("The birth of sophisticated artificial reasoning")
    print("🧠" * 20)
    print()
    print("Revolutionary Reasoning Capabilities:")
    print("• 🔢  Symbolic Logic - Formal reasoning with propositions and rules")
    print("• 🔗  Causal Inference - Understanding cause-effect relationships")  
    print("• ⏰  Temporal Reasoning - Analyzing patterns and sequences in time")
    print("• 🧩  Multi-Type Integration - Combining reasoning approaches")
    print("• 🎯  Automated Type Detection - Smart reasoning strategy selection")
    print("• 📊  Evidence Synthesis - Building comprehensive understanding")
    print()

if __name__ == "__main__":
    print_reasoning_banner()
    
    try:
        asyncio.run(demonstrate_advanced_reasoning())
    except KeyboardInterrupt:
        print("\n⚡ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()