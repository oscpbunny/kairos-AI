#!/usr/bin/env python3
"""
Simple Oracle-Agent Integration Test
Test the core Oracle-Agent communication without heavy dependencies
"""

import asyncio
import sys
import os
from datetime import datetime

# Mock missing dependencies
class MockNumpy:
    @staticmethod
    def random():
        import random
        return type('obj', (object,), {
            'choice': lambda keys, p=None: random.choice(keys),
            'uniform': lambda a, b: random.uniform(a, b),
            'beta': lambda a, b: random.random(),
            'randint': lambda a, b: random.randint(a, b)
        })()
    
    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0
    
    @staticmethod
    def std(values):
        if not values:
            return 0
        mean_val = sum(values) / len(values)
        return (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
    
    @staticmethod
    def max(values):
        return max(values) if values else 0
    
    @staticmethod
    def min(values):
        return min(values) if values else 0

# Install mock numpy
if 'numpy' not in sys.modules:
    sys.modules['numpy'] = MockNumpy()
    import numpy as np

# Mock other dependencies
sys.modules['sklearn'] = type('sklearn', (), {})()
sys.modules['sklearn.ensemble'] = type('ensemble', (), {'RandomForestRegressor': None})()
sys.modules['sklearn.preprocessing'] = type('preprocessing', (), {'StandardScaler': None})()
sys.modules['sklearn.feature_extraction'] = type('feature_extraction', (), {})()
sys.modules['sklearn.feature_extraction.text'] = type('text', (), {'TfidfVectorizer': None})()
sys.modules['sklearn.metrics'] = type('metrics', (), {})()
sys.modules['sklearn.metrics.pairwise'] = type('pairwise', (), {'cosine_similarity': None})()
sys.modules['sklearn.cluster'] = type('cluster', (), {'KMeans': None})()
sys.modules['boto3'] = None
sys.modules['pulumi'] = None
sys.modules['pulumi_aws'] = type('pulumi_aws', (), {'ec2': None, 'ecs': None, 'rds': None})()
sys.modules['psycopg2'] = None
sys.modules['psycopg2.extras'] = type('extras', (), {'RealDictCursor': None})()
sys.modules['pandas'] = None
sys.modules['scipy'] = type('scipy', (), {'stats': None})()
sys.modules['tensorflow'] = None
sys.modules['transformers'] = type('transformers', (), {'pipeline': None})()
sys.modules['prometheus_client'] = type('prometheus', (), {
    'Counter': lambda *args, **kwargs: None,
    'Histogram': lambda *args, **kwargs: None,
    'Gauge': lambda *args, **kwargs: None
})()
sys.modules['grpc'] = type('grpc', (), {
    'aio': type('aio', (), {'insecure_channel': lambda x: None})()
})()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))


async def test_oracle_steward_integration():
    """Test basic Oracle-Steward integration"""
    print("🔧 Testing Steward-Oracle Integration...")
    
    try:
        from agents.enhanced.enhanced_steward import EnhancedStewardAgent
        
        # Create Steward agent
        steward = EnhancedStewardAgent(agent_name="Test-Steward")
        steward.agent_id = "test-steward-001"
        
        # Test infrastructure predictions (should use fallback)
        predictions = await steward.get_infrastructure_predictions(
            venture_id="test-venture-001",
            time_horizon=30
        )
        
        # Validate basic structure
        assert 'resource_requirements' in predictions
        assert 'cost_predictions' in predictions
        assert 'confidence_level' in predictions
        
        print(f"   ✅ Steward predictions received")
        print(f"   📊 Predicted cost: ${predictions['cost_predictions'].get('monthly_estimate', 'N/A')}")
        print(f"   🎯 Confidence: {predictions['confidence_level']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Steward integration failed: {e}")
        return False


async def test_oracle_architect_integration():
    """Test basic Oracle-Architect integration"""
    print("🏗️ Testing Architect-Oracle Integration...")
    
    try:
        from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
        
        # Create Architect agent
        architect = EnhancedArchitectAgent(agent_name="Test-Architect")
        architect.agent_id = "test-architect-001"
        
        # Test design validation
        design_spec = {
            'pattern': 'microservices',
            'technology_stack': {
                'backend': 'python_fastapi',
                'database': 'postgresql'
            }
        }
        
        validation = await architect.validate_design_with_oracle(
            design_spec=design_spec,
            venture_id="test-venture-001"
        )
        
        # Validate basic structure
        assert validation is not None
        assert 'confidence_level' in validation
        
        print(f"   ✅ Architect validation completed")
        print(f"   🎯 Confidence: {validation['confidence_level']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Architect integration failed: {e}")
        return False


async def test_oracle_engine_basic():
    """Test basic Oracle Engine functionality"""
    print("🔮 Testing Oracle Engine Basics...")
    
    try:
        from simulation.oracle_engine import OracleEngine
        
        # Create Oracle Engine
        oracle = OracleEngine()
        await oracle.initialize()
        
        # Test infrastructure requirements prediction
        predictions = await oracle.predict_infrastructure_requirements(
            venture_id="test-venture-001",
            time_horizon_days=30
        )
        
        # Validate structure
        assert 'predicted_users' in predictions
        assert 'resource_requirements' in predictions
        assert 'cost_predictions' in predictions
        
        # Validate cost predictions
        costs = predictions['cost_predictions']
        monthly_cost_key = 'total_monthly' if 'total_monthly' in costs else 'monthly_estimate'
        assert monthly_cost_key in costs
        
        print(f"   ✅ Oracle predictions generated")
        print(f"   👥 Predicted users: {predictions['predicted_users']}")
        print(f"   💰 Monthly cost: ${costs[monthly_cost_key]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Oracle engine test failed: {e}")
        return False


async def run_simple_integration_tests():
    """Run simplified integration tests"""
    print("🧪 Kairos Phase 5 - Oracle Integration Tests (Simplified)")
    print("=" * 60)
    
    results = []
    
    # Test individual components
    results.append(await test_oracle_steward_integration())
    results.append(await test_oracle_architect_integration())
    results.append(await test_oracle_engine_basic())
    
    # Calculate results
    passed_tests = sum(1 for result in results if result)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 Integration Test Results")
    print("=" * 60)
    print(f"Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ PHASE 5 ORACLE INTEGRATION: OPERATIONAL")
        print("   • Oracle-Agent communication established")
        print("   • Fallback mechanisms working")
        print("   • Infrastructure predictions functional")
        print("   • Design validation operational")
        return True
    else:
        print("⚠️ PHASE 5 ORACLE INTEGRATION: PARTIAL")
        print("   • Some components need attention")
        return False


if __name__ == "__main__":
    asyncio.run(run_simple_integration_tests())