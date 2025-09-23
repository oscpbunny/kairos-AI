"""
Project Kairos: Test Configuration and Fixtures
Comprehensive test setup for Phase 6 production excellence.
"""

import pytest
import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock heavy dependencies for testing
def setup_mock_dependencies():
    """Setup mock dependencies to avoid external requirements"""
    
    # Mock numpy
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
        def mean(values): return sum(values) / len(values) if values else 0
        @staticmethod
        def std(values): 
            if not values: return 0
            mean_val = sum(values) / len(values)
            return (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
        @staticmethod
        def max(values): return max(values) if values else 0
        @staticmethod
        def min(values): return min(values) if values else 0
    
    # Install mocks
    sys.modules['numpy'] = MockNumpy()
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
    sys.modules['pandas'] = None
    sys.modules['scipy'] = type('scipy', (), {'stats': None})()
    sys.modules['tensorflow'] = None
    sys.modules['transformers'] = type('transformers', (), {'pipeline': None})()
    sys.modules['prometheus_client'] = type('prometheus', (), {
        'Counter': lambda *args, **kwargs: type('MockCounter', (), {'inc': lambda self, *a: None})(),
        'Histogram': lambda *args, **kwargs: type('MockHistogram', (), {'observe': lambda self, *a: None})(),
        'Gauge': lambda *args, **kwargs: type('MockGauge', (), {'set': lambda self, *a: None})()
    })()
    sys.modules['grpc'] = type('grpc', (), {
        'aio': type('aio', (), {'insecure_channel': lambda x: None})()
    })()

# Setup mocks before other imports
setup_mock_dependencies()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files"""
    temp_directory = tempfile.mkdtemp()
    yield temp_directory
    shutil.rmtree(temp_directory, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_database():
    """Mock database connection for testing"""
    class MockConnection:
        def __init__(self):
            self.closed = False
            
        async def cursor(self):
            return MockCursor()
            
        def commit(self):
            pass
            
        def close(self):
            self.closed = True
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            self.close()
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args):
            self.close()
    
    class MockCursor:
        def __init__(self):
            self.results = []
            
        async def execute(self, query, params=None):
            # Mock successful execution
            pass
            
        async def fetchone(self):
            return {'id': 'test-id', 'status': 'success'} if self.results else None
            
        async def fetchall(self):
            return self.results
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args):
            pass
    
    return MockConnection()


@pytest.fixture(scope="function")
def sample_venture():
    """Sample venture data for testing"""
    return {
        'id': 'test-venture-001',
        'name': 'Test E-commerce Platform',
        'objective': 'Build scalable e-commerce platform',
        'target_users': 50000,
        'budget': 100000,
        'timeline_days': 90
    }


@pytest.fixture(scope="function")
def sample_task():
    """Sample task data for testing"""
    return {
        'id': 'test-task-001',
        'venture_id': 'test-venture-001',
        'task_type': 'DEPLOYMENT',
        'description': 'Deploy microservices infrastructure with auto-scaling',
        'cc_bounty': 1000,
        'status': 'BOUNTY_POSTED',
        'requirements': {
            'compute_requirements': {'instances': 5, 'type': 't3.medium'},
            'storage_requirements': {'size_gb': 500, 'type': 'gp3'},
            'performance_requirements': {'latency_target': 200}
        }
    }


@pytest.fixture(scope="function")
def sample_design_spec():
    """Sample design specification for testing"""
    return {
        'pattern': 'microservices',
        'technology_stack': {
            'backend': 'python_fastapi',
            'database': 'postgresql',
            'cache': 'redis',
            'queue': 'rabbitmq'
        },
        'expected_scale': {
            'users': 10000,
            'requests_per_second': 100,
            'data_size_gb': 50
        },
        'performance_requirements': {
            'latency_target': 300,
            'throughput_target': 1000,
            'availability_target': 0.999
        }
    }


@pytest.fixture(scope="function")
async def mock_oracle_engine():
    """Mock Oracle Engine for testing"""
    class MockOracleEngine:
        def __init__(self):
            self.initialized = False
            
        async def initialize(self):
            self.initialized = True
            
        async def predict_infrastructure_requirements(self, venture_id: str, time_horizon_days: int = 30, current_infrastructure: Dict = None):
            return {
                'venture_id': venture_id,
                'time_horizon_days': time_horizon_days,
                'predicted_users': 5000 * (time_horizon_days / 30),
                'resource_requirements': {
                    'compute': {'instances': 3, 'type': 't3.medium'},
                    'storage': {'size_gb': 200, 'type': 'gp3'},
                    'database': {'type': 'postgresql', 'size': 'db.t3.medium'}
                },
                'cost_predictions': {
                    'monthly_estimate': 800,
                    'annual_estimate': 9600,
                    'total_monthly': 800
                },
                'scaling_recommendations': {
                    'auto_scaling': True,
                    'scale_up_threshold': 75,
                    'scale_down_threshold': 25
                },
                'confidence_level': 0.85,
                'prediction_source': 'mock_oracle'
            }
            
        async def simulate_design_performance(self, design_spec: Dict, user_scenarios: list, load_patterns: Dict):
            return {
                'design_spec': design_spec,
                'performance_scenarios': [
                    {
                        'scenario': 'normal_load',
                        'performance_metrics': {
                            'response_time_ms': 150,
                            'throughput_rps': 800,
                            'cpu_utilization_percent': 45,
                            'memory_utilization_percent': 60,
                            'error_rate_percent': 0.1
                        }
                    }
                ],
                'overall_assessment': {
                    'overall_grade': 8.5,
                    'performance_class': 'Excellent'
                },
                'potential_bottlenecks': [],
                'improvement_recommendations': [
                    {'category': 'Performance', 'recommendation': 'Add caching layer'}
                ],
                'simulation_confidence': 0.88
            }
    
    return MockOracleEngine()


@pytest.fixture(scope="function")
async def mock_enhanced_steward():
    """Mock Enhanced Steward Agent for testing"""
    from agents.enhanced.enhanced_steward import EnhancedStewardAgent
    
    steward = EnhancedStewardAgent(agent_name="Test-Steward")
    steward.agent_id = "test-steward-001"
    
    # Mock database connection method
    async def mock_get_db_connection():
        return None
    steward.get_db_connection = mock_get_db_connection
    
    return steward


@pytest.fixture(scope="function")
async def mock_enhanced_architect():
    """Mock Enhanced Architect Agent for testing"""
    from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
    
    architect = EnhancedArchitectAgent(agent_name="Test-Architect")
    architect.agent_id = "test-architect-001"
    
    # Mock database connection method
    async def mock_get_db_connection():
        return None
    architect.get_db_connection = mock_get_db_connection
    
    return architect


@pytest.fixture(scope="function")
def performance_test_data():
    """Test data for performance testing"""
    return {
        'small_load': {'concurrent_users': 10, 'duration_seconds': 30},
        'medium_load': {'concurrent_users': 100, 'duration_seconds': 60},
        'large_load': {'concurrent_users': 1000, 'duration_seconds': 120},
        'stress_load': {'concurrent_users': 5000, 'duration_seconds': 180}
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    os.environ['KAIROS_ENV'] = 'test'
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_NAME'] = 'kairos_test'
    os.environ['DB_USER'] = 'test_user'
    os.environ['DB_PASSWORD'] = 'test_password'
    os.environ['REDIS_HOST'] = 'localhost'
    os.environ['LOG_LEVEL'] = 'INFO'


@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for all tests"""
    with patch('psycopg2.connect') as mock_connect:
        mock_connect.return_value = Mock()
        yield


# Utility functions for tests
class TestHelpers:
    @staticmethod
    def assert_prediction_structure(prediction: Dict[str, Any]):
        """Assert that a prediction has the expected structure"""
        assert 'venture_id' in prediction
        assert 'resource_requirements' in prediction
        assert 'cost_predictions' in prediction
        assert 'confidence_level' in prediction
        assert 0 <= prediction['confidence_level'] <= 1
        
    @staticmethod
    def assert_validation_structure(validation: Dict[str, Any]):
        """Assert that a validation has the expected structure"""
        assert 'confidence_level' in validation
        assert 0 <= validation['confidence_level'] <= 1
        
    @staticmethod
    def create_mock_task_result(task_id: str, success: bool = True) -> Dict[str, Any]:
        """Create a mock task result"""
        return {
            'task_id': task_id,
            'status': 'completed' if success else 'failed',
            'quality_score': 0.9 if success else 0.3,
            'execution_time': 120 if success else 300,
            'deliverables': {'artifacts': ['test_artifact']} if success else {}
        }

# Make helper class available as fixture
@pytest.fixture
def test_helpers():
    return TestHelpers