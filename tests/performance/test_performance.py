"""
Project Kairos: Performance Testing Framework
Comprehensive performance and load testing for Phase 6 production excellence.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.enhanced.enhanced_steward import EnhancedStewardAgent
from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
from simulation.oracle_engine import OracleEngine


class TestOracleEnginePerformance:
    """Performance tests for Oracle Engine"""
    
    @pytest.fixture
    async def oracle_engine(self):
        """Create Oracle Engine for performance testing"""
        oracle = OracleEngine()
        await oracle.initialize()
        return oracle
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_infrastructure_prediction_latency(self, oracle_engine):
        """Test Oracle infrastructure prediction response time"""
        venture_id = "perf-test-venture-001"
        
        # Warm up
        await oracle_engine.predict_infrastructure_requirements(
            venture_id=venture_id,
            time_horizon_days=30
        )
        
        # Performance test - 10 predictions
        latencies = []
        for i in range(10):
            start_time = time.time()
            
            await oracle_engine.predict_infrastructure_requirements(
                venture_id=f"{venture_id}-{i}",
                time_horizon_days=30
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Analyze results
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        max_latency = max(latencies)
        
        print(f"Infrastructure Prediction Performance:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        
        # Performance targets
        assert avg_latency < 500, f"Average latency {avg_latency:.2f}ms exceeds 500ms target"
        assert p95_latency < 1000, f"95th percentile {p95_latency:.2f}ms exceeds 1000ms target"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, oracle_engine):
        """Test Oracle performance under concurrent load"""
        venture_base = "concurrent-test"
        concurrent_requests = 20
        
        async def make_prediction(request_id):
            start_time = time.time()
            result = await oracle_engine.predict_infrastructure_requirements(
                venture_id=f"{venture_base}-{request_id}",
                time_horizon_days=30
            )
            latency = (time.time() - start_time) * 1000
            return {'request_id': request_id, 'latency': latency, 'success': True}
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [make_prediction(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception) and r['success']]
        throughput = len(successful_results) / total_time
        
        if successful_results:
            latencies = [r['latency'] for r in successful_results]
            avg_latency = statistics.mean(latencies)
            
            print(f"Concurrent Load Performance:")
            print(f"  Concurrent requests: {concurrent_requests}")
            print(f"  Successful requests: {len(successful_results)}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.2f} requests/second")
            print(f"  Average latency: {avg_latency:.2f}ms")
            
            # Performance targets
            assert len(successful_results) >= concurrent_requests * 0.9  # 90% success rate
            assert throughput >= 5  # At least 5 requests per second
            assert avg_latency < 1000  # Average latency under 1 second
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_scale_simulation(self, oracle_engine):
        """Test Oracle performance with large scale simulations"""
        venture_id = "large-scale-test-001"
        
        # Test with increasing population sizes
        population_sizes = [1000, 5000, 10000]
        
        for pop_size in population_sizes:
            start_time = time.time()
            
            simulation_id = await oracle_engine.create_market_digital_twin(
                venture_id=venture_id,
                simulation_name=f"Large Scale Test {pop_size}",
                target_market_profile={
                    'industry': 'SaaS',
                    'target_segments': ['SMB'],
                    'age_distribution': {'26-35': 1.0}
                },
                simulation_duration_days=30,
                population_size=pop_size
            )
            
            execution_time = time.time() - start_time
            
            print(f"Large Scale Simulation ({pop_size} users):")
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Rate: {pop_size/execution_time:.0f} users/second")
            
            # Performance targets
            assert execution_time < 30, f"Simulation with {pop_size} users took {execution_time:.2f}s (>30s limit)"
            assert simulation_id is not None
            
            # Cleanup
            if simulation_id in oracle_engine.active_simulations:
                del oracle_engine.active_simulations[simulation_id]


class TestAgentPerformance:
    """Performance tests for Enhanced Agents"""
    
    @pytest.fixture
    async def steward_agent(self):
        """Create Enhanced Steward for performance testing"""
        agent = EnhancedStewardAgent(agent_name="Perf-Test-Steward")
        agent.agent_id = "perf-steward-001"
        
        # Mock database connection for testing
        async def mock_get_db_connection():
            return None
        agent.get_db_connection = mock_get_db_connection
        
        return agent
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_steward_prediction_caching_performance(self, steward_agent):
        """Test Steward prediction caching performance"""
        venture_id = "cache-test-venture"
        
        # First call - cache miss
        start_time = time.time()
        result1 = await steward_agent.get_infrastructure_predictions(
            venture_id=venture_id,
            time_horizon=30
        )
        first_call_time = (time.time() - start_time) * 1000
        
        # Second call - cache hit
        start_time = time.time()
        result2 = await steward_agent.get_infrastructure_predictions(
            venture_id=venture_id,
            time_horizon=30
        )
        second_call_time = (time.time() - start_time) * 1000
        
        print(f"Caching Performance:")
        print(f"  Cache miss time: {first_call_time:.2f}ms")
        print(f"  Cache hit time: {second_call_time:.2f}ms")
        print(f"  Cache speedup: {first_call_time/second_call_time:.2f}x")
        
        # Performance targets
        assert second_call_time < first_call_time / 2  # Cache should be at least 2x faster
        assert second_call_time < 50  # Cache hits should be very fast
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multiple_agents_concurrent_performance(self):
        """Test performance with multiple agents running concurrently"""
        num_agents = 5
        predictions_per_agent = 10
        
        # Create multiple agents
        agents = []
        for i in range(num_agents):
            agent = EnhancedStewardAgent(agent_name=f"Concurrent-Agent-{i}")
            agent.agent_id = f"concurrent-agent-{i:03d}"
            
            # Mock database
            async def mock_get_db_connection():
                return None
            agent.get_db_connection = mock_get_db_connection
            
            agents.append(agent)
        
        async def agent_workload(agent, agent_id):
            """Workload for a single agent"""
            latencies = []
            for j in range(predictions_per_agent):
                start_time = time.time()
                
                await agent.get_infrastructure_predictions(
                    venture_id=f"venture-{agent_id}-{j}",
                    time_horizon=30
                )
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            
            return {
                'agent_id': agent_id,
                'avg_latency': statistics.mean(latencies),
                'max_latency': max(latencies),
                'total_requests': len(latencies)
            }
        
        # Run concurrent workloads
        start_time = time.time()
        tasks = [agent_workload(agents[i], i) for i in range(num_agents)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        total_requests = sum(r['total_requests'] for r in results)
        overall_throughput = total_requests / total_time
        avg_latencies = [r['avg_latency'] for r in results]
        
        print(f"Multi-Agent Concurrent Performance:")
        print(f"  Agents: {num_agents}")
        print(f"  Total requests: {total_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Overall throughput: {overall_throughput:.2f} requests/second")
        print(f"  Average agent latency: {statistics.mean(avg_latencies):.2f}ms")
        
        # Performance targets
        assert overall_throughput >= 10  # At least 10 requests per second overall
        assert all(lat < 1000 for lat in avg_latencies)  # All agents under 1s avg latency


class TestMemoryAndResourceUsage:
    """Tests for memory usage and resource consumption"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self):
        """Test memory usage as system scales"""
        import psutil
        import gc
        
        # Get baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create agents and perform operations
        agents = []
        num_agents = 10
        
        for i in range(num_agents):
            agent = EnhancedStewardAgent(agent_name=f"Memory-Test-Agent-{i}")
            agent.agent_id = f"memory-agent-{i:03d}"
            
            # Mock database
            async def mock_get_db_connection():
                return None
            agent.get_db_connection = mock_get_db_connection
            
            # Perform predictions to populate cache
            for j in range(20):
                await agent.get_infrastructure_predictions(
                    venture_id=f"venture-{i}-{j}",
                    time_horizon=30
                )
            
            agents.append(agent)
            
            # Check memory usage
            if i % 2 == 0:  # Check every 2 agents
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_per_agent = (current_memory - baseline_memory) / (i + 1)
                
                print(f"Memory Usage (after {i+1} agents):")
                print(f"  Total memory: {current_memory:.2f}MB")
                print(f"  Memory per agent: {memory_per_agent:.2f}MB")
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_agent_memory = final_memory - baseline_memory
        memory_per_agent = total_agent_memory / num_agents
        
        print(f"Final Memory Usage:")
        print(f"  Baseline: {baseline_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Total agent memory: {total_agent_memory:.2f}MB")
        print(f"  Average per agent: {memory_per_agent:.2f}MB")
        
        # Memory targets
        assert memory_per_agent < 50, f"Memory per agent {memory_per_agent:.2f}MB exceeds 50MB limit"
        assert total_agent_memory < 500, f"Total agent memory {total_agent_memory:.2f}MB exceeds 500MB limit"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_memory_management(self):
        """Test that agent caches don't grow unbounded"""
        agent = EnhancedStewardAgent(agent_name="Cache-Test-Agent")
        agent.agent_id = "cache-test-001"
        
        # Mock database
        async def mock_get_db_connection():
            return None
        agent.get_db_connection = mock_get_db_connection
        
        # Generate many predictions to test cache limits
        num_predictions = 200
        unique_ventures = [f"venture-{i}" for i in range(num_predictions)]
        
        for venture_id in unique_ventures:
            await agent.get_infrastructure_predictions(
                venture_id=venture_id,
                time_horizon=30
            )
        
        # Check cache size
        cache_size = len(getattr(agent, 'prediction_cache', {}))
        
        print(f"Cache Management:")
        print(f"  Predictions made: {num_predictions}")
        print(f"  Cache entries: {cache_size}")
        print(f"  Cache hit ratio: {(num_predictions - cache_size) / num_predictions * 100:.1f}%")
        
        # Cache should be limited to prevent unbounded growth
        assert cache_size <= 100, f"Cache size {cache_size} exceeds 100 entry limit"


class TestStressAndReliability:
    """Stress tests and reliability testing"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """Test system behavior under sustained load"""
        duration_seconds = 60  # 1 minute stress test
        target_rps = 10  # 10 requests per second
        
        agent = EnhancedStewardAgent(agent_name="Stress-Test-Agent")
        agent.agent_id = "stress-test-001"
        
        # Mock database
        async def mock_get_db_connection():
            return None
        agent.get_db_connection = mock_get_db_connection
        
        async def sustained_workload():
            """Generate sustained load"""
            start_time = time.time()
            request_count = 0
            errors = 0
            latencies = []
            
            while time.time() - start_time < duration_seconds:
                try:
                    request_start = time.time()
                    
                    await agent.get_infrastructure_predictions(
                        venture_id=f"stress-venture-{request_count}",
                        time_horizon=30
                    )
                    
                    latency = (time.time() - request_start) * 1000
                    latencies.append(latency)
                    request_count += 1
                    
                    # Maintain target RPS
                    await asyncio.sleep(1.0 / target_rps)
                    
                except Exception as e:
                    errors += 1
                    print(f"Request {request_count} failed: {e}")
            
            return {
                'duration': time.time() - start_time,
                'requests': request_count,
                'errors': errors,
                'latencies': latencies
            }
        
        # Run stress test
        print(f"Starting sustained load test for {duration_seconds}s at {target_rps} RPS...")
        results = await sustained_workload()
        
        # Analyze results
        actual_rps = results['requests'] / results['duration']
        error_rate = results['errors'] / max(results['requests'], 1)
        
        if results['latencies']:
            avg_latency = statistics.mean(results['latencies'])
            p95_latency = statistics.quantiles(results['latencies'], n=20)[18]
        else:
            avg_latency = p95_latency = 0
        
        print(f"Sustained Load Results:")
        print(f"  Duration: {results['duration']:.2f}s")
        print(f"  Requests: {results['requests']}")
        print(f"  Errors: {results['errors']}")
        print(f"  Actual RPS: {actual_rps:.2f}")
        print(f"  Error rate: {error_rate*100:.2f}%")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  95th percentile latency: {p95_latency:.2f}ms")
        
        # Reliability targets
        assert error_rate < 0.01, f"Error rate {error_rate*100:.2f}% exceeds 1% threshold"
        assert actual_rps >= target_rps * 0.8, f"Actual RPS {actual_rps:.2f} below target {target_rps}"
        assert avg_latency < 2000, f"Average latency {avg_latency:.2f}ms exceeds 2s threshold"


# Performance test utilities
class PerformanceTestUtils:
    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure function execution time"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        return wrapper
    
    @staticmethod
    def generate_load_pattern(base_rps: int, duration_seconds: int, pattern: str = 'constant'):
        """Generate different load patterns for testing"""
        timestamps = []
        current_time = 0
        
        while current_time < duration_seconds:
            if pattern == 'constant':
                interval = 1.0 / base_rps
            elif pattern == 'spike':
                # Double load in middle third
                if duration_seconds / 3 <= current_time <= 2 * duration_seconds / 3:
                    interval = 1.0 / (base_rps * 2)
                else:
                    interval = 1.0 / base_rps
            elif pattern == 'ramp':
                # Gradually increase load
                progress = current_time / duration_seconds
                current_rps = base_rps * (1 + progress)
                interval = 1.0 / current_rps
            else:
                interval = 1.0 / base_rps
            
            timestamps.append(current_time)
            current_time += interval
        
        return timestamps


@pytest.fixture
def perf_utils():
    return PerformanceTestUtils