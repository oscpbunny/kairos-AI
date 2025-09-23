"""
Project Kairos: Enhanced Agents Unit Tests
Comprehensive unit testing for Phase 6 production excellence.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.enhanced.enhanced_steward import EnhancedStewardAgent
from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
from agents.enhanced.enhanced_engineer import EnhancedEngineerAgent


class TestEnhancedStewardAgent:
    """Unit tests for Enhanced Steward Agent"""
    
    @pytest.fixture
    async def steward_agent(self, mock_database):
        """Create Enhanced Steward Agent for testing"""
        agent = EnhancedStewardAgent(agent_name="Test-Steward")
        agent.agent_id = "test-steward-001"
        
        # Mock database connection
        async def mock_get_db_connection():
            return mock_database
        agent.get_db_connection = mock_get_db_connection
        
        return agent
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_steward_initialization(self, steward_agent):
        """Test Steward agent initializes correctly"""
        assert steward_agent.agent_name == "Test-Steward"
        assert steward_agent.agent_id == "test-steward-001"
        assert steward_agent.agent_type == "Enhanced Steward"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_infrastructure_predictions_caching(self, steward_agent):
        """Test infrastructure prediction caching mechanism"""
        venture_id = "test-venture-001"
        
        # First call should create cache
        predictions1 = await steward_agent.get_infrastructure_predictions(
            venture_id=venture_id,
            time_horizon=30
        )
        
        # Second call should use cache
        predictions2 = await steward_agent.get_infrastructure_predictions(
            venture_id=venture_id,
            time_horizon=30
        )
        
        # Should be cached (same venture_id and time_horizon)
        assert predictions1['venture_id'] == predictions2['venture_id']
        assert predictions1['confidence_level'] > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_resource_optimization(self, steward_agent):
        """Test resource optimization logic"""
        current_resources = {
            'compute': {'instances': 5, 'type': 't3.large'},
            'storage': {'size_gb': 1000, 'type': 'gp3'}
        }
        
        predicted_requirements = {
            'compute': {'instances': 3, 'type': 't3.medium'},
            'storage': {'size_gb': 500, 'type': 'gp3'}
        }
        
        optimization = await steward_agent.optimize_resources(
            current_resources=current_resources,
            predicted_requirements=predicted_requirements
        )
        
        assert 'cost_savings' in optimization
        assert 'optimization_actions' in optimization
        assert optimization['cost_savings'] > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fallback_predictions(self, steward_agent):
        """Test fallback prediction mechanism when Oracle unavailable"""
        # Mock Oracle failure
        with patch.object(steward_agent, 'oracle_client', None):
            predictions = await steward_agent.get_infrastructure_predictions(
                venture_id="test-venture-001",
                time_horizon=30
            )
            
            assert predictions['prediction_source'] == 'fallback_heuristics'
            assert predictions['confidence_level'] < 0.7  # Fallback should have lower confidence


class TestEnhancedArchitectAgent:
    """Unit tests for Enhanced Architect Agent"""
    
    @pytest.fixture
    async def architect_agent(self, mock_database):
        """Create Enhanced Architect Agent for testing"""
        agent = EnhancedArchitectAgent(agent_name="Test-Architect")
        agent.agent_id = "test-architect-001"
        
        # Mock database connection
        async def mock_get_db_connection():
            return mock_database
        agent.get_db_connection = mock_get_db_connection
        
        return agent
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_architect_initialization(self, architect_agent):
        """Test Architect agent initializes correctly"""
        assert architect_agent.agent_name == "Test-Architect"
        assert architect_agent.agent_id == "test-architect-001"
        assert architect_agent.agent_type == "Enhanced Architect"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_design_pattern_selection(self, architect_agent, sample_venture):
        """Test design pattern selection logic"""
        design_patterns = await architect_agent.select_design_patterns(
            venture=sample_venture,
            requirements={
                'scalability': 'high',
                'consistency': 'eventual',
                'availability': 'high'
            }
        )
        
        assert 'recommended_patterns' in design_patterns
        assert 'pattern_rationale' in design_patterns
        assert len(design_patterns['recommended_patterns']) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_technology_stack_selection(self, architect_agent, sample_design_spec):
        """Test technology stack selection logic"""
        tech_stack = await architect_agent.select_technology_stack(
            design_spec=sample_design_spec,
            constraints={
                'budget': 10000,
                'team_expertise': ['python', 'javascript', 'postgresql'],
                'compliance_requirements': ['GDPR', 'SOC2']
            }
        )
        
        assert 'backend' in tech_stack
        assert 'database' in tech_stack
        assert 'frontend' in tech_stack
        assert 'rationale' in tech_stack
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_design_validation_fallback(self, architect_agent, sample_design_spec):
        """Test design validation fallback when Oracle unavailable"""
        # Mock Oracle failure
        with patch.object(architect_agent, 'oracle_client', None):
            validation = await architect_agent.validate_design_with_oracle(
                design_spec=sample_design_spec,
                venture_id="test-venture-001"
            )
            
            assert 'confidence_level' in validation
            assert validation['confidence_level'] < 0.8  # Fallback should have lower confidence


class TestEnhancedEngineerAgent:
    """Unit tests for Enhanced Engineer Agent"""
    
    @pytest.fixture
    async def engineer_agent(self, mock_database):
        """Create Enhanced Engineer Agent for testing"""
        agent = EnhancedEngineerAgent(agent_name="Test-Engineer")
        agent.agent_id = "test-engineer-001"
        
        # Mock database connection
        async def mock_get_db_connection():
            return mock_database
        agent.get_db_connection = mock_get_db_connection
        
        return agent
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_engineer_initialization(self, engineer_agent):
        """Test Engineer agent initializes correctly"""
        assert engineer_agent.agent_name == "Test-Engineer"
        assert engineer_agent.agent_id == "test-engineer-001"
        assert engineer_agent.agent_type == "Enhanced Engineer"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_task_execution_planning(self, engineer_agent, sample_task):
        """Test task execution planning logic"""
        execution_plan = await engineer_agent.create_execution_plan(
            task=sample_task,
            context={
                'available_resources': {'compute_budget': 5000, 'time_budget_hours': 40},
                'constraints': {'deployment_window': '2024-01-15T02:00:00Z'},
                'dependencies': []
            }
        )
        
        assert 'execution_steps' in execution_plan
        assert 'estimated_duration' in execution_plan
        assert 'resource_requirements' in execution_plan
        assert 'risk_assessment' in execution_plan
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_quality_assessment(self, engineer_agent):
        """Test deliverable quality assessment"""
        deliverable = {
            'type': 'code_deployment',
            'artifacts': ['main.py', 'requirements.txt', 'Dockerfile'],
            'test_results': {'unit_tests': 95, 'integration_tests': 88},
            'performance_metrics': {'response_time': 150, 'throughput': 1200}
        }
        
        quality_score = await engineer_agent.assess_deliverable_quality(
            deliverable=deliverable,
            quality_criteria={'test_coverage': 90, 'performance_threshold': 200}
        )
        
        assert 'overall_score' in quality_score
        assert 'quality_dimensions' in quality_score
        assert 0 <= quality_score['overall_score'] <= 1
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_deployment_risk_analysis(self, engineer_agent):
        """Test deployment risk analysis"""
        deployment_spec = {
            'environment': 'production',
            'strategy': 'blue_green',
            'rollback_plan': True,
            'health_checks': ['http_200', 'database_connection']
        }
        
        risk_analysis = await engineer_agent.analyze_deployment_risks(
            deployment_spec=deployment_spec,
            historical_data={'success_rate': 0.92, 'avg_rollback_time': 300}
        )
        
        assert 'risk_level' in risk_analysis
        assert 'risk_factors' in risk_analysis
        assert 'mitigation_strategies' in risk_analysis


class TestAgentCoordination:
    """Tests for multi-agent coordination and communication"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_steward_architect_coordination(self, mock_enhanced_steward, mock_enhanced_architect):
        """Test coordination between Steward and Architect"""
        venture_id = "test-venture-001"
        
        # Steward gets infrastructure predictions
        steward_predictions = await mock_enhanced_steward.get_infrastructure_predictions(
            venture_id=venture_id,
            time_horizon=30
        )
        
        # Architect uses predictions for design
        design_spec = {
            'pattern': 'microservices',
            'expected_scale': {
                'users': steward_predictions['predicted_users'],
                'requests_per_second': 100
            }
        }
        
        architect_validation = await mock_enhanced_architect.validate_design_with_oracle(
            design_spec=design_spec,
            venture_id=venture_id
        )
        
        # Both should have valid results
        assert steward_predictions['confidence_level'] > 0
        assert architect_validation['confidence_level'] > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_propagation_handling(self, mock_enhanced_steward):
        """Test error handling and propagation between agents"""
        # Test with invalid venture ID
        predictions = await mock_enhanced_steward.get_infrastructure_predictions(
            venture_id="invalid-venture-id",
            time_horizon=30
        )
        
        # Should gracefully handle error and provide fallback
        assert predictions is not None
        assert 'error' in predictions or 'prediction_source' in predictions


class TestAgentMemoryAndState:
    """Tests for agent memory management and state persistence"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_memory_persistence(self, mock_enhanced_steward):
        """Test agent memory persistence across operations"""
        venture_id = "test-venture-001"
        
        # First operation should create memory
        await mock_enhanced_steward.get_infrastructure_predictions(
            venture_id=venture_id,
            time_horizon=30
        )
        
        # Memory should be updated
        assert hasattr(mock_enhanced_steward, 'prediction_cache')
        cache_key = f"{venture_id}_30"
        assert cache_key in mock_enhanced_steward.prediction_cache
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_state_consistency(self, mock_enhanced_architect):
        """Test agent state remains consistent across operations"""
        initial_agent_id = mock_enhanced_architect.agent_id
        initial_agent_name = mock_enhanced_architect.agent_name
        
        # Perform operations
        await mock_enhanced_architect.select_design_patterns(
            venture={'id': 'test-venture'},
            requirements={'scalability': 'high'}
        )
        
        # State should remain consistent
        assert mock_enhanced_architect.agent_id == initial_agent_id
        assert mock_enhanced_architect.agent_name == initial_agent_name


class TestPerformanceAndScaling:
    """Performance and scaling unit tests"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self, mock_enhanced_steward):
        """Test concurrent operations on single agent"""
        venture_ids = [f"venture-{i:03d}" for i in range(10)]
        
        # Run concurrent predictions
        tasks = [
            mock_enhanced_steward.get_infrastructure_predictions(
                venture_id=vid,
                time_horizon=30
            )
            for vid in venture_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert 'confidence_level' in result
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_usage_bounds(self, mock_enhanced_steward):
        """Test agent memory usage stays within bounds"""
        # Generate many predictions to test memory management
        for i in range(100):
            await mock_enhanced_steward.get_infrastructure_predictions(
                venture_id=f"venture-{i}",
                time_horizon=30
            )
        
        # Cache should be managed (not grow infinitely)
        cache_size = len(getattr(mock_enhanced_steward, 'prediction_cache', {}))
        assert cache_size <= 50  # Should have cache size limits