#!/usr/bin/env python3
"""
End-to-End Workflow Validation Helper Methods
Supporting methods for comprehensive workflow testing.

Author: Kairos Development Team
Version: 2.0
"""

import asyncio
import json
import logging
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class WorkflowHelpers:
    """Helper methods for E2E workflow validation"""
    
    def __init__(self, validator):
        """Initialize with reference to main validator"""
        self.validator = validator
    
    async def oracle_guided_venture_planning(self, venture_config: Dict[str, Any]) -> Dict[str, Any]:
        """Plan venture using Oracle predictions"""
        try:
            # Request Oracle prediction for venture planning
            planning_request = {
                'scenario_type': 'venture_planning',
                'parameters': {
                    'complexity': venture_config.get('priority', 3),
                    'estimated_cc': venture_config.get('estimated_cc', 1000),
                    'task_count': len(venture_config.get('tasks', []))
                },
                'prediction_horizon': timedelta(weeks=4)
            }
            
            prediction = await self.validator.oracle_engine.generate_prediction(planning_request)
            
            # Optimize venture based on prediction
            optimized_config = venture_config.copy()
            
            # Adjust CC allocation based on Oracle recommendation
            if 'recommendations' in prediction:
                rec = prediction['recommendations']
                if 'cost_adjustment' in rec:
                    optimized_config['estimated_cc'] *= rec['cost_adjustment']
                
                # Reorder tasks based on priority recommendations
                if 'task_priority_order' in rec and 'tasks' in optimized_config:
                    tasks = optimized_config['tasks']
                    optimized_tasks = []
                    
                    # Sort by Oracle recommended priority
                    for i, task in enumerate(tasks):
                        task['oracle_priority'] = rec['task_priority_order'].get(str(i), i)
                        optimized_tasks.append(task)
                    
                    optimized_config['tasks'] = sorted(optimized_tasks, 
                                                     key=lambda t: t.get('oracle_priority', 999))
            
            return {
                'optimized_config': optimized_config,
                'predictions_used': 1,
                'optimization_applied': True
            }
            
        except Exception as e:
            logger.error(f"Oracle-guided planning failed: {e}")
            return {
                'optimized_config': venture_config,
                'predictions_used': 0,
                'optimization_applied': False,
                'error': str(e)
            }
    
    async def create_execution_plan(self, task_assignments: Dict[str, Any], scenario) -> List[Dict[str, Any]]:
        """Create coordinated execution plan for tasks"""
        try:
            # Simple dependency-based execution plan
            # Phase 1: Architecture tasks
            # Phase 2: Infrastructure tasks
            # Phase 3: Development tasks
            
            phases = []
            
            # Group tasks by agent type
            architect_tasks = {}
            steward_tasks = {}
            engineer_tasks = {}
            
            for task_id, agent in task_assignments.items():
                if 'architect' in agent.agent_id:
                    architect_tasks[task_id] = agent
                elif 'steward' in agent.agent_id:
                    steward_tasks[task_id] = agent
                elif 'engineer' in agent.agent_id:
                    engineer_tasks[task_id] = agent
            
            # Create execution phases
            if architect_tasks:
                phases.append(architect_tasks)
            if steward_tasks:
                phases.append(steward_tasks)
            if engineer_tasks:
                phases.append(engineer_tasks)
            
            return phases
            
        except Exception as e:
            logger.error(f"Execution plan creation failed: {e}")
            # Return all tasks in single phase as fallback
            return [task_assignments]
    
    async def execute_coordinated_task(self, task_id: str, agent, scenario) -> Dict[str, Any]:
        """Execute a task with coordination"""
        try:
            start_time = datetime.now()
            result = {
                'task_id': task_id,
                'agent_id': agent.agent_id,
                'oracle_used': False,
                'completed': False,
                'transactions': 0,
                'coordination_events': 0
            }
            
            # Check if Oracle consultation needed
            if scenario.venture_config.get('require_oracle', False):
                prediction = await self.validator._request_oracle_prediction(task_id, 'execution')
                result['oracle_used'] = True
            
            # Execute task through standard workflow
            bid_result = await self.validator._agent_bid_on_task(agent, task_id)
            if bid_result['success']:
                result['transactions'] += 1
                
                execution_result = await self.validator._execute_task(agent, task_id)
                result['completed'] = execution_result['success']
                result['coordination_events'] += 1
            
            end_time = datetime.now()
            result['execution_time'] = (end_time - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"Coordinated task execution failed: {e}")
            return {
                'task_id': task_id,
                'error': str(e),
                'oracle_used': False,
                'completed': False,
                'transactions': 0
            }
    
    async def assess_coordination_quality(self, task_assignments: Dict[str, Any], 
                                        result: Dict[str, Any]) -> float:
        """Assess quality of multi-agent coordination"""
        try:
            # Factors for coordination quality:
            # 1. Task completion rate
            # 2. Agent utilization balance
            # 3. Communication efficiency
            # 4. Resource sharing
            
            completion_rate = result.get('tasks_completed', 0) / max(len(task_assignments), 1)
            
            # Calculate agent utilization balance
            agent_usage = {}
            for agent in task_assignments.values():
                agent_id = agent.agent_id
                agent_usage[agent_id] = agent_usage.get(agent_id, 0) + 1
            
            if agent_usage:
                usage_values = list(agent_usage.values())
                usage_balance = 1 - (max(usage_values) - min(usage_values)) / max(max(usage_values), 1)
            else:
                usage_balance = 1.0
            
            # Base coordination score
            coordination_score = (completion_rate * 0.6 + usage_balance * 0.4)
            
            # Bonus for successful multi-agent coordination
            if len(set(agent.agent_id for agent in task_assignments.values())) > 1:
                coordination_score *= 1.1  # 10% bonus for actual multi-agent work
            
            return min(1.0, coordination_score)
            
        except Exception as e:
            logger.error(f"Coordination assessment failed: {e}")
            return 0.5  # Default moderate score
    
    async def oracle_guided_agent_selection(self, task_config: Dict[str, Any], 
                                          prediction: Dict[str, Any]) -> Any:
        """Select agent based on Oracle recommendation"""
        try:
            # Default selection based on role
            required_role = task_config.get('required_role', 'engineer')
            default_agent = self.validator._select_agent_for_task(required_role)
            
            # Check if Oracle has agent recommendations
            if 'recommendations' in prediction:
                rec = prediction['recommendations']
                if 'preferred_agent_type' in rec:
                    preferred_type = rec['preferred_agent_type']
                    oracle_agent = self.validator._select_agent_for_task(preferred_type)
                    
                    # Use Oracle recommendation if confidence is high
                    confidence = prediction.get('confidence_scores', {}).get('overall', 0)
                    if confidence > 0.8:
                        return oracle_agent
            
            return default_agent
            
        except Exception as e:
            logger.error(f"Oracle-guided agent selection failed: {e}")
            return self.validator._select_agent_for_task('engineer')
    
    async def monitored_task_execution(self, task_id: str, agent, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with monitoring and adaptation"""
        try:
            start_time = datetime.now()
            result = {
                'task_id': task_id,
                'completed': False,
                'transactions': 0,
                'decisions_made': 1,
                'adaptation_needed': False,
                'adaptation_data': {},
                'monitoring_data': {}
            }
            
            # Execute task with monitoring
            bid_result = await self.validator._agent_bid_on_task(agent, task_id)
            if bid_result['success']:
                result['transactions'] += 1
                
                # Monitor execution
                execution_result = await self.validator._execute_task(agent, task_id)
                result['completed'] = execution_result['success']
                
                # Check if adaptation needed based on prediction
                if prediction.get('confidence_scores', {}).get('overall', 1.0) < 0.7:
                    result['adaptation_needed'] = True
                    result['adaptation_data'] = {
                        'low_confidence': True,
                        'recommended_changes': prediction.get('recommendations', {})
                    }
                
                # Record monitoring data
                end_time = datetime.now()
                result['monitoring_data'] = {
                    'execution_time': (end_time - start_time).total_seconds(),
                    'prediction_accuracy': self._assess_prediction_accuracy(prediction, execution_result),
                    'agent_performance': execution_result.get('execution_time', 0)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Monitored task execution failed: {e}")
            return {
                'task_id': task_id,
                'completed': False,
                'transactions': 0,
                'decisions_made': 0,
                'error': str(e)
            }
    
    def _assess_prediction_accuracy(self, prediction: Dict[str, Any], 
                                  execution_result: Dict[str, Any]) -> float:
        """Assess how accurate the Oracle prediction was"""
        try:
            # Compare predicted vs actual execution time
            predicted_time = prediction.get('analysis', {}).get('estimated_duration', 2.0)
            actual_time = execution_result.get('execution_time', 0)
            
            if predicted_time > 0:
                time_accuracy = 1 - abs(predicted_time - actual_time) / predicted_time
                return max(0, min(1, time_accuracy))
            
            return 0.8  # Default moderate accuracy
            
        except Exception as e:
            logger.debug(f"Prediction accuracy assessment failed: {e}")
            return 0.5
    
    async def simulate_black_swan_event(self, venture_id: str) -> Dict[str, Any]:
        """Simulate a Black Swan event and measure response"""
        try:
            logger.info(f"🚨 Simulating Black Swan event for venture {venture_id}")
            
            # Choose random Black Swan event type
            event_types = ['market_crash', 'supply_chain_disruption', 'cyber_attack', 'natural_disaster']
            event_type = random.choice(event_types)
            
            # Create Black Swan prediction
            black_swan_request = {
                'scenario_type': 'crisis',
                'parameters': {
                    'event_type': event_type,
                    'severity': random.uniform(0.7, 0.95),
                    'affected_systems': ['infrastructure', 'development', 'economy'],
                    'response_time_limit': 60  # seconds
                },
                'prediction_horizon': timedelta(hours=1),
                'urgency': 'critical'
            }
            
            start_time = datetime.now()
            prediction = await self.validator.oracle_engine.generate_prediction(black_swan_request)
            
            # Test agent responses
            emergency_decisions = 0
            
            # Steward response
            if self.validator.steward:
                steward_response = await self.validator.steward._handle_black_swan_event(prediction)
                if steward_response.get('emergency_action'):
                    emergency_decisions += 1
            
            # Architect response
            if self.validator.architect:
                architect_response = await self.validator.architect._handle_black_swan_event(prediction)
                if architect_response.get('system_adaptation'):
                    emergency_decisions += 1
            
            # Engineer response
            if self.validator.engineer:
                engineer_response = await self.validator.engineer._handle_black_swan_event(prediction)
                if engineer_response.get('continuity_plan'):
                    emergency_decisions += 1
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            result = {
                'event_type': event_type,
                'oracle_predictions': 1,
                'emergency_decisions': emergency_decisions,
                'response_time': response_time,
                'agents_responded': emergency_decisions,
                'system_resilience_score': min(1.0, emergency_decisions / 3.0)  # 3 agents max
            }
            
            logger.info(f"🎯 Black Swan response: {emergency_decisions}/3 agents responded in {response_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Black Swan simulation failed: {e}")
            return {
                'event_type': 'unknown',
                'oracle_predictions': 0,
                'emergency_decisions': 0,
                'error': str(e)
            }
    
    async def execute_adaptation_cycle(self, task_id: str, adaptation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute adaptation cycle based on monitoring data"""
        try:
            logger.info(f"🔄 Executing adaptation cycle for task {task_id}")
            
            result = {
                'task_id': task_id,
                'oracle_consultations': 0,
                'adaptations_applied': 0,
                'success': False
            }
            
            # Request Oracle consultation for adaptation
            adaptation_request = {
                'scenario_type': 'adaptation',
                'parameters': adaptation_data,
                'prediction_horizon': timedelta(minutes=30)
            }
            
            adaptation_prediction = await self.validator.oracle_engine.generate_prediction(adaptation_request)
            result['oracle_consultations'] += 1
            
            # Apply adaptations based on prediction
            if 'recommendations' in adaptation_prediction:
                recommendations = adaptation_prediction['recommendations']
                
                # Simulate applying adaptations
                if 'resource_reallocation' in recommendations:
                    result['adaptations_applied'] += 1
                    
                if 'approach_modification' in recommendations:
                    result['adaptations_applied'] += 1
                    
                if 'timeline_adjustment' in recommendations:
                    result['adaptations_applied'] += 1
            
            result['success'] = result['adaptations_applied'] > 0
            
            logger.info(f"✅ Applied {result['adaptations_applied']} adaptations for task {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Adaptation cycle failed: {e}")
            return {
                'task_id': task_id,
                'oracle_consultations': 0,
                'adaptations_applied': 0,
                'success': False,
                'error': str(e)
            }
    
    async def create_concurrent_venture(self, venture_config: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Create and execute a venture concurrently"""
        try:
            start_time = datetime.now()
            
            # Create venture
            venture_id = await self.validator._create_test_venture(venture_config)
            
            result = {
                'venture_id': venture_id,
                'venture_index': index,
                'tasks_created': 0,
                'tasks_completed': 0,
                'oracle_predictions': 0,
                'cc_transactions': 0,
                'decisions_made': 0,
                'response_times': []
            }
            
            # Create and execute tasks
            for i, task_config in enumerate(venture_config.get('tasks', [])):
                task_start = datetime.now()
                
                # Create task
                task_id = await self.validator._create_test_task(venture_id, task_config)
                result['tasks_created'] += 1
                
                # Select agent (round-robin for load balancing)
                agents = [self.validator.steward, self.validator.architect, self.validator.engineer]
                agent = agents[i % len(agents)]
                
                # Execute task
                bid_result = await self.validator._agent_bid_on_task(agent, task_id)
                if bid_result['success']:
                    result['cc_transactions'] += 1
                    
                    execution_result = await self.validator._execute_task(agent, task_id)
                    if execution_result['success']:
                        result['tasks_completed'] += 1
                        result['decisions_made'] += 1
                
                task_end = datetime.now()
                result['response_times'].append((task_end - task_start).total_seconds())
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Concurrent venture {index} completed: {result['tasks_completed']}/{result['tasks_created']} tasks in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Concurrent venture {index} failed: {e}")
            return {
                'venture_index': index,
                'tasks_created': 0,
                'tasks_completed': 0,
                'oracle_predictions': 0,
                'cc_transactions': 0,
                'decisions_made': 0,
                'response_times': [],
                'error': str(e)
            }
    
    async def assess_system_performance_under_load(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system performance under extreme load"""
        try:
            assessment = {
                'throughput_score': 0.0,
                'reliability_score': 0.0,
                'scalability_score': 0.0,
                'resource_efficiency': 0.0
            }
            
            # Calculate throughput score
            if result['max_concurrent_tasks'] > 0:
                completion_rate = result['tasks_completed'] / result['max_concurrent_tasks']
                assessment['throughput_score'] = min(1.0, completion_rate)
            
            # Calculate reliability score
            if result['concurrent_workflows'] > 0:
                success_rate = (result['concurrent_workflows'] - result['system_errors']) / result['concurrent_workflows']
                assessment['reliability_score'] = max(0.0, success_rate)
            
            # Calculate scalability score based on response times
            if result['average_response_time'] > 0:
                # Good if average response time is under 3 seconds
                scalability_score = max(0, 1 - (result['average_response_time'] - 3.0) / 10.0)
                assessment['scalability_score'] = min(1.0, max(0.0, scalability_score))
            
            # Calculate resource efficiency
            if result['cc_transactions'] > 0:
                efficiency = result['tasks_completed'] / result['cc_transactions']
                assessment['resource_efficiency'] = min(1.0, efficiency)
            
            # Overall performance score
            scores = list(assessment.values())
            assessment['overall_performance'] = sum(scores) / len(scores) if scores else 0.0
            
            return assessment
            
        except Exception as e:
            logger.error(f"Performance assessment failed: {e}")
            return {
                'throughput_score': 0.0,
                'reliability_score': 0.0,
                'scalability_score': 0.0,
                'resource_efficiency': 0.0,
                'overall_performance': 0.0,
                'error': str(e)
            }