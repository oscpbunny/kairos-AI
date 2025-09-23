#!/usr/bin/env python3
"""
Project Kairos: Health Check System
Basic monitoring and health checks for Oracle and Agent operations
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED" 
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"

class HealthCheckResult:
    def __init__(self, component: str, status: HealthStatus, response_time: float, 
                 message: str = "", details: Dict[str, Any] = None):
        self.component = component
        self.status = status
        self.response_time = response_time
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'status': self.status.value,
            'response_time_ms': round(self.response_time * 1000, 2),
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'healthy': self.status == HealthStatus.HEALTHY
        }

class HealthChecker:
    """Health check system for Kairos components"""
    
    def __init__(self):
        self.checks = {}
        self.history = []
        self.max_history = 100
    
    async def check_oracle_engine(self) -> HealthCheckResult:
        """Check Oracle Engine health"""
        start_time = time.time()
        
        try:
            # Mock Oracle Engine health check without dependencies
            from simulation.oracle_engine import OracleEngine
            
            oracle = OracleEngine()
            await oracle.initialize()
            
            # Test basic functionality
            predictions = await oracle.predict_infrastructure_requirements(
                venture_id="health-check-001",
                time_horizon_days=7
            )
            
            response_time = time.time() - start_time
            
            if predictions and 'confidence_level' in predictions:
                confidence = predictions.get('confidence_level', 0)
                if confidence >= 0.6:
                    status = HealthStatus.HEALTHY
                    message = f"Oracle operational (confidence: {confidence:.2f})"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"Oracle functional but low confidence: {confidence:.2f}"
            else:
                status = HealthStatus.DEGRADED
                message = "Oracle responding but predictions incomplete"
            
            return HealthCheckResult(
                component="Oracle Engine",
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'prediction_confidence': predictions.get('confidence_level', 0),
                    'prediction_source': predictions.get('prediction_source', 'unknown')
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component="Oracle Engine",
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                message=f"Oracle Engine failed: {str(e)[:100]}",
                details={'error': str(e)}
            )
    
    async def check_enhanced_steward(self) -> HealthCheckResult:
        """Check Enhanced Steward Agent health"""
        start_time = time.time()
        
        try:
            from agents.enhanced.enhanced_steward import EnhancedStewardAgent
            
            steward = EnhancedStewardAgent(agent_name="Health-Check-Steward")
            steward.agent_id = "health-check-steward"
            
            # Test infrastructure predictions
            predictions = await steward.get_infrastructure_predictions(
                venture_id="health-check-001",
                time_horizon=7
            )
            
            response_time = time.time() - start_time
            
            if predictions and 'confidence_level' in predictions:
                confidence = predictions.get('confidence_level', 0)
                if confidence >= 0.5:
                    status = HealthStatus.HEALTHY
                    message = "Steward Agent operational"
                else:
                    status = HealthStatus.DEGRADED 
                    message = f"Steward functional but low confidence: {confidence:.2f}"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Steward not responding properly"
            
            return HealthCheckResult(
                component="Enhanced Steward",
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'oracle_connection': steward.oracle_client is not None,
                    'aws_clients': len(steward.aws_clients) > 0,
                    'prediction_confidence': predictions.get('confidence_level', 0)
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component="Enhanced Steward",
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                message=f"Steward Agent failed: {str(e)[:100]}",
                details={'error': str(e)}
            )
    
    async def check_enhanced_architect(self) -> HealthCheckResult:
        """Check Enhanced Architect Agent health"""
        start_time = time.time()
        
        try:
            from agents.enhanced.enhanced_architect import EnhancedArchitectAgent
            
            architect = EnhancedArchitectAgent(agent_name="Health-Check-Architect")
            architect.agent_id = "health-check-architect"
            
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
                venture_id="health-check-001"
            )
            
            response_time = time.time() - start_time
            
            if validation and 'confidence_level' in validation:
                confidence = validation.get('confidence_level', 0)
                if confidence >= 0.5:
                    status = HealthStatus.HEALTHY
                    message = "Architect Agent operational"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"Architect functional but low confidence: {confidence:.2f}"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Architect not responding properly"
            
            return HealthCheckResult(
                component="Enhanced Architect",
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'oracle_connection': architect.oracle_client is not None,
                    'pattern_recognition': architect.pattern_vectors is not None,
                    'design_confidence': validation.get('confidence_level', 0)
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component="Enhanced Architect",
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                message=f"Architect Agent failed: {str(e)[:100]}",
                details={'error': str(e)}
            )
    
    async def check_system_integration(self) -> HealthCheckResult:
        """Check overall system integration health"""
        start_time = time.time()
        
        try:
            # Run all component checks
            oracle_result = await self.check_oracle_engine()
            steward_result = await self.check_enhanced_steward()
            architect_result = await self.check_enhanced_architect()
            
            response_time = time.time() - start_time
            
            # Determine overall health
            healthy_components = sum(1 for result in [oracle_result, steward_result, architect_result] 
                                   if result.status == HealthStatus.HEALTHY)
            total_components = 3
            
            if healthy_components == total_components:
                status = HealthStatus.HEALTHY
                message = "All components operational"
            elif healthy_components >= 2:
                status = HealthStatus.DEGRADED
                message = f"{healthy_components}/{total_components} components healthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Multiple component failures: {healthy_components}/{total_components} healthy"
            
            return HealthCheckResult(
                component="System Integration",
                status=status,
                response_time=response_time,
                message=message,
                details={
                    'oracle_status': oracle_result.status.value,
                    'steward_status': steward_result.status.value,
                    'architect_status': architect_result.status.value,
                    'healthy_components': healthy_components,
                    'total_components': total_components
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component="System Integration",
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                message=f"Integration check failed: {str(e)[:100]}",
                details={'error': str(e)}
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results"""
        results = {}
        
        # Run checks in parallel for better performance
        oracle_task = asyncio.create_task(self.check_oracle_engine())
        steward_task = asyncio.create_task(self.check_enhanced_steward())
        architect_task = asyncio.create_task(self.check_enhanced_architect())
        integration_task = asyncio.create_task(self.check_system_integration())
        
        # Wait for all checks to complete
        oracle_result = await oracle_task
        steward_result = await steward_task
        architect_result = await architect_task
        integration_result = await integration_task
        
        results = {
            'oracle_engine': oracle_result,
            'enhanced_steward': steward_result,
            'enhanced_architect': architect_result,
            'system_integration': integration_result
        }
        
        # Store in history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'results': {k: v.to_dict() for k, v in results.items()}
        })
        
        # Trim history if too large
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return results
    
    def get_system_status(self, results: Dict[str, HealthCheckResult]) -> Dict[str, Any]:
        """Generate overall system status report"""
        healthy_count = sum(1 for result in results.values() 
                          if result.status == HealthStatus.HEALTHY)
        total_count = len(results)
        
        if healthy_count == total_count:
            overall_status = "OPERATIONAL"
            status_emoji = "✅"
        elif healthy_count >= total_count * 0.75:
            overall_status = "DEGRADED"
            status_emoji = "⚠️"
        else:
            overall_status = "CRITICAL"
            status_emoji = "❌"
        
        return {
            'overall_status': overall_status,
            'status_emoji': status_emoji,
            'healthy_components': healthy_count,
            'total_components': total_count,
            'success_rate': (healthy_count / total_count) * 100,
            'components': {k: v.to_dict() for k, v in results.items()},
            'timestamp': datetime.now().isoformat()
        }

async def main():
    """Run health checks and display results"""
    print("🏥 Kairos Health Check System")
    print("=" * 50)
    
    checker = HealthChecker()
    results = await checker.run_all_checks()
    status_report = checker.get_system_status(results)
    
    # Display results
    print(f"\n{status_report['status_emoji']} SYSTEM STATUS: {status_report['overall_status']}")
    print(f"Healthy Components: {status_report['healthy_components']}/{status_report['total_components']} ({status_report['success_rate']:.1f}%)")
    print(f"Check Time: {status_report['timestamp']}")
    
    print(f"\n📊 Component Health Details:")
    print("-" * 50)
    
    for component_name, result in results.items():
        status_icon = "✅" if result.status == HealthStatus.HEALTHY else "⚠️" if result.status == HealthStatus.DEGRADED else "❌"
        print(f"{status_icon} {result.component}: {result.status.value}")
        print(f"   Response Time: {result.response_time * 1000:.1f}ms")
        print(f"   Message: {result.message}")
        
        if result.details:
            print(f"   Details: {json.dumps(result.details, indent=6)}")
        print()
    
    if status_report['success_rate'] >= 75:
        print("🎉 Phase 5 Oracle Integration: Systems are operational!")
    else:
        print("🚨 Phase 5 Oracle Integration: Some systems need attention")
    
    return status_report['success_rate'] >= 75

if __name__ == "__main__":
    asyncio.run(main())