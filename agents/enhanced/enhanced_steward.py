"""
Project Kairos: Enhanced Steward Agent
The advanced Resource Broker with cognitive substrate integration and predictive capabilities.

Key Capabilities:
- Predictive resource scaling using machine learning
- Multi-cloud cost optimization and workload placement
- Infrastructure-as-Code deployment automation
- Real-time performance monitoring and alerting
- Intelligent bidding on infrastructure tasks
- Collaboration with other agents for resource sharing
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import os

try:
    import boto3
except ImportError:
    boto3 = None
    
import numpy as np
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    RandomForestRegressor = None
    StandardScaler = None
    
try:
    import pulumi
    from pulumi_aws import ec2, ecs, rds
except ImportError:
    pulumi = None
    ec2 = None
    ecs = None
    rds = None

from .agent_base import (
    KairosAgentBase, 
    AgentType, 
    DecisionType, 
    TaskBid, 
    AgentCapability,
    TaskType
)

logger = logging.getLogger('EnhancedSteward')

class ResourceType:
    COMPUTE = "COMPUTE"
    STORAGE = "STORAGE"
    DATABASE = "DATABASE"
    NETWORK = "NETWORK"
    CONTAINER = "CONTAINER"
    SERVERLESS = "SERVERLESS"

class WorkloadPriority:
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"

class EnhancedStewardAgent(KairosAgentBase):
    """
    Advanced Resource Broker Agent with predictive intelligence
    """
    
    def __init__(self, agent_name: str = "Enhanced-Steward", initial_cc_balance: int = 5000):
        super().__init__(
            agent_name=agent_name,
            agent_type=AgentType.STEWARD,
            specialization="Advanced Resource Management & Infrastructure Automation",
            initial_cc_balance=initial_cc_balance
        )
        
        # Initialize capabilities specific to the Steward
        self._initialize_steward_capabilities()
        
        # AWS Clients
        self.aws_clients = self._setup_aws_clients()
        
        # Machine Learning Models
        self.ml_models = {
            'demand_predictor': None,
            'cost_optimizer': None,
            'performance_forecaster': None
        }
        
        # Resource monitoring and prediction
        self.resource_metrics_history = []
        self.workload_patterns = {}
        self.cost_optimization_targets = {}
        
        # Infrastructure state tracking
        self.active_infrastructure = {}
        self.pending_deployments = {}
        self.resource_pools = {
            'compute': [],
            'storage': [],
            'database': [],
            'network': []
        }
        
        # Predictive scaling parameters
        self.scaling_thresholds = {
            'cpu_utilization': {'scale_up': 75, 'scale_down': 25},
            'memory_utilization': {'scale_up': 80, 'scale_down': 30},
            'network_io': {'scale_up': 70, 'scale_down': 20}
        }
        
        # Economic intelligence for infrastructure
        self.pricing_models = {}
        self.demand_forecasts = {}
        
        # Oracle Engine integration
        self.oracle_client = None
        self.oracle_predictions_cache = {}
        self.last_oracle_sync = None
        
    def _initialize_steward_capabilities(self):
        """Initialize Steward-specific capabilities"""
        base_time = datetime.now()
        
        self.capabilities = {
            'aws_infrastructure': AgentCapability(
                name='AWS Infrastructure Management',
                proficiency_level=0.95,
                experience_points=2500,
                last_used=base_time,
                success_rate=0.94
            ),
            'cost_optimization': AgentCapability(
                name='Multi-cloud Cost Optimization',
                proficiency_level=0.90,
                experience_points=2000,
                last_used=base_time,
                success_rate=0.92
            ),
            'predictive_scaling': AgentCapability(
                name='Predictive Resource Scaling',
                proficiency_level=0.88,
                experience_points=1800,
                last_used=base_time,
                success_rate=0.89
            ),
            'performance_monitoring': AgentCapability(
                name='Real-time Performance Monitoring',
                proficiency_level=0.93,
                experience_points=2200,
                last_used=base_time,
                success_rate=0.95
            ),
            'infrastructure_automation': AgentCapability(
                name='Infrastructure-as-Code Automation',
                proficiency_level=0.87,
                experience_points=1600,
                last_used=base_time,
                success_rate=0.91
            ),
            'workload_placement': AgentCapability(
                name='Intelligent Workload Placement',
                proficiency_level=0.85,
                experience_points=1400,
                last_used=base_time,
                success_rate=0.88
            )
        }
    
    def _setup_aws_clients(self) -> Dict[str, Any]:
        """Setup AWS service clients"""
        if not boto3:
            logger.warning("boto3 not available - AWS clients will not be initialized")
            return {}
            
        try:
            session = boto3.Session(
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )
            
            return {
                'ec2': session.client('ec2'),
                'ecs': session.client('ecs'),
                'rds': session.client('rds'),
                'cloudwatch': session.client('cloudwatch'),
                'autoscaling': session.client('autoscaling'),
                'pricing': session.client('pricing', region_name='us-east-1'),
                'cost_explorer': session.client('ce'),
                'lambda': session.client('lambda')
            }
        except Exception as e:
            logger.warning(f"Failed to setup AWS clients: {e}")
            return {}
    
    async def evaluate_task_fit(self, task: Dict[str, Any]) -> float:
        """Evaluate how well this agent fits a specific task"""
        task_description = task.get('description', '').lower()
        task_requirements = task.get('requirements', {})
        task_type = task.get('task_type', '')
        
        fit_score = 0.0
        
        # Infrastructure and resource management tasks
        infrastructure_keywords = [
            'infrastructure', 'aws', 'cloud', 'deployment', 'scaling', 'monitoring',
            'performance', 'cost', 'optimization', 'resource', 'server', 'database',
            'network', 'container', 'kubernetes', 'docker', 'terraform', 'pulumi'
        ]
        
        for keyword in infrastructure_keywords:
            if keyword in task_description:
                fit_score += 0.1
        
        # Task type matching
        if task_type in ['DEPLOYMENT', 'MONITORING', 'OPTIMIZATION']:
            fit_score += 0.3
        
        # Resource requirements analysis
        if 'compute_requirements' in task_requirements:
            fit_score += 0.2
        if 'storage_requirements' in task_requirements:
            fit_score += 0.15
        if 'performance_requirements' in task_requirements:
            fit_score += 0.2
        
        # Cost considerations
        if task.get('cc_bounty', 0) >= 500:  # High-value infrastructure tasks
            fit_score += 0.1
        
        return min(fit_score, 1.0)
    
    async def generate_task_bid(self, task: Dict[str, Any]) -> Optional[TaskBid]:
        """Generate intelligent bid for infrastructure tasks"""
        try:
            fit_score = await self.evaluate_task_fit(task)
            
            if fit_score < 0.4:  # Don't bid on tasks we're not well suited for
                return None
            
            task_complexity = self._assess_task_complexity(task)
            market_demand = await self._assess_market_demand(task)
            
            # Base bid calculation
            base_cost = self._calculate_base_cost(task, task_complexity)
            market_adjustment = base_cost * (0.5 + market_demand * 0.5)
            confidence_adjustment = fit_score * 0.2
            
            final_bid = int(market_adjustment * (1 + confidence_adjustment))
            
            # Ensure we don't bid more than we can afford
            max_affordable = int(self.cognitive_cycles_balance * 0.6)
            final_bid = min(final_bid, max_affordable)
            
            # Estimated completion time based on complexity
            base_time = 60  # 1 hour base
            estimated_time = int(base_time * (1 + task_complexity))
            
            # Risk assessment
            risk_factors = self._identify_risk_factors(task)
            
            return TaskBid(
                task_id=task['id'],
                bid_amount_cc=final_bid,
                estimated_completion_time=estimated_time,
                proposed_approach=self._generate_approach_description(task),
                confidence_score=fit_score,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Failed to generate bid: {e}")
            return None
    
    def _assess_task_complexity(self, task: Dict[str, Any]) -> float:
        """Assess task complexity (0.0 to 2.0)"""
        complexity = 0.0
        
        description = task.get('description', '').lower()
        requirements = task.get('requirements', {})
        
        # Multi-service complexity
        services = ['ec2', 'rds', 'ecs', 'lambda', 'vpc', 'elb', 'cloudfront']
        service_count = sum(1 for service in services if service in description)
        complexity += service_count * 0.2
        
        # Scale requirements
        if 'high_availability' in description or 'ha' in description:
            complexity += 0.3
        if 'multi_region' in description or 'disaster_recovery' in description:
            complexity += 0.4
        if 'auto_scaling' in description or 'autoscaling' in description:
            complexity += 0.2
        
        # Performance requirements
        if requirements.get('performance_tier') == 'high':
            complexity += 0.3
        elif requirements.get('performance_tier') == 'critical':
            complexity += 0.5
        
        return min(complexity, 2.0)
    
    async def _assess_market_demand(self, task: Dict[str, Any]) -> float:
        """Assess current market demand for similar tasks (0.0 to 1.0)"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                # Check number of bidders on similar tasks
                await cur.execute(
                    """
                    SELECT COUNT(DISTINCT agent_id) as bidder_count
                    FROM Task_Bids tb
                    JOIN Tasks t ON tb.task_id = t.id
                    WHERE t.task_type = %s
                    AND t.created_at > NOW() - INTERVAL '24 hours';
                    """,
                    (task.get('task_type', 'DEPLOYMENT'),)
                )
                
                result = await cur.fetchone()
                bidder_count = result['bidder_count'] if result else 0
                
                # Normalize demand (more bidders = higher demand)
                demand = min(bidder_count / 10.0, 1.0)
                
            conn.close()
            return demand
            
        except Exception as e:
            logger.error(f"Failed to assess market demand: {e}")
            return 0.5  # Default moderate demand
    
    def _calculate_base_cost(self, task: Dict[str, Any], complexity: float) -> int:
        """Calculate base cost for infrastructure task"""
        base_bounty = task.get('cc_bounty', 1000)
        
        # Adjust for complexity
        complexity_multiplier = 1.0 + (complexity * 0.5)
        
        # Adjust for urgency
        urgency_multiplier = 1.0
        if task.get('priority') == 'HIGH':
            urgency_multiplier = 1.2
        elif task.get('priority') == 'CRITICAL':
            urgency_multiplier = 1.5
        
            return int(base_bounty * complexity_multiplier * urgency_multiplier * 0.8)  # Competitive pricing
    
    async def get_infrastructure_predictions(self, venture_id: str, time_horizon: int = 30) -> Dict[str, Any]:
        """Get infrastructure predictions from Oracle Engine"""
        try:
            # Import here to avoid circular imports
            from simulation.oracle_engine import OracleEngine
            
            if not self.oracle_client:
                self.oracle_client = OracleEngine()
                await self.oracle_client.initialize()
            
            # Check cache first (valid for 10 minutes)
            cache_key = f"{venture_id}_{time_horizon}"
            if (cache_key in self.oracle_predictions_cache and 
                self.last_oracle_sync and 
                (datetime.now() - self.last_oracle_sync).total_seconds() < 600):
                return self.oracle_predictions_cache[cache_key]
            
            # Generate infrastructure demand predictions
            predictions = await self.oracle_client.predict_infrastructure_requirements(
                venture_id=venture_id,
                time_horizon_days=time_horizon,
                current_infrastructure=self.active_infrastructure
            )
            
            # Cache the predictions
            self.oracle_predictions_cache[cache_key] = predictions
            self.last_oracle_sync = datetime.now()
            
            logger.info(f"Steward received infrastructure predictions for venture {venture_id}")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get infrastructure predictions: {e}")
            # Return fallback predictions based on heuristics
            return self._generate_fallback_predictions(venture_id, time_horizon)
    
    def _generate_fallback_predictions(self, venture_id: str, time_horizon: int) -> Dict[str, Any]:
        """Generate fallback infrastructure predictions when Oracle is unavailable"""
        # Simple heuristic-based predictions
        return {
            'resource_requirements': {
                'compute': {'instances': 2, 'type': 't3.medium', 'scaling_factor': 1.5},
                'storage': {'size_gb': 100, 'type': 'gp3', 'growth_rate': 0.1},
                'database': {'type': 'postgresql', 'size': 'small', 'connections': 100}
            },
            'cost_predictions': {
                'monthly_estimate': 500,
                'scaling_cost': 300,
                'optimization_potential': 0.15
            },
            'scaling_recommendations': {
                'auto_scaling': True,
                'scale_up_threshold': 75,
                'scale_down_threshold': 25
            },
            'risk_factors': ['Limited prediction accuracy - Oracle unavailable'],
            'confidence_level': 0.6,
            'prediction_source': 'fallback_heuristics'
        }
    
    def _identify_risk_factors(self, task: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors for the task"""
        risks = []
        
        description = task.get('description', '').lower()
        requirements = task.get('requirements', {})
        
        if 'production' in description:
            risks.append('Production environment changes')
        
        if 'migration' in description:
            risks.append('Data migration complexity')
        
        if 'legacy' in description:
            risks.append('Legacy system integration')
        
        if requirements.get('downtime_tolerance') == 'zero':
            risks.append('Zero-downtime requirement')
        
        if 'multi_region' in description:
            risks.append('Cross-region coordination')
        
        return risks
    
    def _generate_approach_description(self, task: Dict[str, Any]) -> str:
        """Generate approach description for the task"""
        task_type = task.get('task_type', 'DEPLOYMENT')
        description = task.get('description', '')
        
        if task_type == 'DEPLOYMENT':
            return f"Automated Infrastructure-as-Code deployment using Pulumi/Terraform with CI/CD integration, monitoring setup, and rollback capabilities."
        elif task_type == 'MONITORING':
            return f"Comprehensive monitoring solution with CloudWatch, custom dashboards, intelligent alerting, and performance optimization recommendations."
        elif task_type == 'OPTIMIZATION':
            return f"AI-driven cost and performance optimization using historical data analysis, predictive scaling, and resource right-sizing recommendations."
        else:
            return f"Strategic infrastructure approach leveraging automation, monitoring, and optimization best practices."
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process infrastructure task with comprehensive approach"""
        task_id = task['id']
        task_type = task.get('task_type', 'DEPLOYMENT')
        
        logger.info(f"Steward processing task {task_id}: {task_type}")
        
        try:
            # Record decision to take on this task
            await self.record_decision(
                venture_id=task.get('venture_id'),
                decision_type=DecisionType.OPERATIONAL,
                triggered_by_event=f"Task assignment: {task_id}",
                rationale=f"Assigned to process {task_type} task based on specialization match",
                confidence_level=0.85,
                expected_outcomes={
                    'task_completion': 'successful',
                    'quality_score': 0.9,
                    'resource_efficiency': 'optimized'
                }
            )
            
            # Process based on task type
            if task_type == 'DEPLOYMENT':
                result = await self._handle_deployment_task(task)
            elif task_type == 'MONITORING':
                result = await self._handle_monitoring_task(task)
            elif task_type == 'OPTIMIZATION':
                result = await self._handle_optimization_task(task)
            else:
                result = await self._handle_general_infrastructure_task(task)
            
            # Update capability experience
            capability_name = self._get_relevant_capability(task_type)
            if capability_name in self.capabilities:
                self.capabilities[capability_name].experience_points += 100
                self.capabilities[capability_name].last_used = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process task {task_id}: {e}")
            raise
    
    async def _handle_deployment_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle infrastructure deployment tasks"""
        deployment_id = str(uuid.uuid4())
        
        # Parse requirements
        requirements = task.get('requirements', {})
        infrastructure_spec = requirements.get('infrastructure', {})
        
        # Create deployment plan
        deployment_plan = await self._create_deployment_plan(infrastructure_spec)
        
        # Validate resources
        resource_validation = await self._validate_resource_requirements(deployment_plan)
        
        if not resource_validation['valid']:
            return {
                'status': 'failed',
                'error': resource_validation['error'],
                'quality_score': 0.2
            }
        
        # Execute deployment
        deployment_result = await self._execute_deployment(deployment_plan, deployment_id)
        
        # Setup monitoring
        await self._setup_infrastructure_monitoring(deployment_id, deployment_plan)
        
        return {
            'status': 'completed',
            'deployment_id': deployment_id,
            'infrastructure_endpoints': deployment_result.get('endpoints', []),
            'monitoring_dashboard': f"https://cloudwatch.aws.amazon.com/dashboards/{deployment_id}",
            'cost_estimate': deployment_result.get('monthly_cost_estimate'),
            'deliverables': {
                'deployment_plan': deployment_plan,
                'resource_ids': deployment_result.get('resource_ids', {}),
                'access_credentials': deployment_result.get('credentials', {}),
                'documentation': f"Deployment documentation generated for {deployment_id}"
            },
            'quality_score': 0.92
        }
    
    async def _handle_monitoring_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle monitoring setup and optimization tasks"""
        monitoring_id = str(uuid.uuid4())
        
        requirements = task.get('requirements', {})
        target_resources = requirements.get('target_resources', [])
        
        # Create monitoring configuration
        monitoring_config = await self._create_monitoring_configuration(target_resources)
        
        # Deploy monitoring infrastructure
        monitoring_result = await self._deploy_monitoring_solution(monitoring_config, monitoring_id)
        
        # Setup alerting
        alert_config = await self._setup_intelligent_alerting(monitoring_config)
        
        return {
            'status': 'completed',
            'monitoring_id': monitoring_id,
            'dashboard_urls': monitoring_result.get('dashboards', []),
            'alert_configurations': alert_config,
            'metrics_collected': monitoring_config.get('metrics', []),
            'deliverables': {
                'monitoring_configuration': monitoring_config,
                'dashboard_definitions': monitoring_result.get('dashboard_configs'),
                'alert_rules': alert_config,
                'runbook': f"Monitoring runbook for {monitoring_id}"
            },
            'quality_score': 0.90
        }
    
    async def _handle_optimization_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cost and performance optimization tasks"""
        optimization_id = str(uuid.uuid4())
        
        requirements = task.get('requirements', {})
        optimization_targets = requirements.get('optimization_targets', ['cost', 'performance'])
        
        # Analyze current infrastructure
        current_analysis = await self._analyze_current_infrastructure(requirements)
        
        # Generate optimization recommendations
        recommendations = await self._generate_optimization_recommendations(
            current_analysis, optimization_targets
        )
        
        # Implement optimizations (if approved)
        implementation_result = await self._implement_optimizations(recommendations)
        
        # Calculate savings/improvements
        impact_analysis = await self._calculate_optimization_impact(
            current_analysis, implementation_result
        )
        
        return {
            'status': 'completed',
            'optimization_id': optimization_id,
            'recommendations_implemented': len(implementation_result.get('completed', [])),
            'cost_savings_projected': impact_analysis.get('cost_savings'),
            'performance_improvement': impact_analysis.get('performance_gains'),
            'deliverables': {
                'optimization_report': current_analysis,
                'recommendations': recommendations,
                'implementation_results': implementation_result,
                'impact_analysis': impact_analysis,
                'ongoing_monitoring': f"Optimization monitoring for {optimization_id}"
            },
            'quality_score': 0.88
        }
    
    async def _handle_general_infrastructure_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general infrastructure management tasks"""
        task_id = str(uuid.uuid4())
        
        # Analyze task requirements
        analysis = await self._analyze_general_task(task)
        
        # Execute appropriate actions
        execution_result = await self._execute_general_actions(analysis)
        
        return {
            'status': 'completed',
            'task_id': task_id,
            'actions_completed': execution_result.get('completed_actions', []),
            'deliverables': {
                'task_analysis': analysis,
                'execution_results': execution_result,
                'recommendations': analysis.get('recommendations', [])
            },
            'quality_score': 0.85
        }
    
    def _get_relevant_capability(self, task_type: str) -> str:
        """Get relevant capability name for task type"""
        mapping = {
            'DEPLOYMENT': 'infrastructure_automation',
            'MONITORING': 'performance_monitoring',
            'OPTIMIZATION': 'cost_optimization'
        }
        return mapping.get(task_type, 'aws_infrastructure')
    
    async def _create_deployment_plan(self, infrastructure_spec: Dict) -> Dict:
        """Create detailed deployment plan"""
        plan = {
            'id': str(uuid.uuid4()),
            'compute_resources': [],
            'storage_resources': [],
            'network_resources': [],
            'database_resources': [],
            'security_groups': [],
            'estimated_cost': 0,
            'deployment_steps': []
        }
        
        # Process compute requirements
        if 'compute' in infrastructure_spec:
            compute_config = infrastructure_spec['compute']
            plan['compute_resources'] = await self._plan_compute_resources(compute_config)
        
        # Process storage requirements
        if 'storage' in infrastructure_spec:
            storage_config = infrastructure_spec['storage']
            plan['storage_resources'] = await self._plan_storage_resources(storage_config)
        
        # Process network requirements
        if 'network' in infrastructure_spec:
            network_config = infrastructure_spec['network']
            plan['network_resources'] = await self._plan_network_resources(network_config)
        
        # Add security configurations
        plan['security_groups'] = await self._plan_security_configurations(infrastructure_spec)
        
        # Calculate estimated costs
        plan['estimated_cost'] = await self._estimate_infrastructure_cost(plan)
        
        return plan
    
    async def _plan_compute_resources(self, compute_config: Dict) -> List[Dict]:
        """Plan compute resource deployment"""
        resources = []
        
        instance_type = compute_config.get('instance_type', 't3.medium')
        instance_count = compute_config.get('count', 1)
        
        for i in range(instance_count):
            resource = {
                'type': 'ec2_instance',
                'instance_type': instance_type,
                'name': f"kairos-instance-{i+1}",
                'tags': {
                    'Project': 'Kairos',
                    'Environment': compute_config.get('environment', 'development'),
                    'ManagedBy': 'Enhanced-Steward'
                }
            }
            resources.append(resource)
        
        return resources
    
    async def _plan_storage_resources(self, storage_config: Dict) -> List[Dict]:
        """Plan storage resource deployment"""
        resources = []
        
        if storage_config.get('type') == 'ebs':
            volume = {
                'type': 'ebs_volume',
                'size': storage_config.get('size_gb', 100),
                'volume_type': storage_config.get('volume_type', 'gp3'),
                'encrypted': storage_config.get('encrypted', True)
            }
            resources.append(volume)
        
        return resources
    
    async def _plan_network_resources(self, network_config: Dict) -> List[Dict]:
        """Plan network resource deployment"""
        resources = []
        
        # VPC
        vpc = {
            'type': 'vpc',
            'cidr_block': network_config.get('vpc_cidr', '10.0.0.0/16'),
            'enable_dns_hostnames': True,
            'enable_dns_support': True
        }
        resources.append(vpc)
        
        # Subnets
        subnet_count = network_config.get('subnet_count', 2)
        for i in range(subnet_count):
            subnet = {
                'type': 'subnet',
                'cidr_block': f"10.0.{i+1}.0/24",
                'availability_zone': f"us-east-1{'a' if i % 2 == 0 else 'b'}",
                'map_public_ip_on_launch': network_config.get('public_subnets', True)
            }
            resources.append(subnet)
        
        return resources
    
    async def _plan_security_configurations(self, infrastructure_spec: Dict) -> List[Dict]:
        """Plan security group configurations"""
        security_groups = []
        
        # Default web security group
        web_sg = {
            'name': 'kairos-web-sg',
            'description': 'Security group for web services',
            'ingress_rules': [
                {'port': 80, 'protocol': 'tcp', 'cidr': '0.0.0.0/0'},
                {'port': 443, 'protocol': 'tcp', 'cidr': '0.0.0.0/0'}
            ]
        }
        security_groups.append(web_sg)
        
        # Database security group
        if 'database' in infrastructure_spec:
            db_sg = {
                'name': 'kairos-db-sg',
                'description': 'Security group for database services',
                'ingress_rules': [
                    {'port': 5432, 'protocol': 'tcp', 'source_sg': 'kairos-web-sg'}
                ]
            }
            security_groups.append(db_sg)
        
        return security_groups
    
    async def _estimate_infrastructure_cost(self, plan: Dict) -> float:
        """Estimate monthly infrastructure cost"""
        total_cost = 0.0
        
        # Estimate compute costs
        for resource in plan['compute_resources']:
            instance_type = resource.get('instance_type', 't3.medium')
            # Simple cost estimation (would use AWS Pricing API in production)
            hourly_cost = self._get_instance_hourly_cost(instance_type)
            monthly_cost = hourly_cost * 24 * 30
            total_cost += monthly_cost
        
        # Estimate storage costs
        for resource in plan['storage_resources']:
            if resource['type'] == 'ebs_volume':
                gb_cost = 0.10  # Approximate cost per GB per month
                total_cost += resource['size'] * gb_cost
        
        return total_cost
    
    def _get_instance_hourly_cost(self, instance_type: str) -> float:
        """Get approximate hourly cost for instance type"""
        # Simplified pricing (would use actual AWS pricing in production)
        pricing = {
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            't3.xlarge': 0.1664,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'c5.large': 0.085,
            'c5.xlarge': 0.17
        }
        return pricing.get(instance_type, 0.05)
    
    async def _validate_resource_requirements(self, plan: Dict) -> Dict:
        """Validate resource requirements and constraints"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check AWS service limits
            if len(plan['compute_resources']) > 20:  # EC2 instance limit check
                validation_result['errors'].append('Exceeds EC2 instance limit')
                validation_result['valid'] = False
            
            # Check estimated cost against budget
            if plan['estimated_cost'] > 10000:  # $10k monthly limit
                validation_result['warnings'].append(f"High cost estimate: ${plan['estimated_cost']:.2f}/month")
            
            # Validate network configuration
            for resource in plan['network_resources']:
                if resource['type'] == 'vpc' and not resource.get('cidr_block'):
                    validation_result['errors'].append('VPC missing CIDR block')
                    validation_result['valid'] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                'valid': False,
                'error': f'Validation failed: {str(e)}'
            }
    
    async def _execute_deployment(self, plan: Dict, deployment_id: str) -> Dict:
        """Execute infrastructure deployment"""
        # In a real implementation, this would use Pulumi/Terraform
        deployment_result = {
            'status': 'completed',
            'deployment_id': deployment_id,
            'resource_ids': {},
            'endpoints': [],
            'credentials': {},
            'monthly_cost_estimate': plan['estimated_cost']
        }
        
        # Simulate deployment process
        logger.info(f"Executing deployment {deployment_id}")
        await asyncio.sleep(2)  # Simulate deployment time
        
        # Generate mock resource IDs
        for i, resource in enumerate(plan['compute_resources']):
            resource_id = f"i-{uuid.uuid4().hex[:17]}"
            deployment_result['resource_ids'][f"instance_{i}"] = resource_id
            deployment_result['endpoints'].append(f"http://ec2-{resource_id}.compute.amazonaws.com")
        
        return deployment_result
    
    async def _setup_infrastructure_monitoring(self, deployment_id: str, plan: Dict):
        """Setup monitoring for deployed infrastructure"""
        logger.info(f"Setting up monitoring for deployment {deployment_id}")
        
        # Would setup CloudWatch dashboards, alarms, etc.
        monitoring_config = {
            'dashboards': [f"kairos-{deployment_id}"],
            'alarms': ['cpu-utilization', 'memory-utilization', 'disk-usage'],
            'log_groups': [f"/aws/ec2/kairos-{deployment_id}"]
        }
        
        return monitoring_config
    
    async def _create_monitoring_configuration(self, target_resources: List) -> Dict:
        """Create comprehensive monitoring configuration"""
        config = {
            'metrics': [
                'CPUUtilization',
                'MemoryUtilization',
                'DiskReadOps',
                'DiskWriteOps',
                'NetworkIn',
                'NetworkOut'
            ],
            'log_sources': [
                '/var/log/application.log',
                '/var/log/system.log'
            ],
            'dashboard_widgets': [
                'cpu_usage_chart',
                'memory_usage_chart',
                'disk_io_chart',
                'network_io_chart'
            ],
            'alert_thresholds': {
                'cpu_critical': 90,
                'memory_critical': 85,
                'disk_critical': 80
            }
        }
        
        return config
    
    async def _deploy_monitoring_solution(self, config: Dict, monitoring_id: str) -> Dict:
        """Deploy monitoring solution"""
        logger.info(f"Deploying monitoring solution {monitoring_id}")
        
        result = {
            'monitoring_id': monitoring_id,
            'dashboards': [
                f"https://cloudwatch.aws.amazon.com/dashboards/kairos-{monitoring_id}",
                f"https://grafana.kairos.com/d/{monitoring_id}/overview"
            ],
            'dashboard_configs': config
        }
        
        return result
    
    async def _setup_intelligent_alerting(self, config: Dict) -> Dict:
        """Setup intelligent alerting with ML-based anomaly detection"""
        alert_config = {
            'static_alerts': [],
            'anomaly_detection': [],
            'notification_channels': ['email', 'slack', 'pagerduty']
        }
        
        # Static threshold alerts
        for metric, threshold in config['alert_thresholds'].items():
            alert = {
                'name': f"kairos_{metric}_alert",
                'metric': metric,
                'threshold': threshold,
                'comparison': 'GreaterThanThreshold',
                'evaluation_periods': 2
            }
            alert_config['static_alerts'].append(alert)
        
        # Anomaly detection alerts
        for metric in ['CPUUtilization', 'NetworkIn', 'NetworkOut']:
            anomaly_alert = {
                'name': f"kairos_{metric}_anomaly",
                'metric': metric,
                'detector_type': 'ML_ANOMALY_DETECTOR',
                'sensitivity': 'HIGH'
            }
            alert_config['anomaly_detection'].append(anomaly_alert)
        
        return alert_config
    
    async def _analyze_current_infrastructure(self, requirements: Dict) -> Dict:
        """Analyze current infrastructure for optimization opportunities"""
        analysis = {
            'resource_utilization': {},
            'cost_breakdown': {},
            'performance_metrics': {},
            'optimization_opportunities': []
        }
        
        # Simulate infrastructure analysis
        analysis['resource_utilization'] = {
            'cpu_avg': 45.2,
            'memory_avg': 62.8,
            'storage_used': 78.5,
            'network_avg': 32.1
        }
        
        analysis['cost_breakdown'] = {
            'compute': 1250.00,
            'storage': 180.00,
            'network': 95.00,
            'other': 75.00,
            'total_monthly': 1600.00
        }
        
        # Identify optimization opportunities
        if analysis['resource_utilization']['cpu_avg'] < 50:
            analysis['optimization_opportunities'].append({
                'type': 'right_sizing',
                'description': 'CPU utilization low, consider smaller instances',
                'potential_savings': 300.00
            })
        
        return analysis
    
    async def _generate_optimization_recommendations(self, analysis: Dict, targets: List) -> List[Dict]:
        """Generate AI-driven optimization recommendations"""
        recommendations = []
        
        if 'cost' in targets:
            # Cost optimization recommendations
            if analysis['resource_utilization']['cpu_avg'] < 50:
                recommendations.append({
                    'type': 'cost_optimization',
                    'action': 'downsize_instances',
                    'description': 'Downsize over-provisioned instances',
                    'expected_savings': 300.00,
                    'risk_level': 'low',
                    'implementation_effort': 'medium'
                })
            
            recommendations.append({
                'type': 'cost_optimization',
                'action': 'reserved_instances',
                'description': 'Purchase reserved instances for stable workloads',
                'expected_savings': 450.00,
                'risk_level': 'low',
                'implementation_effort': 'low'
            })
        
        if 'performance' in targets:
            # Performance optimization recommendations
            if analysis['resource_utilization']['memory_avg'] > 80:
                recommendations.append({
                    'type': 'performance_optimization',
                    'action': 'increase_memory',
                    'description': 'Increase memory for memory-intensive workloads',
                    'expected_improvement': '25% response time reduction',
                    'risk_level': 'low',
                    'implementation_effort': 'low'
                })
        
        return recommendations
    
    async def _implement_optimizations(self, recommendations: List[Dict]) -> Dict:
        """Implement optimization recommendations"""
        implementation_result = {
            'completed': [],
            'failed': [],
            'pending': []
        }
        
        for rec in recommendations:
            try:
                # Simulate implementation
                await asyncio.sleep(0.5)
                
                if rec['risk_level'] == 'low':
                    implementation_result['completed'].append({
                        'recommendation_id': rec.get('id', str(uuid.uuid4())),
                        'action': rec['action'],
                        'status': 'completed',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    implementation_result['pending'].append({
                        'recommendation_id': rec.get('id', str(uuid.uuid4())),
                        'action': rec['action'],
                        'status': 'requires_approval',
                        'reason': 'High risk change requires manual approval'
                    })
                    
            except Exception as e:
                implementation_result['failed'].append({
                    'recommendation_id': rec.get('id', str(uuid.uuid4())),
                    'error': str(e)
                })
        
        return implementation_result
    
    async def _calculate_optimization_impact(self, analysis: Dict, implementation: Dict) -> Dict:
        """Calculate the impact of implemented optimizations"""
        impact = {
            'cost_savings': 0.0,
            'performance_gains': {},
            'risk_mitigation': []
        }
        
        # Calculate savings from completed optimizations
        for completed in implementation['completed']:
            if completed['action'] == 'downsize_instances':
                impact['cost_savings'] += 300.00
            elif completed['action'] == 'reserved_instances':
                impact['cost_savings'] += 450.00
        
        impact['performance_gains'] = {
            'response_time_improvement': '15%',
            'throughput_increase': '8%',
            'resource_efficiency': '22%'
        }
        
        return impact
    
    async def _analyze_general_task(self, task: Dict) -> Dict:
        """Analyze general infrastructure task requirements"""
        analysis = {
            'task_type': task.get('task_type', 'GENERAL'),
            'complexity_score': self._assess_task_complexity(task),
            'required_actions': [],
            'recommendations': []
        }
        
        description = task.get('description', '').lower()
        
        if 'backup' in description:
            analysis['required_actions'].append('setup_backup_strategy')
        if 'security' in description:
            analysis['required_actions'].append('security_audit')
        if 'performance' in description:
            analysis['required_actions'].append('performance_analysis')
        
        return analysis
    
    async def _execute_general_actions(self, analysis: Dict) -> Dict:
        """Execute general infrastructure actions"""
        result = {
            'completed_actions': [],
            'failed_actions': [],
            'recommendations_generated': []
        }
        
        for action in analysis['required_actions']:
            try:
                # Simulate action execution
                await asyncio.sleep(1)
                
                result['completed_actions'].append({
                    'action': action,
                    'status': 'completed',
                    'output': f"Successfully executed {action}",
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                result['failed_actions'].append({
                    'action': action,
                    'error': str(e)
                })
        
        return result
    
    async def agent_specific_processing(self):
        """Steward-specific processing that runs each cycle"""
        try:
            # Update resource metrics
            await self._collect_resource_metrics()
            
            # Perform predictive scaling analysis
            await self._analyze_scaling_needs()
            
            # Check for cost optimization opportunities
            await self._monitor_cost_optimization()
            
            # Update infrastructure health status
            await self._update_infrastructure_health()
            
            # Train ML models if enough data
            await self._update_ml_models()
            
        except Exception as e:
            logger.error(f"Error in Steward specific processing: {e}")
    
    async def _collect_resource_metrics(self):
        """Collect current resource utilization metrics"""
        if not self.aws_clients.get('cloudwatch'):
            return
        
        try:
            # Collect metrics from CloudWatch
            current_metrics = {
                'timestamp': datetime.now(),
                'cpu_utilization': np.random.normal(45, 15),  # Mock data
                'memory_utilization': np.random.normal(60, 20),
                'network_io': np.random.normal(30, 10),
                'disk_io': np.random.normal(25, 8)
            }
            
            self.resource_metrics_history.append(current_metrics)
            
            # Keep only last 1000 data points
            if len(self.resource_metrics_history) > 1000:
                self.resource_metrics_history = self.resource_metrics_history[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")
    
    async def _analyze_scaling_needs(self):
        """Analyze if infrastructure scaling is needed"""
        if len(self.resource_metrics_history) < 10:
            return
        
        try:
            # Get recent metrics
            recent_metrics = self.resource_metrics_history[-10:]
            
            avg_cpu = np.mean([m['cpu_utilization'] for m in recent_metrics])
            avg_memory = np.mean([m['memory_utilization'] for m in recent_metrics])
            
            scaling_decision = None
            
            if avg_cpu > self.scaling_thresholds['cpu_utilization']['scale_up']:
                scaling_decision = 'scale_up'
            elif avg_cpu < self.scaling_thresholds['cpu_utilization']['scale_down']:
                scaling_decision = 'scale_down'
            
            if scaling_decision:
                logger.info(f"Scaling recommendation: {scaling_decision} (CPU: {avg_cpu:.1f}%)")
                
                # In production, would trigger auto-scaling actions
                await self._record_scaling_decision(scaling_decision, avg_cpu, avg_memory)
                
        except Exception as e:
            logger.error(f"Failed to analyze scaling needs: {e}")
    
    async def _record_scaling_decision(self, decision: str, cpu_avg: float, memory_avg: float):
        """Record scaling decision for future analysis"""
        scaling_record = {
            'timestamp': datetime.now(),
            'decision': decision,
            'cpu_avg': cpu_avg,
            'memory_avg': memory_avg,
            'rationale': f"Auto-scaling triggered due to {decision} conditions"
        }
        
        # Would store in database for analysis
        logger.info(f"Scaling decision recorded: {scaling_record}")
    
    async def _monitor_cost_optimization(self):
        """Monitor for cost optimization opportunities"""
        try:
            # Check for unused resources
            unused_resources = await self._identify_unused_resources()
            
            if unused_resources:
                logger.info(f"Found {len(unused_resources)} potentially unused resources")
                
                # Create optimization recommendations
                for resource in unused_resources:
                    recommendation = {
                        'resource_id': resource['id'],
                        'type': 'cost_optimization',
                        'action': 'terminate_unused',
                        'potential_savings': resource.get('monthly_cost', 0),
                        'last_activity': resource.get('last_activity')
                    }
                    
                    # Would send to optimization queue
                    logger.info(f"Cost optimization opportunity: {recommendation}")
                    
        except Exception as e:
            logger.error(f"Failed to monitor cost optimization: {e}")
    
    async def _identify_unused_resources(self) -> List[Dict]:
        """Identify potentially unused resources"""
        # Mock implementation
        unused = []
        
        # Simulate finding unused resources
        if np.random.random() < 0.3:  # 30% chance of finding unused resources
            unused.append({
                'id': f"i-{uuid.uuid4().hex[:17]}",
                'type': 'ec2_instance',
                'monthly_cost': 75.00,
                'last_activity': datetime.now() - timedelta(days=7)
            })
        
        return unused
    
    async def _update_infrastructure_health(self):
        """Update overall infrastructure health status"""
        try:
            if self.resource_metrics_history:
                latest_metrics = self.resource_metrics_history[-1]
                
                health_score = 100
                
                # Deduct points for high utilization
                if latest_metrics['cpu_utilization'] > 80:
                    health_score -= 20
                if latest_metrics['memory_utilization'] > 85:
                    health_score -= 15
                
                # Update health in database
                await self._update_agent_health_score(health_score)
                
        except Exception as e:
            logger.error(f"Failed to update infrastructure health: {e}")
    
    async def _update_agent_health_score(self, health_score: int):
        """Update agent health score in database"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE Agents 
                    SET metadata = COALESCE(metadata, '{}')::jsonb || %s::jsonb
                    WHERE id = %s;
                    """,
                    (json.dumps({'health_score': health_score}), self.agent_id)
                )
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update health score: {e}")
    
    async def _update_ml_models(self):
        """Update machine learning models with new data"""
        if len(self.resource_metrics_history) < 50:  # Need minimum data
            return
        
        try:
            # Prepare training data
            features = []
            targets = []
            
            for i in range(10, len(self.resource_metrics_history)):
                # Use last 10 data points as features
                feature_window = self.resource_metrics_history[i-10:i]
                feature_vector = [
                    np.mean([m['cpu_utilization'] for m in feature_window]),
                    np.mean([m['memory_utilization'] for m in feature_window]),
                    np.mean([m['network_io'] for m in feature_window])
                ]
                features.append(feature_vector)
                
                # Target is next data point's CPU utilization
                targets.append(self.resource_metrics_history[i]['cpu_utilization'])
            
            if len(features) >= 20:  # Minimum for training
                # Train demand predictor
                if self.ml_models['demand_predictor'] is None:
                    self.ml_models['demand_predictor'] = RandomForestRegressor(n_estimators=100)
                
                X = np.array(features)
                y = np.array(targets)
                
                self.ml_models['demand_predictor'].fit(X, y)
                
                logger.info("Updated ML demand prediction model")
                
        except Exception as e:
            logger.error(f"Failed to update ML models: {e}")
    
    async def predict_resource_demand(self, time_horizon_minutes: int = 60) -> Dict:
        """Predict resource demand for given time horizon"""
        if self.ml_models['demand_predictor'] is None or len(self.resource_metrics_history) < 10:
            return {'error': 'Insufficient data for prediction'}
        
        try:
            # Prepare current state as features
            recent_metrics = self.resource_metrics_history[-10:]
            current_features = [[
                np.mean([m['cpu_utilization'] for m in recent_metrics]),
                np.mean([m['memory_utilization'] for m in recent_metrics]),
                np.mean([m['network_io'] for m in recent_metrics])
            ]]
            
            # Predict
            prediction = self.ml_models['demand_predictor'].predict(current_features)[0]
            
            return {
                'predicted_cpu_utilization': prediction,
                'time_horizon_minutes': time_horizon_minutes,
                'confidence': 0.75,  # Would calculate actual confidence
                'recommendation': 'scale_up' if prediction > 80 else 'maintain' if prediction > 40 else 'scale_down'
            }
            
        except Exception as e:
            logger.error(f"Failed to predict resource demand: {e}")
            return {'error': str(e)}
    
    # Oracle Integration Methods
    async def _process_oracle_prediction(self, prediction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process Oracle predictions for infrastructure decisions"""
        try:
            # Extract infrastructure-specific predictions
            recommendations = prediction.get('recommendations', {})
            analysis = prediction.get('analysis', {})
            confidence = prediction.get('confidence_scores', {}).get('overall', 0)
            
            # Determine action based on prediction and current context
            action = 'maintain'  # default
            
            if confidence > 0.8:
                # High confidence predictions
                current_load = context.get('current_load', 0.5)
                expected_growth = context.get('expected_growth', 0.1)
                
                if current_load + expected_growth > 0.8:
                    action = 'scale_up'
                elif recommendations.get('cost_optimization_potential', 0) > 0.2:
                    action = 'optimize_costs'
            
            decision = {
                'action': action,
                'confidence': confidence,
                'reasoning': f"Based on {current_load:.2f} current load and {expected_growth:.2f} expected growth",
                'oracle_prediction_id': prediction.get('prediction_id'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Record decision in causal ledger
            await self._record_oracle_decision(decision, prediction, context)
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to process Oracle prediction: {e}")
            return {
                'action': 'maintain',
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def _handle_black_swan_event(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Black Swan event predictions"""
        try:
            event_params = prediction.get('parameters', {})
            event_type = event_params.get('event_type', 'unknown')
            severity = event_params.get('severity', 0.5)
            
            # Determine emergency response based on event type and severity
            emergency_actions = {
                'market_crash': 'emergency_scale_down',
                'supply_chain_disruption': 'backup_activation', 
                'cyber_attack': 'security_lockdown',
                'natural_disaster': 'disaster_recovery'
            }
            
            emergency_action = emergency_actions.get(event_type, 'general_emergency')
            
            response = {
                'emergency_action': emergency_action,
                'severity_level': severity,
                'event_type': event_type,
                'immediate_steps': [
                    'Scale down non-critical resources',
                    'Activate backup systems',
                    'Alert operations team',
                    'Implement cost controls'
                ],
                'timeline': '15 minutes',
                'confidence': 0.95
            }
            
            # Log emergency response
            logger.warning(f"🚨 Black Swan {event_type} detected - executing {emergency_action}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to handle Black Swan event: {e}")
            return {
                'emergency_action': 'general_emergency',
                'error': str(e)
            }
    
    async def _record_oracle_decision(self, decision: Dict[str, Any], prediction: Dict[str, Any], context: Dict[str, Any]):
        """Record Oracle-based decision in causal ledger"""
        try:
            decision_record = {
                'agent_id': self.agent_id,
                'decision_type': 'oracle_prediction',
                'decision_data': decision,
                'prediction_data': prediction,
                'context': context,
                'timestamp': datetime.now()
            }
            
            # Insert into decisions table
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO Decisions (agent_id, decision_type, decision_data, timestamp)
                    VALUES (%s, %s, %s, %s)
                """, 
                (self.agent_id, 'oracle_infrastructure', 
                 json.dumps(decision_record), datetime.now()))
            conn.commit()
            conn.close()
                
            logger.info(f"📝 Oracle decision recorded: {decision['action']}")
            
        except Exception as e:
            logger.error(f"Failed to record Oracle decision: {e}")
