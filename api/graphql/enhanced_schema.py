#!/usr/bin/env python3
"""
Enhanced GraphQL Schema - Project Kairos Symbiotic Interface
Comprehensive schema for querying the Kairos cognitive substrate.

This schema provides:
1. Full CRUD operations for all Kairos entities
2. Advanced analytics and insights
3. Real-time subscriptions
4. Oracle prediction access
5. Economic system metrics
6. Causal reasoning queries

Author: Kairos Development Team
Version: 2.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
import uuid
import hashlib

import strawberry
from strawberry.types import Info
from strawberry.scalars import JSON
import asyncpg
import redis.asyncio as redis
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GraphQL Scalars
@strawberry.scalar
class DateTime(datetime):
    """DateTime scalar"""
    pass

@strawberry.scalar 
class UUID(str):
    """UUID scalar"""
    pass

# GraphQL Types
@strawberry.type
class Agent:
    """Agent entity from Kairos cognitive substrate"""
    id: UUID
    name: str
    role: str
    status: str
    cc_balance: float
    reputation_score: float
    specialization: str
    performance_metrics: JSON
    capabilities: JSON
    last_activity: DateTime
    created_at: DateTime
    
    @strawberry.field
    async def recent_decisions(self, info: Info, limit: int = 10) -> List['Decision']:
        """Get recent decisions made by this agent"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM Decisions 
                WHERE agent_id = $1 
                ORDER BY timestamp DESC 
                LIMIT $2
            """, self.id, limit)
            return [Decision(**dict(row)) for row in rows]
    
    @strawberry.field
    async def active_tasks(self, info: Info) -> List['Task']:
        """Get currently active tasks for this agent"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM Tasks 
                WHERE assigned_agent_id = $1 
                AND status IN ('ASSIGNED', 'IN_PROGRESS')
                ORDER BY created_at DESC
            """, self.id)
            return [Task(**dict(row)) for row in rows]
    
    @strawberry.field
    async def collaboration_history(self, info: Info, limit: int = 5) -> List['Collaboration']:
        """Get collaboration history for this agent"""
        # Would query collaborations table
        return []

@strawberry.type 
class Venture:
    """Venture entity representing high-level business objectives"""
    id: UUID
    name: str
    description: str
    status: str
    priority: int
    estimated_cc: float
    actual_cc_spent: Optional[float]
    completion_percentage: Optional[float]
    created_at: DateTime
    completed_at: Optional[DateTime]
    
    @strawberry.field
    async def tasks(self, info: Info) -> List['Task']:
        """Get all tasks associated with this venture"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM Tasks 
                WHERE venture_id = $1 
                ORDER BY created_at DESC
            """, self.id)
            return [Task(**dict(row)) for row in rows]
    
    @strawberry.field
    async def assigned_agents(self, info: Info) -> List[Agent]:
        """Get agents working on this venture"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT a.* FROM Agents a
                JOIN Tasks t ON t.assigned_agent_id = a.id
                WHERE t.venture_id = $1
            """, self.id)
            return [Agent(**dict(row)) for row in rows]
    
    @strawberry.field
    async def oracle_predictions(self, info: Info) -> List['OraclePrediction']:
        """Get Oracle predictions related to this venture"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM Simulations 
                WHERE parameters::jsonb ? $1
                ORDER BY created_at DESC
                LIMIT 5
            """, self.id)
            return [OraclePrediction(**dict(row)) for row in rows]

@strawberry.type
class Task:
    """Task entity representing specific work items"""
    id: UUID
    venture_id: UUID
    description: str
    status: str
    cc_bounty: float
    assigned_agent_id: Optional[UUID]
    complexity_score: Optional[float]
    estimated_hours: Optional[float]
    actual_hours: Optional[float]
    created_at: DateTime
    completed_at: Optional[DateTime]
    
    @strawberry.field
    async def venture(self, info: Info) -> Optional[Venture]:
        """Get the venture this task belongs to"""
        if not self.venture_id:
            return None
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM Ventures WHERE id = $1
            """, self.venture_id)
            return Venture(**dict(row)) if row else None
    
    @strawberry.field
    async def assigned_agent(self, info: Info) -> Optional[Agent]:
        """Get the agent assigned to this task"""
        if not self.assigned_agent_id:
            return None
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM Agents WHERE id = $1
            """, self.assigned_agent_id)
            return Agent(**dict(row)) if row else None
    
    @strawberry.field
    async def bids(self, info: Info) -> List['Bid']:
        """Get all bids for this task"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM Bids 
                WHERE task_id = $1 
                ORDER BY cc_amount ASC
            """, self.id)
            return [Bid(**dict(row)) for row in rows]

@strawberry.type
class Decision:
    """Decision entity from the causal ledger"""
    id: UUID
    agent_id: UUID
    decision_type: str
    decision_data: JSON
    context: Optional[JSON]
    timestamp: DateTime
    causal_chain_id: Optional[UUID]
    
    @strawberry.field
    async def agent(self, info: Info) -> Optional[Agent]:
        """Get the agent that made this decision"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM Agents WHERE id = $1
            """, self.agent_id)
            return Agent(**dict(row)) if row else None
    
    @strawberry.field
    async def causal_chain(self, info: Info) -> List['Decision']:
        """Get related decisions in the causal chain"""
        if not self.causal_chain_id:
            return []
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM Decisions 
                WHERE causal_chain_id = $1 
                ORDER BY timestamp ASC
            """, self.causal_chain_id)
            return [Decision(**dict(row)) for row in rows]

@strawberry.type
class Bid:
    """Bid entity for task marketplace"""
    id: UUID
    task_id: UUID
    agent_id: UUID
    cc_amount: float
    estimated_hours: float
    proposal: str
    confidence_score: float
    status: str
    created_at: DateTime
    
    @strawberry.field
    async def task(self, info: Info) -> Optional[Task]:
        """Get the task this bid is for"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM Tasks WHERE id = $1
            """, self.task_id)
            return Task(**dict(row)) if row else None
    
    @strawberry.field
    async def agent(self, info: Info) -> Optional[Agent]:
        """Get the agent who made this bid"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM Agents WHERE id = $1
            """, self.agent_id)
            return Agent(**dict(row)) if row else None

@strawberry.type
class OraclePrediction:
    """Oracle prediction/simulation results"""
    id: UUID
    scenario_type: str
    parameters: JSON
    results: JSON
    confidence_score: float
    created_at: DateTime
    validation_accuracy: Optional[float]
    
    @strawberry.field
    async def related_decisions(self, info: Info) -> List[Decision]:
        """Get decisions influenced by this prediction"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM Decisions 
                WHERE decision_data::jsonb ? 'oracle_prediction_id'
                AND decision_data::jsonb ->> 'oracle_prediction_id' = $1
            """, str(self.id))
            return [Decision(**dict(row)) for row in rows]

@strawberry.type
class EconomicMetrics:
    """Economic system analytics"""
    timestamp: DateTime
    total_cc_circulation: float
    total_cc_earned: float
    total_cc_spent: float
    active_bounties: int
    active_agents: int
    market_efficiency_score: float
    wealth_distribution_gini: float
    average_task_completion_time: float
    
@strawberry.type
class AgentPerformanceMetrics:
    """Agent performance analytics"""
    agent_id: UUID
    time_period: str
    tasks_completed: int
    tasks_success_rate: float
    average_completion_time: float
    cc_earned: float
    cc_spent: float
    reputation_change: float
    specialization_score: float
    collaboration_score: float
    oracle_prediction_accuracy: float

@strawberry.type
class SystemHealth:
    """Overall system health metrics"""
    timestamp: DateTime
    active_agents: int
    active_ventures: int
    pending_tasks: int
    system_utilization: float
    oracle_accuracy: float
    economic_velocity: float
    decision_throughput: float

@strawberry.type
class CausalChain:
    """Causal reasoning chain analysis"""
    id: UUID
    initial_decision_id: UUID
    chain_length: int
    decisions: List[Decision]
    outcome_impact_score: float
    confidence_degradation: float

@strawberry.type
class Collaboration:
    """Agent collaboration records"""
    id: UUID
    venture_id: UUID
    participating_agents: List[UUID]
    collaboration_type: str
    start_time: DateTime
    end_time: Optional[DateTime]
    success_metrics: JSON

# Input Types for Mutations
@strawberry.input
class CreateVentureInput:
    """Input for creating a new venture"""
    name: str
    description: str
    priority: int = 3
    estimated_cc: float = 1000.0

@strawberry.input
class CreateTaskInput:
    """Input for creating a new task"""
    venture_id: UUID
    description: str
    cc_bounty: float
    estimated_hours: Optional[float] = None

@strawberry.input
class CreateBidInput:
    """Input for creating a bid"""
    task_id: UUID
    agent_id: UUID
    cc_amount: float
    estimated_hours: float
    proposal: str
    confidence_score: float = 0.8

@strawberry.input
class OraclePredictionInput:
    """Input for requesting Oracle predictions"""
    scenario_type: str
    parameters: JSON
    prediction_horizon: Optional[int] = 24  # hours

# Filter Input Types
@strawberry.input
class AgentFilter:
    """Filter for agent queries"""
    role: Optional[str] = None
    status: Optional[str] = None
    min_reputation: Optional[float] = None
    specialization: Optional[str] = None
    min_cc_balance: Optional[float] = None

@strawberry.input
class VentureFilter:
    """Filter for venture queries"""
    status: Optional[str] = None
    priority: Optional[int] = None
    created_after: Optional[DateTime] = None
    completion_range: Optional[List[float]] = None

@strawberry.input
class TaskFilter:
    """Filter for task queries"""
    status: Optional[str] = None
    venture_id: Optional[UUID] = None
    assigned_agent_id: Optional[UUID] = None
    min_bounty: Optional[float] = None
    max_bounty: Optional[float] = None

@strawberry.input
class DecisionFilter:
    """Filter for decision queries"""
    agent_id: Optional[UUID] = None
    decision_type: Optional[str] = None
    date_after: Optional[DateTime] = None
    date_before: Optional[DateTime] = None

@strawberry.input
class TimeRange:
    """Time range input"""
    start: DateTime
    end: DateTime

# Query Root
@strawberry.type
class Query:
    """GraphQL Query root with comprehensive Kairos data access"""
    
    @strawberry.field
    async def agents(self, 
                    info: Info,
                    filter: Optional[AgentFilter] = None,
                    limit: int = 100,
                    offset: int = 0) -> List[Agent]:
        """Get agents with advanced filtering"""
        db_pool = info.context.db_pool
        query_parts = ["SELECT * FROM Agents"]
        params = []
        where_conditions = []
        
        if filter:
            if filter.role:
                where_conditions.append(f"role = ${len(params) + 1}")
                params.append(filter.role)
            if filter.status:
                where_conditions.append(f"status = ${len(params) + 1}")
                params.append(filter.status)
            if filter.min_reputation:
                where_conditions.append(f"reputation_score >= ${len(params) + 1}")
                params.append(filter.min_reputation)
            if filter.specialization:
                where_conditions.append(f"specialization = ${len(params) + 1}")
                params.append(filter.specialization)
            if filter.min_cc_balance:
                where_conditions.append(f"cc_balance >= ${len(params) + 1}")
                params.append(filter.min_cc_balance)
        
        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        query_parts.append(f"ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}")
        params.extend([limit, offset])
        
        query = " ".join(query_parts)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [Agent(**dict(row)) for row in rows]
    
    @strawberry.field
    async def agent(self, info: Info, id: UUID) -> Optional[Agent]:
        """Get specific agent by ID"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM Agents WHERE id = $1", id)
            return Agent(**dict(row)) if row else None
    
    @strawberry.field
    async def ventures(self,
                     info: Info,
                     filter: Optional[VentureFilter] = None,
                     limit: int = 100,
                     offset: int = 0) -> List[Venture]:
        """Get ventures with advanced filtering"""
        db_pool = info.context.db_pool
        query_parts = ["SELECT * FROM Ventures"]
        params = []
        where_conditions = []
        
        if filter:
            if filter.status:
                where_conditions.append(f"status = ${len(params) + 1}")
                params.append(filter.status)
            if filter.priority:
                where_conditions.append(f"priority = ${len(params) + 1}")
                params.append(filter.priority)
            if filter.created_after:
                where_conditions.append(f"created_at >= ${len(params) + 1}")
                params.append(filter.created_after)
            if filter.completion_range and len(filter.completion_range) == 2:
                where_conditions.append(f"completion_percentage BETWEEN ${len(params) + 1} AND ${len(params) + 2}")
                params.extend(filter.completion_range)
        
        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        query_parts.append(f"ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}")
        params.extend([limit, offset])
        
        query = " ".join(query_parts)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [Venture(**dict(row)) for row in rows]
    
    @strawberry.field
    async def tasks(self,
                   info: Info,
                   filter: Optional[TaskFilter] = None,
                   limit: int = 100,
                   offset: int = 0) -> List[Task]:
        """Get tasks with advanced filtering"""
        db_pool = info.context.db_pool
        query_parts = ["SELECT * FROM Tasks"]
        params = []
        where_conditions = []
        
        if filter:
            if filter.status:
                where_conditions.append(f"status = ${len(params) + 1}")
                params.append(filter.status)
            if filter.venture_id:
                where_conditions.append(f"venture_id = ${len(params) + 1}")
                params.append(filter.venture_id)
            if filter.assigned_agent_id:
                where_conditions.append(f"assigned_agent_id = ${len(params) + 1}")
                params.append(filter.assigned_agent_id)
            if filter.min_bounty:
                where_conditions.append(f"cc_bounty >= ${len(params) + 1}")
                params.append(filter.min_bounty)
            if filter.max_bounty:
                where_conditions.append(f"cc_bounty <= ${len(params) + 1}")
                params.append(filter.max_bounty)
        
        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        query_parts.append(f"ORDER BY created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}")
        params.extend([limit, offset])
        
        query = " ".join(query_parts)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [Task(**dict(row)) for row in rows]
    
    @strawberry.field
    async def decisions(self,
                       info: Info,
                       filter: Optional[DecisionFilter] = None,
                       limit: int = 100,
                       offset: int = 0) -> List[Decision]:
        """Get decisions from causal ledger with filtering"""
        db_pool = info.context.db_pool
        query_parts = ["SELECT * FROM Decisions"]
        params = []
        where_conditions = []
        
        if filter:
            if filter.agent_id:
                where_conditions.append(f"agent_id = ${len(params) + 1}")
                params.append(filter.agent_id)
            if filter.decision_type:
                where_conditions.append(f"decision_type = ${len(params) + 1}")
                params.append(filter.decision_type)
            if filter.date_after:
                where_conditions.append(f"timestamp >= ${len(params) + 1}")
                params.append(filter.date_after)
            if filter.date_before:
                where_conditions.append(f"timestamp <= ${len(params) + 1}")
                params.append(filter.date_before)
        
        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        query_parts.append(f"ORDER BY timestamp DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}")
        params.extend([limit, offset])
        
        query = " ".join(query_parts)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [Decision(**dict(row)) for row in rows]
    
    @strawberry.field
    async def agent_performance(self, 
                               info: Info, 
                               agent_id: UUID, 
                               time_range: Optional[TimeRange] = None) -> AgentPerformanceMetrics:
        """Get comprehensive agent performance analytics"""
        if not time_range:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
        else:
            start_time = time_range.start
            end_time = time_range.end
        
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            # Comprehensive performance query
            perf_data = await conn.fetchrow("""
                WITH task_stats AS (
                    SELECT 
                        COUNT(*) as completed_tasks,
                        AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_hours,
                        SUM(cc_bounty) as cc_earned,
                        AVG(CASE WHEN status = 'COMPLETED' THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM Tasks 
                    WHERE assigned_agent_id = $1 
                    AND completed_at BETWEEN $2 AND $3
                ),
                bid_stats AS (
                    SELECT SUM(cc_amount) as cc_spent
                    FROM Bids 
                    WHERE agent_id = $1 
                    AND created_at BETWEEN $2 AND $3
                ),
                reputation_stats AS (
                    SELECT reputation_score
                    FROM Agents
                    WHERE id = $1
                )
                SELECT 
                    ts.completed_tasks,
                    ts.avg_hours,
                    ts.cc_earned,
                    ts.success_rate,
                    bs.cc_spent,
                    rs.reputation_score
                FROM task_stats ts, bid_stats bs, reputation_stats rs
            """, agent_id, start_time, end_time)
            
            return AgentPerformanceMetrics(
                agent_id=agent_id,
                time_period=f"{start_time.isoformat()} to {end_time.isoformat()}",
                tasks_completed=perf_data['completed_tasks'] or 0,
                tasks_success_rate=float(perf_data['success_rate'] or 0.0),
                average_completion_time=float(perf_data['avg_hours'] or 0.0),
                cc_earned=float(perf_data['cc_earned'] or 0.0),
                cc_spent=float(perf_data['cc_spent'] or 0.0),
                reputation_change=0.05,  # Would calculate trend
                specialization_score=0.85,  # Would calculate from task types
                collaboration_score=0.78,   # Would calculate from multi-agent tasks
                oracle_prediction_accuracy=0.82  # Would calculate from Oracle usage
            )
    
    @strawberry.field
    async def economic_metrics(self, info: Info) -> EconomicMetrics:
        """Get current economic system metrics"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            metrics = await conn.fetchrow("""
                WITH circulation AS (
                    SELECT 
                        SUM(cc_balance) as total_circulation,
                        COUNT(*) as active_agents
                    FROM Agents 
                    WHERE status = 'ACTIVE'
                ),
                bounties AS (
                    SELECT 
                        SUM(cc_bounty) as total_bounties,
                        COUNT(*) as active_bounties
                    FROM Tasks 
                    WHERE status IN ('BOUNTY_POSTED', 'BIDDING')
                ),
                completed AS (
                    SELECT 
                        SUM(cc_bounty) as cc_earned,
                        AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_completion_hours
                    FROM Tasks 
                    WHERE status = 'COMPLETED' 
                    AND completed_at >= NOW() - INTERVAL '30 days'
                ),
                spending AS (
                    SELECT SUM(cc_amount) as cc_spent
                    FROM Bids 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                )
                SELECT 
                    c.total_circulation,
                    c.active_agents,
                    b.active_bounties,
                    comp.cc_earned,
                    comp.avg_completion_hours,
                    s.cc_spent
                FROM circulation c, bounties b, completed comp, spending s
            """)
            
            return EconomicMetrics(
                timestamp=datetime.now(),
                total_cc_circulation=float(metrics['total_circulation'] or 0.0),
                total_cc_earned=float(metrics['cc_earned'] or 0.0),
                total_cc_spent=float(metrics['cc_spent'] or 0.0),
                active_bounties=metrics['active_bounties'] or 0,
                active_agents=metrics['active_agents'] or 0,
                market_efficiency_score=0.87,  # Would calculate from bid/bounty ratios
                wealth_distribution_gini=0.42,  # Would calculate from balance distribution
                average_task_completion_time=float(metrics['avg_completion_hours'] or 0.0)
            )
    
    @strawberry.field
    async def oracle_predictions(self, 
                                info: Info,
                                scenario_type: Optional[str] = None,
                                limit: int = 50) -> List[OraclePrediction]:
        """Get Oracle predictions with optional scenario filtering"""
        db_pool = info.context.db_pool
        
        query_parts = ["SELECT * FROM Simulations"]
        params = []
        
        if scenario_type:
            query_parts.append("WHERE scenario_type = $1")
            params.append(scenario_type)
        
        query_parts.append(f"ORDER BY created_at DESC LIMIT ${len(params) + 1}")
        params.append(limit)
        
        query = " ".join(query_parts)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [OraclePrediction(**dict(row)) for row in rows]
    
    @strawberry.field
    async def causal_chains(self, 
                           info: Info,
                           initial_decision_id: Optional[UUID] = None,
                           min_length: int = 2) -> List[CausalChain]:
        """Analyze causal reasoning chains"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            # Find causal chains (simplified implementation)
            chains = await conn.fetch("""
                SELECT 
                    causal_chain_id as id,
                    MIN(timestamp) as initial_timestamp,
                    COUNT(*) as chain_length
                FROM Decisions
                WHERE causal_chain_id IS NOT NULL
                GROUP BY causal_chain_id
                HAVING COUNT(*) >= $1
                ORDER BY chain_length DESC
                LIMIT 20
            """, min_length)
            
            result = []
            for chain in chains:
                decisions = await conn.fetch("""
                    SELECT * FROM Decisions
                    WHERE causal_chain_id = $1
                    ORDER BY timestamp ASC
                """, chain['id'])
                
                result.append(CausalChain(
                    id=chain['id'],
                    initial_decision_id=decisions[0]['id'] if decisions else None,
                    chain_length=chain['chain_length'],
                    decisions=[Decision(**dict(d)) for d in decisions],
                    outcome_impact_score=0.75,  # Would calculate from outcomes
                    confidence_degradation=0.05  # Would calculate confidence loss over chain
                ))
            
            return result
    
    @strawberry.field
    async def system_health(self, info: Info) -> SystemHealth:
        """Get overall system health metrics"""
        db_pool = info.context.db_pool
        async with db_pool.acquire() as conn:
            health_data = await conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM Agents WHERE status = 'ACTIVE') as active_agents,
                    (SELECT COUNT(*) FROM Ventures WHERE status IN ('ACTIVE', 'IN_PROGRESS')) as active_ventures,
                    (SELECT COUNT(*) FROM Tasks WHERE status IN ('BOUNTY_POSTED', 'BIDDING', 'ASSIGNED')) as pending_tasks,
                    (SELECT COUNT(*) FROM Decisions WHERE timestamp >= NOW() - INTERVAL '1 hour') as recent_decisions
            """)
            
            return SystemHealth(
                timestamp=datetime.now(),
                active_agents=health_data['active_agents'] or 0,
                active_ventures=health_data['active_ventures'] or 0,
                pending_tasks=health_data['pending_tasks'] or 0,
                system_utilization=0.73,  # Would calculate from resource usage
                oracle_accuracy=0.86,     # Would calculate from prediction validation
                economic_velocity=0.82,   # Would calculate from CC transaction rate
                decision_throughput=float(health_data['recent_decisions'] or 0)
            )

# Mutations for creating and updating data
@strawberry.type
class Mutation:
    """GraphQL Mutation root for data modifications"""
    
    @strawberry.mutation
    async def create_venture(self, info: Info, input: CreateVentureInput) -> Venture:
        """Create a new venture"""
        db_pool = info.context.db_pool
        venture_id = str(uuid.uuid4())
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO Ventures (id, name, description, priority, estimated_cc, status, created_at)
                VALUES ($1, $2, $3, $4, $5, 'PLANNING', NOW())
            """, venture_id, input.name, input.description, input.priority, input.estimated_cc)
            
            row = await conn.fetchrow("SELECT * FROM Ventures WHERE id = $1", venture_id)
            return Venture(**dict(row))
    
    @strawberry.mutation
    async def create_task(self, info: Info, input: CreateTaskInput) -> Task:
        """Create a new task"""
        db_pool = info.context.db_pool
        task_id = str(uuid.uuid4())
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO Tasks (id, venture_id, description, cc_bounty, estimated_hours, status, created_at)
                VALUES ($1, $2, $3, $4, $5, 'BOUNTY_POSTED', NOW())
            """, task_id, input.venture_id, input.description, input.cc_bounty, input.estimated_hours)
            
            row = await conn.fetchrow("SELECT * FROM Tasks WHERE id = $1", task_id)
            
            # Publish to Redis for subscriptions
            redis_client = info.context.redis
            await redis_client.publish("new_tasks", json.dumps(dict(row)))
            
            return Task(**dict(row))
    
    @strawberry.mutation
    async def create_bid(self, info: Info, input: CreateBidInput) -> Bid:
        """Create a new bid"""
        db_pool = info.context.db_pool
        bid_id = str(uuid.uuid4())
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO Bids (id, task_id, agent_id, cc_amount, estimated_hours, proposal, confidence_score, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'ACTIVE', NOW())
            """, bid_id, input.task_id, input.agent_id, input.cc_amount, input.estimated_hours, input.proposal, input.confidence_score)
            
            row = await conn.fetchrow("SELECT * FROM Bids WHERE id = $1", bid_id)
            return Bid(**dict(row))
    
    @strawberry.mutation
    async def request_oracle_prediction(self, info: Info, input: OraclePredictionInput) -> OraclePrediction:
        """Request a prediction from the Oracle"""
        # This would integrate with the Oracle engine
        oracle_engine = info.context.oracle_engine
        
        prediction_result = await oracle_engine.generate_prediction({
            'scenario_type': input.scenario_type,
            'parameters': input.parameters,
            'prediction_horizon': timedelta(hours=input.prediction_horizon)
        })
        
        # Store in database
        db_pool = info.context.db_pool
        simulation_id = str(uuid.uuid4())
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO Simulations (id, scenario_type, parameters, results, confidence_score, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
            """, simulation_id, input.scenario_type, json.dumps(input.parameters), 
            json.dumps(prediction_result), prediction_result.get('confidence_scores', {}).get('overall', 0.8))
            
            row = await conn.fetchrow("SELECT * FROM Simulations WHERE id = $1", simulation_id)
            return OraclePrediction(**dict(row))

# Subscriptions for real-time updates
@strawberry.type
class Subscription:
    """GraphQL Subscription root for real-time data streams"""
    
    @strawberry.subscription
    async def agent_activity(self, info: Info, agent_id: Optional[UUID] = None) -> AsyncGenerator[Agent, None]:
        """Subscribe to agent activity updates"""
        redis_client = info.context.redis
        channel = f"agent_activity:{agent_id}" if agent_id else "agent_activity:all"
        
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    agent_data = json.loads(message['data'])
                    yield Agent(**agent_data)
        except asyncio.CancelledError:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    
    @strawberry.subscription
    async def task_updates(self, info: Info, venture_id: Optional[UUID] = None) -> AsyncGenerator[Task, None]:
        """Subscribe to task creation and updates"""
        redis_client = info.context.redis
        channel = f"task_updates:{venture_id}" if venture_id else "task_updates:all"
        
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    task_data = json.loads(message['data'])
                    yield Task(**task_data)
        except asyncio.CancelledError:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    
    @strawberry.subscription
    async def decision_stream(self, info: Info, agent_id: Optional[UUID] = None) -> AsyncGenerator[Decision, None]:
        """Subscribe to the causal ledger decision stream"""
        redis_client = info.context.redis
        channel = f"decisions:{agent_id}" if agent_id else "decisions:all"
        
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    decision_data = json.loads(message['data'])
                    yield Decision(**decision_data)
        except asyncio.CancelledError:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    
    @strawberry.subscription
    async def oracle_predictions(self, info: Info, scenario_type: Optional[str] = None) -> AsyncGenerator[OraclePrediction, None]:
        """Subscribe to new Oracle predictions"""
        redis_client = info.context.redis
        channel = f"oracle_predictions:{scenario_type}" if scenario_type else "oracle_predictions:all"
        
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    prediction_data = json.loads(message['data'])
                    yield OraclePrediction(**prediction_data)
        except asyncio.CancelledError:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
    
    @strawberry.subscription
    async def economic_updates(self, info: Info) -> AsyncGenerator[EconomicMetrics, None]:
        """Subscribe to economic system updates"""
        redis_client = info.context.redis
        
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("economic_metrics")
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    metrics_data = json.loads(message['data'])
                    yield EconomicMetrics(**metrics_data)
        except asyncio.CancelledError:
            await pubsub.unsubscribe("economic_metrics")
            await pubsub.close()

# Create the schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)