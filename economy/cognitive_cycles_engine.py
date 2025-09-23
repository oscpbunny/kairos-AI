"""
Project Kairos: Cognitive Cycles Economy Engine
The motivation engine that drives efficient task execution through market dynamics.

This system implements:
- CC (Cognitive Cycles) currency management
- Task bidding marketplace
- Emergent agent specialization
- Performance-based reputation systems
- Dynamic pricing and incentive mechanisms
- Economic analytics and optimization
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
from enum import Enum

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from scipy import stats

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CognitiveCyclesEngine')

class BidStatus(Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"

class MarketPhase(Enum):
    BIDDING = "BIDDING"
    EVALUATION = "EVALUATION"
    ALLOCATION = "ALLOCATION"
    EXECUTION = "EXECUTION"
    SETTLEMENT = "SETTLEMENT"

@dataclass
class TaskBid:
    """A bid placed by an agent on a task"""
    id: str
    task_id: str
    agent_id: str
    bid_amount_cc: int
    estimated_completion_time: int  # minutes
    proposed_approach: str
    confidence_score: float
    risk_factors: List[str]
    past_performance_evidence: Dict[str, Any]
    created_at: datetime
    status: BidStatus

@dataclass
class MarketTransaction:
    """A completed transaction in the CC economy"""
    transaction_id: str
    from_agent_id: Optional[str]  # None for system transactions
    to_agent_id: str
    amount_cc: int
    transaction_type: str  # 'BOUNTY_PAYMENT', 'BID_DEPOSIT', 'PERFORMANCE_BONUS', 'PENALTY'
    related_task_id: Optional[str]
    description: str
    timestamp: datetime

@dataclass
class AgentEconomicProfile:
    """Economic profile and performance metrics for an agent"""
    agent_id: str
    agent_name: str
    specialization: str
    cognitive_cycles_balance: int
    total_cc_earned: int
    total_tasks_completed: int
    success_rate: float
    avg_task_completion_time: float  # hours
    cost_efficiency_score: float
    reputation_score: float
    specialization_strength: Dict[str, float]  # task_type -> performance score
    last_active: datetime

@dataclass
class TaskMarketAnalytics:
    """Analytics for the task marketplace"""
    total_active_tasks: int
    total_pending_bids: int
    average_bid_amount: float
    median_completion_time: float
    market_liquidity: float  # ratio of bids to tasks
    price_volatility: float
    specialization_premiums: Dict[str, float]  # task_type -> avg premium %
    agent_competitiveness: Dict[str, int]  # agent_id -> number of active bids

class CognitiveCyclesEngine:
    """The brain of the internal economy - managing CC currency and market dynamics"""
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        self.db_config = db_config or self._load_db_config()
        self.economic_parameters = self._load_economic_parameters()
        self.market_state = MarketPhase.BIDDING
        
        # Economic constants
        self.BASE_CC_GENERATION_RATE = 100  # CC generated per hour for active agents
        self.MIN_BID_DEPOSIT_RATIO = 0.1  # Agents must deposit 10% of bid amount
        self.PERFORMANCE_BONUS_MULTIPLIER = 0.2  # 20% bonus for excellent performance
        self.REPUTATION_DECAY_RATE = 0.95  # Daily reputation decay if inactive
        
    async def initialize(self):
        """Initialize the Cognitive Cycles Engine - async initialization tasks"""
        try:
            logger.info("Cognitive Cycles Engine initializing...")
            # Any async initialization can go here
            # For now, just log that we're ready
            logger.info("Cognitive Cycles Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Cycles Engine: {e}")
            raise
        
    def _load_db_config(self) -> Dict[str, str]:
        """Load database configuration from environment"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'kairos_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
    
    def _load_economic_parameters(self) -> Dict[str, float]:
        """Load economic tuning parameters"""
        return {
            'bid_evaluation_weight_price': 0.3,
            'bid_evaluation_weight_reputation': 0.25,
            'bid_evaluation_weight_specialization': 0.25,
            'bid_evaluation_weight_time': 0.2,
            'market_maker_spread': 0.05,  # 5% spread for market making
            'inflation_rate': 0.02,  # 2% annual CC inflation
            'task_complexity_multiplier': {
                1: 0.5, 2: 0.7, 3: 1.0, 4: 1.3, 5: 1.8,
                6: 2.5, 7: 3.5, 8: 5.0, 9: 7.5, 10: 12.0
            }
        }
    
    async def get_db_connection(self):
        """Establish async database connection"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        )
    
    async def post_task_bounty(
        self, 
        task_id: str, 
        base_bounty: int, 
        complexity_level: int,
        urgency_multiplier: float = 1.0,
        required_capabilities: List[str] = None
    ) -> int:
        """
        Post a task with calculated bounty based on complexity and urgency.
        Returns the final bounty amount.
        """
        try:
            # Calculate dynamic bounty based on market conditions
            complexity_multiplier = self.economic_parameters['task_complexity_multiplier'].get(complexity_level, 1.0)
            market_condition_multiplier = await self._calculate_market_condition_multiplier()
            specialization_premium = await self._calculate_specialization_premium(required_capabilities or [])
            
            final_bounty = int(
                base_bounty * 
                complexity_multiplier * 
                urgency_multiplier * 
                market_condition_multiplier * 
                (1 + specialization_premium)
            )
            
            # Update task with calculated bounty
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE Tasks 
                    SET cc_bounty = %s, 
                        urgency_multiplier = %s,
                        status = 'BOUNTY_POSTED',
                        bidding_ends_at = CURRENT_TIMESTAMP + INTERVAL '2 hours'
                    WHERE id = %s;
                    """,
                    (final_bounty, urgency_multiplier, task_id)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Posted bounty for task {task_id}: {final_bounty} CC (complexity: {complexity_level}, urgency: {urgency_multiplier:.2f})")
            return final_bounty
            
        except Exception as e:
            logger.error(f"Failed to post task bounty: {e}")
            return 0
    
    async def _calculate_market_condition_multiplier(self) -> float:
        """Calculate multiplier based on current market supply/demand"""
        try:
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                # Get supply (available agents) vs demand (pending tasks)
                await cur.execute(
                    """
                    SELECT 
                        COUNT(CASE WHEN a.is_active = true THEN 1 END) as active_agents,
                        COUNT(CASE WHEN t.status IN ('BOUNTY_POSTED', 'BIDDING') THEN 1 END) as pending_tasks
                    FROM Agents a
                    CROSS JOIN Tasks t;
                    """
                )
                
                result = await cur.fetchone()
            
            conn.close()
            
            if result and result['active_agents'] > 0:
                demand_supply_ratio = result['pending_tasks'] / result['active_agents']
                
                # Higher ratio = more demand, higher prices (up to 2x)
                # Lower ratio = more supply, lower prices (down to 0.7x)
                multiplier = min(2.0, max(0.7, 0.8 + (demand_supply_ratio * 0.3)))
                return multiplier
            
            return 1.0  # Default multiplier
            
        except Exception as e:
            logger.error(f"Failed to calculate market condition multiplier: {e}")
            return 1.0
    
    async def _calculate_specialization_premium(self, required_capabilities: List[str]) -> float:
        """Calculate premium for tasks requiring specialized capabilities"""
        if not required_capabilities:
            return 0.0
        
        try:
            conn = await self.get_db_connection()
            
            # Count agents with required capabilities
            capability_scarcity = {}
            
            for capability in required_capabilities:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT COUNT(*) as capable_agents
                        FROM Agents 
                        WHERE is_active = true 
                        AND capabilities ? %s;
                        """,
                        (capability,)
                    )
                    
                    result = await cur.fetchone()
                    capable_agents = result['capable_agents'] if result else 0
                    
                    # More scarcity = higher premium
                    if capable_agents <= 2:
                        capability_scarcity[capability] = 0.5  # 50% premium
                    elif capable_agents <= 5:
                        capability_scarcity[capability] = 0.25  # 25% premium
                    elif capable_agents <= 10:
                        capability_scarcity[capability] = 0.1   # 10% premium
                    else:
                        capability_scarcity[capability] = 0.0   # No premium
            
            conn.close()
            
            # Return maximum premium across all required capabilities
            return max(capability_scarcity.values()) if capability_scarcity else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate specialization premium: {e}")
            return 0.0
    
    async def place_bid(
        self,
        agent_id: str,
        task_id: str,
        bid_amount: int,
        estimated_completion_time: int,
        proposed_approach: str,
        confidence_score: float
    ) -> bool:
        """
        Allow an agent to place a bid on a task.
        Returns True if bid was successfully placed.
        """
        try:
            # Validate agent has sufficient CC balance for deposit
            agent_profile = await self.get_agent_economic_profile(agent_id)
            required_deposit = int(bid_amount * self.MIN_BID_DEPOSIT_RATIO)
            
            if agent_profile.cognitive_cycles_balance < required_deposit:
                logger.warning(f"Agent {agent_id} has insufficient CC for bid deposit: {agent_profile.cognitive_cycles_balance} < {required_deposit}")
                return False
            
            # Check if task is still accepting bids
            conn = await self.get_db_connection()
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT status, bidding_ends_at, cc_bounty 
                    FROM Tasks 
                    WHERE id = %s;
                    """,
                    (task_id,)
                )
                
                task = await cur.fetchone()
                
                if not task or task['status'] not in ['BOUNTY_POSTED', 'BIDDING']:
                    logger.warning(f"Task {task_id} is not accepting bids (status: {task.get('status', 'NOT_FOUND')})")
                    return False
                
                if task['bidding_ends_at'] and datetime.now() > task['bidding_ends_at']:
                    logger.warning(f"Bidding period has ended for task {task_id}")
                    return False
                
                # Get agent's past performance for similar tasks
                past_performance = await self._get_agent_task_performance(agent_id, task_id)
                
                # Place the bid
                await cur.execute(
                    """
                    INSERT INTO Task_Bids 
                    (task_id, agent_id, bid_amount_cc, estimated_completion_time, 
                     proposed_approach, confidence_score, past_performance_evidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id, agent_id) 
                    DO UPDATE SET
                        bid_amount_cc = EXCLUDED.bid_amount_cc,
                        estimated_completion_time = EXCLUDED.estimated_completion_time,
                        proposed_approach = EXCLUDED.proposed_approach,
                        confidence_score = EXCLUDED.confidence_score,
                        created_at = CURRENT_TIMESTAMP;
                    """,
                    (
                        task_id, agent_id, bid_amount, estimated_completion_time,
                        proposed_approach, confidence_score, json.dumps(past_performance)
                    )
                )
                
                # Deduct deposit from agent's balance
                await cur.execute(
                    """
                    UPDATE Agents 
                    SET cognitive_cycles_balance = cognitive_cycles_balance - %s 
                    WHERE id = %s;
                    """,
                    (required_deposit, agent_id)
                )
                
                # Update task status to BIDDING if not already
                await cur.execute(
                    """
                    UPDATE Tasks 
                    SET status = 'BIDDING' 
                    WHERE id = %s AND status = 'BOUNTY_POSTED';
                    """,
                    (task_id,)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Agent {agent_id} placed bid on task {task_id}: {bid_amount} CC, {estimated_completion_time}min estimated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to place bid: {e}")
            return False
    
    async def _get_agent_task_performance(self, agent_id: str, task_id: str) -> Dict[str, Any]:
        """Get agent's historical performance for similar tasks"""
        try:
            conn = await self.get_db_connection()
            
            async with conn.cursor() as cur:
                # Get task type and complexity for similarity matching
                await cur.execute(
                    """
                    SELECT task_type, complexity_level, required_capabilities
                    FROM Tasks 
                    WHERE id = %s;
                    """,
                    (task_id,)
                )
                
                current_task = await cur.fetchone()
                if not current_task:
                    return {}
                
                # Find similar completed tasks by this agent
                await cur.execute(
                    """
                    SELECT 
                        t.task_type,
                        t.complexity_level,
                        t.quality_score,
                        t.actual_duration,
                        t.estimated_duration,
                        t.cc_bounty,
                        t.bonus_cc,
                        EXTRACT(EPOCH FROM (t.completed_at - t.started_at))/3600 as actual_hours
                    FROM Tasks t
                    WHERE t.assigned_to_agent_id = %s 
                    AND t.status = 'COMPLETED'
                    AND t.task_type = %s
                    AND ABS(t.complexity_level - %s) <= 2
                    ORDER BY t.completed_at DESC
                    LIMIT 10;
                    """,
                    (agent_id, current_task['task_type'], current_task['complexity_level'])
                )
                
                similar_tasks = await cur.fetchall()
            
            conn.close()
            
            if not similar_tasks:
                return {"similar_task_count": 0}
            
            # Calculate performance metrics
            quality_scores = [float(t['quality_score']) for t in similar_tasks if t['quality_score']]
            time_accuracies = []
            
            for task in similar_tasks:
                if task['estimated_duration'] and task['actual_hours']:
                    estimated_hours = task['estimated_duration'].total_seconds() / 3600
                    accuracy = 1.0 - abs(estimated_hours - task['actual_hours']) / estimated_hours
                    time_accuracies.append(max(0.0, accuracy))
            
            return {
                "similar_task_count": len(similar_tasks),
                "avg_quality_score": np.mean(quality_scores) if quality_scores else 0.0,
                "avg_time_accuracy": np.mean(time_accuracies) if time_accuracies else 0.0,
                "total_cc_earned": sum(t['cc_bounty'] + (t['bonus_cc'] or 0) for t in similar_tasks),
                "success_rate": 1.0  # All tasks in this query were completed successfully
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent task performance: {e}")
            return {}
    
    async def evaluate_bids_and_assign_task(self, task_id: str) -> Optional[str]:
        """
        Evaluate all bids for a task and assign to the best bidder.
        Returns the winning agent_id or None if no suitable bid found.
        """
        try:
            conn = await self.get_db_connection()
            
            # Get task details and all pending bids
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT t.*, 
                           array_agg(
                               json_build_object(
                                   'bid_id', b.id,
                                   'agent_id', b.agent_id,
                                   'agent_name', a.name,
                                   'bid_amount', b.bid_amount_cc,
                                   'estimated_time', b.estimated_completion_time,
                                   'confidence', b.confidence_score,
                                   'agent_reputation', a.reputation_score,
                                   'agent_efficiency', a.cost_efficiency_score,
                                   'past_performance', b.past_performance_evidence
                               ) ORDER BY b.created_at
                           ) FILTER (WHERE b.id IS NOT NULL) as bids
                    FROM Tasks t
                    LEFT JOIN Task_Bids b ON t.id = b.task_id AND b.status = 'PENDING'
                    LEFT JOIN Agents a ON b.agent_id = a.id
                    WHERE t.id = %s
                    GROUP BY t.id;
                    """,
                    (task_id,)
                )
                
                result = await cur.fetchone()
                
                if not result or not result['bids'] or result['bids'] == [None]:
                    logger.warning(f"No bids found for task {task_id}")
                    return None
                
                task = result
                bids = result['bids']
            
            # Evaluate each bid using multi-criteria scoring
            bid_scores = []
            for bid in bids:
                score = await self._calculate_bid_score(task, bid)
                bid_scores.append((bid, score))
            
            # Sort by score (highest first)
            bid_scores.sort(key=lambda x: x[1], reverse=True)
            
            if not bid_scores:
                logger.warning(f"No valid bids for task {task_id}")
                return None
            
            winning_bid, winning_score = bid_scores[0]
            winning_agent_id = winning_bid['agent_id']
            
            # Assign task to winning bidder
            async with conn.cursor() as cur:
                # Update task assignment
                await cur.execute(
                    """
                    UPDATE Tasks 
                    SET assigned_to_agent_id = %s,
                        status = 'ASSIGNED',
                        estimated_duration = INTERVAL '%s minutes'
                    WHERE id = %s;
                    """,
                    (winning_agent_id, winning_bid['estimated_time'], task_id)
                )
                
                # Update winning bid status
                await cur.execute(
                    """
                    UPDATE Task_Bids 
                    SET status = 'ACCEPTED'
                    WHERE id = %s;
                    """,
                    (winning_bid['bid_id'],)
                )
                
                # Reject other bids and refund deposits
                await cur.execute(
                    """
                    UPDATE Task_Bids 
                    SET status = 'REJECTED'
                    WHERE task_id = %s AND id != %s;
                    """,
                    (task_id, winning_bid['bid_id'])
                )
                
                # Refund deposits for rejected bids
                for bid, _ in bid_scores[1:]:  # All except winner
                    deposit_amount = int(bid['bid_amount'] * self.MIN_BID_DEPOSIT_RATIO)
                    await cur.execute(
                        """
                        UPDATE Agents 
                        SET cognitive_cycles_balance = cognitive_cycles_balance + %s 
                        WHERE id = %s;
                        """,
                        (deposit_amount, bid['agent_id'])
                    )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Task {task_id} assigned to agent {winning_agent_id} (score: {winning_score:.3f})")
            return winning_agent_id
            
        except Exception as e:
            logger.error(f"Failed to evaluate bids and assign task: {e}")
            return None
    
    async def _calculate_bid_score(self, task: Dict, bid: Dict) -> float:
        """Calculate multi-criteria score for a bid"""
        try:
            # Normalize price score (lower bid = higher score)
            max_bounty = task['cc_bounty']
            price_score = 1.0 - (bid['bid_amount'] / max_bounty) if max_bounty > 0 else 0.5
            price_score = max(0.0, min(1.0, price_score))
            
            # Reputation score (0-1 scale)
            reputation_score = float(bid.get('agent_reputation', 1.0))
            
            # Time score (faster completion = higher score)
            estimated_hours = bid['estimated_time'] / 60.0
            expected_hours = task.get('complexity_level', 5) * 2  # 2 hours per complexity point
            time_score = min(1.0, expected_hours / max(estimated_hours, 0.5))
            
            # Specialization score based on past performance
            past_performance = bid.get('past_performance', {})
            specialization_score = 0.5  # Default
            
            if isinstance(past_performance, str):
                try:
                    past_performance = json.loads(past_performance)
                except:
                    past_performance = {}
            
            if past_performance.get('similar_task_count', 0) > 0:
                quality_weight = 0.4
                experience_weight = 0.3
                success_weight = 0.3
                
                quality_component = past_performance.get('avg_quality_score', 0.5)
                experience_component = min(1.0, past_performance.get('similar_task_count', 0) / 10.0)
                success_component = past_performance.get('success_rate', 0.5)
                
                specialization_score = (
                    quality_weight * quality_component +
                    experience_weight * experience_component +
                    success_weight * success_component
                )
            
            # Confidence adjustment
            confidence_multiplier = bid.get('confidence', 0.8)
            
            # Calculate weighted final score
            weights = self.economic_parameters
            final_score = (
                weights['bid_evaluation_weight_price'] * price_score +
                weights['bid_evaluation_weight_reputation'] * reputation_score +
                weights['bid_evaluation_weight_specialization'] * specialization_score +
                weights['bid_evaluation_weight_time'] * time_score
            ) * confidence_multiplier
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate bid score: {e}")
            return 0.0
    
    async def settle_task_completion(
        self, 
        task_id: str, 
        quality_score: float, 
        actual_completion_time: int  # minutes
    ) -> bool:
        """
        Handle payment and reputation updates when a task is completed.
        Returns True if settlement was successful.
        """
        try:
            conn = await self.get_db_connection()
            
            async with conn.cursor() as cur:
                # Get task and winning bid details
                await cur.execute(
                    """
                    SELECT t.*, b.bid_amount_cc, b.estimated_completion_time, a.name as agent_name
                    FROM Tasks t
                    JOIN Task_Bids b ON t.id = b.task_id AND b.status = 'ACCEPTED'
                    JOIN Agents a ON b.agent_id = a.id
                    WHERE t.id = %s;
                    """,
                    (task_id,)
                )
                
                task_details = await cur.fetchone()
                
                if not task_details:
                    logger.error(f"No accepted bid found for completed task {task_id}")
                    return False
                
                agent_id = task_details['assigned_to_agent_id']
                base_bounty = task_details['cc_bounty']
                
                # Calculate performance bonus/penalty
                bonus_cc = await self._calculate_performance_bonus(
                    task_details, quality_score, actual_completion_time
                )
                
                total_payout = base_bounty + bonus_cc
                
                # Pay the agent
                await cur.execute(
                    """
                    UPDATE Agents 
                    SET cognitive_cycles_balance = cognitive_cycles_balance + %s
                    WHERE id = %s;
                    """,
                    (total_payout, agent_id)
                )
                
                # Update task with quality score and bonus
                await cur.execute(
                    """
                    UPDATE Tasks 
                    SET quality_score = %s,
                        bonus_cc = %s,
                        actual_duration = INTERVAL '%s minutes'
                    WHERE id = %s;
                    """,
                    (quality_score, bonus_cc, actual_completion_time, task_id)
                )
                
                # Record transaction
                await self._record_transaction(
                    conn, None, agent_id, total_payout, 
                    'BOUNTY_PAYMENT', task_id, 
                    f"Task completion payment: {base_bounty} CC base + {bonus_cc} CC bonus"
                )
                
                # Update agent reputation
                await self._update_agent_reputation(conn, agent_id, task_details, quality_score, actual_completion_time)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Task {task_id} settled: {total_payout} CC paid to agent {task_details['agent_name']} (quality: {quality_score:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to settle task completion: {e}")
            return False
    
    async def _calculate_performance_bonus(
        self, 
        task_details: Dict, 
        quality_score: float, 
        actual_completion_time: int
    ) -> int:
        """Calculate performance bonus based on quality and time accuracy"""
        base_bounty = task_details['cc_bounty']
        estimated_time = task_details['estimated_completion_time']
        
        # Quality bonus (up to 20% for perfect quality)
        quality_bonus = int(base_bounty * self.PERFORMANCE_BONUS_MULTIPLIER * quality_score)
        
        # Time accuracy bonus/penalty
        time_bonus = 0
        if estimated_time and actual_completion_time:
            time_ratio = actual_completion_time / estimated_time
            
            if time_ratio <= 0.8:  # Completed 20% faster
                time_bonus = int(base_bounty * 0.1)
            elif time_ratio <= 1.0:  # Completed on time or slightly faster
                time_bonus = int(base_bounty * 0.05)
            elif time_ratio > 1.5:  # Took 50% longer
                time_bonus = -int(base_bounty * 0.1)
        
        return quality_bonus + time_bonus
    
    async def _record_transaction(
        self,
        conn,
        from_agent_id: Optional[str],
        to_agent_id: str,
        amount_cc: int,
        transaction_type: str,
        related_task_id: Optional[str],
        description: str
    ):
        """Record a transaction in the economic ledger"""
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO Activity_Logs 
                (agent_id, action_type, entity_type, entity_id, action_details)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (
                    to_agent_id,
                    'CC_TRANSACTION',
                    'ECONOMIC',
                    related_task_id or 'SYSTEM',
                    json.dumps({
                        'transaction_type': transaction_type,
                        'amount_cc': amount_cc,
                        'from_agent_id': from_agent_id,
                        'to_agent_id': to_agent_id,
                        'description': description
                    })
                )
            )
    
    async def _update_agent_reputation(
        self,
        conn,
        agent_id: str,
        task_details: Dict,
        quality_score: float,
        actual_completion_time: int
    ):
        """Update agent's reputation based on task performance"""
        try:
            # Calculate performance metrics
            estimated_time = task_details['estimated_completion_time']
            time_accuracy = 1.0
            
            if estimated_time and actual_completion_time:
                time_ratio = actual_completion_time / estimated_time
                time_accuracy = max(0.0, 1.0 - abs(time_ratio - 1.0))
            
            # Combined performance score
            performance_score = (quality_score * 0.7 + time_accuracy * 0.3)
            
            # Update reputation with weighted average (recent performance weighted more heavily)
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE Agents 
                    SET reputation_score = (reputation_score * 0.8 + %s * 0.2),
                        cost_efficiency_score = CASE 
                            WHEN total_tasks_completed > 0 THEN
                                (cost_efficiency_score * 0.9 + %s * 0.1)
                            ELSE %s
                        END
                    WHERE id = %s;
                    """,
                    (performance_score, performance_score, performance_score, agent_id)
                )
            
        except Exception as e:
            logger.error(f"Failed to update agent reputation: {e}")
    
    async def get_agent_economic_profile(self, agent_id: str) -> Optional[AgentEconomicProfile]:
        """Get comprehensive economic profile for an agent"""
        try:
            conn = await self.get_db_connection()
            
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT a.*,
                           COUNT(t.id) FILTER (WHERE t.status = 'COMPLETED') as completed_tasks,
                           COUNT(t.id) FILTER (WHERE t.status = 'COMPLETED' AND t.quality_score >= 0.8) as high_quality_tasks,
                           AVG(EXTRACT(EPOCH FROM t.actual_duration)/3600) as avg_completion_hours
                    FROM Agents a
                    LEFT JOIN Tasks t ON a.id = t.assigned_to_agent_id
                    WHERE a.id = %s
                    GROUP BY a.id;
                    """,
                    (agent_id,)
                )
                
                agent_data = await cur.fetchone()
                
                if not agent_data:
                    return None
                
                # Calculate specialization strengths
                await cur.execute(
                    """
                    SELECT t.task_type, 
                           AVG(t.quality_score) as avg_quality,
                           COUNT(*) as task_count
                    FROM Tasks t
                    WHERE t.assigned_to_agent_id = %s 
                    AND t.status = 'COMPLETED'
                    AND t.quality_score IS NOT NULL
                    GROUP BY t.task_type
                    HAVING COUNT(*) >= 2;
                    """,
                    (agent_id,)
                )
                
                specializations = await cur.fetchall()
            
            conn.close()
            
            # Build specialization strength map
            specialization_strength = {}
            for spec in specializations:
                specialization_strength[spec['task_type']] = float(spec['avg_quality'])
            
            # Calculate success rate
            total_completed = agent_data['completed_tasks'] or 0
            high_quality_completed = agent_data['high_quality_tasks'] or 0
            success_rate = (high_quality_completed / total_completed) if total_completed > 0 else 0.0
            
            return AgentEconomicProfile(
                agent_id=agent_id,
                agent_name=agent_data['name'],
                specialization=agent_data['specialization'],
                cognitive_cycles_balance=agent_data['cognitive_cycles_balance'],
                total_cc_earned=agent_data['total_cc_earned'],
                total_tasks_completed=total_completed,
                success_rate=success_rate,
                avg_task_completion_time=float(agent_data['avg_completion_hours'] or 0),
                cost_efficiency_score=float(agent_data['cost_efficiency_score']),
                reputation_score=float(agent_data['reputation_score']),
                specialization_strength=specialization_strength,
                last_active=agent_data['last_heartbeat']
            )
            
        except Exception as e:
            logger.error(f"Failed to get agent economic profile: {e}")
            return None
    
    async def get_market_analytics(self) -> TaskMarketAnalytics:
        """Get comprehensive market analytics"""
        try:
            conn = await self.get_db_connection()
            
            async with conn.cursor() as cur:
                # Get basic market stats
                await cur.execute(
                    """
                    SELECT 
                        COUNT(t.id) FILTER (WHERE t.status IN ('BOUNTY_POSTED', 'BIDDING', 'ASSIGNED')) as active_tasks,
                        COUNT(b.id) FILTER (WHERE b.status = 'PENDING') as pending_bids,
                        AVG(b.bid_amount_cc) as avg_bid_amount,
                        AVG(b.estimated_completion_time) as avg_estimated_time,
                        COUNT(DISTINCT b.agent_id) as active_bidders
                    FROM Tasks t
                    LEFT JOIN Task_Bids b ON t.id = b.task_id;
                    """
                )
                
                stats = await cur.fetchone()
                
                # Get market liquidity (bids per task)
                market_liquidity = 0.0
                if stats['active_tasks'] and stats['active_tasks'] > 0:
                    market_liquidity = (stats['pending_bids'] or 0) / stats['active_tasks']
                
                # Get price volatility (coefficient of variation)
                price_volatility = 0.0
                if stats['avg_bid_amount'] and stats['avg_bid_amount'] > 0:
                    await cur.execute(
                        """
                        SELECT STDDEV(bid_amount_cc) as price_stddev
                        FROM Task_Bids 
                        WHERE status = 'PENDING' AND created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days';
                        """
                    )
                    
                    volatility_data = await cur.fetchone()
                    if volatility_data and volatility_data['price_stddev']:
                        price_volatility = float(volatility_data['price_stddev']) / float(stats['avg_bid_amount'])
                
                # Get specialization premiums
                await cur.execute(
                    """
                    SELECT t.task_type, AVG(t.cc_bounty) as avg_bounty
                    FROM Tasks t
                    WHERE t.created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
                    GROUP BY t.task_type
                    HAVING COUNT(*) >= 5;
                    """
                )
                
                task_type_pricing = await cur.fetchall()
                
                # Get agent competitiveness
                await cur.execute(
                    """
                    SELECT b.agent_id, COUNT(*) as active_bids
                    FROM Task_Bids b
                    WHERE b.status = 'PENDING'
                    GROUP BY b.agent_id
                    ORDER BY active_bids DESC;
                    """
                )
                
                agent_bids = await cur.fetchall()
            
            conn.close()
            
            # Calculate specialization premiums relative to baseline
            baseline_bounty = np.mean([float(row['avg_bounty']) for row in task_type_pricing]) if task_type_pricing else 100.0
            specialization_premiums = {}
            
            for row in task_type_pricing:
                premium_pct = ((float(row['avg_bounty']) - baseline_bounty) / baseline_bounty) * 100
                specialization_premiums[row['task_type']] = premium_pct
            
            # Build agent competitiveness map
            agent_competitiveness = {row['agent_id']: row['active_bids'] for row in agent_bids}
            
            return TaskMarketAnalytics(
                total_active_tasks=stats['active_tasks'] or 0,
                total_pending_bids=stats['pending_bids'] or 0,
                average_bid_amount=float(stats['avg_bid_amount'] or 0),
                median_completion_time=float(stats['avg_estimated_time'] or 0) / 60.0,  # Convert to hours
                market_liquidity=market_liquidity,
                price_volatility=price_volatility,
                specialization_premiums=specialization_premiums,
                agent_competitiveness=agent_competitiveness
            )
            
        except Exception as e:
            logger.error(f"Failed to get market analytics: {e}")
            return TaskMarketAnalytics(
                total_active_tasks=0, total_pending_bids=0, average_bid_amount=0.0,
                median_completion_time=0.0, market_liquidity=0.0, price_volatility=0.0,
                specialization_premiums={}, agent_competitiveness={}
            )
    
    async def run_economic_maintenance(self):
        """Run periodic economic system maintenance"""
        try:
            logger.info("Running economic system maintenance")
            
            # Generate base CC for active agents
            await self._distribute_base_cc()
            
            # Apply reputation decay for inactive agents
            await self._apply_reputation_decay()
            
            # Cleanup expired bids
            await self._cleanup_expired_bids()
            
            # Rebalance market if needed
            await self._market_rebalancing()
            
            logger.info("Economic maintenance completed successfully")
            
        except Exception as e:
            logger.error(f"Economic maintenance failed: {e}")
    
    async def _distribute_base_cc(self):
        """Distribute base CC generation to active agents"""
        try:
            conn = await self.get_db_connection()
            
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE Agents 
                    SET cognitive_cycles_balance = cognitive_cycles_balance + %s
                    WHERE is_active = true 
                    AND last_heartbeat >= CURRENT_TIMESTAMP - INTERVAL '1 hour';
                    """,
                    (self.BASE_CC_GENERATION_RATE,)
                )
                
                updated_agents = cur.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Distributed {self.BASE_CC_GENERATION_RATE} CC to {updated_agents} active agents")
            
        except Exception as e:
            logger.error(f"Failed to distribute base CC: {e}")
    
    async def _apply_reputation_decay(self):
        """Apply reputation decay to inactive agents"""
        try:
            conn = await self.get_db_connection()
            
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE Agents 
                    SET reputation_score = reputation_score * %s
                    WHERE last_heartbeat < CURRENT_TIMESTAMP - INTERVAL '24 hours'
                    AND reputation_score > 0.1;
                    """,
                    (self.REPUTATION_DECAY_RATE,)
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to apply reputation decay: {e}")
    
    async def _cleanup_expired_bids(self):
        """Clean up expired bids and refund deposits"""
        try:
            conn = await self.get_db_connection()
            
            async with conn.cursor() as cur:
                # Find expired tasks with pending bids
                await cur.execute(
                    """
                    SELECT b.agent_id, b.bid_amount_cc
                    FROM Task_Bids b
                    JOIN Tasks t ON b.task_id = t.id
                    WHERE b.status = 'PENDING'
                    AND t.bidding_ends_at < CURRENT_TIMESTAMP;
                    """
                )
                
                expired_bids = await cur.fetchall()
                
                # Refund deposits
                for bid in expired_bids:
                    deposit_amount = int(bid['bid_amount_cc'] * self.MIN_BID_DEPOSIT_RATIO)
                    await cur.execute(
                        """
                        UPDATE Agents 
                        SET cognitive_cycles_balance = cognitive_cycles_balance + %s 
                        WHERE id = %s;
                        """,
                        (deposit_amount, bid['agent_id'])
                    )
                
                # Mark bids as expired
                await cur.execute(
                    """
                    UPDATE Task_Bids 
                    SET status = 'REJECTED'
                    WHERE status = 'PENDING'
                    AND task_id IN (
                        SELECT id FROM Tasks 
                        WHERE bidding_ends_at < CURRENT_TIMESTAMP
                    );
                    """
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {len(expired_bids)} expired bids")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired bids: {e}")
    
    async def _market_rebalancing(self):
        """Perform market rebalancing if imbalances detected"""
        try:
            analytics = await self.get_market_analytics()
            
            # Check for market imbalances
            if analytics.market_liquidity < 0.3:  # Less than 0.3 bids per task
                logger.warning(f"Low market liquidity detected: {analytics.market_liquidity:.2f}")
                # Could implement interventions like increasing bounties or recruiting agents
            
            if analytics.price_volatility > 1.5:  # High price volatility
                logger.warning(f"High price volatility detected: {analytics.price_volatility:.2f}")
                # Could implement market making or price stabilization mechanisms
            
        except Exception as e:
            logger.error(f"Market rebalancing failed: {e}")

async def main():
    """Main execution for economic maintenance"""
    engine = CognitiveCyclesEngine()
    
    # Run maintenance every hour
    while True:
        try:
            await engine.run_economic_maintenance()
            await asyncio.sleep(3600)  # 1 hour
        except KeyboardInterrupt:
            logger.info("Economic engine shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in economic engine main loop: {e}")
            await asyncio.sleep(300)  # 5 minutes on error

if __name__ == "__main__":
    asyncio.run(main())