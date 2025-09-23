# Project Kairos: Internal Economy System
# Cognitive Cycles (CC) - The Motivation Engine
# Filename: cognitive_cycles.py

import os
import json
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from decimal import Decimal

logger = logging.getLogger(__name__)

class TransactionType(Enum):
    """Types of Cognitive Cycle transactions."""
    TASK_BOUNTY = "task_bounty"
    TASK_COMPLETION = "task_completion"
    PERFORMANCE_BONUS = "performance_bonus"
    EFFICIENCY_PENALTY = "efficiency_penalty"
    RESOURCE_USAGE = "resource_usage"
    MARKET_TRADE = "market_trade"
    SYSTEM_GRANT = "system_grant"

@dataclass
class CCTransaction:
    """Represents a Cognitive Cycles transaction."""
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None
    amount: Decimal = Decimal("0.00")
    transaction_type: TransactionType = TransactionType.SYSTEM_GRANT
    reference_id: Optional[str] = None  # Task ID, Bid ID, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TaskBid:
    """Represents a bid on a task by an agent."""
    bid_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_id: str = ""
    bid_amount: Decimal = Decimal("0.00")
    estimated_completion_time: timedelta = timedelta(hours=1)
    confidence_score: float = 0.0
    specialization_bonus: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent."""
    agent_id: str
    total_tasks_completed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_completion_time: timedelta = timedelta(0)
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    specialization_scores: Dict[str, float] = field(default_factory=dict)
    cc_earned_total: Decimal = Decimal("0.00")
    cc_spent_total: Decimal = Decimal("0.00")
    reputation: float = 0.0

class CognitiveCyclesEngine:
    """
    The Motivation Engine: Manages the internal economy of Cognitive Cycles.
    Drives efficiency and prioritization through economic incentives.
    """
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.engine_id = str(uuid.uuid4())
        
        # Economic parameters
        self.base_cc_generation_rate = Decimal("100.00")  # CC generated per hour
        self.task_completion_multiplier = Decimal("1.5")
        self.quality_bonus_threshold = 0.8
        self.efficiency_bonus_threshold = 0.9
        self.reputation_decay_rate = 0.01  # Per day
        
        # Market dynamics
        self.bid_adjustment_factor = 0.1  # How much bids adjust based on competition
        self.market_volatility = 0.2  # Random variation in task values
        
        # Performance tracking
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.transaction_ledger: List[CCTransaction] = []
        
    async def generate_cognitive_cycles(self) -> Decimal:
        """
        Generate new Cognitive Cycles based on system performance and resource availability.
        This is the "mining" mechanism of the internal economy.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check system performance metrics
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT a.id) as active_agents,
                        COUNT(t.id) FILTER (WHERE t.status = 'COMPLETED') as completed_tasks,
                        COUNT(t.id) FILTER (WHERE t.status = 'PENDING') as pending_tasks,
                        AVG(EXTRACT(EPOCH FROM (t.completed_at - t.created_at))/3600) 
                            FILTER (WHERE t.status = 'COMPLETED') as avg_completion_hours
                    FROM Agents a
                    LEFT JOIN Tasks t ON t.created_at > NOW() - INTERVAL '1 hour'
                """)
                
                metrics = cur.fetchone()
                
                # Calculate generation rate based on system activity
                activity_factor = Decimal(str(min(metrics['active_agents'] / 10.0, 2.0)))
                efficiency_factor = Decimal(str(min(metrics['completed_tasks'] / 
                                                   max(metrics['pending_tasks'], 1), 1.5)))
                
                generation_amount = (self.base_cc_generation_rate * 
                                   activity_factor * 
                                   efficiency_factor)
                
                # Distribute to Resource Broker for allocation
                cur.execute("""
                    UPDATE Agents 
                    SET wallet_balance_cc = wallet_balance_cc + %s
                    WHERE name = 'ResourceBroker-Prime'
                    RETURNING id
                """, (generation_amount,))
                
                broker = cur.fetchone()
                if broker:
                    # Record transaction
                    await self._record_transaction(
                        conn,
                        CCTransaction(
                            to_agent=broker['id'],
                            amount=generation_amount,
                            transaction_type=TransactionType.SYSTEM_GRANT,
                            metadata={"reason": "Periodic CC generation", "metrics": dict(metrics)}
                        )
                    )
                
                conn.commit()
                logger.info(f"Generated {generation_amount} CC")
                return generation_amount
                
        except Exception as e:
            logger.error(f"Error generating Cognitive Cycles: {e}")
            return Decimal("0.00")
        finally:
            if conn:
                conn.close()
    
    async def post_task_bounty(self, task_id: str, bounty_amount: Decimal, 
                               architect_agent_id: str) -> bool:
        """
        Post a bounty for a task, making it available for bidding.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            with conn.cursor() as cur:
                # Check if architect has sufficient CC
                cur.execute("""
                    SELECT wallet_balance_cc FROM Agents WHERE id = %s
                """, (architect_agent_id,))
                
                balance = cur.fetchone()
                if not balance or balance[0] < bounty_amount:
                    logger.warning(f"Insufficient CC balance for agent {architect_agent_id}")
                    return False
                
                # Deduct from architect's wallet
                cur.execute("""
                    UPDATE Agents 
                    SET wallet_balance_cc = wallet_balance_cc - %s
                    WHERE id = %s
                """, (bounty_amount, architect_agent_id))
                
                # Set task bounty
                cur.execute("""
                    UPDATE Tasks 
                    SET bounty_cc = %s, status = 'OPEN_FOR_BIDDING'
                    WHERE id = %s
                """, (bounty_amount, task_id))
                
                # Record transaction
                await self._record_transaction(
                    conn,
                    CCTransaction(
                        from_agent=architect_agent_id,
                        amount=bounty_amount,
                        transaction_type=TransactionType.TASK_BOUNTY,
                        reference_id=task_id,
                        metadata={"action": "post_bounty"}
                    )
                )
                
                conn.commit()
                logger.info(f"Posted bounty of {bounty_amount} CC for task {task_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error posting task bounty: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    async def submit_bid(self, task_id: str, agent_id: str) -> Optional[TaskBid]:
        """
        Submit a bid for a task. The bid amount is calculated based on:
        - Agent's specialization and past performance
        - Current market conditions
        - Task complexity
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get task details
                cur.execute("""
                    SELECT bounty_cc, description, created_by_agent_id 
                    FROM Tasks WHERE id = %s AND status = 'OPEN_FOR_BIDDING'
                """, (task_id,))
                
                task = cur.fetchone()
                if not task:
                    logger.warning(f"Task {task_id} not available for bidding")
                    return None
                
                # Get agent metrics
                metrics = await self._get_agent_metrics(agent_id)
                
                # Calculate bid amount based on agent's capabilities
                base_bid = Decimal(str(task['bounty_cc']))
                
                # Adjust based on specialization
                specialization_factor = Decimal(str(1.0 - metrics.specialization_scores.get(
                    self._classify_task(task['description']), 0.0) * 0.3))
                
                # Adjust based on efficiency
                efficiency_factor = Decimal(str(1.0 - metrics.efficiency_score * 0.2))
                
                # Add market volatility
                volatility = Decimal(str(np.random.normal(1.0, self.market_volatility)))
                volatility = max(Decimal("0.8"), min(Decimal("1.2"), volatility))
                
                bid_amount = base_bid * specialization_factor * efficiency_factor * volatility
                bid_amount = bid_amount.quantize(Decimal("0.01"))
                
                # Estimate completion time
                avg_time = metrics.average_completion_time.total_seconds() / 3600
                estimated_hours = max(1.0, avg_time * float(efficiency_factor))
                
                # Create bid
                bid = TaskBid(
                    task_id=task_id,
                    agent_id=agent_id,
                    bid_amount=bid_amount,
                    estimated_completion_time=timedelta(hours=estimated_hours),
                    confidence_score=metrics.reputation,
                    specialization_bonus=float(specialization_factor)
                )
                
                # Check for existing bid
                cur.execute("""
                    SELECT id FROM Bids WHERE task_id = %s AND agent_id = %s
                """, (task_id, agent_id))
                
                existing_bid = cur.fetchone()
                
                if existing_bid:
                    # Update existing bid
                    cur.execute("""
                        UPDATE Bids 
                        SET bid_amount_cc = %s, created_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (bid_amount, existing_bid['id']))
                else:
                    # Insert new bid
                    cur.execute("""
                        INSERT INTO Bids (task_id, agent_id, bid_amount_cc)
                        VALUES (%s, %s, %s)
                        RETURNING id
                    """, (task_id, agent_id, bid_amount))
                
                conn.commit()
                logger.info(f"Agent {agent_id} bid {bid_amount} CC on task {task_id}")
                return bid
                
        except Exception as e:
            logger.error(f"Error submitting bid: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    async def evaluate_bids(self, task_id: str) -> Optional[str]:
        """
        Evaluate all bids for a task and select the winner.
        Returns the winning agent ID.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all bids for the task
                cur.execute("""
                    SELECT b.*, a.wallet_balance_cc, a.type
                    FROM Bids b
                    JOIN Agents a ON b.agent_id = a.id
                    WHERE b.task_id = %s
                    ORDER BY b.bid_amount_cc ASC, b.created_at ASC
                """, (task_id,))
                
                bids = cur.fetchall()
                
                if not bids:
                    logger.warning(f"No bids for task {task_id}")
                    return None
                
                # Evaluate each bid
                best_score = -1
                winner = None
                
                for bid in bids:
                    # Get agent metrics
                    metrics = await self._get_agent_metrics(bid['agent_id'])
                    
                    # Calculate bid score
                    price_score = 1.0 / (1.0 + float(bid['bid_amount_cc']) / 100.0)
                    reputation_score = metrics.reputation
                    efficiency_score = metrics.efficiency_score
                    
                    # Weighted scoring
                    total_score = (price_score * 0.4 + 
                                 reputation_score * 0.4 + 
                                 efficiency_score * 0.2)
                    
                    if total_score > best_score:
                        best_score = total_score
                        winner = bid
                
                if winner:
                    # Assign task to winner
                    cur.execute("""
                        UPDATE Tasks 
                        SET assigned_to_agent_id = %s, 
                            status = 'IN_PROGRESS',
                            assigned_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (winner['agent_id'], task_id))
                    
                    # Record the winning bid
                    cur.execute("""
                        INSERT INTO Decisions (venture_id, agent_id, triggered_by_event, 
                                             rationale, consulted_data_sources)
                        SELECT t.venture_id, %s, 'BID_EVALUATION', %s, %s
                        FROM Tasks t WHERE t.id = %s
                    """, (
                        winner['agent_id'],
                        f"Won bid for task with amount {winner['bid_amount_cc']} CC",
                        json.dumps({"bid_score": best_score, "competing_bids": len(bids)}),
                        task_id
                    ))
                    
                    conn.commit()
                    logger.info(f"Agent {winner['agent_id']} won task {task_id}")
                    return winner['agent_id']
                
        except Exception as e:
            logger.error(f"Error evaluating bids: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    async def complete_task(self, task_id: str, agent_id: str, 
                           quality_score: float = 1.0) -> Decimal:
        """
        Complete a task and distribute the bounty to the agent.
        Returns the amount earned (including bonuses).
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get task details
                cur.execute("""
                    SELECT bounty_cc, assigned_at, created_by_agent_id
                    FROM Tasks 
                    WHERE id = %s AND assigned_to_agent_id = %s AND status = 'IN_PROGRESS'
                """, (task_id, agent_id))
                
                task = cur.fetchone()
                if not task:
                    logger.warning(f"Task {task_id} not found or not assigned to {agent_id}")
                    return Decimal("0.00")
                
                # Calculate completion time
                completion_time = datetime.utcnow() - task['assigned_at']
                
                # Calculate earnings
                base_amount = Decimal(str(task['bounty_cc']))
                
                # Quality bonus
                quality_bonus = Decimal("0.00")
                if quality_score >= self.quality_bonus_threshold:
                    quality_bonus = base_amount * Decimal("0.2")
                
                # Efficiency bonus (if completed quickly)
                efficiency_bonus = Decimal("0.00")
                if completion_time < timedelta(hours=2):
                    efficiency_bonus = base_amount * Decimal("0.1")
                
                total_amount = base_amount + quality_bonus + efficiency_bonus
                
                # Update agent's wallet
                cur.execute("""
                    UPDATE Agents 
                    SET wallet_balance_cc = wallet_balance_cc + %s
                    WHERE id = %s
                """, (total_amount, agent_id))
                
                # Update task status
                cur.execute("""
                    UPDATE Tasks 
                    SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (task_id,))
                
                # Record transaction
                await self._record_transaction(
                    conn,
                    CCTransaction(
                        to_agent=agent_id,
                        amount=total_amount,
                        transaction_type=TransactionType.TASK_COMPLETION,
                        reference_id=task_id,
                        metadata={
                            "base_amount": str(base_amount),
                            "quality_bonus": str(quality_bonus),
                            "efficiency_bonus": str(efficiency_bonus),
                            "quality_score": quality_score,
                            "completion_time_hours": completion_time.total_seconds() / 3600
                        }
                    )
                )
                
                # Update agent metrics
                await self._update_agent_metrics(agent_id, task_id, True, 
                                                completion_time, quality_score)
                
                conn.commit()
                logger.info(f"Task {task_id} completed. Agent {agent_id} earned {total_amount} CC")
                return total_amount
                
        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return Decimal("0.00")
        finally:
            if conn:
                conn.close()
    
    async def _get_agent_metrics(self, agent_id: str) -> AgentPerformanceMetrics:
        """Get or calculate performance metrics for an agent."""
        if agent_id in self.agent_metrics:
            return self.agent_metrics[agent_id]
        
        metrics = AgentPerformanceMetrics(agent_id=agent_id)
        
        try:
            conn = psycopg2.connect(**self.db_config)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get historical performance
                cur.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed,
                        COUNT(*) FILTER (WHERE status = 'FAILED') as failed,
                        AVG(EXTRACT(EPOCH FROM (completed_at - assigned_at))/3600) 
                            FILTER (WHERE status = 'COMPLETED') as avg_hours,
                        SUM(bounty_cc) FILTER (WHERE status = 'COMPLETED') as total_earned
                    FROM Tasks 
                    WHERE assigned_to_agent_id = %s
                """, (agent_id,))
                
                data = cur.fetchone()
                
                if data:
                    metrics.total_tasks_completed = data['completed'] or 0
                    metrics.successful_tasks = data['completed'] or 0
                    metrics.failed_tasks = data['failed'] or 0
                    metrics.average_completion_time = timedelta(hours=data['avg_hours'] or 1)
                    metrics.cc_earned_total = Decimal(str(data['total_earned'] or 0))
                    
                    # Calculate derived metrics
                    if metrics.total_tasks_completed > 0:
                        metrics.quality_score = metrics.successful_tasks / metrics.total_tasks_completed
                        metrics.efficiency_score = min(1.0, 2.0 / max(data['avg_hours'] or 1, 0.1))
                        metrics.reputation = (metrics.quality_score * 0.7 + 
                                            metrics.efficiency_score * 0.3)
                
                # Cache the metrics
                self.agent_metrics[agent_id] = metrics
                
        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
        finally:
            if conn:
                conn.close()
        
        return metrics
    
    async def _update_agent_metrics(self, agent_id: str, task_id: str, 
                                   success: bool, completion_time: timedelta, 
                                   quality_score: float):
        """Update agent performance metrics after task completion."""
        metrics = await self._get_agent_metrics(agent_id)
        
        # Update counts
        metrics.total_tasks_completed += 1
        if success:
            metrics.successful_tasks += 1
        else:
            metrics.failed_tasks += 1
        
        # Update average completion time (exponential moving average)
        alpha = 0.2  # Smoothing factor
        new_avg = (alpha * completion_time + 
                  (1 - alpha) * metrics.average_completion_time)
        metrics.average_completion_time = new_avg
        
        # Update quality score
        metrics.quality_score = (metrics.quality_score * 0.9 + quality_score * 0.1)
        
        # Update efficiency score
        target_time = timedelta(hours=2)  # Target completion time
        efficiency = min(1.0, target_time.total_seconds() / completion_time.total_seconds())
        metrics.efficiency_score = (metrics.efficiency_score * 0.9 + efficiency * 0.1)
        
        # Update reputation
        metrics.reputation = (metrics.quality_score * 0.7 + metrics.efficiency_score * 0.3)
        
        # Update cache
        self.agent_metrics[agent_id] = metrics
    
    async def _record_transaction(self, conn, transaction: CCTransaction):
        """Record a CC transaction in the database."""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO CCTransactions 
                    (id, from_agent_id, to_agent_id, amount, transaction_type, 
                     reference_id, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    transaction.transaction_id,
                    transaction.from_agent,
                    transaction.to_agent,
                    transaction.amount,
                    transaction.transaction_type.value,
                    transaction.reference_id,
                    json.dumps(transaction.metadata),
                    transaction.timestamp
                ))
            
            # Also keep in memory for quick access
            self.transaction_ledger.append(transaction)
            
        except Exception as e:
            logger.error(f"Error recording transaction: {e}")
    
    def _classify_task(self, description: str) -> str:
        """Classify a task based on its description."""
        # Simple keyword-based classification
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['code', 'implement', 'develop', 'fix']):
            return 'engineering'
        elif any(word in description_lower for word in ['simulate', 'predict', 'analyze']):
            return 'analysis'
        elif any(word in description_lower for word in ['test', 'validate', 'verify']):
            return 'quality_assurance'
        elif any(word in description_lower for word in ['plan', 'design', 'architect']):
            return 'architecture'
        else:
            return 'general'

# Main execution
async def main():
    """Main execution loop for the Cognitive Cycles Engine."""
    db_config = {
        "dbname": os.environ.get("DB_NAME", "kairos_db"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASSWORD", "password"),
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": os.environ.get("DB_PORT", "5432")
    }
    
    engine = CognitiveCyclesEngine(db_config)
    logger.info("Cognitive Cycles Engine initialized")
    
    while True:
        try:
            # Generate new CC every hour
            await engine.generate_cognitive_cycles()
            
            # Process pending task auctions
            # (In production, this would be more sophisticated)
            
            await asyncio.sleep(3600)  # Run every hour
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received. CC Engine terminating.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())