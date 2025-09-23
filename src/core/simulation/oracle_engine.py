# Project Kairos: Pre-Cognitive Simulation Engine
# The Oracle - Exploring Possible Futures
# Filename: oracle_engine.py

import os
import json
import asyncio
import uuid
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from multiprocessing import Pool, cpu_count
import pickle
import hashlib

logger = logging.getLogger(__name__)

class PersonaType(Enum):
    """Types of simulated user personas."""
    EARLY_ADOPTER = "early_adopter"
    MAINSTREAM_USER = "mainstream_user"
    ENTERPRISE_BUYER = "enterprise_buyer"
    TECHNICAL_USER = "technical_user"
    CASUAL_USER = "casual_user"
    POWER_USER = "power_user"
    SKEPTIC = "skeptic"

class EventType(Enum):
    """Types of black swan events."""
    COMPETITOR_LAUNCH = "competitor_launch"
    MARKET_CRASH = "market_crash"
    VIRAL_GROWTH = "viral_growth"
    SECURITY_BREACH = "security_breach"
    REGULATORY_CHANGE = "regulatory_change"
    TECHNOLOGY_DISRUPTION = "technology_disruption"
    PARTNERSHIP_OPPORTUNITY = "partnership_opportunity"

@dataclass
class UserPersona:
    """Represents a simulated user in the market digital twin."""
    persona_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    persona_type: PersonaType = PersonaType.MAINSTREAM_USER
    age: int = 30
    income_level: float = 50000.0
    tech_savviness: float = 0.5  # 0 to 1
    price_sensitivity: float = 0.5  # 0 to 1
    feature_preferences: Dict[str, float] = field(default_factory=dict)
    brand_loyalty: float = 0.3
    social_influence: float = 0.5
    adoption_threshold: float = 0.6
    current_satisfaction: float = 0.0
    churn_probability: float = 0.0
    lifetime_value: float = 0.0
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ProductVariant:
    """Represents a variant of the product being tested."""
    variant_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    pricing_model: Dict[str, float] = field(default_factory=dict)
    marketing_strategy: Dict[str, Any] = field(default_factory=dict)
    target_segments: List[PersonaType] = field(default_factory=list)
    development_cost: float = 0.0
    time_to_market: timedelta = timedelta(days=90)

@dataclass
class SimulationResult:
    """Results from a simulation run."""
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    variant_id: str = ""
    total_users: int = 0
    active_users: int = 0
    revenue: float = 0.0
    costs: float = 0.0
    profit: float = 0.0
    market_share: float = 0.0
    user_satisfaction: float = 0.0
    viral_coefficient: float = 0.0
    retention_rate: float = 0.0
    growth_rate: float = 0.0
    risk_events_survived: int = 0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BlackSwanEvent:
    """Represents a high-impact, low-probability event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.MARKET_CRASH
    probability: float = 0.01
    impact_magnitude: float = 0.5  # -1 to 1, negative is bad
    duration: timedelta = timedelta(days=30)
    affected_segments: List[PersonaType] = field(default_factory=list)
    market_effects: Dict[str, float] = field(default_factory=dict)

class MarketSimulator:
    """
    The Oracle: Pre-Cognitive Simulation Engine.
    Creates digital twins of markets and simulates thousands of futures.
    """
    
    def __init__(self, db_config: Dict[str, str], num_personas: int = 10000):
        self.db_config = db_config
        self.num_personas = num_personas
        self.simulation_id = str(uuid.uuid4())
        
        # Market parameters
        self.market_size = 1000000  # Total addressable market
        self.competitor_strength = 0.3  # Competitor market control
        self.market_growth_rate = 0.15  # Annual growth rate
        self.network_effect_threshold = 0.1  # When network effects kick in
        
        # Simulation parameters
        self.time_steps_per_day = 4
        self.monte_carlo_runs = 100
        self.confidence_level = 0.95
        
        # Initialize market
        self.personas: List[UserPersona] = []
        self.active_simulations: Dict[str, Any] = {}
        self.simulation_cache: Dict[str, SimulationResult] = {}
        
        # Initialize persona population
        self._initialize_personas()
    
    def _initialize_personas(self):
        """Create a diverse population of user personas."""
        persona_distribution = {
            PersonaType.EARLY_ADOPTER: 0.025,
            PersonaType.MAINSTREAM_USER: 0.35,
            PersonaType.ENTERPRISE_BUYER: 0.05,
            PersonaType.TECHNICAL_USER: 0.15,
            PersonaType.CASUAL_USER: 0.30,
            PersonaType.POWER_USER: 0.075,
            PersonaType.SKEPTIC: 0.10
        }
        
        for persona_type, proportion in persona_distribution.items():
            count = int(self.num_personas * proportion)
            for _ in range(count):
                self.personas.append(self._create_persona(persona_type))
    
    def _create_persona(self, persona_type: PersonaType) -> UserPersona:
        """Create a single user persona with realistic attributes."""
        # Base attributes by persona type
        base_attributes = {
            PersonaType.EARLY_ADOPTER: {
                "tech_savviness": np.random.normal(0.9, 0.05),
                "price_sensitivity": np.random.normal(0.3, 0.1),
                "adoption_threshold": np.random.normal(0.3, 0.1),
                "brand_loyalty": np.random.normal(0.2, 0.1)
            },
            PersonaType.MAINSTREAM_USER: {
                "tech_savviness": np.random.normal(0.5, 0.15),
                "price_sensitivity": np.random.normal(0.7, 0.1),
                "adoption_threshold": np.random.normal(0.6, 0.1),
                "brand_loyalty": np.random.normal(0.5, 0.15)
            },
            PersonaType.ENTERPRISE_BUYER: {
                "tech_savviness": np.random.normal(0.7, 0.1),
                "price_sensitivity": np.random.normal(0.4, 0.15),
                "adoption_threshold": np.random.normal(0.7, 0.1),
                "brand_loyalty": np.random.normal(0.6, 0.1)
            },
            PersonaType.TECHNICAL_USER: {
                "tech_savviness": np.random.normal(0.95, 0.03),
                "price_sensitivity": np.random.normal(0.5, 0.15),
                "adoption_threshold": np.random.normal(0.4, 0.1),
                "brand_loyalty": np.random.normal(0.3, 0.1)
            },
            PersonaType.CASUAL_USER: {
                "tech_savviness": np.random.normal(0.3, 0.1),
                "price_sensitivity": np.random.normal(0.8, 0.1),
                "adoption_threshold": np.random.normal(0.7, 0.1),
                "brand_loyalty": np.random.normal(0.4, 0.15)
            },
            PersonaType.POWER_USER: {
                "tech_savviness": np.random.normal(0.8, 0.1),
                "price_sensitivity": np.random.normal(0.4, 0.1),
                "adoption_threshold": np.random.normal(0.5, 0.1),
                "brand_loyalty": np.random.normal(0.7, 0.1)
            },
            PersonaType.SKEPTIC: {
                "tech_savviness": np.random.normal(0.6, 0.15),
                "price_sensitivity": np.random.normal(0.6, 0.15),
                "adoption_threshold": np.random.normal(0.8, 0.1),
                "brand_loyalty": np.random.normal(0.3, 0.15)
            }
        }
        
        attrs = base_attributes.get(persona_type, base_attributes[PersonaType.MAINSTREAM_USER])
        
        # Create persona with constrained random attributes
        persona = UserPersona(
            persona_type=persona_type,
            age=int(np.random.normal(35, 12)),
            income_level=max(20000, np.random.lognormal(10.5, 0.8)),
            tech_savviness=np.clip(attrs["tech_savviness"], 0, 1),
            price_sensitivity=np.clip(attrs["price_sensitivity"], 0, 1),
            adoption_threshold=np.clip(attrs["adoption_threshold"], 0, 1),
            brand_loyalty=np.clip(attrs["brand_loyalty"], 0, 1),
            social_influence=np.clip(np.random.normal(0.5, 0.2), 0, 1)
        )
        
        # Generate feature preferences
        features = ["performance", "usability", "price", "support", "integration", "security"]
        for feature in features:
            persona.feature_preferences[feature] = np.clip(np.random.normal(0.5, 0.2), 0, 1)
        
        return persona
    
    async def simulate_product_launch(self, variant: ProductVariant, 
                                     duration_days: int = 365,
                                     include_black_swans: bool = True) -> SimulationResult:
        """
        Simulate a product launch with a specific variant.
        Returns aggregated results from multiple Monte Carlo runs.
        """
        # Check cache first
        cache_key = self._get_cache_key(variant, duration_days, include_black_swans)
        if cache_key in self.simulation_cache:
            logger.info(f"Using cached simulation for variant {variant.variant_id}")
            return self.simulation_cache[cache_key]
        
        logger.info(f"Starting simulation for variant: {variant.name}")
        
        # Run Monte Carlo simulations in parallel
        with Pool(processes=min(cpu_count(), 8)) as pool:
            sim_args = [(variant, duration_days, include_black_swans, seed) 
                       for seed in range(self.monte_carlo_runs)]
            
            results = pool.starmap(self._run_single_simulation, sim_args)
        
        # Aggregate results
        aggregated = self._aggregate_simulation_results(results, variant.variant_id)
        
        # Cache result
        self.simulation_cache[cache_key] = aggregated
        
        # Record in database
        await self._record_simulation_result(aggregated, variant)
        
        return aggregated
    
    def _run_single_simulation(self, variant: ProductVariant, 
                               duration_days: int,
                               include_black_swans: bool,
                               seed: int) -> Dict[str, Any]:
        """Run a single simulation instance."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize simulation state
        adopted_users: Set[str] = set()
        churned_users: Set[str] = set()
        daily_revenue = []
        daily_costs = []
        daily_users = []
        satisfaction_scores = []
        
        # Generate black swan events if enabled
        black_swan_events = []
        if include_black_swans:
            black_swan_events = self._generate_black_swan_events(duration_days)
        
        # Simulate each day
        for day in range(duration_days):
            # Check for black swan events
            active_events = [e for e in black_swan_events 
                           if e['start_day'] <= day < e['end_day']]
            
            # Calculate market conditions
            market_modifier = 1.0
            for event in active_events:
                market_modifier *= (1 + event['impact'])
            
            # Simulate user interactions for the day
            day_revenue = 0
            day_satisfaction = []
            new_adoptions = 0
            
            for persona in random.sample(self.personas, 
                                        min(1000, len(self.personas))):
                if persona.persona_id not in adopted_users and \
                   persona.persona_id not in churned_users:
                    # Check for adoption
                    adoption_prob = self._calculate_adoption_probability(
                        persona, variant, len(adopted_users), market_modifier
                    )
                    
                    if random.random() < adoption_prob:
                        adopted_users.add(persona.persona_id)
                        new_adoptions += 1
                
                elif persona.persona_id in adopted_users:
                    # Calculate satisfaction and potential churn
                    satisfaction = self._calculate_satisfaction(persona, variant)
                    day_satisfaction.append(satisfaction)
                    
                    # Generate revenue
                    revenue = self._calculate_daily_revenue(persona, variant)
                    day_revenue += revenue * market_modifier
                    
                    # Check for churn
                    churn_prob = self._calculate_churn_probability(
                        persona, satisfaction, day
                    )
                    
                    if random.random() < churn_prob:
                        adopted_users.remove(persona.persona_id)
                        churned_users.add(persona.persona_id)
            
            # Record daily metrics
            daily_revenue.append(day_revenue)
            daily_costs.append(variant.development_cost / duration_days)
            daily_users.append(len(adopted_users))
            if day_satisfaction:
                satisfaction_scores.append(np.mean(day_satisfaction))
            
            # Network effects
            if len(adopted_users) / self.num_personas > self.network_effect_threshold:
                # Boost adoption rate due to network effects
                market_modifier *= 1.1
        
        # Calculate final metrics
        total_revenue = sum(daily_revenue)
        total_costs = sum(daily_costs)
        final_users = len(adopted_users)
        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0
        
        retention_rate = 1 - (len(churned_users) / max(len(adopted_users) + len(churned_users), 1))
        
        growth_rate = 0
        if len(daily_users) > 30:
            early_users = np.mean(daily_users[:30])
            late_users = np.mean(daily_users[-30:])
            if early_users > 0:
                growth_rate = (late_users - early_users) / early_users
        
        return {
            "total_revenue": total_revenue,
            "total_costs": total_costs,
            "final_users": final_users,
            "retention_rate": retention_rate,
            "satisfaction": avg_satisfaction,
            "growth_rate": growth_rate,
            "black_swan_survived": len(black_swan_events)
        }
    
    def _calculate_adoption_probability(self, persona: UserPersona, 
                                       variant: ProductVariant,
                                       current_adopters: int,
                                       market_modifier: float) -> float:
        """Calculate the probability of a persona adopting the product."""
        # Base probability from persona type match
        base_prob = 0.01
        if persona.persona_type in variant.target_segments:
            base_prob = 0.05
        
        # Adjust for feature preferences
        feature_match = 0
        for feature, importance in persona.feature_preferences.items():
            if feature in variant.features:
                feature_match += importance * variant.features[feature]
        
        feature_match /= max(len(persona.feature_preferences), 1)
        
        # Price sensitivity adjustment
        price_factor = 1.0
        if "monthly" in variant.pricing_model:
            affordability = variant.pricing_model["monthly"] / (persona.income_level / 12)
            price_factor = 1 / (1 + affordability * persona.price_sensitivity * 10)
        
        # Social influence (network effects)
        social_factor = 1 + (current_adopters / self.num_personas) * persona.social_influence
        
        # Combine factors
        adoption_prob = base_prob * feature_match * price_factor * social_factor * market_modifier
        
        # Apply adoption threshold
        if adoption_prob < persona.adoption_threshold * 0.1:
            adoption_prob = 0
        
        return min(adoption_prob, 1.0)
    
    def _calculate_satisfaction(self, persona: UserPersona, 
                               variant: ProductVariant) -> float:
        """Calculate user satisfaction with the product."""
        satisfaction = 0.5  # Base satisfaction
        
        # Feature satisfaction
        for feature, importance in persona.feature_preferences.items():
            if feature in variant.features:
                satisfaction += importance * variant.features[feature] * 0.3
        
        # Price satisfaction
        if "monthly" in variant.pricing_model:
            price_value = 1 / (1 + variant.pricing_model["monthly"] / 100)
            satisfaction += price_value * persona.price_sensitivity * 0.2
        
        # Tech savviness alignment
        if "complexity" in variant.features:
            complexity_match = 1 - abs(variant.features["complexity"] - persona.tech_savviness)
            satisfaction += complexity_match * 0.2
        
        return min(max(satisfaction, 0), 1)
    
    def _calculate_daily_revenue(self, persona: UserPersona, 
                                variant: ProductVariant) -> float:
        """Calculate daily revenue from a user."""
        if "monthly" in variant.pricing_model:
            return variant.pricing_model["monthly"] / 30
        elif "annual" in variant.pricing_model:
            return variant.pricing_model["annual"] / 365
        elif "one_time" in variant.pricing_model:
            # One-time payment already made
            return 0
        else:
            # Usage-based or other model
            return variant.pricing_model.get("daily", 0)
    
    def _calculate_churn_probability(self, persona: UserPersona, 
                                    satisfaction: float, day: int) -> float:
        """Calculate the probability of a user churning."""
        # Base churn rate
        base_churn = 0.002  # 0.2% daily = ~6% monthly
        
        # Satisfaction factor
        satisfaction_factor = 2 * (1 - satisfaction)
        
        # Time factor (users more likely to churn early)
        time_factor = 1.5 if day < 30 else 0.8
        
        # Loyalty factor
        loyalty_factor = 1 - persona.brand_loyalty * 0.5
        
        return base_churn * satisfaction_factor * time_factor * loyalty_factor
    
    def _generate_black_swan_events(self, duration_days: int) -> List[Dict[str, Any]]:
        """Generate random black swan events for the simulation period."""
        events = []
        
        event_types = [
            (EventType.COMPETITOR_LAUNCH, 0.05, -0.3, 60),
            (EventType.MARKET_CRASH, 0.02, -0.5, 90),
            (EventType.VIRAL_GROWTH, 0.03, 0.8, 30),
            (EventType.SECURITY_BREACH, 0.02, -0.6, 14),
            (EventType.REGULATORY_CHANGE, 0.04, -0.2, 180),
            (EventType.PARTNERSHIP_OPPORTUNITY, 0.03, 0.4, 60)
        ]
        
        for event_type, probability, impact, duration in event_types:
            if random.random() < probability:
                start_day = random.randint(30, max(31, duration_days - duration))
                events.append({
                    "type": event_type.value,
                    "start_day": start_day,
                    "end_day": start_day + duration,
                    "impact": impact
                })
        
        return events
    
    def _aggregate_simulation_results(self, results: List[Dict[str, Any]], 
                                     variant_id: str) -> SimulationResult:
        """Aggregate results from multiple simulation runs."""
        # Extract metrics
        revenues = [r["total_revenue"] for r in results]
        costs = [r["total_costs"] for r in results]
        users = [r["final_users"] for r in results]
        retention = [r["retention_rate"] for r in results]
        satisfaction = [r["satisfaction"] for r in results]
        growth = [r["growth_rate"] for r in results]
        black_swans = [r["black_swan_survived"] for r in results]
        
        # Calculate statistics
        mean_revenue = np.mean(revenues)
        mean_profit = mean_revenue - np.mean(costs)
        
        # Calculate confidence intervals
        revenue_ci = np.percentile(revenues, [2.5, 97.5])
        
        # Calculate viral coefficient (simplified)
        mean_users = np.mean(users)
        viral_coeff = 0
        if mean_users > 100:
            # Estimate based on growth pattern
            viral_coeff = max(0, np.mean(growth) / 10)
        
        return SimulationResult(
            variant_id=variant_id,
            total_users=int(mean_users),
            active_users=int(mean_users * np.mean(retention)),
            revenue=mean_revenue,
            costs=np.mean(costs),
            profit=mean_profit,
            market_share=mean_users / self.market_size,
            user_satisfaction=np.mean(satisfaction),
            viral_coefficient=viral_coeff,
            retention_rate=np.mean(retention),
            growth_rate=np.mean(growth),
            risk_events_survived=int(np.mean(black_swans)),
            confidence_interval=tuple(revenue_ci),
            metadata={
                "simulation_runs": len(results),
                "duration_days": len(results[0]) if results else 0,
                "persona_count": self.num_personas
            }
        )
    
    def _get_cache_key(self, variant: ProductVariant, duration: int, 
                      black_swans: bool) -> str:
        """Generate a cache key for simulation results."""
        key_data = {
            "features": variant.features,
            "pricing": variant.pricing_model,
            "duration": duration,
            "black_swans": black_swans
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def _record_simulation_result(self, result: SimulationResult, 
                                       variant: ProductVariant):
        """Record simulation results in the database."""
        try:
            conn = psycopg2.connect(**self.db_config)
            with conn.cursor() as cur:
                # Get current venture ID
                cur.execute("""
                    SELECT id FROM Ventures 
                    WHERE status = 'IN_PROGRESS' 
                    ORDER BY created_at DESC LIMIT 1
                """)
                venture = cur.fetchone()
                venture_id = venture[0] if venture else None
                
                # Record the simulation decision
                cur.execute("""
                    INSERT INTO Decisions 
                    (venture_id, agent_id, triggered_by_event, rationale, consulted_data_sources)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    venture_id,
                    self.simulation_id,
                    'MARKET_SIMULATION',
                    f"Simulated {variant.name}: {result.total_users} users, "
                    f"${result.revenue:.2f} revenue, {result.retention_rate:.1%} retention",
                    json.dumps({
                        "variant": variant.name,
                        "results": {
                            "users": result.total_users,
                            "revenue": result.revenue,
                            "profit": result.profit,
                            "retention": result.retention_rate,
                            "satisfaction": result.user_satisfaction,
                            "market_share": result.market_share,
                            "confidence_interval": result.confidence_interval
                        },
                        "metadata": result.metadata
                    })
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording simulation result: {e}")

class OracleEngine:
    """
    Main interface to the Pre-Cognitive Simulation Engine.
    Coordinates market simulations and A/B testing.
    """
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.market_simulator = MarketSimulator(db_config)
        self.engine_id = str(uuid.uuid4())
    
    async def run_ab_test(self, variants: List[ProductVariant], 
                         duration_days: int = 365) -> Dict[str, SimulationResult]:
        """
        Run A/B testing on multiple product variants.
        Returns results for each variant.
        """
        logger.info(f"Starting A/B test with {len(variants)} variants")
        
        results = {}
        for variant in variants:
            result = await self.market_simulator.simulate_product_launch(
                variant, duration_days, include_black_swans=True
            )
            results[variant.variant_id] = result
        
        # Find the winner
        best_variant = max(results.items(), key=lambda x: x[1].profit)
        
        logger.info(f"A/B test complete. Winner: {best_variant[0]} "
                   f"with profit ${best_variant[1].profit:.2f}")
        
        return results
    
    async def test_resilience(self, variant: ProductVariant, 
                             num_scenarios: int = 50) -> Dict[str, Any]:
        """
        Test product resilience against various black swan scenarios.
        """
        logger.info(f"Testing resilience for variant: {variant.name}")
        
        survival_rate = 0
        impact_scores = []
        
        for _ in range(num_scenarios):
            result = await self.market_simulator.simulate_product_launch(
                variant, duration_days=365, include_black_swans=True
            )
            
            # Consider it "survived" if still profitable
            if result.profit > 0:
                survival_rate += 1
            
            # Calculate impact score
            baseline = await self.market_simulator.simulate_product_launch(
                variant, duration_days=365, include_black_swans=False
            )
            
            impact = (baseline.profit - result.profit) / max(baseline.profit, 1)
            impact_scores.append(impact)
        
        return {
            "survival_rate": survival_rate / num_scenarios,
            "average_impact": np.mean(impact_scores),
            "worst_case_impact": max(impact_scores),
            "resilience_score": 1 - np.mean(impact_scores)
        }

# Main execution
async def main():
    """Main execution for the Oracle Engine."""
    db_config = {
        "dbname": os.environ.get("DB_NAME", "kairos_db"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASSWORD", "password"),
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": os.environ.get("DB_PORT", "5432")
    }
    
    oracle = OracleEngine(db_config)
    
    # Example: Create product variants for testing
    variants = [
        ProductVariant(
            name="Premium Version",
            features={"performance": 0.9, "usability": 0.7, "security": 0.95},
            pricing_model={"monthly": 49.99},
            target_segments=[PersonaType.ENTERPRISE_BUYER, PersonaType.POWER_USER]
        ),
        ProductVariant(
            name="Freemium Version",
            features={"performance": 0.6, "usability": 0.9, "security": 0.7},
            pricing_model={"monthly": 0, "premium": 19.99},
            target_segments=[PersonaType.CASUAL_USER, PersonaType.MAINSTREAM_USER]
        ),
        ProductVariant(
            name="Professional Version",
            features={"performance": 0.8, "usability": 0.8, "security": 0.85},
            pricing_model={"monthly": 29.99},
            target_segments=[PersonaType.TECHNICAL_USER, PersonaType.POWER_USER]
        )
    ]
    
    # Run A/B testing
    results = await oracle.run_ab_test(variants)
    
    for variant_id, result in results.items():
        logger.info(f"Variant {variant_id}: Revenue ${result.revenue:.2f}, "
                   f"Users: {result.total_users}, Retention: {result.retention_rate:.1%}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())