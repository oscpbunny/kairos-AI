-- Project Kairos: Enhanced Causal Ledger Schema
-- Phase 2: The Cognitive Substrate & Internal Economy
-- Language: PostgreSQL

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For similarity searches
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For advanced indexing

-- =====================================================
-- CORE ENTITIES: The Foundation of the ADO
-- =====================================================

-- The Ventures table: Strategic missions given to the ADO
CREATE TABLE IF NOT EXISTS Ventures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    objective TEXT NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('DIRECTIVE_RECEIVED', 'IN_PROGRESS', 'COMPLETED', 'ARCHIVED')),
    strategic_priority INTEGER DEFAULT 5 CHECK (strategic_priority BETWEEN 1 AND 10),
    estimated_budget_cc BIGINT, -- Estimated cognitive cycles needed
    actual_cost_cc BIGINT DEFAULT 0, -- Actual cognitive cycles spent
    success_metrics JSONB, -- KPIs and success criteria
    market_context JSONB, -- Market research and competitive analysis
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    plan_generated_at TIMESTAMP WITH TIME ZONE
);

-- Enhanced Agents table: The cognitive workforce with specialization tracking
CREATE TABLE IF NOT EXISTS Agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    specialization VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL CHECK (agent_type IN ('STEWARD', 'ARCHITECT', 'ENGINEER', 'STRATEGIST', 'EMPATHY', 'ETHICIST', 'QA')),
    cognitive_cycles_balance BIGINT DEFAULT 1000 NOT NULL,
    performance_metrics JSONB DEFAULT '{}', -- Success rate, speed, quality scores
    capabilities JSONB DEFAULT '{}', -- Dynamic list of capabilities
    hardware_preference VARCHAR(50) CHECK (hardware_preference IN ('CPU', 'GPU', 'TPU', 'ANY')),
    cost_efficiency_score DECIMAL(5,3) DEFAULT 1.000, -- Higher = more efficient
    reputation_score DECIMAL(5,3) DEFAULT 1.000, -- Based on peer reviews and outcomes
    total_tasks_completed INTEGER DEFAULT 0,
    total_cc_earned BIGINT DEFAULT 0,
    avg_task_completion_time INTERVAL,
    is_active BOOLEAN DEFAULT TRUE,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- THE CAUSAL LEDGER: The Immutable Mind
-- =====================================================

-- Enhanced Decisions table: The central nervous system of the ADO
CREATE TABLE IF NOT EXISTS Decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venture_id UUID NOT NULL REFERENCES Ventures(id),
    agent_id UUID REFERENCES Agents(id),
    parent_decision_id UUID REFERENCES Decisions(id), -- For decision chains
    decision_type VARCHAR(100) NOT NULL, -- 'STRATEGIC', 'OPERATIONAL', 'REACTIVE', 'PREDICTIVE'
    triggered_by_event VARCHAR(255) NOT NULL,
    rationale TEXT NOT NULL,
    confidence_level DECIMAL(3,2) CHECK (confidence_level BETWEEN 0.00 AND 1.00),
    consulted_data_sources JSONB,
    alternative_options JSONB, -- Paths not taken and why
    risk_assessment JSONB, -- Identified risks and mitigations
    expected_outcomes JSONB, -- Predictions and success criteria
    actual_outcomes JSONB, -- Results after execution
    impact_score DECIMAL(5,2), -- Measured impact (-100 to +100)
    cognitive_cycles_invested BIGINT DEFAULT 0,
    decision_latency INTERVAL, -- Time taken to make decision
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    validated_at TIMESTAMP WITH TIME ZONE, -- When outcomes were verified
    lessons_learned TEXT
);

-- Enhanced Tasks table: Actionable work units with bidding system
CREATE TABLE IF NOT EXISTS Tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID NOT NULL REFERENCES Decisions(id),
    venture_id UUID NOT NULL REFERENCES Ventures(id),
    parent_task_id UUID REFERENCES Tasks(id),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    task_type VARCHAR(50) NOT NULL, -- 'ANALYSIS', 'DEVELOPMENT', 'TESTING', 'DEPLOYMENT', 'RESEARCH'
    complexity_level INTEGER CHECK (complexity_level BETWEEN 1 AND 10),
    required_capabilities JSONB, -- Skills/tools needed
    preferred_hardware VARCHAR(50) CHECK (preferred_hardware IN ('CPU', 'GPU', 'TPU', 'ANY')),
    cc_bounty BIGINT NOT NULL DEFAULT 0,
    bonus_cc BIGINT DEFAULT 0, -- Additional rewards for exceptional work
    urgency_multiplier DECIMAL(3,2) DEFAULT 1.00, -- Affects bounty calculation
    status VARCHAR(50) NOT NULL DEFAULT 'BOUNTY_POSTED' CHECK (
        status IN ('BOUNTY_POSTED', 'BIDDING', 'ASSIGNED', 'IN_PROGRESS', 'UNDER_REVIEW', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),
    assigned_to_agent_id UUID REFERENCES Agents(id),
    created_by_agent_id UUID NOT NULL REFERENCES Agents(id),
    quality_score DECIMAL(3,2) CHECK (quality_score BETWEEN 0.00 AND 1.00),
    deliverables JSONB, -- Expected outputs
    actual_deliverables JSONB, -- What was actually delivered
    dependencies JSONB, -- Other tasks this depends on
    estimated_duration INTERVAL,
    actual_duration INTERVAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    bidding_ends_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    reviewed_at TIMESTAMP WITH TIME ZONE
);

-- Task Bidding System: The core of the internal economy
CREATE TABLE IF NOT EXISTS Task_Bids (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES Tasks(id),
    agent_id UUID NOT NULL REFERENCES Agents(id),
    bid_amount_cc BIGINT NOT NULL,
    estimated_completion_time INTERVAL,
    proposed_approach TEXT,
    confidence_score DECIMAL(3,2) CHECK (confidence_score BETWEEN 0.00 AND 1.00),
    risk_factors JSONB,
    past_performance_evidence JSONB, -- Links to previous similar tasks
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'ACCEPTED', 'REJECTED', 'WITHDRAWN')),
    UNIQUE(task_id, agent_id)
);

-- =====================================================
-- SIMULATION ENGINE: The Oracle
-- =====================================================

-- Market Simulations: Digital twins of target markets
CREATE TABLE IF NOT EXISTS Market_Simulations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venture_id UUID NOT NULL REFERENCES Ventures(id),
    simulation_name VARCHAR(255) NOT NULL,
    market_parameters JSONB NOT NULL, -- Demographics, economics, competition
    user_personas JSONB NOT NULL, -- Detailed user archetypes
    simulation_duration INTERVAL,
    simulation_scale INTEGER, -- Number of simulated users
    random_seed BIGINT, -- For reproducible results
    status VARCHAR(20) DEFAULT 'CREATED' CHECK (status IN ('CREATED', 'RUNNING', 'COMPLETED', 'FAILED')),
    results JSONB, -- Outcome metrics and insights
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Simulation Experiments: A/B tests within simulations
CREATE TABLE IF NOT EXISTS Simulation_Experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    simulation_id UUID NOT NULL REFERENCES Market_Simulations(id),
    experiment_name VARCHAR(255) NOT NULL,
    hypothesis TEXT NOT NULL,
    control_parameters JSONB NOT NULL,
    variant_parameters JSONB NOT NULL,
    success_metrics JSONB NOT NULL,
    results JSONB,
    confidence_interval DECIMAL(5,4),
    statistical_significance DECIMAL(5,4),
    recommendation TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Black Swan Events: Stress testing scenarios
CREATE TABLE IF NOT EXISTS Simulation_Events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    simulation_id UUID NOT NULL REFERENCES Market_Simulations(id),
    event_name VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- 'COMPETITOR', 'REGULATORY', 'ECONOMIC', 'TECHNICAL'
    severity INTEGER CHECK (severity BETWEEN 1 AND 10),
    probability DECIMAL(5,4) CHECK (probability BETWEEN 0.0001 AND 1.0000),
    impact_parameters JSONB NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE,
    market_response JSONB,
    recovery_time INTERVAL,
    lessons_learned TEXT
);

-- =====================================================
-- RESOURCE MANAGEMENT: The Steward's Domain
-- =====================================================

-- Infrastructure Resources: Cloud resources under management
CREATE TABLE IF NOT EXISTS Infrastructure_Resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_type VARCHAR(50) NOT NULL, -- 'COMPUTE', 'STORAGE', 'NETWORK', 'DATABASE'
    provider VARCHAR(50) NOT NULL, -- 'AWS', 'GCP', 'AZURE'
    resource_id VARCHAR(255) NOT NULL, -- Provider-specific ID
    configuration JSONB NOT NULL,
    cost_per_hour DECIMAL(10,6),
    performance_metrics JSONB,
    utilization_percentage DECIMAL(5,2),
    status VARCHAR(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'STOPPED', 'TERMINATED', 'ERROR')),
    assigned_to_agent_id UUID REFERENCES Agents(id),
    provisioned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    terminated_at TIMESTAMP WITH TIME ZONE
);

-- Cost Tracking: Real-time expense monitoring
CREATE TABLE IF NOT EXISTS Cost_Records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venture_id UUID REFERENCES Ventures(id),
    resource_id UUID REFERENCES Infrastructure_Resources(id),
    cost_type VARCHAR(50) NOT NULL, -- 'COMPUTE', 'STORAGE', 'NETWORK', 'OTHER'
    amount DECIMAL(12,6) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    billing_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    billing_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    usage_metrics JSONB,
    cost_center VARCHAR(100),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance Metrics: System and agent performance tracking
CREATE TABLE IF NOT EXISTS Performance_Metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN ('AGENT', 'TASK', 'VENTURE', 'SYSTEM')),
    entity_id UUID NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20),
    measurement_context JSONB,
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- VIRTUAL WORKSPACE: Collaborative Environment
-- =====================================================

-- Virtual Workspace Files: Project artifacts and deliverables
CREATE TABLE IF NOT EXISTS Virtual_Workspace_Files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venture_id UUID NOT NULL REFERENCES Ventures(id),
    task_id UUID REFERENCES Tasks(id),
    created_by_agent_id UUID NOT NULL REFERENCES Agents(id),
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT,
    content_hash VARCHAR(64), -- SHA-256 hash for integrity
    file_content TEXT, -- For text files
    binary_content BYTEA, -- For binary files
    version INTEGER DEFAULT 1,
    parent_version_id UUID REFERENCES Virtual_Workspace_Files(id),
    metadata JSONB,
    access_permissions JSONB DEFAULT '{"public": true}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- COMMUNICATION & COLLABORATION
-- =====================================================

-- Agent Communications: Inter-agent messaging and coordination
CREATE TABLE IF NOT EXISTS Agent_Communications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_agent_id UUID NOT NULL REFERENCES Agents(id),
    to_agent_id UUID REFERENCES Agents(id), -- NULL for broadcasts
    venture_id UUID REFERENCES Ventures(id),
    task_id UUID REFERENCES Tasks(id),
    message_type VARCHAR(50) NOT NULL, -- 'REQUEST', 'RESPONSE', 'NOTIFICATION', 'COLLABORATION'
    subject VARCHAR(255),
    content TEXT NOT NULL,
    priority INTEGER DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    requires_response BOOLEAN DEFAULT FALSE,
    parent_message_id UUID REFERENCES Agent_Communications(id),
    read_at TIMESTAMP WITH TIME ZONE,
    responded_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge Base: Shared organizational memory
CREATE TABLE IF NOT EXISTS Knowledge_Entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    tags TEXT[],
    created_by_agent_id UUID NOT NULL REFERENCES Agents(id),
    relevance_score DECIMAL(3,2) DEFAULT 1.00,
    usage_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    is_validated BOOLEAN DEFAULT FALSE,
    validation_source VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- AUDIT & COMPLIANCE
-- =====================================================

-- Activity Logs: Comprehensive audit trail
CREATE TABLE IF NOT EXISTS Activity_Logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES Agents(id),
    action_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action_details JSONB NOT NULL,
    ip_address INET,
    user_agent TEXT,
    session_id UUID,
    severity VARCHAR(20) DEFAULT 'INFO' CHECK (severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Core entity indexes
CREATE INDEX IF NOT EXISTS idx_ventures_status ON Ventures(status);
CREATE INDEX IF NOT EXISTS idx_ventures_priority ON Ventures(strategic_priority DESC);
CREATE INDEX IF NOT EXISTS idx_agents_type_active ON Agents(agent_type, is_active);
CREATE INDEX IF NOT EXISTS idx_agents_specialization ON Agents(specialization);
CREATE INDEX IF NOT EXISTS idx_agents_performance ON Agents(performance_metrics) USING GIN;

-- Decision tracking indexes
CREATE INDEX IF NOT EXISTS idx_decisions_venture ON Decisions(venture_id);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON Decisions(agent_id);
CREATE INDEX IF NOT EXISTS idx_decisions_type_time ON Decisions(decision_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_parent ON Decisions(parent_decision_id);

-- Task management indexes
CREATE INDEX IF NOT EXISTS idx_tasks_status ON Tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_bounty ON Tasks(cc_bounty DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_agent ON Tasks(assigned_to_agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_venture ON Tasks(venture_id);
CREATE INDEX IF NOT EXISTS idx_tasks_complexity ON Tasks(complexity_level DESC);

-- Bidding system indexes
CREATE INDEX IF NOT EXISTS idx_bids_task ON Task_Bids(task_id);
CREATE INDEX IF NOT EXISTS idx_bids_agent ON Task_Bids(agent_id);
CREATE INDEX IF NOT EXISTS idx_bids_amount ON Task_Bids(bid_amount_cc ASC);

-- Simulation indexes
CREATE INDEX IF NOT EXISTS idx_simulations_venture ON Market_Simulations(venture_id);
CREATE INDEX IF NOT EXISTS idx_simulations_status ON Market_Simulations(status);
CREATE INDEX IF NOT EXISTS idx_experiments_simulation ON Simulation_Experiments(simulation_id);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_performance_entity ON Performance_Metrics(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_performance_metric ON Performance_Metrics(metric_name, measured_at DESC);

-- Communication indexes
CREATE INDEX IF NOT EXISTS idx_communications_from_agent ON Agent_Communications(from_agent_id);
CREATE INDEX IF NOT EXISTS idx_communications_to_agent ON Agent_Communications(to_agent_id);
CREATE INDEX IF NOT EXISTS idx_communications_unread ON Agent_Communications(to_agent_id, read_at) WHERE read_at IS NULL;

-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_content_search ON Knowledge_Entries USING GIN(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON Knowledge_Entries USING GIN(tags);

-- =====================================================
-- FUNCTIONS & TRIGGERS
-- =====================================================

-- Function to update agent performance metrics
CREATE OR REPLACE FUNCTION update_agent_performance()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'COMPLETED' AND OLD.status != 'COMPLETED' THEN
        UPDATE Agents 
        SET 
            total_tasks_completed = total_tasks_completed + 1,
            total_cc_earned = total_cc_earned + NEW.cc_bounty + COALESCE(NEW.bonus_cc, 0),
            cognitive_cycles_balance = cognitive_cycles_balance + NEW.cc_bounty + COALESCE(NEW.bonus_cc, 0),
            avg_task_completion_time = (
                COALESCE(avg_task_completion_time * total_tasks_completed, INTERVAL '0') + 
                (NEW.completed_at - NEW.started_at)
            ) / (total_tasks_completed + 1)
        WHERE id = NEW.assigned_to_agent_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update agent performance
CREATE TRIGGER trigger_update_agent_performance
    AFTER UPDATE ON Tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_agent_performance();

-- Function to update venture costs
CREATE OR REPLACE FUNCTION update_venture_costs()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'COMPLETED' AND OLD.status != 'COMPLETED' THEN
        UPDATE Ventures
        SET actual_cost_cc = actual_cost_cc + NEW.cc_bounty + COALESCE(NEW.bonus_cc, 0)
        WHERE id = NEW.venture_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to track venture costs
CREATE TRIGGER trigger_update_venture_costs
    AFTER UPDATE ON Tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_venture_costs();

-- Function to automatically log activities
CREATE OR REPLACE FUNCTION log_significant_activities()
RETURNS TRIGGER AS $$
BEGIN
    -- Log task assignments
    IF TG_TABLE_NAME = 'Tasks' AND NEW.assigned_to_agent_id IS NOT NULL AND OLD.assigned_to_agent_id IS NULL THEN
        INSERT INTO Activity_Logs (agent_id, action_type, entity_type, entity_id, action_details)
        VALUES (NEW.assigned_to_agent_id, 'TASK_ASSIGNED', 'TASK', NEW.id, 
                jsonb_build_object('task_title', NEW.title, 'bounty_cc', NEW.cc_bounty));
    END IF;
    
    -- Log decision making
    IF TG_TABLE_NAME = 'Decisions' AND TG_OP = 'INSERT' THEN
        INSERT INTO Activity_Logs (agent_id, action_type, entity_type, entity_id, action_details)
        VALUES (NEW.agent_id, 'DECISION_MADE', 'DECISION', NEW.id,
                jsonb_build_object('decision_type', NEW.decision_type, 'confidence', NEW.confidence_level));
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for activity logging
CREATE TRIGGER trigger_log_task_activities
    AFTER INSERT OR UPDATE ON Tasks
    FOR EACH ROW
    EXECUTE FUNCTION log_significant_activities();

CREATE TRIGGER trigger_log_decision_activities
    AFTER INSERT ON Decisions
    FOR EACH ROW
    EXECUTE FUNCTION log_significant_activities();

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Agent Performance Dashboard
CREATE OR REPLACE VIEW agent_performance_dashboard AS
SELECT 
    a.id,
    a.name,
    a.specialization,
    a.cognitive_cycles_balance,
    a.total_tasks_completed,
    a.total_cc_earned,
    a.cost_efficiency_score,
    a.reputation_score,
    CASE 
        WHEN a.total_tasks_completed > 0 THEN a.total_cc_earned::DECIMAL / a.total_tasks_completed
        ELSE 0
    END as avg_cc_per_task,
    COUNT(t.id) FILTER (WHERE t.status = 'IN_PROGRESS') as active_tasks,
    COUNT(t.id) FILTER (WHERE t.status = 'COMPLETED' AND t.completed_at > CURRENT_TIMESTAMP - INTERVAL '7 days') as tasks_completed_last_week,
    a.last_heartbeat
FROM Agents a
LEFT JOIN Tasks t ON a.id = t.assigned_to_agent_id
WHERE a.is_active = TRUE
GROUP BY a.id, a.name, a.specialization, a.cognitive_cycles_balance, a.total_tasks_completed, 
         a.total_cc_earned, a.cost_efficiency_score, a.reputation_score, a.last_heartbeat;

-- Venture Progress Overview
CREATE OR REPLACE VIEW venture_progress_overview AS
SELECT 
    v.id,
    v.name,
    v.status,
    v.strategic_priority,
    v.estimated_budget_cc,
    v.actual_cost_cc,
    COUNT(t.id) as total_tasks,
    COUNT(t.id) FILTER (WHERE t.status = 'COMPLETED') as completed_tasks,
    COUNT(t.id) FILTER (WHERE t.status = 'IN_PROGRESS') as active_tasks,
    COUNT(t.id) FILTER (WHERE t.status = 'BOUNTY_POSTED') as pending_tasks,
    CASE 
        WHEN COUNT(t.id) > 0 THEN 
            (COUNT(t.id) FILTER (WHERE t.status = 'COMPLETED')::DECIMAL / COUNT(t.id)) * 100
        ELSE 0
    END as completion_percentage,
    v.created_at,
    v.completed_at
FROM Ventures v
LEFT JOIN Tasks t ON v.id = t.venture_id
GROUP BY v.id, v.name, v.status, v.strategic_priority, v.estimated_budget_cc, 
         v.actual_cost_cc, v.created_at, v.completed_at;

-- Task Market Overview
CREATE OR REPLACE VIEW task_market_overview AS
SELECT 
    t.id,
    t.title,
    t.task_type,
    t.complexity_level,
    t.cc_bounty,
    t.status,
    t.urgency_multiplier,
    COUNT(b.id) as bid_count,
    MIN(b.bid_amount_cc) as lowest_bid,
    MAX(b.bid_amount_cc) as highest_bid,
    AVG(b.bid_amount_cc) as average_bid,
    t.bidding_ends_at,
    t.created_at
FROM Tasks t
LEFT JOIN Task_Bids b ON t.id = b.task_id AND b.status = 'PENDING'
WHERE t.status IN ('BOUNTY_POSTED', 'BIDDING')
GROUP BY t.id, t.title, t.task_type, t.complexity_level, t.cc_bounty, 
         t.status, t.urgency_multiplier, t.bidding_ends_at, t.created_at
ORDER BY t.cc_bounty DESC, t.created_at ASC;

-- =====================================================
-- SEED DATA: Initialize the Enhanced System
-- =====================================================

-- Insert enhanced genesis agents
INSERT INTO Agents (name, specialization, agent_type, cognitive_cycles_balance, capabilities, hardware_preference) VALUES
('Steward-Alpha', 'RESOURCE_MANAGEMENT', 'STEWARD', 50000, 
 '{"cost_optimization", "infrastructure_management", "performance_monitoring", "capacity_planning"}', 'ANY'),
('Architect-Prime', 'STRATEGIC_PLANNING', 'ARCHITECT', 25000,
 '{"system_design", "task_decomposition", "workflow_optimization", "dependency_analysis"}', 'CPU'),
('Engineer-Omega', 'CODE_GENERATION', 'ENGINEER', 15000,
 '{"full_stack_development", "api_design", "database_optimization", "testing"}', 'GPU'),
('Strategist-Sigma', 'MARKET_ANALYSIS', 'STRATEGIST', 30000,
 '{"market_simulation", "predictive_modeling", "competitive_analysis", "trend_forecasting"}', 'TPU'),
('Empathy-Beta', 'USER_RESEARCH', 'EMPATHY', 20000,
 '{"user_persona_modeling", "behavioral_analysis", "journey_mapping", "feedback_synthesis"}', 'CPU'),
('Ethicist-Gamma', 'COMPLIANCE_ETHICS', 'ETHICIST', 18000,
 '{"ethical_review", "bias_detection", "privacy_compliance", "fairness_analysis"}', 'CPU'),
('QA-Delta', 'QUALITY_ASSURANCE', 'QA', 12000,
 '{"automated_testing", "performance_testing", "security_auditing", "code_review"}', 'GPU')
ON CONFLICT (name) DO NOTHING;

-- Insert sample knowledge entries
INSERT INTO Knowledge_Entries (title, content, category, tags, created_by_agent_id) VALUES
('Cognitive Cycles Economy Fundamentals',
 'The CC economy operates on scarcity principles. Agents earn CC by completing tasks efficiently and lose CC when bidding. Higher complexity tasks command higher bounties. Performance history affects bidding power.',
 'ECONOMICS',
 ARRAY['cognitive_cycles', 'economy', 'bidding', 'fundamentals'],
 (SELECT id FROM Agents WHERE name = 'Steward-Alpha')),
('Task Complexity Assessment Guidelines',
 'Task complexity ranges from 1-10. Level 1: Simple data retrieval. Level 5: API development. Level 10: Full system architecture. Consider dependencies, required expertise, and time estimates.',
 'METHODOLOGY',
 ARRAY['tasks', 'complexity', 'assessment', 'guidelines'],
 (SELECT id FROM Agents WHERE name = 'Architect-Prime')),
('Market Simulation Best Practices',
 'Effective simulations require: 1) Diverse user personas, 2) Realistic economic constraints, 3) Competitive landscape modeling, 4) Black swan event injection, 5) Statistical significance validation.',
 'SIMULATION',
 ARRAY['simulation', 'market_analysis', 'best_practices', 'methodology'],
 (SELECT id FROM Agents WHERE name = 'Strategist-Sigma'))
ON CONFLICT DO NOTHING;

-- Create indexes on JSONB columns for better performance
CREATE INDEX IF NOT EXISTS idx_agents_capabilities ON Agents USING GIN(capabilities);
CREATE INDEX IF NOT EXISTS idx_ventures_metrics ON Ventures USING GIN(success_metrics);
CREATE INDEX IF NOT EXISTS idx_decisions_data_sources ON Decisions USING GIN(consulted_data_sources);
CREATE INDEX IF NOT EXISTS idx_tasks_deliverables ON Tasks USING GIN(deliverables);
CREATE INDEX IF NOT EXISTS idx_simulations_parameters ON Market_Simulations USING GIN(market_parameters);

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================

\echo 'Project Kairos Enhanced Schema Successfully Deployed!'
\echo 'The Cognitive Substrate is now operational.'
\echo 'Genesis agents have been seeded with enhanced capabilities.'
\echo 'Internal economy and simulation engine are ready.'
\echo 'The ADO is prepared for prescient operation.'