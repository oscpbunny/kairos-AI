-- Project Kairos: Migration to Enhanced Schema
-- This script upgrades the existing database to the enhanced Kairos architecture
-- Run this AFTER backing up your existing database

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; 
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================
-- BACKUP EXISTING DATA
-- =====================================================

-- Create backup tables for existing data
CREATE TABLE IF NOT EXISTS agents_backup AS SELECT * FROM Agents;
CREATE TABLE IF NOT EXISTS ventures_backup AS SELECT * FROM Ventures;
CREATE TABLE IF NOT EXISTS decisions_backup AS SELECT * FROM Decisions;
CREATE TABLE IF NOT EXISTS tasks_backup AS SELECT * FROM Tasks;

-- =====================================================
-- ENHANCE EXISTING TABLES
-- =====================================================

-- Enhance Ventures table
ALTER TABLE Ventures 
    ADD COLUMN IF NOT EXISTS strategic_priority INTEGER DEFAULT 5 CHECK (strategic_priority BETWEEN 1 AND 10),
    ADD COLUMN IF NOT EXISTS estimated_budget_cc BIGINT,
    ADD COLUMN IF NOT EXISTS actual_cost_cc BIGINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS success_metrics JSONB,
    ADD COLUMN IF NOT EXISTS market_context JSONB,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE;

-- Enhance Agents table
ALTER TABLE Agents 
    ADD COLUMN IF NOT EXISTS agent_type VARCHAR(50) CHECK (agent_type IN ('STEWARD', 'ARCHITECT', 'ENGINEER', 'STRATEGIST', 'EMPATHY', 'ETHICIST', 'QA')),
    ADD COLUMN IF NOT EXISTS performance_metrics JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS capabilities JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS hardware_preference VARCHAR(50) CHECK (hardware_preference IN ('CPU', 'GPU', 'TPU', 'ANY')),
    ADD COLUMN IF NOT EXISTS cost_efficiency_score DECIMAL(5,3) DEFAULT 1.000,
    ADD COLUMN IF NOT EXISTS reputation_score DECIMAL(5,3) DEFAULT 1.000,
    ADD COLUMN IF NOT EXISTS total_tasks_completed INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS total_cc_earned BIGINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_task_completion_time INTERVAL,
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

-- Update existing agents with default agent types
UPDATE Agents SET agent_type = 'STEWARD' WHERE name LIKE 'Steward%' AND agent_type IS NULL;
UPDATE Agents SET agent_type = 'ARCHITECT' WHERE name LIKE 'Architect%' AND agent_type IS NULL;
UPDATE Agents SET agent_type = 'ENGINEER' WHERE name LIKE 'Engineer%' AND agent_type IS NULL;

-- Rename cognitive_cycles_balance if it exists as wallet_balance_cc
DO $$ 
BEGIN 
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'agents' AND column_name = 'wallet_balance_cc') THEN
        ALTER TABLE Agents RENAME COLUMN wallet_balance_cc TO cognitive_cycles_balance;
    END IF;
END $$;

-- Enhance Decisions table
ALTER TABLE Decisions 
    ADD COLUMN IF NOT EXISTS parent_decision_id UUID REFERENCES Decisions(id),
    ADD COLUMN IF NOT EXISTS decision_type VARCHAR(100) DEFAULT 'OPERATIONAL',
    ADD COLUMN IF NOT EXISTS confidence_level DECIMAL(3,2) CHECK (confidence_level BETWEEN 0.00 AND 1.00),
    ADD COLUMN IF NOT EXISTS alternative_options JSONB,
    ADD COLUMN IF NOT EXISTS risk_assessment JSONB,
    ADD COLUMN IF NOT EXISTS expected_outcomes JSONB,
    ADD COLUMN IF NOT EXISTS actual_outcomes JSONB,
    ADD COLUMN IF NOT EXISTS impact_score DECIMAL(5,2),
    ADD COLUMN IF NOT EXISTS cognitive_cycles_invested BIGINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS decision_latency INTERVAL,
    ADD COLUMN IF NOT EXISTS validated_at TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS lessons_learned TEXT;

-- Enhance Tasks table
ALTER TABLE Tasks 
    ADD COLUMN IF NOT EXISTS title VARCHAR(255),
    ADD COLUMN IF NOT EXISTS task_type VARCHAR(50) DEFAULT 'DEVELOPMENT',
    ADD COLUMN IF NOT EXISTS complexity_level INTEGER CHECK (complexity_level BETWEEN 1 AND 10),
    ADD COLUMN IF NOT EXISTS required_capabilities JSONB,
    ADD COLUMN IF NOT EXISTS preferred_hardware VARCHAR(50) CHECK (preferred_hardware IN ('CPU', 'GPU', 'TPU', 'ANY')),
    ADD COLUMN IF NOT EXISTS bonus_cc BIGINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS urgency_multiplier DECIMAL(3,2) DEFAULT 1.00,
    ADD COLUMN IF NOT EXISTS quality_score DECIMAL(3,2) CHECK (quality_score BETWEEN 0.00 AND 1.00),
    ADD COLUMN IF NOT EXISTS deliverables JSONB,
    ADD COLUMN IF NOT EXISTS actual_deliverables JSONB,
    ADD COLUMN IF NOT EXISTS dependencies JSONB,
    ADD COLUMN IF NOT EXISTS estimated_duration INTERVAL,
    ADD COLUMN IF NOT EXISTS actual_duration INTERVAL,
    ADD COLUMN IF NOT EXISTS bidding_ends_at TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP WITH TIME ZONE;

-- Update existing tasks
UPDATE Tasks SET title = LEFT(description, 255) WHERE title IS NULL;

-- Rename existing bounty column if needed
DO $$ 
BEGIN 
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tasks' AND column_name = 'bounty_cc') THEN
        ALTER TABLE Tasks RENAME COLUMN bounty_cc TO cc_bounty;
    END IF;
END $$;

-- Update task status values to new enum
UPDATE Tasks SET status = 'BOUNTY_POSTED' WHERE status = 'PENDING';

-- Add decision_id column and populate it
ALTER TABLE Tasks ADD COLUMN IF NOT EXISTS decision_id UUID;

-- Create placeholder decisions for existing tasks without decision_id
WITH task_decisions AS (
    INSERT INTO Decisions (venture_id, agent_id, decision_type, triggered_by_event, rationale, created_at)
    SELECT 
        t.venture_id,
        t.created_by_agent_id,
        'OPERATIONAL',
        'TASK_CREATION_MIGRATION',
        'Migrated from legacy task system: ' || t.description,
        t.created_at
    FROM Tasks t
    WHERE t.decision_id IS NULL
    RETURNING id, venture_id
)
UPDATE Tasks t 
SET decision_id = td.id
FROM task_decisions td
WHERE t.venture_id = td.venture_id AND t.decision_id IS NULL;

-- Make decision_id NOT NULL after populating
ALTER TABLE Tasks ALTER COLUMN decision_id SET NOT NULL;
ALTER TABLE Tasks ADD CONSTRAINT fk_tasks_decision_id FOREIGN KEY (decision_id) REFERENCES Decisions(id);

-- =====================================================
-- CREATE NEW TABLES
-- =====================================================

-- Task Bidding System
CREATE TABLE IF NOT EXISTS Task_Bids (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES Tasks(id),
    agent_id UUID NOT NULL REFERENCES Agents(id),
    bid_amount_cc BIGINT NOT NULL,
    estimated_completion_time INTERVAL,
    proposed_approach TEXT,
    confidence_score DECIMAL(3,2) CHECK (confidence_score BETWEEN 0.00 AND 1.00),
    risk_factors JSONB,
    past_performance_evidence JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'ACCEPTED', 'REJECTED', 'WITHDRAWN')),
    UNIQUE(task_id, agent_id)
);

-- Market Simulations
CREATE TABLE IF NOT EXISTS Market_Simulations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venture_id UUID NOT NULL REFERENCES Ventures(id),
    simulation_name VARCHAR(255) NOT NULL,
    market_parameters JSONB NOT NULL,
    user_personas JSONB NOT NULL,
    simulation_duration INTERVAL,
    simulation_scale INTEGER,
    random_seed BIGINT,
    status VARCHAR(20) DEFAULT 'CREATED' CHECK (status IN ('CREATED', 'RUNNING', 'COMPLETED', 'FAILED')),
    results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Simulation Experiments
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

-- Simulation Events (Black Swan)
CREATE TABLE IF NOT EXISTS Simulation_Events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    simulation_id UUID NOT NULL REFERENCES Market_Simulations(id),
    event_name VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity INTEGER CHECK (severity BETWEEN 1 AND 10),
    probability DECIMAL(5,4) CHECK (probability BETWEEN 0.0001 AND 1.0000),
    impact_parameters JSONB NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE,
    market_response JSONB,
    recovery_time INTERVAL,
    lessons_learned TEXT
);

-- Infrastructure Resources
CREATE TABLE IF NOT EXISTS Infrastructure_Resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_type VARCHAR(50) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    configuration JSONB NOT NULL,
    cost_per_hour DECIMAL(10,6),
    performance_metrics JSONB,
    utilization_percentage DECIMAL(5,2),
    status VARCHAR(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'STOPPED', 'TERMINATED', 'ERROR')),
    assigned_to_agent_id UUID REFERENCES Agents(id),
    provisioned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    terminated_at TIMESTAMP WITH TIME ZONE
);

-- Cost Records
CREATE TABLE IF NOT EXISTS Cost_Records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venture_id UUID REFERENCES Ventures(id),
    resource_id UUID REFERENCES Infrastructure_Resources(id),
    cost_type VARCHAR(50) NOT NULL,
    amount DECIMAL(12,6) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    billing_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    billing_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    usage_metrics JSONB,
    cost_center VARCHAR(100),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance Metrics
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

-- Virtual Workspace Files
CREATE TABLE IF NOT EXISTS Virtual_Workspace_Files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venture_id UUID NOT NULL REFERENCES Ventures(id),
    task_id UUID REFERENCES Tasks(id),
    created_by_agent_id UUID NOT NULL REFERENCES Agents(id),
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT,
    content_hash VARCHAR(64),
    file_content TEXT,
    binary_content BYTEA,
    version INTEGER DEFAULT 1,
    parent_version_id UUID REFERENCES Virtual_Workspace_Files(id),
    metadata JSONB,
    access_permissions JSONB DEFAULT '{"public": true}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Communications
CREATE TABLE IF NOT EXISTS Agent_Communications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_agent_id UUID NOT NULL REFERENCES Agents(id),
    to_agent_id UUID REFERENCES Agents(id),
    venture_id UUID REFERENCES Ventures(id),
    task_id UUID REFERENCES Tasks(id),
    message_type VARCHAR(50) NOT NULL,
    subject VARCHAR(255),
    content TEXT NOT NULL,
    priority INTEGER DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    requires_response BOOLEAN DEFAULT FALSE,
    parent_message_id UUID REFERENCES Agent_Communications(id),
    read_at TIMESTAMP WITH TIME ZONE,
    responded_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge Base
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

-- Activity Logs
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
-- CREATE INDEXES
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
CREATE INDEX IF NOT EXISTS idx_tasks_decision ON Tasks(decision_id);

-- Bidding system indexes
CREATE INDEX IF NOT EXISTS idx_bids_task ON Task_Bids(task_id);
CREATE INDEX IF NOT EXISTS idx_bids_agent ON Task_Bids(agent_id);
CREATE INDEX IF NOT EXISTS idx_bids_amount ON Task_Bids(bid_amount_cc ASC);

-- Other indexes
CREATE INDEX IF NOT EXISTS idx_simulations_venture ON Market_Simulations(venture_id);
CREATE INDEX IF NOT EXISTS idx_simulations_status ON Market_Simulations(status);
CREATE INDEX IF NOT EXISTS idx_experiments_simulation ON Simulation_Experiments(simulation_id);
CREATE INDEX IF NOT EXISTS idx_performance_entity ON Performance_Metrics(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_performance_metric ON Performance_Metrics(metric_name, measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_communications_from_agent ON Agent_Communications(from_agent_id);
CREATE INDEX IF NOT EXISTS idx_communications_to_agent ON Agent_Communications(to_agent_id);
CREATE INDEX IF NOT EXISTS idx_communications_unread ON Agent_Communications(to_agent_id, read_at) WHERE read_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_content_search ON Knowledge_Entries USING GIN(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON Knowledge_Entries USING GIN(tags);

-- JSONB indexes
CREATE INDEX IF NOT EXISTS idx_agents_capabilities ON Agents USING GIN(capabilities);
CREATE INDEX IF NOT EXISTS idx_ventures_metrics ON Ventures USING GIN(success_metrics);
CREATE INDEX IF NOT EXISTS idx_decisions_data_sources ON Decisions USING GIN(consulted_data_sources);
CREATE INDEX IF NOT EXISTS idx_tasks_deliverables ON Tasks USING GIN(deliverables);
CREATE INDEX IF NOT EXISTS idx_simulations_parameters ON Market_Simulations USING GIN(market_parameters);

-- =====================================================
-- CREATE FUNCTIONS & TRIGGERS
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
DROP TRIGGER IF EXISTS trigger_update_agent_performance ON Tasks;
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
DROP TRIGGER IF EXISTS trigger_update_venture_costs ON Tasks;
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
DROP TRIGGER IF EXISTS trigger_log_task_activities ON Tasks;
CREATE TRIGGER trigger_log_task_activities
    AFTER INSERT OR UPDATE ON Tasks
    FOR EACH ROW
    EXECUTE FUNCTION log_significant_activities();

DROP TRIGGER IF EXISTS trigger_log_decision_activities ON Decisions;
CREATE TRIGGER trigger_log_decision_activities
    AFTER INSERT ON Decisions
    FOR EACH ROW
    EXECUTE FUNCTION log_significant_activities();

-- =====================================================
-- CREATE VIEWS
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
-- MIGRATE EXISTING AGENT DATA
-- =====================================================

-- Update existing agents with enhanced capabilities based on their specialization
UPDATE Agents SET 
    capabilities = CASE 
        WHEN specialization = 'RESOURCE_MANAGEMENT' THEN 
            '{"cost_optimization", "infrastructure_management", "performance_monitoring"}'::jsonb
        WHEN specialization LIKE '%PLANNING%' THEN 
            '{"system_design", "task_decomposition", "workflow_optimization"}'::jsonb
        WHEN specialization LIKE '%GENERATION%' THEN 
            '{"code_development", "api_design", "testing"}'::jsonb
        ELSE '{}'::jsonb
    END,
    hardware_preference = 'ANY',
    last_heartbeat = CURRENT_TIMESTAMP
WHERE capabilities = '{}'::jsonb;

-- =====================================================
-- ADD ENHANCED AGENTS
-- =====================================================

-- Insert enhanced genesis agents (skip if they already exist)
INSERT INTO Agents (name, specialization, agent_type, cognitive_cycles_balance, capabilities, hardware_preference) VALUES
('Strategist-Sigma', 'MARKET_ANALYSIS', 'STRATEGIST', 30000,
 '{"market_simulation", "predictive_modeling", "competitive_analysis", "trend_forecasting"}'::jsonb, 'TPU'),
('Empathy-Beta', 'USER_RESEARCH', 'EMPATHY', 20000,
 '{"user_persona_modeling", "behavioral_analysis", "journey_mapping", "feedback_synthesis"}'::jsonb, 'CPU'),
('Ethicist-Gamma', 'COMPLIANCE_ETHICS', 'ETHICIST', 18000,
 '{"ethical_review", "bias_detection", "privacy_compliance", "fairness_analysis"}'::jsonb, 'CPU'),
('QA-Delta', 'QUALITY_ASSURANCE', 'QA', 12000,
 '{"automated_testing", "performance_testing", "security_auditing", "code_review"}'::jsonb, 'GPU')
ON CONFLICT (name) DO NOTHING;

-- =====================================================
-- SEED KNOWLEDGE BASE
-- =====================================================

-- Insert foundational knowledge entries
INSERT INTO Knowledge_Entries (title, content, category, tags, created_by_agent_id) VALUES
('Cognitive Cycles Economy Fundamentals',
 'The CC economy operates on scarcity principles. Agents earn CC by completing tasks efficiently and lose CC when bidding. Higher complexity tasks command higher bounties. Performance history affects bidding power.',
 'ECONOMICS',
 ARRAY['cognitive_cycles', 'economy', 'bidding', 'fundamentals'],
 (SELECT id FROM Agents WHERE agent_type = 'STEWARD' LIMIT 1)),
('Task Complexity Assessment Guidelines',
 'Task complexity ranges from 1-10. Level 1: Simple data retrieval. Level 5: API development. Level 10: Full system architecture. Consider dependencies, required expertise, and time estimates.',
 'METHODOLOGY',
 ARRAY['tasks', 'complexity', 'assessment', 'guidelines'],
 (SELECT id FROM Agents WHERE agent_type = 'ARCHITECT' LIMIT 1)),
('Market Simulation Best Practices',
 'Effective simulations require: 1) Diverse user personas, 2) Realistic economic constraints, 3) Competitive landscape modeling, 4) Black swan event injection, 5) Statistical significance validation.',
 'SIMULATION',
 ARRAY['simulation', 'market_analysis', 'best_practices', 'methodology'],
 (SELECT id FROM Agents WHERE agent_type = 'STRATEGIST' LIMIT 1))
ON CONFLICT DO NOTHING;

-- =====================================================
-- CLEANUP & VALIDATION
-- =====================================================

-- Update task counts for existing agents
UPDATE Agents SET 
    total_tasks_completed = (SELECT COUNT(*) FROM Tasks WHERE assigned_to_agent_id = Agents.id AND status = 'COMPLETED'),
    total_cc_earned = COALESCE((SELECT SUM(cc_bounty + COALESCE(bonus_cc, 0)) FROM Tasks WHERE assigned_to_agent_id = Agents.id AND status = 'COMPLETED'), 0)
WHERE total_tasks_completed = 0;

-- Cleanup old tables if needed (uncomment if you want to remove backup tables)
-- DROP TABLE IF EXISTS agents_backup;
-- DROP TABLE IF EXISTS ventures_backup;
-- DROP TABLE IF EXISTS decisions_backup;
-- DROP TABLE IF EXISTS tasks_backup;

-- =====================================================
-- FINAL MESSAGE
-- =====================================================

\echo 'Enhanced Kairos Schema Migration Completed Successfully!'
\echo 'The system has been upgraded to support:'
\echo '  - Advanced Cognitive Substrate with Internal Economy'
\echo '  - Market Simulation Engine (The Oracle)'
\echo '  - Enhanced Resource Management'
\echo '  - Comprehensive Causal Tracing'
\echo '  - Agent Communication & Knowledge Base'
\echo '  - Performance Analytics & Audit Trails'
\echo ''
\echo 'Your existing data has been preserved and enhanced.'
\echo 'The ADO is now ready for prescient autonomous operation!'