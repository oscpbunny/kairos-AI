-- Project Kairos: Causal Ledger Data Model
-- Milestone 1.1: The Immutable Mind
-- Language: PostgreSQL

-- This script enables the UUID generation function, which is required for primary keys.
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- The Ventures table represents high-level strategic goals or products
-- that the ADO is pursuing. It's the "mission" a human gives to the system.
CREATE TABLE Ventures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    objective TEXT NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('DIRECTIVE_RECEIVED', 'IN_PROGRESS', 'COMPLETED', 'ARCHIVED')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- The Agents table keeps a record of all autonomous agents within the swarm.
-- It also tracks their "wealth" in the internal economy.
CREATE TABLE Agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE, -- e.g., 'Steward-01', 'Engineer-Alpha'
    specialization VARCHAR(100) NOT NULL, -- e.g., 'RESOURCE_MANAGEMENT', 'CODE_GENERATION'
    cognitive_cycles_balance BIGINT DEFAULT 1000 NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- This is the central table: The Causal Ledger.
-- It records the "why" behind every significant action, linking intent to outcome.
CREATE TABLE Decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venture_id UUID NOT NULL REFERENCES Ventures(id),
    agent_id UUID REFERENCES Agents(id), -- Can be NULL if triggered by a system process or user
    triggered_by_event VARCHAR(255) NOT NULL, -- e.g., 'USER_DIRECTIVE', 'HOURLY_AUDIT_SCHEDULE', 'TASK_BOUNTY_POSTED'
    rationale TEXT NOT NULL, -- The agent's explicit reasoning for the decision.
    consulted_data_sources JSONB, -- Links to reports, simulation IDs, user feedback, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- The Tasks table represents discrete units of work derived from a Decision.
-- This is the "what" that agents execute.
CREATE TABLE Tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID NOT NULL REFERENCES Decisions(id),
    assigned_to_agent_id UUID REFERENCES Agents(id), -- The agent who won the bid/was assigned
    status VARCHAR(50) NOT NULL CHECK (status IN ('BOUNTY_POSTED', 'IN_PROGRESS', 'COMPLETED', 'FAILED')),
    description TEXT NOT NULL,
    cc_bounty BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Initial Genesis Agent Entry
-- We seed the database with the first agent, the Steward.
INSERT INTO Agents (name, specialization, cognitive_cycles_balance)
VALUES ('Steward-01', 'RESOURCE_MANAGEMENT', 10000);

-- Display a confirmation message
\echo "Project Kairos Causal Ledger schema created successfully."
\echo "Genesis agent 'Steward-01' has been seeded."

