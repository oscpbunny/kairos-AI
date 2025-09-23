-- Project Kairos: Schema Update for Phase 3
-- Filename: kairos_schema_update_p3.sql

-- Add a wallet balance to agents. Default to 1000 CC for existing and new agents.
ALTER TABLE Agents ADD COLUMN IF NOT EXISTS wallet_balance_cc NUMERIC(12, 2) NOT NULL DEFAULT 1000.00;

-- Add a bounty to tasks.
ALTER TABLE Tasks ADD COLUMN IF NOT EXISTS bounty_cc NUMERIC(10, 2) NOT NULL DEFAULT 0.00;

-- Create a new table to store bids from agents on tasks.
CREATE TABLE IF NOT EXISTS Bids (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES Tasks(id),
    agent_id UUID NOT NULL REFERENCES Agents(id),
    bid_amount_cc NUMERIC(10, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- An agent can only bid once per task.
    UNIQUE(task_id, agent_id)
);

-- Add indexes for performance.
CREATE INDEX IF NOT EXISTS idx_bids_task_id ON Bids(task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_bounty_cc ON Tasks(bounty_cc);

COMMENT ON COLUMN Agents.wallet_balance_cc IS 'The current amount of Cognitive Cycles (CC) the agent possesses.';
COMMENT ON COLUMN Tasks.bounty_cc IS 'The reward offered for completing this task.';
COMMENT ON TABLE Bids IS 'Records bids placed by agents on specific tasks.';

