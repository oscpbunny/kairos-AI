# Project Kairos: Specialist Agent
# Milestone 2.2: The First Engineer
# Filename: engineer_agent.py

import os
import psycopg2
import time
import json
import random

# --- Configuration ---
DB_NAME = os.environ.get("DB_NAME", "kairos_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

AGENT_NAME = "Engineer-01"
CHECK_INTERVAL_SECONDS = 10 # Check for new tasks every 10 seconds

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        print("Engineer Agent: Database connection established.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Engineer Agent: Error connecting to database. Details: {e}")
        return None

def get_agent_id(conn, agent_name):
    """Retrieves the UUID of the agent from the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM Agents WHERE name = %s;", (agent_name,))
        result = cur.fetchone()
        if result: return result[0]
        else: raise Exception(f"Agent '{agent_name}' not found.")

def claim_task(conn, agent_id):
    """Atomically finds and claims a PENDING task."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # FOR UPDATE SKIP LOCKED ensures that this agent doesn't wait for a task
        # locked by another agent, making it safe for multiple Engineer agents
        # to run concurrently in the future.
        cur.execute(
            """
            UPDATE Tasks
            SET 
                status = 'IN_PROGRESS', 
                assigned_to_agent_id = %s,
                started_at = CURRENT_TIMESTAMP
            WHERE id = (
                SELECT id FROM Tasks
                WHERE status = 'PENDING'
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, description, venture_id;
            """,
            (agent_id,)
        )
        task = cur.fetchone()
        conn.commit()
        return task

def simulate_work():
    """Simulates the agent performing a task."""
    work_time = random.uniform(3, 8) # Simulate work for 3 to 8 seconds
    print(f"Simulating work for {work_time:.2f} seconds...")
    time.sleep(work_time)
    return work_time

def complete_task(conn, task_id):
    """Marks a task as COMPLETED in the database."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE Tasks SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP WHERE id = %s;",
            (task_id,)
        )
    conn.commit()

def record_execution_decision(conn, task, agent_id, work_time):
    """Records the decision to execute and complete a task."""
    rationale = f"Task '{task['description'][:50]}...' completed after {work_time:.2f} seconds of simulated work."
    data_sources = {"source": "Task Queue", "task_id": str(task['id'])}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO Decisions (venture_id, agent_id, triggered_by_event, rationale, consulted_data_sources)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (task['venture_id'], agent_id, 'TASK_COMPLETED', rationale, json.dumps(data_sources))
        )
    conn.commit()

def main_loop():
    """The main operational loop for the Engineer agent."""
    print(f"Starting Engineer Agent v0.1 ({AGENT_NAME})...")
    conn = get_db_connection()
    if not conn: return
    
    agent_id = get_agent_id(conn, AGENT_NAME)
    print(f"Agent '{AGENT_NAME}' identified with ID: {agent_id}")

    while True:
        print(f"\n--- Engineer cycle started at {time.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
        task = claim_task(conn, agent_id)

        if not task:
            print("No pending tasks found. Standing by.")
        else:
            print(f"Claimed task: '{task['description']}'")
            
            # 1. Simulate the work being done
            work_time = simulate_work()
            
            # 2. Mark the task as complete
            complete_task(conn, task['id'])
            print(f"Task ID {task['id']} marked as COMPLETED.")

            # 3. Record the decision in the Causal Ledger
            record_execution_decision(conn, task, agent_id, work_time)
            print("Execution decision recorded.")

        print(f"Cycle complete. Sleeping for {CHECK_INTERVAL_SECONDS} seconds...")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nShutdown signal received. Engineer Agent is terminating.")
