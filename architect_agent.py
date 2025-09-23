# Project Kairos: Specialist Agent
# Milestone 2.1: The Architect
# Filename: architect_agent.py

import os
import psycopg2
import time
import json

# --- Configuration ---
DB_NAME = os.environ.get("DB_NAME", "kairos_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

AGENT_NAME = "Architect-01"
CHECK_INTERVAL_SECONDS = 20 # Check for new ventures every 20 seconds

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        print("Architect Agent: Database connection established.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Architect Agent: Error connecting to database. Details: {e}")
        return None

def get_agent_id(conn, agent_name):
    """Retrieves the UUID of the agent from the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM Agents WHERE name = %s;", (agent_name,))
        result = cur.fetchone()
        if result:
            return result[0]
        else:
            raise Exception(f"Agent '{agent_name}' not found. Please run schema update.")

def get_unplanned_venture(conn):
    """Finds the most recent active venture that has not yet been planned."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, objective FROM Ventures 
            WHERE status = 'IN_PROGRESS' AND plan_generated_at IS NULL 
            ORDER BY created_at DESC LIMIT 1;
            """
        )
        return cur.fetchone()

def decompose_objective(objective):
    """
    A simple heuristic for task decomposition.
    In a real system, this would be a call to a planning model or LLM.
    """
    # Simple decomposition: split by sentences. We also filter out empty strings.
    tasks = [task.strip() + '.' for task in objective.split('.') if task.strip()]
    return tasks

def create_tasks(conn, venture_id, agent_id, task_descriptions):
    """Inserts a list of new tasks into the Tasks table."""
    with conn.cursor() as cur:
        for desc in task_descriptions:
            cur.execute(
                """
                INSERT INTO Tasks (venture_id, created_by_agent_id, description, status)
                VALUES (%s, %s, %s, 'PENDING');
                """,
                (venture_id, agent_id, desc)
            )
    print(f"Successfully created {len(task_descriptions)} tasks for Venture ID: {venture_id}")

def record_planning_decision(conn, venture_id, agent_id, task_count):
    """Records the decision to create a plan in the Causal Ledger."""
    rationale = f"Decomposed the venture objective into {task_count} actionable tasks based on sentence structure analysis."
    data_sources = {"source": "Internal Heuristic", "method": "Sentence Splitting"}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO Decisions (venture_id, agent_id, triggered_by_event, rationale, consulted_data_sources)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (venture_id, agent_id, 'NEW_VENTURE_DETECTED', rationale, json.dumps(data_sources))
        )
    print("Successfully recorded planning decision.")

def mark_venture_as_planned(conn, venture_id):
    """Updates the venture to mark that the plan has been generated."""
    with conn.cursor() as cur:
        cur.execute("UPDATE Ventures SET plan_generated_at = CURRENT_TIMESTAMP WHERE id = %s;", (venture_id,))

def main_loop():
    """The main operational loop for the Architect agent."""
    print(f"Starting Architect Agent v0.1 ({AGENT_NAME})...")
    
    conn = get_db_connection()
    if not conn:
        return

    try:
        agent_id = get_agent_id(conn, AGENT_NAME)
        print(f"Agent '{AGENT_NAME}' identified with ID: {agent_id}")
    except Exception as e:
        print(e)
        conn.close()
        return

    while True:
        print(f"\n--- Architect cycle started at {time.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
        venture = get_unplanned_venture(conn)
        
        if not venture:
            print("No unplanned ventures found. Standing by.")
        else:
            print(f"Found unplanned Venture ID: {venture['id']}. Objective: '{venture['objective']}'")
            
            # 1. Decompose the objective into tasks
            task_descriptions = decompose_objective(venture['objective'])
            
            if task_descriptions:
                # 2. Record the decision
                record_planning_decision(conn, venture['id'], agent_id, len(task_descriptions))
                
                # 3. Create the tasks in the database
                create_tasks(conn, venture['id'], agent_id, task_descriptions)
                
                # 4. Mark the venture as planned
                mark_venture_as_planned(conn, venture['id'])
                
                conn.commit()
                print("Planning process complete for venture.")
            else:
                print("Objective resulted in no actionable tasks. Skipping.")
                # Mark as planned to avoid re-processing a blank objective
                mark_venture_as_planned(conn, venture['id'])
                conn.commit()


        print(f"Cycle complete. Sleeping for {CHECK_INTERVAL_SECONDS} seconds...")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nShutdown signal received. Architect Agent is terminating.")
