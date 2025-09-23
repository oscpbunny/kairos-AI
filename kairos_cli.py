# Project Kairos: Command-Line Interface
# Milestone 2.1: Viewing the Architect's Plan
# Filename: kairos_cli.py

import os
import psycopg2
import argparse
import json
from psycopg2.extras import RealDictCursor

# --- Configuration ---
# Uses the same environment variables as the agent for consistency.
DB_NAME = os.environ.get("DB_NAME", "kairos")
DB_USER = os.environ.get("DB_USER", "kairos")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "kairos_password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to the database. Details: {e}")
        return None

def create_venture(conn, name, objective):
    """Inserts a new venture into the database, marking it as the active mission."""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO Ventures (name, objective, status)
                VALUES (%s, %s, 'IN_PROGRESS')
                RETURNING id;
                """,
                (name, objective)
            )
            new_venture = cur.fetchone()
            conn.commit()
            print("✅ New Venture created successfully!")
            print(f"   - ID: {new_venture['id']}")
            print(f"   - Name: {name}")
            print("Agents will now begin work on this directive.")
    except Exception as e:
        print(f"❌ Error creating venture: {e}")
        conn.rollback()

def view_decisions(conn, limit):
    """Retrieves and displays the most recent decisions from the Causal Ledger."""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    d.created_at, a.name AS agent_name, v.name AS venture_name,
                    d.rationale, d.consulted_data_sources
                FROM Decisions d
                JOIN Agents a ON d.agent_id = a.id
                JOIN Ventures v ON d.venture_id = v.id
                ORDER BY d.created_at DESC
                LIMIT %s;
                """,
                (limit,)
            )
            decisions = cur.fetchall()
            if not decisions:
                print("No decisions found in the Causal Ledger yet.")
                return
            print(f"--- Displaying Last {len(decisions)} Decisions ---")
            for decision in decisions:
                print(f"\nTimestamp: {decision['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"  Venture: {decision['venture_name']}")
                print(f"  Agent: {decision['agent_name']}")
                print(f"  Rationale: {decision['rationale']}")
                data_sources = json.dumps(decision['consulted_data_sources'], indent=2)
                print(f"  Data Sources:\n{data_sources}")
            print("\n--- End of Log ---")
    except Exception as e:
        print(f"❌ Error viewing decisions: {e}")

def view_tasks(conn, limit):
    """Retrieves and displays the most recent tasks for the active venture."""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # First, find the active venture
            cur.execute("SELECT id, name FROM Ventures WHERE status = 'IN_PROGRESS' ORDER BY created_at DESC LIMIT 1;")
            active_venture = cur.fetchone()
            if not active_venture:
                print("No active venture found.")
                return
            
            print(f"--- Displaying Tasks for Venture: '{active_venture['name']}' ---")

            cur.execute(
                """
                SELECT t.status, t.description, a.name as agent_name, t.created_at
                FROM Tasks t
                JOIN Agents a ON t.created_by_agent_id = a.id
                WHERE t.venture_id = %s
                ORDER BY t.created_at ASC
                LIMIT %s;
                """,
                (active_venture['id'], limit)
            )
            tasks = cur.fetchall()

            if not tasks:
                print("No tasks found for this venture yet. The Architect may still be planning.")
                return

            for i, task in enumerate(tasks):
                print(f"\n{i+1}. [{task['status']}] {task['description']}")
                print(f"   - Created by: {task['agent_name']} at {task['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print("\n--- End of Task List ---")

    except Exception as e:
        print(f"❌ Error viewing tasks: {e}")


def main():
    parser = argparse.ArgumentParser(description="Kairos Command-Line Interface.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # create-venture command
    parser_create = subparsers.add_parser('create-venture', help='Create a new strategic venture for the ADO.')
    parser_create.add_argument('--name', type=str, required=True, help='A short, descriptive name for the venture.')
    parser_create.add_argument('--objective', type=str, required=True, help='The detailed objective or goal of the venture.')
    
    # view-decisions command
    parser_view = subparsers.add_parser('view-decisions', help='View the most recent decisions made by the agent swarm.')
    parser_view.add_argument('--limit', type=int, default=10, help='Number of recent decisions to display.')

    # NEW: view-tasks command
    parser_tasks = subparsers.add_parser('view-tasks', help="View the Architect's plan for the active venture.")
    parser_tasks.add_argument('--limit', type=int, default=25, help='Number of tasks to display.')

    args = parser.parse_args()
    conn = get_db_connection()
    if not conn: return

    if args.command == 'create-venture':
        create_venture(conn, args.name, args.objective)
    elif args.command == 'view-decisions':
        view_decisions(conn, args.limit)
    elif args.command == 'view-tasks':
        view_tasks(conn, args.limit)
    
    conn.close()

if __name__ == "__main__":
    main()

