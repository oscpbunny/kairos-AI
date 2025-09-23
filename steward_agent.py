# Project Kairos: Genesis Agent
# Milestone 1.2: The Steward v0.1 (with Dev Mode)
# Filename: steward_agent.py

import os
import psycopg2
import time
import json
from datetime import datetime

# --- AWS SDK ---
# boto3 is only required if not in DEV_MODE.
# We use a try-except block to make it optional for local development.
try:
    import boto3
except ImportError:
    boto3 = None

# --- Configuration ---
# It's recommended to use environment variables for security.
DB_NAME = os.environ.get("DB_NAME", "kairos_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

# --- Development Mode ---
# Set to "true" to use a mock billing API instead of a real cloud provider.
DEV_MODE = os.environ.get("DEV_MODE", "true").lower() == "true"

# The name of the agent as seeded in the database.
AGENT_NAME = "Steward-01"

# The agent will perform its check at this interval.
# In dev mode, it runs frequently for easy observation.
CHECK_INTERVAL_SECONDS = 15 if DEV_MODE else 3600 # 15 seconds in dev, 1 hour in prod

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print("Database connection established successfully.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to the database. Please check configuration. Details: {e}")
        return None

def get_agent_id(conn, agent_name):
    """Retrieves the UUID of the agent from the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM Agents WHERE name = %s;", (agent_name,))
        result = cur.fetchone()
        if result:
            return result[0]
        else:
            raise Exception(f"Agent '{agent_name}' not found in the database.")

def get_active_venture_id(conn):
    """Retrieves the UUID of the most recent 'IN_PROGRESS' venture."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM Ventures WHERE status = 'IN_PROGRESS' ORDER BY created_at DESC LIMIT 1;")
        result = cur.fetchone()
        return result[0] if result else None

def get_mock_billing_info():
    """
    Generates fake billing data for development/testing purposes.
    Simulates a cost that slowly increases over time.
    """
    # Use the current minute to create a pseudo-random, slowly increasing cost
    current_second = datetime.utcnow().second
    base_cost = 12.34
    increment = current_second * 0.01 
    mock_cost = base_cost + increment

    today = datetime.utcnow()
    start_of_month = today.replace(day=1).strftime('%Y-%m-%d')
    end_of_month = today.strftime('%Y-%m-%d')

    print("--- RUNNING IN DEV MODE: Using mock billing data. ---")
    return {"amount": mock_cost, "unit": "USD", "time_period_start": start_of_month, "time_period_end": end_of_month}

def get_aws_billing_info():
    """
    Fetches the current month-to-date billing information from AWS Cost Explorer.
    """
    if not boto3:
        raise ImportError("boto3 is not installed, but is required when not in DEV_MODE.")
    try:
        client = boto3.client('ce')
        today = datetime.utcnow()
        start_of_month = today.replace(day=1).strftime('%Y-%m-%d')
        end_of_month = today.strftime('%Y-%m-%d')

        response = client.get_cost_and_usage(
            TimePeriod={'Start': start_of_month, 'End': end_of_month},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost']
        )
        amount = response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount']
        unit = response['ResultsByTime'][0]['Total']['UnblendedCost']['Unit']
        return {"amount": float(amount), "unit": unit, "time_period_start": start_of_month, "time_period_end": end_of_month}

    except Exception as e:
        print(f"Error fetching AWS billing data: {e}")
        return None

def record_decision(conn, venture_id, agent_id, rationale, data_sources):
    """Inserts a new decision into the Causal Ledger."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO Decisions (venture_id, agent_id, triggered_by_event, rationale, consulted_data_sources)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (venture_id, agent_id, 'PERIODIC_COST_AUDIT', rationale, json.dumps(data_sources))
        )
    conn.commit()
    print("Successfully recorded new decision in the Causal Ledger.")

def main_loop():
    """The main operational loop for the Steward agent."""
    print(f"Starting Steward Agent v0.1 ({AGENT_NAME})...")
    if DEV_MODE:
        print("INFO: Running in Development Mode. Cloud provider APIs will be mocked.")
    
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
        print(f"\n--- New cycle started at {datetime.utcnow().isoformat()} UTC ---")
        venture_id = get_active_venture_id(conn)
        
        if not venture_id:
            print("No active venture found. The ADO has no current mission. Standing by.")
        else:
            print(f"Active Venture ID: {venture_id}. Proceeding with cost audit.")
            
            billing_data = get_mock_billing_info() if DEV_MODE else get_aws_billing_info()
            
            if billing_data:
                cost = billing_data['amount']
                unit = billing_data['unit']
                
                if DEV_MODE:
                    rationale = f"Scheduled DEV MODE cost audit. Simulated month-to-date cost is {cost:.2f} {unit}."
                    data_sources = {"source": "Mock Billing API", "parameters": {"mode": "development"}}
                else:
                    rationale = f"Scheduled hourly cost audit. Current month-to-date unblended cost is {cost:.2f} {unit}."
                    data_sources = {
                        "source": "AWS Cost Explorer API",
                        "parameters": {
                            "TimePeriod": {
                                "Start": billing_data['time_period_start'],
                                "End": billing_data['time_period_end']
                            },
                            "Granularity": "MONTHLY",
                            "Metrics": ["UnblendedCost"]
                        }
                    }
                
                record_decision(conn, venture_id, agent_id, rationale, data_sources)
            else:
                print("Could not retrieve billing data. Skipping decision recording for this cycle.")

        print(f"Cycle complete. Sleeping for {CHECK_INTERVAL_SECONDS} seconds...")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nShutdown signal received. Steward Agent is terminating.")

