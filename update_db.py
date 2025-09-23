import psycopg2
import os

# Configuration
DB_NAME = os.environ.get("DB_NAME", "kairos_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

def update_database():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        with conn.cursor() as cur:
            # Read and execute the update schema
            with open('kairos_schema_update_p2.sql', 'r') as f:
                sql = f.read()
            cur.execute(sql)
        conn.commit()
        conn.close()
        print("Schema update applied successfully.")
    except Exception as e:
        print(f"Error applying schema update: {e}")

if __name__ == "__main__":
    update_database()
