import psycopg2
import os

# Configuration
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

def setup_database():
    # Connect to default postgres database to create kairos_db
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE DATABASE kairos_db;")
        conn.close()
        print("Database kairos_db created successfully.")
    except psycopg2.errors.DuplicateDatabase:
        print("Database kairos_db already exists.")
    except Exception as e:
        print(f"Error creating database: {e}")
        return

    # Now connect to kairos_db and execute schema
    try:
        conn = psycopg2.connect(
            dbname='kairos_db',
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        with conn.cursor() as cur:
            # Read and execute the schema
            with open('kairos_schema.sql', 'r') as f:
                sql = f.read()
            cur.execute(sql)
        conn.commit()
        conn.close()
        print("Schema applied successfully.")
    except Exception as e:
        print(f"Error applying schema: {e}")

if __name__ == "__main__":
    setup_database()
