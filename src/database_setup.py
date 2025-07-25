import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

class DatabaseSetup:
    def __init__(self) -> None:
        # Get variance from .env
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "cocktails_db")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "your_password")

    def create_database(self) -> None:
        """Create database if it doesn't exist"""
        try:
            # Connect to the database
            connection = psycopg2.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                dbname=self.db_name,
                sslmode="require"
            )
            # Set the isolation level to AUTOCOMMIT to allow database creation without using `connection.commit()`
            connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Create a cursor to execute SQL queries
            cursor = connection.cursor()
            
            # Check if database exists
            cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{self.db_name}'")
            exists = cursor.fetchone()

            if not exists:
                cursor.execute(f"CREATE DATABASE {self.db_name}")
                print(f"Database '{self.db_name}' created successfully.")
            else:
                print(f"Database '{self.db_name}' already exists.")

            # Close the cursor and connection
            cursor.close()
            connection.close()

        except Exception as e:
            print(f"Error creating database: {e}")

    def setup_pgvector(self) -> None:
        """Setup pgvector extension and create tables"""
        try:
            # Connect to the database
            connection = psycopg2.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                dbname=self.db_name,
                sslmode="require"
            )

            cursor = connection.cursor()

            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create cocktails table with vector embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cocktails (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    ingredients TEXT NOT NULL,
                    recipe TEXT,
                    glass VARCHAR(100),
                    category VARCHAR(100),
                    alcoholic VARCHAR(50),
                    embedding vector(384)
                    )
            """)

            # Create index for vector similarity search
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS cocktails_embedding_idx
                ON cocktails USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            connection.commit()
            cursor.close()
            connection.close()

            print("Database tables and pgvector extension set up sucessfully.")
        
        except Exception as e:
            print(f"Error setting up pgvector: {e}")

    def get_connection(self) -> psycopg2.extensions.connection:
        """Get database connection"""
        try:
            connection = psycopg2.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                dbname=self.db_name,
                sslmode="require"
            )
            return connection
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None
        
if __name__ == "__main__":
    db_setup = DatabaseSetup()
    db_setup.create_database()
    db_setup.setup_pgvector()