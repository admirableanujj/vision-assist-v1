#app/precheck_db.py
"""
Database Schema Precheck & Seeding Migration Script

Verifies the PostgreSQL service connection, configures the relational database schema,
and populates default seeding data (roles, admin, standard user, cameras, zones) if empty.

Usage:
    python precheck_db.py

Dependencies:
    psycopg2-binary>=2.9.9
    
__original_author__ = "Anujj Saxena"    
"""
import os
import time
import uuid
import hashlib
import psycopg2
import requests
from app.core.logging_config import get_logger

logger = get_logger(__name__)

__author__ = "Anujj Saxena"
__license__ = "MIT"
__version__ = "1.0.3"

def _hash_password(password: str, salt: str) -> str:
    """Helper to generate a secure SHA-256 hash incorporating a salt."""
    salted_pass = password.encode('utf-8') + salt.encode('utf-8')
    hashed = hashlib.sha256(salted_pass).hexdigest()
    return f"{salt}:{hashed}"

def run_migrations_and_seeding():
    db_host = os.getenv("POSTGRES_HOST", "vision_assist_db")
    db_name = os.getenv("POSTGRES_DB", "vision_assist")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    password_file_path = os.getenv("POSTGRES_PASSWORD_FILE")
    # Safely check for the secret file path first
    secret_path = "/run/secrets/postgres_password"
    if os.path.exists(secret_path) or (password_file_path and os.path.exists(password_file_path)):
        with open(secret_path, "r") as f:
            db_password = f.read().strip()
    else:
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")

    # 1. Connect to PostgreSQL with retry loop
    max_retries = 10
    conn = None
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=db_host,
                database=db_name,
                user=db_user,
                password=db_password,
                connect_timeout=3
            )
            logger.info("Database link verified.")
            break
        except psycopg2.OperationalError as err:
            logger.warning(f"Database connection attempt {i+1}/{max_retries} failed ({err}). Retrying in 2s...")
            time.sleep(2)

    if not conn:
        logger.error("Could not connect to PostgreSQL database. Exiting precheck.")
        exit(1)

    # 2. Table Definitions (DDL)
    schema_queries = [
        # Roles Table (Renamed role_desciption -> role_description)
        """
        CREATE TABLE IF NOT EXISTS roles (
            id BIGSERIAL PRIMARY KEY,
            role_name VARCHAR(50) UNIQUE NOT NULL,
            role_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_modified_by VARCHAR(50),
            last_modified_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Users Table
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            guid_user VARCHAR(36) UNIQUE,
            username VARCHAR(50) UNIQUE NOT NULL,
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            email VARCHAR(100) UNIQUE,
            role INT DEFAULT 1,
            status VARCHAR(20) DEFAULT 'active',
            is_active BOOLEAN DEFAULT TRUE,
            created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_modified_by VARCHAR(50),
            last_modified_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # User Login Table
        """
        CREATE TABLE IF NOT EXISTS user_login (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT UNIQUE REFERENCES users(id) ON DELETE CASCADE,
            hashed_password VARCHAR(256) NOT NULL,
            created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_modified_by VARCHAR(50),
            last_modified_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # User Login History Table
        """
        CREATE TABLE IF NOT EXISTS user_login_history (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            login_status VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_modified_by VARCHAR(50),
            last_modified_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Items Table
        """
        CREATE TABLE IF NOT EXISTS items (
            id VARCHAR(100) PRIMARY KEY,
            user_id BIGINT UNIQUE REFERENCES users(id) ON DELETE CASCADE,
            owner_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            item_name VARCHAR(100) NOT NULL,
            description TEXT,
            object_class VARCHAR(50),
            home_zone_id BIGINT,
            created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_modified_by VARCHAR(50),
            last_modified_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding VARCHAR(512)
        );
        """,
        # Zones Table
        """
        CREATE TABLE IF NOT EXISTS zones (
            id SERIAL PRIMARY KEY,
            user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            name VARCHAR(100) NOT NULL
            );
            """,        
        # Cameras Table
        """
        CREATE TABLE IF NOT EXISTS cameras (
            id VARCHAR(100) PRIMARY KEY,
            owner_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            user_id BIGINT UNIQUE REFERENCES users(id) ON DELETE CASCADE,            
            name VARCHAR(100) NOT NULL,
            source VARCHAR(255),
            location VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            zone_id INTEGER REFERENCES zones(id) ON DELETE SET NULL
        );
        """,                
        # Detections Table
        """
        CREATE TABLE IF NOT EXISTS detections (
            id VARCHAR(100) PRIMARY KEY,
            camera_id VARCHAR(100) REFERENCES cameras(id) ON DELETE CASCADE,
            object_class VARCHAR(50) NOT NULL,
            confidence REAL,
            zone_name VARCHAR(100),
            bx1 REAL,
            by1 REAL,
            bx2 REAL,
            by2 REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Reminders Table
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id VARCHAR(100) PRIMARY KEY,
            owner_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            remind_at TIMESTAMP,
            done BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Alerts Table
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id VARCHAR(100) PRIMARY KEY,
            owner_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            message TEXT NOT NULL,
            item_name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            read BOOLEAN DEFAULT FALSE
        );
        """,
        # Query Logs Table
        """
        CREATE TABLE IF NOT EXISTS query_logs (
            id VARCHAR(100) PRIMARY KEY,
            owner_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            item_id VARCHAR(100) REFERENCES items(id) ON DELETE SET NULL,
            text TEXT,
            intent VARCHAR(50),
            found BOOLEAN,
            latency_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Item Embeddings Table
        """
        CREATE TABLE IF NOT EXISTS item_embeddings (
            id VARCHAR(100) PRIMARY KEY,
            item_id VARCHAR(100) REFERENCES items(id) ON DELETE CASCADE,
            vector TEXT NOT NULL,
            model VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]

    try:
        # Create Tables
        with conn.cursor() as cur:
            for query in schema_queries:
                cur.execute(query)
        conn.commit()
        logger.info("DB migrations completed successfully.")

# 3. Seed Initial Base Data
        with conn.cursor() as cur:
            # Seed default system roles
            cur.execute("""
                INSERT INTO roles (id, role_name, role_description)
                VALUES 
                    (1, 'User', 'Standard client privileges'),
                    (2, 'Admin', 'Global system administrative control')
                ON CONFLICT (id) DO NOTHING;
            """)

            # Verify if we need to seed the users
            cur.execute("SELECT COUNT(*) FROM users;")
            if cur.fetchone()[0] == 0:
                logger.info("No existing users found. Seeding default accounts...")
                
                # --- A. Seed Admin User (Role: 2) ---
                admin_guid = str(uuid.uuid4())
                admin_username = "admin"
                admin_password = "adminpassword123"
                admin_salt = uuid.uuid4().hex
                hashed_admin_password = _hash_password(admin_password, admin_salt)

                cur.execute("""
                    INSERT INTO users (guid_user, username, first_name, last_name, email, role, is_active)
                    VALUES (%s, %s, %s, %s, %s, 2, TRUE) RETURNING id;
                """, (admin_guid, admin_username, "System", "Administrator", "admin@visionassist.local"))
                
                admin_user_id = cur.fetchone()[0]

                cur.execute("""
                    INSERT INTO user_login (user_id, hashed_password)
                    VALUES (%s, %s);
                """, (admin_user_id, hashed_admin_password))

                # --- B. Seed Standard User (Role: 1) ---
                user_guid = str(uuid.uuid4())
                user_username = "test_user"
                user_password = "test_user"
                user_salt = uuid.uuid4().hex
                hashed_user_password = _hash_password(user_password, user_salt)

                cur.execute("""
                    INSERT INTO users (guid_user, username, first_name, last_name, email, role, is_active)
                    VALUES (%s, %s, %s, %s, %s, 1, TRUE) RETURNING id;
                """, (user_guid, user_username, "Standard", "User", "user@visionassist.local"))
                
                std_user_id = cur.fetchone()[0]

                cur.execute("""
                    INSERT INTO user_login (user_id, hashed_password)
                    VALUES (%s, %s);
                """, (std_user_id, hashed_user_password))

                # --- C. Seed Zones Safely with RETURNING ---
                cur.execute("""
                    INSERT INTO zones (user_id, name)
                    VALUES (%s, 'Main Living Zone')
                    RETURNING id;
                """, (admin_user_id,))
                generated_admin_zone_id = cur.fetchone()[0]

                cur.execute("""
                    INSERT INTO zones (user_id, name)
                    VALUES (%s, 'Master Bedroom')
                    RETURNING id;
                """, (admin_user_id,))

                cur.execute("""
                    INSERT INTO zones (user_id, name)
                    VALUES (%s, 'Zone1: Master Bedroom')
                    RETURNING id;
                """, (std_user_id,))
                generated_std_zone_id = cur.fetchone()[0]

                cur.execute("""
                    INSERT INTO zones (user_id, name)
                    VALUES (%s, 'Zone2: Kitchen Zone')
                    RETURNING id;
                """, (std_user_id,))

                # --- D. Seed Cameras with Unique IDs ---
                cur.executemany("""
                    INSERT INTO cameras (id, owner_id, name, source, location, zone_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET zone_id = EXCLUDED.zone_id;
                """, [
                    ("cam_admin_living", admin_user_id, "Living Room Cam", "rtsp://192.168.1.50/stream", "Living Room", generated_admin_zone_id),
                    ("cam_admin_kitchen", admin_user_id, "Kitchen Cam", "rtsp://192.168.1.51/stream", "Kitchen", generated_admin_zone_id),
                    ("cam_std_living", std_user_id, "Living Room Cam", "rtsp://192.168.1.50/stream", "Living Room", generated_std_zone_id),
                    ("cam_std_kitchen", std_user_id, "Kitchen Cam", "rtsp://192.168.1.51/stream", "Kitchen", generated_std_zone_id)
                ])

                logger.info("Default seeding data successfully inserted.")
            else:
                logger.info("Table datasets contain prior objects. Skipping seeding phase.")
        
        conn.commit()

    except Exception as e:
        logger.exception(f"Seeding process failed: {e}")
        conn.rollback()
    finally:
        conn.close()

def verify_ollama_model(model_name="llama3"):
    ollama_host = os.getenv("OLLAMA_HOST", "vision_assist_llm_local")
    endpoint = f"http://{ollama_host}:11434/api/tags"
    try:
        response = requests.get(endpoint, timeout=3)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            # Match exact or tag-less (e.g. 'llama3:latest' or 'llama3')
            if any(model_name in m for m in models):
                logger.info(f"Local LLM status check: '{model_name}' is downloaded and ready.")
                return True
            logger.warning(f"Local model '{model_name}' is missing on the Ollama instance ({endpoint}).")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot reach the Ollama service container at {endpoint}: {e}")
    return False

if __name__ == "__main__":
    run_migrations_and_seeding()
    verify_ollama_model()