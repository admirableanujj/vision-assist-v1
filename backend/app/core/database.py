import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Read environment variables mirroring precheck_db.py
db_host = os.getenv("POSTGRES_HOST", "vision_assist_db")
db_name = os.getenv("POSTGRES_DB", "vision_assist")
db_user = os.getenv("POSTGRES_USER", "postgres")

# Check for Docker secrets or environment password
secret_path = "/run/secrets/postgres_password"
if os.path.exists(secret_path):
    with open(secret_path, "r") as f:
        db_password = f.read().strip()
else:
    db_password = os.getenv("POSTGRES_PASSWORD", "postgres")

# Construct DATABASE_URL pointing to the Docker service container instead of localhost
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:5432/{db_name}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()