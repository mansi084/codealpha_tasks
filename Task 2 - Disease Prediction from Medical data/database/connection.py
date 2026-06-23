import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DB_DIR = "database"
os.makedirs(DB_DIR, exist_ok=True)

# Standard DATABASE_URL check, fallback to workspace SQLite database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_DIR}/disease_prediction.db")

# Fix for Heroku/Render postgresql:// vs postgres:// URLs
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print(f"Connecting to database via URL: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")

# SQLite specific config requirements
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency injection helper for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes tables in database."""
    from database.models import Base as ModelBase
    ModelBase.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")
