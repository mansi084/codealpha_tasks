import os
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from database.models import Base

# Load config to get db path
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

db_url = config['database']['connection_string']

# Ensure directory for SQLite exists
if db_url.startswith("sqlite:///"):
    db_file_path = db_url.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_file_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

engine = create_engine(db_url, connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {})
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

def init_db():
    """Initialize the database schema."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get scoped database session."""
    db = Session()
    try:
        yield db
    finally:
        db.close()
