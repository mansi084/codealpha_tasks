import os
import sys
import argparse
import subprocess
import time

def start_backend():
    print("Starting FastAPI Backend Server on http://localhost:8000...")
    # Run uvicorn. Run in the current python environment
    cmd = [sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", "8000"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nBackend server stopped.")

def start_frontend():
    print("Starting Streamlit Frontend Dashboard...")
    cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nFrontend stopped.")

def run_tests():
    print("Running Pytest Suite...")
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    subprocess.run(cmd)

def run_train():
    print("Running Model Training Pipeline...")
    cmd = [sys.executable, "-m", "training.train"]
    subprocess.run(cmd)

def run_retrain():
    print("Running Model Retraining Pipeline...")
    cmd = [sys.executable, "-m", "training.retrain"]
    subprocess.run(cmd)

def start_all():
    print("Starting credit scoring platform (Backend + Frontend)...")
    
    # 1. Start FastAPI backend as a subprocess
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL, # keep stdout quiet
        stderr=subprocess.DEVNULL
    )
    print("FastAPI Backend started in background at http://127.0.0.1:8000")
    
    # Wait a moment for uvicorn to initialize
    time.sleep(2)
    
    # 2. Start Streamlit frontend
    print("Starting Streamlit Dashboard...")
    frontend_cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py"]
    try:
        subprocess.run(frontend_cmd)
    except KeyboardInterrupt:
        print("Stopping platform...")
    finally:
        # Terminate backend process when frontend is stopped
        backend_proc.terminate()
        backend_proc.wait()
        print("Backend and Frontend stopped.")

def main():
    parser = argparse.ArgumentParser(description="Credit Scoring and Creditworthiness Prediction System CLI")
    parser.add_argument(
        "action",
        choices=["backend", "frontend", "train", "retrain", "test", "all"],
        help="Action to perform: run backend API, run frontend UI, run model training, run retraining pipeline, run tests, or run both (all)"
    )
    
    args = parser.parse_args()
    
    if args.action == "backend":
        start_backend()
    elif args.action == "frontend":
        start_frontend()
    elif args.action == "train":
        run_train()
    elif args.action == "retrain":
        run_retrain()
    elif args.action == "test":
        run_tests()
    elif args.action == "all":
        start_all()

if __name__ == "__main__":
    main()
