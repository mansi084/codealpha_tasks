import os
import yaml
import logging
from data.generate_dataset import generate_synthetic_data
from training.train import train_pipeline
from database.db import Session
from database.models import UserActivityLog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_retraining")

def retrain_system():
    logger.info("Initializing system retraining...")
    db = Session()
    try:
        # Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        raw_data_path = config['paths']['raw_data']
        
        # 1. Regenerate/update the dataset
        logger.info("Regenerating training data...")
        df_new = generate_synthetic_data(num_samples=3000) # generate slightly more samples
        df_new.to_csv(raw_data_path, index=False)
        logger.info(f"Updated raw dataset at {raw_data_path}")
        
        # 2. Rerun training pipeline
        new_version = train_pipeline(raw_data_path)
        logger.info(f"Retraining completed successfully. New active model version: {new_version}")
        
        db.add(UserActivityLog(
            action="System Retrained", 
            details=f"Dataset regenerated (3000 samples) and training pipeline completed. New version: {new_version}"
        ))
        db.commit()
        return new_version
    except Exception as e:
        logger.exception("Error occurred during retraining process")
        db.add(UserActivityLog(
            action="System Retraining - Failed", 
            details=f"Exception: {str(e)}"
        ))
        db.commit()
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    retrain_system()
