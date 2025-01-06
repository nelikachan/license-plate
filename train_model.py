from ultralytics import YOLO
import logging
from pathlib import Path
import shutil
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    """
    Train YOLOv8 model on the license plate dataset
    """
    try:
        data_yaml = Path("License-Plate-Recognition-4/data.yaml")
        if not data_yaml.exists():
            raise FileNotFoundError(
                "Dataset not found. Please run prepare_dataset.py first."
            )
        
        model = YOLO('yolov8n.pt')  
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = {
            'data': str(data_yaml),
            'epochs': 20, 
            'imgsz': 640,
            'batch': 16,
            'name': 'license_plate_detector',
            'patience': 5, 
            'save': True,  
            'save_period': 5,  
            'device': '0' if torch.cuda.is_available() else 'cpu',
            'workers': 8,  
            'exist_ok': True,  
            'pretrained': True,  
            'optimizer': 'auto', 
            'verbose': True,  
            'seed': 42,  
        }
        

        logger.info("Starting model training...")
        logger.info(f"Training parameters: {training_args}")
        
        results = model.train(**training_args)
        
        runs_dir = Path("runs/detect/license_plate_detector")
        best_model_path = runs_dir / "weights/best.pt"
        last_model_path = runs_dir / "weights/last.pt"
        target_model_path = Path("models/best.pt")
        
        if best_model_path.exists():
            shutil.copy2(best_model_path, target_model_path)
            logger.info(f"Best model saved to {target_model_path}")
        elif last_model_path.exists():
            shutil.copy2(last_model_path, target_model_path)
            logger.info(f"Last model saved to {target_model_path} (best model not found)")
        else:
            logger.error("No model files found after training")
            raise FileNotFoundError("Model files not found after training")
        
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            logger.info("Training metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}")
        
        logger.info("Training completed successfully!")
        return target_model_path
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
