import os
from pathlib import Path
import logging
from roboflow import Roboflow
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dataset():
    """
    Download the License Plate Recognition dataset from Roboflow
    Returns:
        str: Path to the downloaded dataset
    """
    try:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not found in environment variables")
            
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
        dataset = project.version(4).download("yolov8")
        
        logger.info(f"Dataset downloaded successfully to: {dataset.location}")
        return dataset.location
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def prepare_dataset(dataset_path):
    """
    Prepare the dataset for training by organizing files
    Args:
        dataset_path: Path to the downloaded dataset
    """
    try:
        dataset_path = Path(dataset_path)
        
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_file = dataset_path / "data.yaml"
        if not yaml_file.exists():
            raise FileNotFoundError(f"Dataset YAML file not found at {yaml_file}")
            
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        config['path'] = str(dataset_path.absolute())
        config['train'] = str((dataset_path / 'train' / 'images').absolute())
        config['val'] = str((dataset_path / 'valid' / 'images').absolute())
        config['test'] = str((dataset_path / 'test' / 'images').absolute())
        
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info("Dataset preparation completed successfully!")
        logger.info(f"Dataset configuration saved to: {yaml_file}")
        logger.info(f"Training images: {len(list(Path(config['train']).glob('*.jpg')))} images")
        logger.info(f"Validation images: {len(list(Path(config['val']).glob('*.jpg')))} images")
        logger.info(f"Test images: {len(list(Path(config['test']).glob('*.jpg')))} images")
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        dataset_path = download_dataset()
        
        prepare_dataset(dataset_path)
        
        logger.info("Dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
