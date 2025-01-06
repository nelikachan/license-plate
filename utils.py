import cv2
import tempfile
import os
from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)

TEMP_FILES = set()

def save_upload_file(uploaded_file):
    """
    Save uploaded file to temporary directory
    Args:
        uploaded_file: StreamlitUploadedFile object
    Returns:
        str: Path to saved file
    """
    try:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            TEMP_FILES.add(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def get_video_info(video_path):
    """
    Get video file information
    Args:
        video_path: Path to video file
    Returns:
        dict: Video information including duration, fps, frame count
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'frame_count': frame_count,
            'fps': fps,
            'duration': duration,
            'width': width,
            'height': height
        }
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        raise

def cleanup_temp_files():
    """
    Clean up temporary files created during processing
    """
    try:
        for temp_file in TEMP_FILES:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {temp_file}: {str(e)}")
        TEMP_FILES.clear()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def ensure_dir(directory):
    """
    Ensure directory exists, create if it doesn't
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def create_video_writer(video_path, fps, width, height):
    """
    Create VideoWriter object
    Args:
        video_path: Path to save video
        fps: Frames per second
        width: Frame width
        height: Frame height
    Returns:
        cv2.VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(video_path, fourcc, fps, (width, height))
