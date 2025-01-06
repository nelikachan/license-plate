import cv2
import numpy as np
import easyocr
import torch
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LicensePlateOCR:
    def __init__(self):
        """Initialize the OCR reader"""
        self.reader = easyocr.Reader(['en'])  
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"OCR initialized on device: {self.device}")
        
    def preprocess_plate(self, plate_img):
        """
        Preprocess the license plate image for better OCR accuracy
        Args:
            plate_img: numpy array of the license plate image
        Returns:
            preprocessed image
        """
        try:

            if plate_img is None or plate_img.size == 0:
                logger.error("Empty image received")
                return None
            
            original_height, original_width = plate_img.shape[:2]
            logger.info(f"Original image size: {original_width}x{original_height}")
            
            min_height = 100  
            if original_height < min_height:
                scale = min_height / original_height
                new_width = int(original_width * scale)
                new_height = min_height
                plate_img = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                logger.info(f"Resized image to: {new_width}x{new_height}")
            
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            variants = []
            
            thresh1 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            variants.append(thresh1)
            
            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(thresh2)
            
            _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            variants.append(thresh3)
            

            processed_variants = []
            for variant in variants:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                cleaned = cv2.morphologyEx(variant, cv2.MORPH_CLOSE, kernel)
                cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
                processed_variants.append(cleaned)
            
            logger.info(f"Generated {len(processed_variants)} preprocessing variants")
            return processed_variants
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            if plate_img is not None:
                return [plate_img] 
            return None
    
    def clean_plate_text(self, text):
        """
        Clean and format the recognized text
        Args:
            text: recognized text string
        Returns:
            cleaned text string
        """
        if not text:
            return ""
            
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        text = text.replace('O', '0').replace('I', '1').replace('S', '5')
        
        return text
    
    def read_plate(self, plate_img, conf_threshold=0.0):  
        """
        Read text from license plate image
        Args:
            plate_img: numpy array of the license plate image
            conf_threshold: confidence threshold for OCR results
        Returns:
            tuple: (text, confidence)
        """
        try:
            if plate_img is None or plate_img.size == 0:
                logger.error("Empty plate image")
                return "", 0.0
            
            results = self.reader.readtext(plate_img)
            
            if not results:
                return "", 0.0
            
            best_result = max(results, key=lambda x: x[2])
            text = best_result[1]
            conf = best_result[2]
            
            cleaned_text = self.clean_plate_text(text)
            logger.info(f"Original text: '{text}', Cleaned text: '{cleaned_text}', Confidence: {conf}")
            
            return cleaned_text, conf
            
        except Exception as e:
            logger.error(f"Error in OCR: {str(e)}")
            return "", 0.0
    
    def extract_plate(self, frame, bbox):
        """
        Extract license plate region from frame
        Args:
            frame: full frame
            bbox: bounding box coordinates (x1, y1, x2, y2)
        Returns:
            plate image
        """
        x1, y1, x2, y2 = map(int, bbox)
        plate = frame[y1:y2, x1:x2]
        return plate
