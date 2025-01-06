import cv2
import numpy as np
from ultralytics import YOLO
import logging
from sort.sort import Sort
from ocr_reader import LicensePlateOCR

logger = logging.getLogger(__name__)

class LicensePlateDetector:
    def __init__(self, model_path='models/best.pt'):
        """
        Initialize detector with model
        Args:
            model_path: path to YOLOv8 model
        """
        try:
            self.vehicle_model = YOLO('yolov8n.pt')
            self.plate_model = YOLO(model_path)
            self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
            self.ocr = LicensePlateOCR()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def process_frame(self, frame):
        """
        Process a single frame
        Args:
            frame: numpy array of frame
        Returns:
            tuple: (processed frame, detections)
        """
        if frame is None or frame.size == 0:
            logger.error("Empty frame received")
            return None, []

        try:

            height, width = frame.shape[:2]
            
            vehicle_results = self.vehicle_model(frame, classes=[2,3,5,7], conf=0.5)[0]  
            vehicle_detections = []
            
            for r in vehicle_results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                vehicle_detections.append([x1, y1, x2, y2, conf])
            
            if len(vehicle_detections) > 0:
                tracked_vehicles = self.tracker.update(np.array(vehicle_detections))
            else:
                tracked_vehicles = np.empty((0, 5))

            plate_results = self.plate_model(frame)[0]
            detections = []

            for vehicle in tracked_vehicles:
                vehicle_id = int(vehicle[4])
                vehicle_bbox = vehicle[:4]
                
                cv2.rectangle(frame, 
                            (int(vehicle_bbox[0]), int(vehicle_bbox[1])), 
                            (int(vehicle_bbox[2]), int(vehicle_bbox[3])), 
                            (0, 255, 0), 2)
                
                cv2.putText(frame, 
                          f'Vehicle {vehicle_id}', 
                          (int(vehicle_bbox[0]), int(vehicle_bbox[1]-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                          (0, 255, 0), 2)

                for plate_box in plate_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = plate_box
                    
                    if (x1 >= vehicle_bbox[0] and x2 <= vehicle_bbox[2] and 
                        y1 >= vehicle_bbox[1] and y2 <= vehicle_bbox[3]):
                        
                        plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        if plate_img is not None and plate_img.size > 0:
                            plate_text, ocr_conf = self.ocr.read_plate(plate_img)
                            
                            if plate_text and ocr_conf > 0.85:
                                cv2.rectangle(frame, 
                                            (int(x1), int(y1)), 
                                            (int(x2), int(y2)), 
                                            (255, 0, 0), 2)
                                
                                cv2.putText(frame, 
                                          plate_text, 
                                          (int(x1), int(y1-10)), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.9, (255, 0, 0), 2)
                                
                                detections.append({
                                    'vehicle_id': vehicle_id,
                                    'plate_text': plate_text,
                                    'confidence': ocr_conf,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                })

            return frame, detections

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, []
