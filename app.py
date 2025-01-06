import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from detector import LicensePlateDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(video_file):
    """
    Process video file and detect license plates
    Args:
        video_file: uploaded video file
    Returns:
        None
    """
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        detector = LicensePlateDetector()
        
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        table_placeholder = st.empty()
        stop_button_placeholder = st.empty()
        
        results = []
        frame_count = 0
        total_detections = 0
        high_conf_detections = 0
        
        if 'stop_processing' not in st.session_state:
            st.session_state.stop_processing = False
            
        if stop_button_placeholder.button('Stop Processing', key='stop_button'):
            st.session_state.stop_processing = True
        
        while cap.isOpened() and not st.session_state.stop_processing:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            processed_frame, detections = detector.process_frame(frame)
            
            if processed_frame is not None:
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(rgb_frame)
                
                total_detections += len(detections)
                high_conf_detections += len([d for d in detections if float(d['confidence']) > 0.85])
                
                for det in detections:
                    conf = float(det['confidence'])
                    if conf > 0.85:  
                        results.append({
                            'Frame': frame_count,
                            'Vehicle ID': det['vehicle_id'],
                            'Plate Number': det['plate_text'],
                            'Confidence': f"{conf:.2%}" 
                        })
                
                info_text = f"""
                Processed frames: {frame_count}
                Total detections: {total_detections}
                High confidence detections (>85%): {high_conf_detections}
                """
                info_placeholder.text(info_text)
                
                if results:
                    df = pd.DataFrame(results)
                    df = df.sort_values(['Confidence', 'Frame'], ascending=[False, True])
                    table_placeholder.dataframe(df)
        
        cap.release()
        
        st.session_state.stop_processing = False
        
        if results:
            st.success(f"Processing completed! Found {high_conf_detections} high confidence detections.")
        else:
            st.warning("No high confidence detections found.")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        st.error(f"Error processing video: {str(e)}")

def main():
    st.title("License plate recognition system")
    st.write("Detects and recognizes license plates with confidence > 85%")
    
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        if st.button('Process Video', key='start_button'):
            process_video(video_file)

if __name__ == "__main__":
    main()
