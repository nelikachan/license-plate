# License Plate Recognition System

A comprehensive system for detecting vehicles, tracking them, and recognizing their license plates in video streams using deep learning techniques.

## Overview

This project combines YOLOv8 for object detection, SORT algorithm for object tracking, and EasyOCR for text recognition to create a robust license plate recognition system.

## Key Features

- Vehicle detection and tracking
- License plate detection
- Real-time license plate text recognition
- Web interface for video processing
- High confidence filtering (>85%)
- Vehicle-plate association

## Project Structure

```
license-plate-recognition-system/
├── models/                     # Trained models directory
│   └── best.pt                # Best trained model
├── runs/                      # Training artifacts
├── License-Plate-Recognition-4/# Dataset directory
│   ├── train/                 # Training data
│   ├── valid/                 # Validation data
│   ├── test/                  # Test data
│   └── data.yaml             # Dataset configuration
├── sort/                      # SORT tracking algorithm
├── app.py                     # Streamlit web interface
├── detector.py               # Main detection logic
├── ocr_reader.py            # OCR processing
├── prepare_dataset.py       # Dataset preparation
├── train_model.py           # Model training
└── requirements.txt         # Project dependencies
```

## Dataset

The project uses the "License Plate Recognition" dataset from Roboflow Universe:
- **Source**: Roboflow Universe
- **Project**: license-plate-recognition-rxg4e
- **Version**: 4
- **Format**: YOLOv8
- **Contents**:
  - Training images with license plate annotations
  - Validation images for model evaluation
  - Test images for final testing
  - Labels in YOLO format

### Dataset Preparation

1. Set up your Roboflow API key in `.env`:
```
ROBOFLOW_API_KEY=your_api_key
```

2. Run dataset preparation:
```bash
python prepare_dataset.py
```

## Model Training

The system uses YOLOv8 for both vehicle and license plate detection:

### Training Configuration
- Base model: YOLOv8n (nano)
- Input size: 640x640
- Batch size: 16
- Epochs: 20
- Early stopping patience: 5
- Optimizer: Auto
- Device: GPU (if available) or CPU

### Training Process

1. Ensure dataset is prepared
2. Run training:
```bash
python train_model.py
```

The training process will:
- Initialize YOLOv8 model
- Train on the prepared dataset
- Save the best model to `models/best.pt`
- Generate training metrics and visualizations in `runs/`

## License Plate Recognition

The system combines multiple components:

### 1. Vehicle Detection and Tracking
- Uses YOLOv8n for vehicle detection (cars, trucks, buses, motorcycles)
- Implements SORT algorithm for vehicle tracking
- Assigns unique IDs to tracked vehicles

### 2. License Plate Detection
- Uses custom-trained YOLOv8 model
- Detects license plates within vehicle bounding boxes
- Associates plates with specific vehicles

### 3. Text Recognition
- Uses EasyOCR for text extraction
- Implements preprocessing for better recognition:
  - Size normalization
  - Contrast enhancement
  - Multiple thresholding techniques
- Filters results by confidence (>85%)
- Applies text cleaning and formatting

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web interface:
```bash
streamlit run app.py
```

3. Through the web interface:
   - Upload a video file
   - Click "Process Video"
   - View real-time processing results
   - Monitor detection statistics
   - Review recognized plates in the results table

## Results Display

The system provides:
- Real-time video display with annotations
- Vehicle tracking boxes with IDs
- License plate detections with recognized text
- Confidence scores for each recognition
- Summary statistics of processing
- Filtered results table (>85% confidence)

## Dependencies

Key dependencies include:
- ultralytics>=8.0.0 (YOLOv8)
- opencv-python>=4.8.0
- streamlit>=1.26.0
- easyocr>=1.7.1
- torch>=2.0.0
- filterpy>=1.4.5 (SORT tracking)
- Additional dependencies in requirements.txt

## Notes

- The system performs best with clear, well-lit video footage
- GPU is recommended for optimal performance
- Recognition accuracy depends on:
  - Video quality
  - Lighting conditions
  - License plate visibility
  - Vehicle speed and angle
