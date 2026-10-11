"""
Configuration and Taxonomy for GreenSM Vehicle Detection Pipeline
"""

from pathlib import Path

# GreenSM 5-Class Label Taxonomy (Mentor-Aligned)
# 0: greensm (VinFast electric taxis/rental cars - cyan/logo)
# 1: car (general passenger cars, conventional taxis)
# 2: motorcycle (all motorcycles/scooters)
# 3: bus (city buses, coaches)
# 4: truck (trucks, pickups, heavy vehicles)
GREENSM_CLASSES = [
    "greensm",
    "car",
    "motorcycle",
    "bus",
    "truck"
]

CLASS_COLORS = {
    "greensm": "#00FFFF",  # Cyan
    "car": "#FF0000",      # Red
    "motorcycle": "#00FF00", # Green
    "bus": "#9C27B0",      # Purple
    "truck": "#FF9800"     # Orange
}

# COCO Dataset Vehicle IDs (YOLO11n COCO pretrained)
# 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Mapping COCO Vehicle IDs to GreenSM 5-Class IDs
# Note: Class 0 (greensm) is reserved for human annotators on CVAT
COCO_TO_GREENSM_MAP = {
    2: 1,  # car -> 1
    3: 2,  # motorcycle -> 2
    5: 3,  # bus -> 3
    7: 4   # truck -> 4
}

# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# Default Pipeline Thresholds
DEFAULT_HD_THRESH = 5            # Hamming distance for pHash duplicate clustering
DEFAULT_CONF_THRESH = 0.25        # YOLO detection confidence threshold
DEFAULT_MODEL_PATH = "yolo11n.pt" # Pretrained model weights
DEFAULT_BATCH_SIZE = 50          # Number of images per CVAT zip batch

# Data Profiling Thresholds (Stage 2)
NIGHT_V_THRESH = 65.0            # HSV V-channel mean threshold
NIGHT_P90_THRESH = 100.0         # 90th percentile ambient light threshold
NIGHT_EXTREME_THRESH = 50.0      # Extreme darkness threshold
BLUR_THRESH = 100.0              # Laplacian variance threshold
SMALL_OBJECT_RATIO = 0.01        # BBox area / Image area < 1%
OCCLUSION_IOA_THRESH = 0.30      # Intersection over Area > 30%
DENSE_SCENE_COUNT = 12           # Vehicle count > 12

# Project Directory Defaults
DEFAULT_INPUT_DIR = Path("data/inputs")
DEFAULT_VIDEO_DIR = DEFAULT_INPUT_DIR / "videos"
DEFAULT_IMAGE_DIR = DEFAULT_INPUT_DIR / "images"
DEFAULT_OUTPUT_DIR = Path("data/outputs")
