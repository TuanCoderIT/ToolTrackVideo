"""
Configuration and Taxonomy for GreenSM Vehicle Detection Pipeline
"""

from pathlib import Path

# GreenSM 7-Class Label Taxonomy (CVAT Aligned)
# 0: greensm (VinFast electric taxis/rentals - cyan/logo) [MANUAL LABEL ON CVAT]
# 1: car (general passenger cars, conventional taxis)
# 2: motorcycle (all motorcycles/scooters)
# 3: bike (bicycles)
# 4: person (pedestrians)
# 5: bus (city buses, coaches)
# 6: truck (trucks, pickups, heavy vehicles)
GREENSM_CLASSES = [
    "greensm",
    "car",
    "motorcycle",
    "bike",
    "person",
    "bus",
    "truck"
]

CLASS_COLORS = {
    "greensm": "#00FFFF",     # Cyan
    "car": "#FF0000",         # Red
    "motorcycle": "#00FF00",  # Green
    "bike": "#E91E63",        # Pink / Magenta
    "person": "#3F51B5",      # Indigo / Blue
    "bus": "#9C27B0",         # Purple
    "truck": "#FF9800"        # Orange
}

# Raw CVAT Label Specification for Direct Import
CVAT_LABELS_SPEC = [
    {"name": "greensm", "color": "#00FFFF", "type": "any", "attributes": []},
    {"name": "car", "color": "#FF0000", "type": "any", "attributes": []},
    {"name": "motorcycle", "color": "#00FF00", "type": "any", "attributes": []},
    {"name": "bike", "color": "#E91E63", "type": "any", "attributes": []},
    {"name": "person", "color": "#3F51B5", "type": "any", "attributes": []},
    {"name": "bus", "color": "#9C27B0", "type": "any", "attributes": []},
    {"name": "truck", "color": "#FF9800", "type": "any", "attributes": []}
]

# COCO Dataset Target IDs (YOLO11n COCO pretrained)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_TARGET_CLASS_IDS = {0, 1, 2, 3, 5, 7}
COCO_VEHICLE_CLASS_IDS = COCO_TARGET_CLASS_IDS  # Backward-compatible alias

# Mapping COCO IDs to Project 7-Class IDs
# NOTE: Class 0 ("greensm") is intentionally omitted from automated pre-labels
# and will be labeled manually by human annotators on CVAT.
COCO_TO_GREENSM_MAP = {
    0: 4,  # COCO person -> 4 (person)
    1: 3,  # COCO bicycle -> 3 (bike)
    2: 1,  # COCO car -> 1 (car)
    3: 2,  # COCO motorcycle -> 2 (motorcycle)
    5: 5,  # COCO bus -> 5 (bus)
    7: 6   # COCO truck -> 6 (truck)
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
