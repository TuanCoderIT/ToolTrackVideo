"""
Utility functions: Standard Naming, Hashing, and Geometry Calculations
"""

import re
import hashlib
import uuid
from pathlib import Path


def sanitize_slug(name):
    """
    Sanitize text to a clean slug:
    Lowercase, alphanumeric and underscores only.
    """
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_').lower()
    return clean


def generate_image_id(identifier_str=None):
    """
    Generate an 8-character hex ID string (e.g. 'a4f89c2e').
    Uses MD5 of identifier if provided (deterministic), else random hex.
    """
    if identifier_str:
        return hashlib.md5(str(identifier_str).encode('utf-8')).hexdigest()[:8]
    return uuid.uuid4().hex[:8]


def generate_video_id(video_name):
    """
    Generate a 4-character ID for grouping frames from the same video (e.g. 'v8a1').
    """
    h = hashlib.md5(str(video_name).encode('utf-8')).hexdigest()[:4]
    return f"v{h[:3]}" if not h[0].isalpha() else h


def get_standard_image_name(source_type, source_name, index, original_stem=None):
    """
    Standard project-wide naming rule:
    - Image: img_<random_8_chars>.jpg (e.g. img_a4f89c2e.jpg)
    - Video: vid_<video_id>_<frame_stt:06d>.jpg (e.g. vid_v8a1_000001.jpg)
    """
    if source_type == "vid":
        vid_code = generate_video_id(source_name)
        return f"vid_{vid_code}_{index:06d}.jpg"
    else:
        key = original_stem if original_stem else f"{source_name}_{index}"
        img_code = generate_image_id(key)
        return f"img_{img_code}.jpg"


def calculate_ioa(box_a_xyxy, box_b_xyxy):
    """
    Calculate Intersection over Area (IoA) for box B covered by box A:
    IoA = Intersection_Area(box_A, box_B) / Area(box_B)
    """
    x1_a, y1_a, x2_a, y2_a = box_a_xyxy
    x1_b, y1_b, x2_b, y2_b = box_b_xyxy

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    if area_b <= 0:
        return 0.0

    return inter_area / area_b
