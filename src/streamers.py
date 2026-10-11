"""
Data Streamers: Memory-efficient Generator Streamers for Video and Images
"""

import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

from .config import (
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    DEFAULT_INPUT_DIR,
    DEFAULT_VIDEO_DIR,
    DEFAULT_IMAGE_DIR,
)
from .utils import get_standard_image_name


def get_exif_timestamp(img_path):
    """Attempt to extract EXIF timestamp from an image file."""
    try:
        pil_img = Image.open(img_path)
        exif = pil_img._getexif()
        if exif:
            for tag_id in (36867, 306):
                if tag_id in exif:
                    dt_str = exif[tag_id]
                    dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                    return dt.timestamp()
    except Exception:
        pass
    return os.path.getmtime(img_path)


def video_frame_streamer(video_path):
    """
    Generator streaming 1 frame per second (1 FPS) from a video file to avoid RAM overflow.
    Uses standardized naming: vid_<video_id>_<idx:06d>.jpg (NO SPACES).
    Yields: dict with keys: 'frame_id', 'timestamp', 'bgr_img', 'name', 'source'
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0

    frame_interval = max(1, int(round(fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = Path(video_path).name
    current_frame_idx = 0
    sampled_count = 0

    print(f"[INFO] Video loaded: {video_name} | FPS: {fps:.2f} | Total Frames: {total_frames} | Frame Step (1 FPS): {frame_interval}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx % frame_interval == 0:
            sampled_count += 1
            sec = current_frame_idx / fps
            frame_name = get_standard_image_name("vid", video_name, sampled_count)
            yield {
                "frame_id": current_frame_idx,
                "timestamp": sec,
                "bgr_img": frame,
                "name": frame_name,
                "source": video_name
            }

        current_frame_idx += 1

    cap.release()


def image_dir_streamer(img_dir_path):
    """
    Generator streaming images sorted by EXIF timestamp / file mtime.
    Uses standardized naming: img_<random_8_chars>.jpg (NO SPACES).
    Yields: dict with keys: 'frame_id', 'timestamp', 'bgr_img', 'name', 'source'
    """
    path_obj = Path(img_dir_path)
    img_files = [
        f for f in path_obj.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not img_files:
        raise RuntimeError(f"No valid image files found in directory: {img_dir_path}")

    print(f"[INFO] Reading {len(img_files)} images from directory: {path_obj.name}")
    print("[INFO] Extracting EXIF/timestamps for sorting & time grouping...")

    img_with_time = []
    for img_path in img_files:
        ts = get_exif_timestamp(img_path)
        img_with_time.append((img_path, ts))

    img_with_time.sort(key=lambda x: (x[1], x[0].name))

    for idx, (img_path, ts) in enumerate(img_with_time, start=1):
        bgr_img = cv2.imread(str(img_path))
        if bgr_img is None:
            continue

        std_name = get_standard_image_name("img", path_obj.name, idx, original_stem=img_path.stem)

        yield {
            "frame_id": idx,
            "timestamp": ts,
            "bgr_img": bgr_img,
            "name": std_name,
            "source": path_obj.name
        }


def auto_detect_inputs():
    """Auto-detect all inputs from data/inputs/videos or data/inputs/images."""
    inputs = []

    # Check for videos in data/inputs/videos
    if DEFAULT_VIDEO_DIR.exists():
        video_files = sorted([
            f for f in DEFAULT_VIDEO_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ])
        inputs.extend(video_files)

    # Check for images in data/inputs/images
    if DEFAULT_IMAGE_DIR.exists():
        image_files = [
            f for f in DEFAULT_IMAGE_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if image_files:
            inputs.append(DEFAULT_IMAGE_DIR)

    # Check root data/inputs directory as fallback
    if DEFAULT_INPUT_DIR.exists():
        root_videos = sorted([
            f for f in DEFAULT_INPUT_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ])
        inputs.extend(root_videos)

    return inputs
