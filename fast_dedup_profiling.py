#!/usr/bin/env python3
"""
GreenSM Vehicle Detection Pipeline - Stage 1 & Stage 2 CLI Tool

Standardized Naming Convention:
- Format: greensm_<source_type>_<clean_identifier>.<ext>
  - Video: greensm_vid_<video_slug>_f<frame_idx:06d>.jpg
  - Image: greensm_img_<image_slug>.jpg
- Rules: Lowercase, alphanumeric and underscores only, NO spaces, standard .jpg extension.

Features:
1. INTRA-VIDEO FASTDEDUP:
   - Generator-based frame streaming (1 FPS sampling for video, EXIF timestamp grouping for image dirs).
   - pHash computation (sliding comparison via Hamming distance).
   - Conflict resolution rule: Keep-Most-Vehicles using YOLO11n COCO predictions.
2. AUTOMATED DATA PROFILING:
   - Re-uses Stage 1 YOLO11n inference results (bounding boxes, confidence, class IDs).
   - Brightness (Day/Night via HSV V-mean).
   - Sharpness/Blur (Laplacian variance).
   - Scale variance (Small vehicle ratio < 1%).
   - High Occlusion (Intersection over Area - IoA > 30%).
   - Density tag (Vehicle count > 12).
   - Exports pre-label `.txt` files in YOLO format for CVAT mapped to GreenSM 5-class taxonomy:
     (0: greensm, 1: car, 2: motorcycle, 3: bus, 4: truck).
   - Outputs aggregated `data_profile.json`.
3. AUTOMATIC CVAT PACKAGING:
   - Native CVAT XML (CVAT 1.1 format) + Split YOLO batches.
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import imagehash
from tqdm import tqdm
from ultralytics import YOLO

from package_for_cvat import create_cvat_zip_package

# Target vehicle class IDs in COCO dataset
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Mapping from COCO vehicle IDs to GreenSM 5-class taxonomy:
# 0: greensm (reserved for CVAT human review)
# 1: car (from COCO 2)
# 2: motorcycle (from COCO 3)
# 3: bus (from COCO 5)
# 4: truck (from COCO 7)
COCO_TO_GREENSM_MAP = {
    2: 1,  # car -> 1
    3: 2,  # motorcycle -> 2
    5: 3,  # bus -> 3
    7: 4   # truck -> 4
}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

DEFAULT_INPUT_DIR = Path("data/inputs")
DEFAULT_VIDEO_DIR = DEFAULT_INPUT_DIR / "videos"
DEFAULT_IMAGE_DIR = DEFAULT_INPUT_DIR / "images"
DEFAULT_OUTPUT_DIR = Path("data/outputs")


def sanitize_slug(name):
    """
    Sanitize text to a clean slug:
    - Lowercase
    - Replace all non-alphanumeric characters (spaces, hyphens, dots, colons) with underscore
    - Strip leading and trailing underscores
    - Collapse multiple underscores into one
    """
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_').lower()
    return clean


import hashlib
import uuid


def generate_image_id(identifier_str=None):
    """Generate 8-character hex string (e.g. 'a4f89c2e')."""
    if identifier_str:
        return hashlib.md5(str(identifier_str).encode('utf-8')).hexdigest()[:8]
    return uuid.uuid4().hex[:8]


def generate_video_id(video_name):
    """Generate 4-character ID for video grouping (e.g. 'v8a1')."""
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="GreenSM CV Pipeline Stage 1 & 2: Intra-Video FastDedup & Automated Data Profiling"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video file (.mp4, .avi, ...) OR directory containing raw images. (Default: Auto-detect all from data/inputs/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory to save deduplicated images, data_profile.json, and pre_labels (Default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--hd_thresh",
        type=int,
        default=5,
        help="Hamming Distance threshold for pHash duplicate clustering (default: 5)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo11n.pt",
        help="Path or name of YOLO model (default: yolo11n.pt)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold (default: 0.25)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device for YOLO inference (e.g., 'cpu', 'cuda', '0'). Default auto-detect."
    )
    parser.add_argument(
        "--no_zip",
        action="store_true",
        help="Skip auto-creation of CVAT ZIP package at the end of pipeline."
    )
    return parser.parse_args()


def auto_detect_inputs():
    """Auto-detect all inputs from data/inputs/videos or data/inputs/images if --input is omitted."""
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
    Uses standardized naming: greensm_vid_<slug>_f<idx:06d>.jpg (NO SPACES).
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
    video_stem = Path(video_path).stem
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
            frame_name = get_standard_image_name("vid", video_stem, sampled_count)
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
    Uses standardized naming: greensm_img_<slug>.jpg (NO SPACES).
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

        # Standardize name: greensm_img_<slug>.jpg
        std_name = get_standard_image_name("img", path_obj.name, idx, original_stem=img_path.stem)

        yield {
            "frame_id": idx,
            "timestamp": ts,
            "bgr_img": bgr_img,
            "name": std_name,
            "source": path_obj.name
        }


def compute_phash(bgr_img):
    """Compute Perceptual Hash (pHash) of an OpenCV BGR image using PIL & imagehash."""
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    return imagehash.phash(pil_img)


def run_yolo_on_frame(model, frame_item, conf_thresh, device):
    """
    Run YOLO11n inference on a single frame and extract vehicle detections.
    Returns: list of dicts [{cls, conf, bbox_xywhn, bbox_xyxy}]
    """
    results = model.predict(
        source=frame_item["bgr_img"],
        conf=conf_thresh,
        device=device if device else None,
        verbose=False
    )

    detections = []
    if not results or len(results) == 0:
        return detections

    res = results[0]

    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Filter for vehicle classes
            if cls_id in VEHICLE_CLASS_IDS:
                xywhn = box.xywhn[0].cpu().numpy().tolist()
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                
                detections.append({
                    "cls": cls_id,
                    "conf": conf,
                    "bbox_xywhn": xywhn,
                    "bbox_xyxy": xyxy
                })

    return detections


def resolve_cluster_keep_most_vehicles(cluster_items, model, conf_thresh, device):
    """
    Keep-Most-Vehicles Conflict Resolution Rule:
    For a cluster of duplicate frames (HD < threshold), run YOLO11n to count vehicles (classes 2,3,5,7).
    Keep the SINGLE frame with the MAX vehicle count (tie-broken by sharpness/Laplacian var).
    Returns: winning frame item with cached YOLO detections.
    """
    best_item = None
    max_count = -1
    best_sharpness = -1.0

    for item in cluster_items:
        if "detections" not in item:
            item["detections"] = run_yolo_on_frame(model, item, conf_thresh, device)

        vehicle_count = len(item["detections"])
        
        gray = cv2.cvtColor(item["bgr_img"], cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        item["laplacian_var"] = sharpness

        if (vehicle_count > max_count) or (vehicle_count == max_count and sharpness > best_sharpness):
            max_count = vehicle_count
            best_sharpness = sharpness
            best_item = item

    return best_item


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


def profile_image(survived_item):
    """
    STAGE 2: AUTOMATED DATA PROFILING
    Computes metadata metrics for survived image using cached YOLO detections.
    """
    bgr_img = survived_item["bgr_img"]

    # 1. Day / Night Tag (Robust Multi-criteria)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    brightness_v = float(np.mean(v_channel))
    p90_brightness = float(np.percentile(v_channel, 90))

    # A scene is truly night if overall brightness < 50.0,
    # OR if brightness < 65.0 AND upper ambient light (p90) < 100.0.
    # (Prevents daytime scenes with dark asphalt road / building shadows from being misclassified as night)
    is_night = bool((brightness_v < 50.0) or (brightness_v < 65.0 and p90_brightness < 100.0))

    # 2. Blur Tag (Laplacian variance < 100)
    if "laplacian_var" in survived_item:
        laplacian_var = survived_item["laplacian_var"]
    else:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blur = bool(laplacian_var < 100.0)

    detections = survived_item.get("detections", [])
    vehicle_count = len(detections)

    # 3. Scale Variance Tag (has_small_object: vehicle box area / img_area < 0.01)
    has_small_object = False
    for det in detections:
        w_norm, h_norm = det["bbox_xywhn"][2], det["bbox_xywhn"][3]
        box_ratio = w_norm * h_norm
        if box_ratio < 0.01:
            has_small_object = True
            break

    # 4. High Occlusion Tag (IoA > 0.30 between any pair of vehicle BBoxes)
    has_high_occlusion = False
    n_det = len(detections)
    for i in range(n_det):
        for j in range(n_det):
            if i == j:
                continue
            ioa = calculate_ioa(detections[i]["bbox_xyxy"], detections[j]["bbox_xyxy"])
            if ioa > 0.30:
                has_high_occlusion = True
                break
        if has_high_occlusion:
            break

    # 5. Density Tag (vehicle_count > 12)
    is_dense_scene = bool(vehicle_count > 12)

    profile = {
        "source": survived_item["source"],
        "vehicle_count": vehicle_count,
        "brightness_v": round(brightness_v, 2),
        "laplacian_var": round(laplacian_var, 2),
        "tags": {
            "is_night": is_night,
            "is_blur": is_blur,
            "has_small_object": has_small_object,
            "has_high_occlusion": has_high_occlusion,
            "is_dense_scene": is_dense_scene
        }
    }

    return profile


def export_pre_labels(survived_item, pre_labels_dir):
    """
    Exports pre-labels in YOLO normalized format (class x_center y_center width height)
    to <pre_labels_dir>/<standard_image_stem>.txt mapped to GreenSM 5-class taxonomy:
    (0: greensm, 1: car, 2: motorcycle, 3: bus, 4: truck)
    """
    img_name = survived_item["name"]
    stem = Path(img_name).stem
    txt_path = Path(pre_labels_dir) / f"{stem}.txt"

    lines = []
    for det in survived_item.get("detections", []):
        coco_cls = det["cls"]
        target_cls = COCO_TO_GREENSM_MAP.get(coco_cls, coco_cls)
        x_c, y_c, w, h = det["bbox_xywhn"]
        lines.append(f"{target_cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def process_single_input(input_path, output_dir, model, args):
    """Process a single video or image directory input."""
    input_path = Path(input_path).resolve()
    is_video = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS
    is_image_dir = input_path.is_dir()

    if is_video:
        streamer = video_frame_streamer(str(input_path))
    elif is_image_dir:
        streamer = image_dir_streamer(str(input_path))
    else:
        print(f"[SKIP] {input_path} is not a valid video or image directory.")
        return

    dedup_images_dir = output_dir / "deduplicated_images"
    pre_labels_dir = output_dir / "pre_labels"
    data_profile_json = output_dir / "data_profile.json"

    dedup_images_dir.mkdir(parents=True, exist_ok=True)
    pre_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" PROCESSING INPUT: {input_path.name}")
    print(f"=======================================================")
    print(f"=== STAGE 1: pHash + Keep-Most-Vehicles (Perceptual Hashing with Object-Density Selection, HD < {args.hd_thresh}) ===")

    current_cluster = []
    current_root_hash = None
    current_root_ts = None

    survived_items = []
    total_processed_frames = 0
    total_clusters = 0

    def process_and_finalize_cluster(cluster_list):
        nonlocal total_clusters
        if not cluster_list:
            return
        total_clusters += 1
        winner = resolve_cluster_keep_most_vehicles(cluster_list, model, args.conf, args.device)
        survived_items.append(winner)

    for frame_item in tqdm(streamer, desc=f"Deduplicating {input_path.name}"):
        total_processed_frames += 1
        frame_hash = compute_phash(frame_item["bgr_img"])
        frame_item["phash"] = frame_hash

        if not current_cluster:
            current_cluster = [frame_item]
            current_root_hash = frame_hash
            current_root_ts = frame_item["timestamp"]
        else:
            hd = frame_hash - current_root_hash
            time_diff = abs(frame_item["timestamp"] - current_root_ts)

            is_same_cluster = (hd < args.hd_thresh)
            if is_image_dir and time_diff >= 1.0:
                is_same_cluster = False

            if is_same_cluster:
                current_cluster.append(frame_item)
            else:
                process_and_finalize_cluster(current_cluster)
                current_cluster = [frame_item]
                current_root_hash = frame_hash
                current_root_ts = frame_item["timestamp"]

    if current_cluster:
        process_and_finalize_cluster(current_cluster)

    dedup_ratio = ((total_processed_frames - len(survived_items)) / total_processed_frames * 100.0) if total_processed_frames > 0 else 0.0
    print(f"[INFO] Stage 1 Completed for {input_path.name}:")
    print(f"       - Total Input Frames Sampled: {total_processed_frames}")
    print(f"       - Formed Clusters: {total_clusters}")
    print(f"       - Survived Frames Kept: {len(survived_items)}")
    print(f"       - Deduplication Ratio: {dedup_ratio:.2f}% reduction")

    print(f"\n[INFO] Saving survived images (Standard Naming, NO SPACES) to: {dedup_images_dir}")
    for item in tqdm(survived_items, desc="Saving Survived Images"):
        save_path = dedup_images_dir / item["name"]
        cv2.imwrite(str(save_path), item["bgr_img"])

    print(f"\n=== STAGE 2: AUTOMATED DATA PROFILING (Using Cached YOLO Inference) ===")

    existing_profile = {}
    if data_profile_json.exists():
        try:
            with open(data_profile_json, "r", encoding="utf-8") as f:
                existing_profile = json.load(f)
        except Exception:
            existing_profile = {}

    for item in tqdm(survived_items, desc="Profiling Survived Images"):
        profile_data = profile_image(item)
        existing_profile[item["name"]] = profile_data
        export_pre_labels(item, pre_labels_dir)

    with open(data_profile_json, "w", encoding="utf-8") as f:
        json.dump(existing_profile, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Updated metadata profile at: {data_profile_json}")
    print(f"[INFO] Pre-labels saved at: {pre_labels_dir}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.input:
        inputs = [Path(args.input).resolve()]
    else:
        inputs = auto_detect_inputs()

    if not inputs:
        print(f"[ERROR] No input specified and no data found in '{DEFAULT_VIDEO_DIR}' or '{DEFAULT_IMAGE_DIR}'.")
        print("        Please place your video in 'data/inputs/videos/' OR images in 'data/inputs/images/',")
        print("        or specify input using: python fast_dedup_profiling.py --input /path/to/data")
        sys.exit(1)

    print(f"[INFO] Found {len(inputs)} input source(s) to process.")
    print(f"[INFO] Loading YOLO model '{args.model_path}'...")
    model = YOLO(args.model_path)

    for idx, input_item in enumerate(inputs, start=1):
        print(f"\n>>> Running [{idx}/{len(inputs)}]: {input_item}")
        process_single_input(input_item, output_dir, model, args)

    print(f"\n[SUCCESS] Pipeline execution finished cleanly!")

    # Auto-package for CVAT unless --no_zip is set
    if not args.no_zip:
        print(f"\n=== AUTOMATIC CVAT PACKAGING ===")
        zip_dir = output_dir / "cvat_uploads"
        create_cvat_zip_package(output_dir, zip_dir, force_all=True)


if __name__ == "__main__":
    main()
