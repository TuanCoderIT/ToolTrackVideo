"""
Stage 2: Automated Data Profiling & Pre-label Generation
Extracts metadata (Day/Night, Blur, Small Object, Occlusion, Density) and exports CVAT pre-labels.
"""

import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from .config import (
    COCO_TO_GREENSM_MAP,
    NIGHT_V_THRESH,
    NIGHT_P90_THRESH,
    NIGHT_EXTREME_THRESH,
    BLUR_THRESH,
    SMALL_OBJECT_RATIO,
    OCCLUSION_IOA_THRESH,
    DENSE_SCENE_COUNT,
)
from .utils import calculate_ioa


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
    is_night = bool(
        (brightness_v < NIGHT_EXTREME_THRESH) or
        (brightness_v < NIGHT_V_THRESH and p90_brightness < NIGHT_P90_THRESH)
    )

    # 2. Blur Tag (Laplacian variance < 100)
    if "laplacian_var" in survived_item:
        laplacian_var = survived_item["laplacian_var"]
    else:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blur = bool(laplacian_var < BLUR_THRESH)

    detections = survived_item.get("detections", [])
    vehicle_count = len(detections)

    # 3. Scale Variance Tag (has_small_object: vehicle box area / img_area < 0.01)
    has_small_object = False
    for det in detections:
        w_norm, h_norm = det["bbox_xywhn"][2], det["bbox_xywhn"][3]
        box_ratio = w_norm * h_norm
        if box_ratio < SMALL_OBJECT_RATIO:
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
            if ioa > OCCLUSION_IOA_THRESH:
                has_high_occlusion = True
                break
        if has_high_occlusion:
            break

    # 5. Density Tag (vehicle_count > 12)
    is_dense_scene = bool(vehicle_count > DENSE_SCENE_COUNT)

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


def run_data_profiling(survived_items, pre_labels_dir, data_profile_json):
    """
    Run Stage 2 profiling and pre-label export for all survived items.
    Updates and saves data_profile.json.
    """
    pre_labels_dir = Path(pre_labels_dir)
    data_profile_json = Path(data_profile_json)

    pre_labels_dir.mkdir(parents=True, exist_ok=True)

    # Load existing profile to merge incremental runs seamlessly
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

    return existing_profile
