"""
Stage 1: Intra-Video FastDedup
Algorithm: Perceptual Hashing with Object-Density Selection (pHash + Keep-Most-Vehicles, HD < 5)
"""

import cv2
from PIL import Image
import imagehash
from tqdm import tqdm

from .config import COCO_VEHICLE_CLASS_IDS, DEFAULT_HD_THRESH, DEFAULT_CONF_THRESH


def compute_phash(bgr_img):
    """Compute Perceptual Hash (pHash) of an OpenCV BGR image using PIL & imagehash."""
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    return imagehash.phash(pil_img)


def run_yolo_on_frame(model, frame_item, conf_thresh=DEFAULT_CONF_THRESH, device=""):
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

            # Filter for COCO vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
            if cls_id in COCO_VEHICLE_CLASS_IDS:
                xywhn = box.xywhn[0].cpu().numpy().tolist()
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                detections.append({
                    "cls": cls_id,
                    "conf": conf,
                    "bbox_xywhn": xywhn,
                    "bbox_xyxy": xyxy
                })

    return detections


def resolve_cluster_keep_most_vehicles(cluster_items, model, conf_thresh=DEFAULT_CONF_THRESH, device=""):
    """
    Keep-Most-Vehicles Conflict Resolution Rule (Object-Density Selection):
    For a cluster of duplicate frames (HD < threshold), run YOLO11n to count vehicles.
    Keep the SINGLE frame with the MAX vehicle count (tie-broken by Laplacian variance sharpness).
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


def run_fast_dedup(streamer, model, hd_thresh=DEFAULT_HD_THRESH, conf_thresh=DEFAULT_CONF_THRESH, device="", is_image_dir=False, desc="Deduplicating"):
    """
    Run Stage 1 Dual-Stream Fast Deduplication on a frame generator.
    Returns: (survived_items, total_processed_frames, total_clusters)
    """
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
        winner = resolve_cluster_keep_most_vehicles(cluster_list, model, conf_thresh, device)
        survived_items.append(winner)

    for frame_item in tqdm(streamer, desc=desc):
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

            is_same_cluster = (hd < hd_thresh)
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

    return survived_items, total_processed_frames, total_clusters
