#!/usr/bin/env python3
"""
GreenSM Vehicle Detection - Pipeline CLI Orchestrator
Executes Stage 1 (FastDedup) and Stage 2 (Automated Data Profiling & CVAT Packaging).
"""

import sys
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from src.config import (
    DEFAULT_HD_THRESH,
    DEFAULT_CONF_THRESH,
    DEFAULT_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_VIDEO_DIR,
    DEFAULT_IMAGE_DIR,
    DEFAULT_OUTPUT_DIR,
    VIDEO_EXTENSIONS,
)
from src.streamers import (
    video_frame_streamer,
    image_dir_streamer,
    auto_detect_inputs,
)
from src.dedup import run_fast_dedup
from src.profiling import run_data_profiling
from src.cvat_packager import create_cvat_zip_package


def parse_args():
    parser = argparse.ArgumentParser(
        description="GreenSM CV Pipeline: Intra-Video FastDedup & Automated Data Profiling"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video file (.mp4, .avi, ...) OR directory containing raw images. (Default: Auto-detect from data/inputs/)"
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
        default=DEFAULT_HD_THRESH,
        help=f"Hamming Distance threshold for pHash duplicate clustering (Default: {DEFAULT_HD_THRESH})."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path or name of YOLO model (Default: {DEFAULT_MODEL_PATH})."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESH,
        help=f"YOLO detection confidence threshold (Default: {DEFAULT_CONF_THRESH})."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device for YOLO inference (e.g. 'cpu', 'cuda', '0'). Default auto-detect."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Maximum images per CVAT ZIP package (Default: {DEFAULT_BATCH_SIZE})."
    )
    parser.add_argument(
        "--no_zip",
        action="store_true",
        help="Skip auto-creation of CVAT ZIP package at the end of pipeline."
    )
    return parser.parse_args()


def process_single_input(input_path, output_dir, model, args):
    """Run pipeline on a single video file or image directory."""
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

    # 1. Run Stage 1 FastDedup
    survived_items, total_frames, total_clusters = run_fast_dedup(
        streamer=streamer,
        model=model,
        hd_thresh=args.hd_thresh,
        conf_thresh=args.conf,
        device=args.device,
        is_image_dir=is_image_dir,
        desc=f"Deduplicating {input_path.name}"
    )

    dedup_ratio = ((total_frames - len(survived_items)) / total_frames * 100.0) if total_frames > 0 else 0.0
    print(f"[INFO] Stage 1 Completed for {input_path.name}:")
    print(f"       - Total Input Frames Sampled: {total_frames}")
    print(f"       - Formed Clusters: {total_clusters}")
    print(f"       - Survived Frames Kept: {len(survived_items)}")
    print(f"       - Deduplication Ratio: {dedup_ratio:.2f}% reduction")

    # 2. Save Survived Images
    print(f"\n[INFO] Saving survived images (Standard Naming, NO SPACES) to: {dedup_images_dir}")
    for item in tqdm(survived_items, desc="Saving Survived Images"):
        save_path = dedup_images_dir / item["name"]
        cv2.imwrite(str(save_path), item["bgr_img"])

    # 3. Run Stage 2 Profiling & Pre-label export
    print(f"\n=== STAGE 2: AUTOMATED DATA PROFILING (Using Cached YOLO Inference) ===")
    run_data_profiling(survived_items, pre_labels_dir, data_profile_json)


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
        print("        or specify input using: python run_pipeline.py --input /path/to/data")
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
        create_cvat_zip_package(output_dir, zip_dir, batch_size=args.batch_size, force_all=True)


if __name__ == "__main__":
    main()
