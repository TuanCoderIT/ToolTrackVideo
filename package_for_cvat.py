#!/usr/bin/env python3
"""
CVAT Packaging Tool for GreenSM Vehicle Detection Project

Features:
- Packages survived images and matching YOLO pre-labels into small, manageable CVAT ZIP files (Default: 50 images per ZIP).
- Generates BOTH:
  1. Split Full ZIP batches (Images + Labels, max 50 images per zip to avoid large file limits).
  2. Ultra-lightweight Annotation-Only ZIP archive (contains ONLY .txt pre-labels, ~100 KB size, instant upload to CVAT).
- Fully compatible with CVAT 'Upload Annotations -> YOLO 1.1'.
- Tracks packaged files in `.zipped_manifest.json` so images are never duplicated.
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path
from datetime import datetime

DEFAULT_OUTPUT_DIR = Path("data/outputs")
DEFAULT_ZIP_DIR = DEFAULT_OUTPUT_DIR / "cvat_uploads"
MANIFEST_FILE = DEFAULT_OUTPUT_DIR / ".zipped_manifest.json"

# GreenSM 5-Class Taxonomy
GREENSM_CLASSES = [
    "greensm",      # 0: GreenSM Electric Taxi / Rental Car
    "car",          # 1: General Cars & tech taxis
    "motorcycle",   # 2: Motorcycles
    "bus",          # 3: Buses
    "truck"         # 4: Trucks
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package deduplicated images and pre-labels into manageable CVAT-ready ZIP archives."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory containing deduplicated_images and pre_labels (Default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--zip_dir",
        type=str,
        default=str(DEFAULT_ZIP_DIR),
        help=f"Directory to save output ZIP archives (Default: {DEFAULT_ZIP_DIR})."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Maximum number of images per ZIP file (Default: 50 images per zip)."
    )
    parser.add_argument(
        "--force_all",
        action="store_true",
        help="Force re-packaging of all images, ignoring manifest tracking."
    )
    return parser.parse_args()


def load_manifest(manifest_path):
    """Load manifest tracking file for previously packaged items."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"zipped_files": [], "batches": []}


def save_manifest(manifest_path, manifest_data):
    """Save manifest tracking file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)


def create_single_zip_batch(zip_path, images_chunk, pre_labels_dir, include_images=True):
    """Helper to create a single zip file with given images and pre-label text files."""
    obj_names_content = "\n".join(GREENSM_CLASSES) + "\n"
    obj_data_content = (
        f"classes = {len(GREENSM_CLASSES)}\n"
        f"train = train.txt\n"
        f"names = obj.names\n"
        f"backup = backup/\n"
    )

    train_txt_lines = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Standard Metadata Files
        zf.writestr("obj.names", obj_names_content)
        zf.writestr("obj.data", obj_data_content)

        for img_file in images_chunk:
            stem = img_file.stem
            img_name = img_file.name
            txt_name = f"{stem}.txt"

            train_txt_lines.append(f"data/obj_train_data/{img_name}\n")

            # Write images if requested
            if include_images and img_file.exists():
                zf.write(img_file, arcname=f"data/obj_train_data/{img_name}")
                zf.write(img_file, arcname=f"obj_train_data/{img_name}")
                zf.write(img_file, arcname=img_name)

            # Write label file (.txt)
            txt_file = pre_labels_dir / txt_name
            if txt_file.exists():
                txt_data = txt_file.read_bytes()
            else:
                txt_data = b""

            zf.writestr(f"data/obj_train_data/{txt_name}", txt_data)
            zf.writestr(f"obj_train_data/{txt_name}", txt_data)
            zf.writestr(txt_name, txt_data)

        zf.writestr("train.txt", "".join(train_txt_lines))


def create_cvat_zip_package(output_dir, zip_dir, batch_size=50, force_all=False):
    output_dir = Path(output_dir).resolve()
    zip_dir = Path(zip_dir).resolve()

    dedup_images_dir = output_dir / "deduplicated_images"
    pre_labels_dir = output_dir / "pre_labels"
    manifest_path = output_dir / ".zipped_manifest.json"

    if not dedup_images_dir.exists():
        print(f"[ERROR] Deduplicated images directory does not exist: {dedup_images_dir}")
        return None

    manifest = load_manifest(manifest_path)
    already_zipped = set(manifest.get("zipped_files", [])) if not force_all else set()

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    all_images = sorted([
        f for f in dedup_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not all_images:
        print(f"[INFO] No images found in {dedup_images_dir}.")
        return None

    new_images = [img for img in all_images if img.name not in already_zipped]

    if not new_images:
        print(f"[INFO] All {len(all_images)} image(s) in '{dedup_images_dir.name}' have ALREADY been packaged in previous ZIP batch(es).")
        print(f"       No new un-zipped images found. Skipping packaging to prevent duplication.")
        return None

    print(f"[INFO] Found {len(new_images)} NEW un-packaged image(s) out of {len(all_images)} total.")
    print(f"[INFO] Batch size limit: {batch_size} images per ZIP file.")

    zip_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Split into chunks of batch_size
    chunks = [new_images[i:i + batch_size] for i in range(0, len(new_images), batch_size)]
    total_chunks = len(chunks)

    created_zip_paths = []
    packaged_names = []

    print(f"\n=== GENERATING {total_chunks} SPLIT ZIP BATCH(ES) ===")
    for idx, chunk in enumerate(chunks, start=1):
        zip_filename = f"cvat_batch_{timestamp_str}_part{idx:02d}_of_{total_chunks:02d}.zip"
        zip_path = zip_dir / zip_filename

        print(f"[INFO] Packaging Part [{idx}/{total_chunks}] ({len(chunk)} images): {zip_filename}")
        create_single_zip_batch(zip_path, chunk, pre_labels_dir, include_images=True)
        created_zip_paths.append(zip_path)

        for img in chunk:
            packaged_names.append(img.name)

    # ALSO GENERATE ANNOTATION-ONLY ZIP (SUPER LIGHTWEIGHT, NO IMAGES)
    anno_only_filename = f"cvat_ANNOTATIONS_ONLY_{timestamp_str}.zip"
    anno_only_path = zip_dir / anno_only_filename
    print(f"\n[INFO] Packaging LIGHTWEIGHT Annotations-Only ZIP (~100 KB): {anno_only_filename}")
    create_single_zip_batch(anno_only_path, new_images, pre_labels_dir, include_images=False)
    created_zip_paths.append(anno_only_path)

    # Update manifest
    manifest["zipped_files"].extend(packaged_names)
    manifest["batches"].append({
        "timestamp": datetime.now().isoformat(),
        "image_count": len(packaged_names),
        "split_parts": total_chunks,
        "annotation_only_file": anno_only_filename
    })
    save_manifest(manifest_path, manifest)

    print(f"\n[SUCCESS] CVAT Packages created successfully!")
    print(f"          Output Folder: {zip_dir}")
    print(f"          - Created {total_chunks} split batch ZIP(s) (Images + Labels, {batch_size} img/zip)")
    print(f"          - Created 1 Annotations-Only ZIP ({anno_only_filename} - Super Fast Upload!)")

    return created_zip_paths


def main():
    args = parse_args()
    create_cvat_zip_package(args.output_dir, args.zip_dir, batch_size=args.batch_size, force_all=args.force_all)


if __name__ == "__main__":
    main()
