#!/usr/bin/env python3
"""
CVAT Packaging Tool - Backward-Compatibility Wrapper
Delegates directly to src.cvat_packager.
"""

import argparse
from src.config import DEFAULT_OUTPUT_DIR, DEFAULT_BATCH_SIZE
from src.cvat_packager import create_cvat_zip_package


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package deduplicated images and pre-labels into CVAT-ready archives."
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
        default=None,
        help="Directory to save output archives (Default: data/outputs/cvat_uploads)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Maximum images per CVAT ZIP package (Default: {DEFAULT_BATCH_SIZE})."
    )
    parser.add_argument(
        "--force_all",
        action="store_true",
        help="Force re-packaging of all images, ignoring manifest tracking."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_cvat_zip_package(
        output_dir=args.output_dir,
        zip_dir=args.zip_dir,
        batch_size=args.batch_size,
        force_all=args.force_all
    )


if __name__ == "__main__":
    main()
