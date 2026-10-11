"""
CVAT Packaging Module: Generates Native XML (CVAT 1.1) and Split YOLO ZIP archives
"""

import json
import zipfile
from pathlib import Path
from datetime import datetime
from PIL import Image

from .config import (
    GREENSM_CLASSES,
    CLASS_COLORS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_OUTPUT_DIR,
)


def load_manifest(manifest_path):
    """Load manifest tracking file for previously packaged items."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"zipped_files": [], "batches": []}


def save_manifest(manifest_path, manifest_data):
    """Save manifest tracking file."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)


def create_cvat_xml_annotations(images_list, dedup_images_dir, pre_labels_dir, xml_path):
    """
    Generate native CVAT 1.1 / CVAT for images 1.1 XML annotation file.
    This format is natively supported by CVAT and avoids all path matching issues of YOLO.
    """
    dedup_images_dir = Path(dedup_images_dir)
    pre_labels_dir = Path(pre_labels_dir)
    xml_path = Path(xml_path)

    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<annotations>',
        '  <version>1.1</version>',
        '  <meta>',
        '    <task>',
        '      <id>1</id>',
        '      <name>GreenSM_PreLabels</name>',
        f'      <size>{len(images_list)}</size>',
        '      <mode>annotation</mode>',
        '      <labels>'
    ]

    for c in GREENSM_CLASSES:
        color = CLASS_COLORS.get(c, "#FFFFFF")
        xml_lines.append(f'        <label><name>{c}</name><color>{color}</color></label>')

    xml_lines.extend([
        '      </labels>',
        '    </task>',
        '  </meta>'
    ])

    total_boxes = 0

    for idx, img_file in enumerate(images_list):
        img_name = img_file.name
        img_path = dedup_images_dir / img_name

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            w, h = 1920, 1080

        xml_lines.append(f'  <image id="{idx}" name="{img_name}" width="{w}" height="{h}">')

        txt_name = f"{img_file.stem}.txt"
        txt_path = pre_labels_dir / txt_name

        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                            label_name = GREENSM_CLASSES[cls_id] if cls_id < len(GREENSM_CLASSES) else "car"

                            xtl = max(0.0, (xc - bw / 2.0) * w)
                            ytl = max(0.0, (yc - bh / 2.0) * h)
                            xbr = min(float(w), (xc + bw / 2.0) * w)
                            ybr = min(float(h), (yc + bh / 2.0) * h)

                            xml_lines.append(
                                f'    <box label="{label_name}" occluded="0" xtl="{xtl:.2f}" ytl="{ytl:.2f}" '
                                f'xbr="{xbr:.2f}" ybr="{ybr:.2f}" z_order="0">\n    </box>'
                            )
                            total_boxes += 1
                        except Exception:
                            continue

        xml_lines.append('  </image>')

    xml_lines.append('</annotations>')

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))

    return total_boxes


def create_single_zip_batch(zip_path, images_chunk, pre_labels_dir, include_images=True):
    """Helper to create a single YOLO 1.1 zip file with exact filename matching."""
    zip_path = Path(zip_path)
    pre_labels_dir = Path(pre_labels_dir)

    obj_names_content = "\n".join(GREENSM_CLASSES) + "\n"
    obj_data_content = (
        f"classes = {len(GREENSM_CLASSES)}\n"
        f"train = train.txt\n"
        f"names = obj.names\n"
        f"backup = backup/\n"
    )

    train_txt_lines = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", obj_names_content)
        zf.writestr("obj.data", obj_data_content)

        for img_file in images_chunk:
            stem = img_file.stem
            img_name = img_file.name
            txt_name = f"{stem}.txt"

            train_txt_lines.append(f"{img_name}\n")
            train_txt_lines.append(f"obj_train_data/{img_name}\n")

            if include_images and img_file.exists():
                zf.write(img_file, arcname=f"obj_train_data/{img_name}")
                zf.write(img_file, arcname=img_name)

            txt_file = pre_labels_dir / txt_name
            if txt_file.exists():
                txt_data = txt_file.read_bytes()
            else:
                txt_data = b""

            zf.writestr(f"obj_train_data/{txt_name}", txt_data)
            zf.writestr(txt_name, txt_data)

        zf.writestr("train.txt", "".join(train_txt_lines))


def create_cvat_zip_package(output_dir=DEFAULT_OUTPUT_DIR, zip_dir=None, batch_size=DEFAULT_BATCH_SIZE, force_all=False):
    """
    Creates CVAT Native XML packages and split YOLO batches.
    """
    output_dir = Path(output_dir).resolve()
    zip_dir = Path(zip_dir).resolve() if zip_dir else output_dir / "cvat_uploads"

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
    zip_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Master CVAT Native XML (ALL Images)
    master_xml_filename = f"cvat_annotations_ALL_{timestamp_str}.xml"
    master_xml_path = zip_dir / master_xml_filename
    master_xml_zip_filename = f"cvat_NATIVE_XML_ALL_{timestamp_str}.zip"
    master_xml_zip_path = zip_dir / master_xml_zip_filename

    total_boxes = create_cvat_xml_annotations(new_images, dedup_images_dir, pre_labels_dir, master_xml_path)
    with zipfile.ZipFile(master_xml_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(master_xml_path, arcname="annotations.xml")

    # 2. Split Batches (50 images/part)
    chunks = [new_images[i:i + batch_size] for i in range(0, len(new_images), batch_size)]
    total_chunks = len(chunks)
    packaged_names = []

    print(f"\n=== GENERATING {total_chunks} SPLIT BATCHES ({batch_size} IMAGES/PART) ===")
    for idx, chunk in enumerate(chunks, start=1):
        # Image + YOLO zip
        zip_filename = f"cvat_batch_{timestamp_str}_part{idx:02d}_of_{total_chunks:02d}.zip"
        zip_path = zip_dir / zip_filename
        create_single_zip_batch(zip_path, chunk, pre_labels_dir, include_images=True)

        # Native XML zip for this part
        part_xml_filename = f"cvat_NATIVE_XML_part{idx:02d}_of_{total_chunks:02d}.xml"
        part_xml_path = zip_dir / part_xml_filename
        part_xml_zip_path = zip_dir / f"cvat_NATIVE_XML_part{idx:02d}_of_{total_chunks:02d}.zip"

        create_cvat_xml_annotations(chunk, dedup_images_dir, pre_labels_dir, part_xml_path)
        with zipfile.ZipFile(part_xml_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(part_xml_path, arcname="annotations.xml")

        print(f"[INFO] Part [{idx}/{total_chunks}] ({len(chunk)} images):")
        print(f"       - Image ZIP: {zip_filename}")
        print(f"       - Native XML ZIP: {part_xml_zip_path.name}")

        for img in chunk:
            packaged_names.append(img.name)

    manifest["zipped_files"].extend(packaged_names)
    manifest["batches"].append({
        "timestamp": datetime.now().isoformat(),
        "image_count": len(packaged_names),
        "total_boxes": total_boxes,
        "master_xml": master_xml_zip_filename
    })
    save_manifest(manifest_path, manifest)

    print(f"\n[SUCCESS] Packaging Completed!")
    print(f"  ⭐ Dùng file XML này nếu Task chứa TOÀN BỘ {len(new_images)} ảnh: {master_xml_zip_path.name}")
    print(f"  ⭐ Dùng file XML part tương ứng nếu bạn tạo Task theo từng Part:")
    for idx in range(1, total_chunks + 1):
        print(f"     - Task Part {idx}: cvat_NATIVE_XML_part{idx:02d}_of_{total_chunks:02d}.zip")

    return master_xml_zip_path
