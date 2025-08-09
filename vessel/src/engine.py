from ultralytics import YOLO
from PIL import Image, ImageOps
from parse_table import parse_table
from parse_nozzles import process_image as process_nozzle_image
import uuid
import os
import time
import io
import requests
from utils import annotate_image

section_model_path = "yolo/section/best.pt"

# LOAD MODELS
# donut_notes = load_model(notes_model_path)
yolo_section = YOLO(section_model_path)

conf = 0.5
table_offset = 10
notes_offset = 15

def get_section_from_image(image_path):
    """
    Extract section name from an image using DONUT model.
    """
    output = yolo_section(image_path, imgsz=1536, verbose=False, conf=conf)

    # Extract detections
    notes_results = []
    table_results = []

    for box, cls_id in zip(output[0].boxes.xyxy, output[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls_id)

        # Map classes
        if class_id == 0:  # notes
            notes_results.append([x1, y1, x2, y2])
        elif class_id == 1:  # table
            table_results.append([x1, y1, x2, y2])

    output = {
        "notes": notes_results,
        "table": table_results
    }

    return output

def get_tables(table_img_paths, bboxes):
    """
    Parse tables from a list of image paths using PPStructureV3.
    """
    parsed_tables = []
    for idx, table_img_path in enumerate(table_img_paths):
        output = parse_table(table_img_path)
        bbox = bboxes[idx]
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]  # Convert to xywh format
        output['bbox'] = bbox
        parsed_tables.append(output)
    return parsed_tables

def save_image(path, image):
    """
    Save an image to the specified path.
    """
    image.save(path)

def analyze_vessel_image(image_path):
    """
    Analyze a vessel image to extract sections, notes, nozzles, and views.
    """
    table_imgs_paths = []
    table_bboxes = []

    initial_time = time.time()

    start_time = time.time()
    sections = get_section_from_image(image_path)
    print("Sections extracted and took:", time.time() - start_time, "seconds")

    start_time = time.time()
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # Handle EXIF orientation
    width, height = img.size

    start_time = time.time()
    all_notes = []
    notes = sections.get("notes", [])
    for note in notes:
        x1, y1, x2, y2 = note
        x1_off = max(0, x1 - notes_offset)
        y1_off = max(0, y1 - notes_offset)
        x2_off = min(width, x2 + notes_offset)
        y2_off = min(height, y2 + notes_offset)

        # Crop the image
        crop = img.crop((x1_off, y1_off, x2_off, y2_off))
        #  Convert crop to bytes
        img_bytes = io.BytesIO()
        crop.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Send to DONUT inference endpoint
        files = {"file": ("crop.png", img_bytes, "image/png")}
        response = requests.post("http://donut_api:8000/infer", files=files)

        notes_dict = response.json()
        notes = {
            "notes": notes_dict
        }
        notes["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        all_notes.append(notes)

    print("Notes extracted and took:", time.time() - start_time, "seconds")
    
    tables = sections.get("table", [])

    start_time = time.time()
    for table in tables:
        id = str(uuid.uuid4())
        x1, y1, x2, y2 = table
        x1_off = max(0, x1 - table_offset)
        y1_off = max(0, y1 - table_offset)
        x2_off = min(width, x2 + table_offset)
        y2_off = min(height, y2 + table_offset)
        crop = img.crop((x1_off, y1_off, x2_off, y2_off))

        table_imgs_paths.append(table_img_path)
        table_bboxes.append(table)

        save_image(path=table_img_path, image=crop)
    
    tables = get_tables(table_imgs_paths, table_bboxes)
    print("Tables extracted and took:", time.time() - start_time, "seconds")

    start_time = time.time()
    nozzles = process_nozzle_image(image_path)
    print("Nozzles extracted and took:", time.time() - start_time, "seconds")

    total_time = time.time() - initial_time

    print(f"Total analysis time: {total_time:.2f} seconds")

    result =  {
        "time": f"{total_time:.2f} Seconds",
        "page":1,
        "tables": tables, # T1
        "nozzles": nozzles, # T2
        "notes": all_notes, # T3
    }

    annotate_image(img, result)  # Annotate the image with results

    return result