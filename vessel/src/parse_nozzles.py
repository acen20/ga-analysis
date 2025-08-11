import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from paddleocr import PaddleOCR
import json
import paddle

paddle.set_device("gpu:0" if paddle.is_compiled_with_cuda() else "cpu")

# Convert NumPy types for JSON serialization
def convert_np(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# Load YOLO models
view_model = YOLO("yolo/view/best.pt")
nozzle_model = YOLO("yolo/nozzle/best.pt")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # use lighter OCR models (optional) — speeds up inference
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    )

def paddle_ocr_text(image, type_="normal"):
    # if type_ == "normal":
    #     h, w = image.shape[:2]
    #     image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    result = ocr.predict(image)
    texts = result[0]['rec_texts']
    if type_ == "nozzle":
        if not len(texts) == 2 and texts:
            texts[-1] = '"'
        if len(texts) == 3:
            texts[1] = texts[1] + texts[2]
            del texts[2]
        elif len(texts) == 4:
            texts[0] = texts[0] + texts[1]
            texts[2] = texts[2] + texts[3]
            del texts[1]
            del texts[2:]

    return " ".join(texts)

def detect_views(image):
    results = view_model(image, imgsz=1536, verbose=False)[0]
    return results.boxes.xyxy.cpu().numpy().astype(int)

def detect_nozzles(view_crop):
    results = nozzle_model(view_crop, imgsz=1024, verbose=False, conf=0.25)[0]
    return results.boxes.xyxy.cpu().numpy().astype(int)

def classify_view(text):
    lower_text = text.lower()
    if "elevation" in lower_text:
        return "Elevation View"
    elif "a\"" in lower_text or "a”" in lower_text or 'a"' in lower_text:
        return "A"
    elif "orientation" in lower_text:
        return "Setting Bolt Orientation"
    else:
        return "Elevation View"

def extract_view_data(image, view_bbox):
    x1, y1, x2, y2 = view_bbox
    expanded = 30
    x1, y1 = max(x1 - expanded, 0), max(y1 - expanded, 0)
    x2, y2 = min(x2 + expanded, image.shape[1]), min(y2 + expanded, image.shape[0])
    view_crop = image[y1:y2, x1:x2]

    # OCR the whole view
    view_text = paddle_ocr_text(view_crop)
    view_type = classify_view(view_text)

    # Detect sub-elements (nozzles)
    nozzle_bboxes = detect_nozzles(view_crop)

    print(f"Detected {len(nozzle_bboxes)} nozzles in {view_type}")

    def process_nozzle(box):
        nx1, ny1, nx2, ny2 = map(int,box)
        nozzle_crop = view_crop[ny1:ny2, nx1:nx2]
        text = paddle_ocr_text(nozzle_crop, type_="nozzle")

        # Convert to full image coords
        fx1, fy1, fx2, fy2 = nx1 + x1, ny1 + y1, nx2 + x1, ny2 + y1
        w, h = fx2 - fx1, fy2 - fy1
        bbox_xywh = (int(fx1), int(fy1), int(w), int(h))

        return {
            "bbox": bbox_xywh,
            "view": view_type,
            "text": text
        }
    
    nozzle_data = [process_nozzle(box) for box in nozzle_bboxes]

    # with ThreadPoolExecutor() as executor:
    #     nozzle_data = list(executor.map(process_nozzle, nozzle_bboxes))

    return nozzle_data

def process_image(image_path):
    image = cv2.imread(str(image_path))
    view_bboxes = detect_views(image)

    print(len(view_bboxes), "views detected")

    all_nozzles = []
    for bbox in view_bboxes:
        view_data = extract_view_data(image, bbox)
        all_nozzles.append(view_data)
    all_nozzles = [item for sublist in all_nozzles for item in sublist]  # Flatten the list
    return all_nozzles

def annotate_image(image, nozzle_data):
    annotated = image.copy()

    for nozzle in nozzle_data:
        x, y, w, h = nozzle["bbox"]
        view_name = nozzle["view"]
        text_label = f"{view_name} | {nozzle['text']}"

        # Draw rectangle
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # green box
            2
        )

        # Draw label background for better readability
        (text_w, text_h), baseline = cv2.getTextSize(
            text_label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, 1
        )
        cv2.rectangle(
            annotated,
            (x, y - text_h - baseline),
            (x + text_w, y),
            (0, 255, 0),
            -1
        )

        # Put text on top of the box
        cv2.putText(
            annotated,
            text_label,
            (x, y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # black text
            1
        )

    return annotated