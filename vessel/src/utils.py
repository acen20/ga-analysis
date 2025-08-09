import cv2
import json
import numpy as np

def annotate_image(img, data):
    """Annotate image with bounding boxes from JSON data."""

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ---- Annotate Tables ----
    for table in data.get("tables", []):
        x, y, w, h = table["bbox"]  # assuming this is xywh
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
        label = table["name"]
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Draw background rectangle (slightly bigger than text)
        cv2.rectangle(img, (x, y - text_height - baseline - 8), (x + text_width, y), (255, 0, 0), thickness=cv2.FILLED)

        # Draw text over rectangle
        cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ---- Annotate Nozzle ----
    if "nozzles" in data:
        for nozzle in data["nozzles"]:
            x, y, w, h = nozzle["bbox"]  # xywh
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
            label = f'{nozzle["text"]} | {nozzle["view"]}'

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw background rectangle (slightly bigger than text)
            cv2.rectangle(img, (x, y - text_height - baseline - 8), (x + text_width, y), (255, 0, 0), thickness=cv2.FILLED)

            # Draw text over rectangle
            cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ---- Annotate Notes ----
    # Here we assume each note dict has "bbox" key in xywh format
    for note in data.get("notes", []):
        if "bbox" in note:
            x, y, w, h = note["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red
            label = json.dumps(note['notes'], indent=2)  # Truncate for display
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)

            # Draw background rectangle (slightly bigger than text)
            cv2.rectangle(img, (x, y - text_height - baseline - 8), (x + text_width, y), (0, 0, 255), thickness=cv2.FILLED)

            # Draw text over rectangle
            cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save annotated image
    cv2.imwrite("annotated.png", img)