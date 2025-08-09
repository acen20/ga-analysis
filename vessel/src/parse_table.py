import os
import json
import re
from PIL import Image
from paddlex import create_pipeline
import paddle
from bs4 import BeautifulSoup
import json
import random
from paddleocr import PPStructureV3

paddle.set_device("gpu:0" if paddle.is_compiled_with_cuda() else "cpu")


pipeline = PPStructureV3(
    use_doc_orientation_classify=False,  # Disable document orientation classification
    use_doc_unwarping=False,             # Disable document unwarping
    use_textline_orientation=False,      # Disable textline orientation classification
    use_formula_recognition=False,       # Disable formula recognition
    use_seal_recognition=False,          # Disable seal text recognition
    use_chart_recognition=False,         # Disable chart parsing
    use_table_recognition=True,          # Enable table recognition
    text_detection_model_name="PP-OCRv5_mobile_det",  # Use the server-side text detection model
    text_recognition_model_name="PP-OCRv5_mobile_rec"  # Use the server-side text recognition model
)


def ppstructure_parse(image_path):
    """Run PPStructureV3 on an image and return parsed JSON."""
    output = pipeline.predict(
        input=image_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        device='cpu'
    )
    if not output:
        return None

    parsed_results = []
    for res in output:
        parsed_results.append(res)

    return parsed_results

def parse_table(table_image_path):
    parsed = ppstructure_parse(table_image_path)

    if not parsed:
        return {"name":"", "rows": []}

    # Table mode
    table = parsed[0]['table_res_list'][0]['pred_html']

    soup = BeautifulSoup(table, "html.parser")
    # Find all rows
    rows = soup.find_all("tr")

    header_text = None
    max_colspan = 0
    header_row_index = None

    # Detect header row (widest colspan)
    for i, tr in enumerate(rows):
        for td in tr.find_all(["td", "th"]):
            colspan = int(td.get("colspan", 1))
            if colspan > max_colspan:
                max_colspan = colspan
                header_text = td.get_text(strip=True)
                header_row_index = i

    # Remove header row from rows list
    if header_row_index is not None:
        rows.pop(header_row_index)

    # Convert table rows into list of lists
    data_rows = []
    for tr in rows:
        row_data = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        # Skip completely empty rows
        if any(cell != "" for cell in row_data):
            data_rows.append(row_data)

    max_search_header = 2
    while not header_text or len(header_text)<=1:
        header_text = " ".join(data_rows[0]).strip()
        del data_rows[0]
        max_search_header -= 1
        if not max_search_header:
            header_text = ""
            break
    # Build JSON
    output = {
        "name": header_text,
        "rows": data_rows
    }
    return output
