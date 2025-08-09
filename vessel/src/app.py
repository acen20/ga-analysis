from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import os
from pdf2image import convert_from_path
from engine import analyze_vessel_image

app = FastAPI()

@app.post("/detect")
async def detect(pdf_file: UploadFile = File(...)):
    # Save uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(await pdf_file.read())
        tmp_pdf_path = tmp_pdf.name

    # Convert first page to image at 300 DPI
    images = convert_from_path(tmp_pdf_path, dpi=300)

    # Save as PNG
    output_path = tmp_pdf_path.replace(".pdf", ".png")
    images[0].save(output_path, "PNG")

    # Clean up PDF file if needed
    os.remove(tmp_pdf_path)

    output = analyze_vessel_image(output_path)

    os.remove(output_path)  # Clean up image file

    return JSONResponse(output, status_code=200)