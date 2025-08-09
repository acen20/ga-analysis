from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils import load_model
from inference import infer as donut_infer
import uvicorn
from PIL import Image
import io

app = FastAPI(title="DONUT Inference API")

# Load model at startup
donut_model = load_model("model")

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    try:
        # Read image into PIL
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference
        result = donut_infer(donut_model, image)

        # Return result as JSON
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

