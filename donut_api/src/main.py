import inference
from donut.model import DonutModel
from PIL import Image
import torch
import json

def start_inference(pretrained_model:DonutModel, image: Image):
    result = inference.infer(pretrained_model, image=image)
    ## SAVE RESULT
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return result
