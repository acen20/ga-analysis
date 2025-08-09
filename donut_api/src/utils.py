from donut.model import DonutModel
import torch

def load_model(model_name:str):

    pretrained_model = DonutModel.from_pretrained(model_name)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    return pretrained_model