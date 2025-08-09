import torch
def infer(pretrained_model, image, debug=True):
    with torch.no_grad():
        output = pretrained_model.inference(image=image, prompt=f"<s_custom>")["predictions"][0]
    return output