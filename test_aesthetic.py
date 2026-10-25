import torch
from PIL import Image
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

print("Loading model. The first run requires a download...")

model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model = model.to(device="cpu", dtype=torch.float32)
model.eval()

image = Image.open("input/person.jpg").convert("RGB")
pixel_values = preprocessor(
    images=image,
    return_tensors="pt",
).pixel_values.to(device="cpu", dtype=torch.float32)

with torch.inference_mode():
    score = model(pixel_values).logits.squeeze().item()

print(f"Aesthetic score: {score:.2f}")