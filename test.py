from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import torch

device = "cpu"

processor = TrOCRProcessor.from_pretrained("qantev/trocr-base-spanish")
model = VisionEncoderDecoderModel.from_pretrained(
    "qantev/trocr-base-spanish"
)

model.to(device)
model.eval()

img = Image.new("RGB", (384, 64), "white")

pixels = processor(
    images=img,
    return_tensors="pt"
).pixel_values.to(device)

with torch.inference_mode():
    ids = model.generate(
        pixels,
        max_new_tokens=50
    )

print("OK")