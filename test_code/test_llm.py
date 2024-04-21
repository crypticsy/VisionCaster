import os
import warnings

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "photo.png")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

with open(image_path, "rb") as f:
    img_url = f.read()
    raw_image = Image.open(f).convert('RGB')



inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)

print(processor.decode(out[0], skip_special_tokens=True))