import os
import warnings

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.abspath(__file__))
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

images = [
    "photo.png",
    "photo2.png",
    "photo3.png"
]

for n, image in enumerate(images):
    image_path = os.path.join(base_dir, image)

    with open(image_path, "rb") as f:
        img_url = f.read()
        raw_image = Image.open(f).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)

    print(f"\nImage {n+1}:")
    print(processor.decode(out[0], skip_special_tokens=True))