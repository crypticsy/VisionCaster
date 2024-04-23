import os
import time
import warnings
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.abspath(__file__))
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

images = [
    "photo_rasp.png",
    # "photo2.png",
    # "photo3.png"
]

total_time = 0  # Initialize a variable to keep track of the total time

for n, image in enumerate(images):
    start_time = time.time()  # Record the start time

    image_path = os.path.join(base_dir, image)

    # Open the image file and process it
    with open(image_path, "rb") as f:
        raw_image = Image.open(f).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)

    # Calculate the time taken to process the image
    elapsed_time = time.time() - start_time
    total_time += elapsed_time  # Add the elapsed time to the total time

    # Output the caption and the time taken for processing this image
    print(f"\nImage {n+1}:")
    print(processor.decode(out[0], skip_special_tokens=True))
    print(f"Processing time: {elapsed_time:.2f} seconds")

# Calculate the average processing time
avg_time = total_time / len(images)
print(f"\nAverage processing time: {avg_time:.2f} seconds")
