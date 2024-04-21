import os
import pygame
import warnings

from PIL import Image
import transformers
from gtts import gTTS

warnings.filterwarnings("ignore")

# Set up directories and image paths
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "photo.png")

# Load processor and model
processor = transformers.BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Open and process the image
with Image.open(image_path).convert('RGB') as raw_image:
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

# Convert text to speech
tts = gTTS(text=caption, lang='en')
tts.save("caption.mp3")

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("caption.mp3")
pygame.mixer.music.play()

# Wait for the audio to finish playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)


# Delete the audio file
os.remove("caption.mp3")

