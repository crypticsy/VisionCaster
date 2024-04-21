# Standard library imports
import os
import time
import json
from datetime import datetime
from subprocess import PIPE

# External library imports for Raspberry Pi hardware control
import board
import digitalio
import RPi.GPIO as GPIO
import adafruit_character_lcd.character_lcd as characterlcd

# Imports for camera and image processing
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from PIL import Image

# Imports for sound and voice synthesis
from gtts import gTTS
import pygame

# Imports for machine learning and model processing
from transformers import BlipProcessor, BlipForConditionalGeneration

# Module for handling warnings
import warnings

# Additional helper functions from time module already imported
from time import sleep



# Suppress less critical warnings during runtime
warnings.filterwarnings("ignore")




# Constants for GPIO
BUTTON_PIN = 16
SHORT_PRESS_TIME = 0.5  # Duration for identifying a short press in seconds (500 milliseconds)
DEBOUNCE_TIME = 0.1     # Time to ignore further changes to avoid bouncing in seconds (100 milliseconds)

# Setup base directory for file paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# Sound files for camera and system sounds
sounds = dict(
    start = os.path.join(base_dir, "sounds", "pi-start.mp3"),
    camera = os.path.join(base_dir, "sounds", "camera-shutter.mp3"),
)

# GPIO setup for button input
GPIO.setmode(GPIO.BCM)  # Set GPIO pin numbering
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Configure button pin with pull-up resistor

# Create and configure the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1920, 1080)}))  # Set camera resolution

# LCD screen setup parameters
lcd_columns = 16  # Number of columns in the LCD display
lcd_rows = 2      # Number of rows in the LCD display

# Pins setup for the LCD on Raspberry Pi
lcd_rs = digitalio.DigitalInOut(board.D25)
lcd_en = digitalio.DigitalInOut(board.D24)
lcd_d4 = digitalio.DigitalInOut(board.D23)
lcd_d5 = digitalio.DigitalInOut(board.D17)
lcd_d6 = digitalio.DigitalInOut(board.D18)
lcd_d7 = digitalio.DigitalInOut(board.D22)

# Initialize the LCD display
lcd = characterlcd.Character_LCD_Mono(
    lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows
)

# Variables to track the state of the button
prev_button_state = GPIO.LOW  # Previous state from the input pin
button_state = None           # Current reading from the input pin
press_time_start = 0          # Start time of a button press
press_time_end = 0            # End time of a button press






def capture_image(filename):
    """Captures an image from the connected camera and saves it as a file."""
    # Start the camera
    picam2.start()
    
    # Allow some time for the camera to adjust settings
    time.sleep(1)  # Sleep for 1 second
    
    # Capture the image
    picam2.capture_file(filename)
    
    # Stop the camera
    picam2.stop()



def analyse_image(filename):
    """Processes an image file to generate a caption using a pre-trained model."""
    # Load pre-trained models
    llm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    llm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    try:
        # Open and process the image
        with Image.open(os.path.join(base_dir, filename)).convert('RGB') as raw_image:
            inputs = llm_processor(raw_image, return_tensors="pt")
            outputs = llm_model.generate(**inputs)
            caption = llm_processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error processing the image."



def convert_text_to_speech(speech_text, musicName=None):
    if musicName:
        #  Play the audio file
        pygame.mixer.init()
        pygame.mixer.music.load(sounds.get(musicName))
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10) # Wait for the audio to finish playing
            
    else:
        """Converts text to speech and plays it back."""
        tts = gTTS(text=speech_text, lang='en')
        tts.save("speech.mp3")
        
        #  Play the audio file
        pygame.mixer.init()
        pygame.mixer.music.load("speech.mp3")
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10) # Wait for the audio to finish playing
        
        # Delete the audio file
        os.remove("speech.mp3")



def display_message(text, sleep_time=0):
    """Displays text on an LCD and clears it after a specified duration."""
    lcd.clear()
    lcd.message = text
    
    if sleep_time > 0:
        sleep(sleep_time)
        lcd.clear()



def save_user_interaction(current_time, caption, filename):
    with open(os.path.join(base_dir, "data", "history.json"), "r") as file:
        data = json.load(file)
    
    # Append the new data to the existing list
    data.append(dict(
        createdAt = current_time.isoformat(),
        caption = caption,
        filename = filename
    ))
    
    # Save the updated data to the file
    json.dump(data, open(os.path.join(base_dir, "data", "history.json"), "w"))




def main():
    """Main function to handle button press logic and process image."""
    global prev_button_state, press_time_start, press_time_end
    
    # Read the state of the switch/button
    button_state = GPIO.input(BUTTON_PIN)
    time.sleep(DEBOUNCE_TIME)   # Sleep to debounce the button

    # Detect button press
    if prev_button_state == GPIO.HIGH and button_state == GPIO.LOW:  # Button is pressed
        press_time_start = time.time()
    elif prev_button_state == GPIO.LOW and button_state == GPIO.HIGH:  # Button is released
        press_time_end = time.time()
        press_duration = press_time_end - press_time_start

        if press_duration < SHORT_PRESS_TIME:
            current_time = datetime.now()
            filename = os.path.join( "data", f"photo_{current_time.strftime('%Y%m%d_%H%M%S')}.png")
            
            display_message("Smile for the camera!")
            capture_image(filename=filename)
            convert_text_to_speech(None, "camera")
            
            convert_text_to_speech("Processing image...")
            display_message("Processing image...")
            caption = analyse_image(filename=filename)
            save_user_interaction(current_time, caption, filename)
            
            display_message(caption)
            convert_text_to_speech(caption)
            lcd.clear()

    prev_button_state = button_state



if __name__ == "__main__":
    try:
        lcd.clear()
        convert_text_to_speech(None, "start")
        display_message("Ready...")
        
        while True:
            main()
    
    except KeyboardInterrupt:
        GPIO.cleanup()
        display_message("Exiting...", 5)