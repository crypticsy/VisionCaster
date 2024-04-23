# Standard library imports
import os
import time
import json
from datetime import datetime
from threading import Thread
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



def play_sound(musicName):
    """Plays a sound file from the sounds dictionary."""
    #  Play the audio file
    pygame.mixer.init()
    pygame.mixer.music.load(sounds.get(musicName))
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)        # Wait for the audio to finish playing



def convert_text_to_speech(speech_text):
    """Converts text to speech and plays it back."""
    tts = gTTS(text=speech_text, lang='en')
    tts.save("speech.mp3")
    
    #  Play the audio file
    pygame.mixer.init()
    pygame.mixer.music.load("speech.mp3")
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)        # Wait for the audio to finish playing
    
    # Delete the audio file
    os.remove("speech.mp3")



def display_message(txt, sleep_time=0):
    """Displays a message on the LCD with a left scrolling effect."""
    lcd.clear()
    txt = txt.strip() + ' '                 # Ensure there's a space after the text
    n = lcd_columns
    
    while True:
        lcd.clear()                         # Clear the display to update the scrolling text
        sleep(0.1)
        lcd.message = txt[:lcd_columns]     # Show the first part of the text on the LCD
        sleep(0.2)                          # Delay to control the speed of the scroll
        txt = txt[1:] + txt[0]              # Rotate text to the left
        
        if n >= len(txt): 
            break
        n += 1
    
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



def process_two_functions_with_threading(func1, args1, func2, args2):
    """Process two functions concurrently."""
    thread1 = Thread(target=func1, args=args1)
    thread2 = Thread(target=func2, args=args2)
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()



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
        
        # Check if the duration of a button press is short
        if press_duration < SHORT_PRESS_TIME:
            # Record the current time when the button press was registered
            current_time = datetime.now()
            # Construct a filename for saving the photo with a timestamp
            filename = os.path.join("data", f"photo_{current_time.strftime('%Y%m%d_%H%M%S')}.png")
            
            # Capture an image using the constructed filename
            capture_image(filename=filename)
            # Simultaneously display a message on the LCD and play a sound
            process_two_functions_with_threading(display_message, ("Smile for the camera!"), play_sound, ("camera"))
            
            # Display a processing message and convert the displayed text to speech concurrently
            process_two_functions_with_threading(convert_text_to_speech, ("Processing image..."), display_message, ("Processing image..."))
            # Analyze the captured image and retrieve a caption
            caption = analyse_image(filename=filename)
            # Log this interaction for future reference or analysis
            save_user_interaction(current_time, caption, filename)
            
            # Display the image caption and play a sound indicating the end of the process
            process_two_functions_with_threading(display_message, (caption), play_sound, ("camera"))
            
            # Clear any previous messages from the LCD
            lcd.clear()
            # Prepare the system for the next interaction by indicating readiness
            process_two_functions_with_threading(display_message, ("Ready..."), play_sound, ("start"))
    
    prev_button_state = button_state



if __name__ == "__main__":
    try:
        lcd.clear()
        process_two_functions_with_threading(display_message, ("Ready..."), play_sound, ("start"))
        while True:
            main()
        
    except KeyboardInterrupt:
        GPIO.cleanup()
        display_message("Exiting...", 5)