import RPi.GPIO as GPIO
import time

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder

BUTTON_PIN = 16
SHORT_PRESS_TIME = 0.5  # 500 milliseconds
DEBOUNCE_TIME = 0.1  # 100 milliseconds

# Variables will change:
prev_button_state = GPIO.LOW  # the previous state from the input pin
button_state = None  # the current reading from the input pin
press_time_start = 0
press_time_end = 0

# Set up GPIO and configure the pull-up resistor
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Create a Picamera2 object
picam2 = Picamera2()

# Configure the camera
picam2.configure(picam2.create_preview_configuration(main={"size": (1920, 1080)}))


def take_picture(filename="photo.png"):
    # Start the camera
    picam2.start()
    
    # Allow some time for the camera to adjust settings
    time.sleep(2)  # Sleep for 2 seconds
    
    # Capture the image
    picam2.capture_file(filename)
    
    # Stop the camera
    picam2.stop()
    
    print(f"Picture saved as {filename}")

try:
    while True:
        # Read the state of the switch/button
        button_state = GPIO.input(BUTTON_PIN)

        # Perform debounce by waiting for DEBOUNCE_TIME
        time.sleep(DEBOUNCE_TIME)

        if prev_button_state == GPIO.HIGH and button_state == GPIO.LOW:  # Button is pressed
            press_time_start = time.time()
        elif prev_button_state == GPIO.LOW and button_state == GPIO.HIGH:  # Button is released
            press_time_end = time.time()

            press_duration = press_time_end - press_time_start

            if press_duration < SHORT_PRESS_TIME:
                print("Smile for the camera!")
                
                # Use the function to take a picture
                take_picture()

        # Save the last state
        prev_button_state = button_state

except KeyboardInterrupt:
    print("\nExiting the program.")
    GPIO.cleanup()




# Incase of redoing the virtual environment
# https://forums.raspberrypi.com/viewtopic.php?t=361758