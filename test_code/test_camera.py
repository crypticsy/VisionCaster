from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from time import sleep

def take_picture(filename="photo.jpg"):
    # Create a Picamera2 object
    picam2 = Picamera2()
    
    # Configure the camera
    picam2.configure(picam2.create_preview_configuration(main={"size": (1920, 1080)}))
    
    # Start the camera
    picam2.start()
    
    # Allow some time for the camera to adjust settings
    sleep(2)  # Sleep for 2 seconds
    
    # Capture the image
    picam2.capture_file(filename)
    
    # Stop the camera
    picam2.stop()
    
    print(f"Picture saved as {filename}")

# Use the function to take a picture
take_picture()
