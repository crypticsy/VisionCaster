import os
import board
import pygame
import digitalio
from gtts import gTTS
from time import sleep
from threading import Thread
import adafruit_character_lcd.character_lcd as characterlcd

# Modify this if you have a different sized character LCD
lcd_columns = 16
lcd_rows = 2

# compatible with all versions of RPI as of Jan. 2019
# v1 - v3B+
lcd_rs = digitalio.DigitalInOut(board.D25)
lcd_en = digitalio.DigitalInOut(board.D24)
lcd_d4 = digitalio.DigitalInOut(board.D23)
lcd_d5 = digitalio.DigitalInOut(board.D17)
lcd_d6 = digitalio.DigitalInOut(board.D18)
lcd_d7 = digitalio.DigitalInOut(board.D22)

# Initialise the lcd class
lcd = characterlcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows)

def display_message():
    # wipe LCD screen before we start
    lcd.clear()
    
    # Display "Hello World" message
    lcd.message = "Hello World"  # This will display "Hello World" on the first line of a 16x2 LCD.
    sleep(2)
    
    # wipe LCD screen before we start
    lcd.clear()

def play_sound():
    caption = "Hello World"
    
    # Convert text to speech
    tts = gTTS(text=caption, lang='en')
    tts.save("testing.mp3")
    
    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load("testing.mp3")
    pygame.mixer.music.play()
    
    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Delete the audio file
    os.remove("testing.mp3")

# Create threads
thread_lcd = Thread(target=display_message)
thread_sound = Thread(target=play_sound)

# Start threads
thread_lcd.start()
thread_sound.start()

# Wait for all threads to complete
thread_lcd.join()
thread_sound.join()
