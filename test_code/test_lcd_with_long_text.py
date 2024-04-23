import board
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd
from time import sleep
from datetime import datetime

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

def DisplayScrollingLeft(txt):
    """
    Displays a message on the LCD with a left scrolling effect.
    """
    txt = txt.strip() + ' '  # Ensure there's a space after the text
    n = lcd_columns
    
    while True:
        lcd.clear()  # Clear the display to update the scrolling text
        sleep(0.1)
        lcd.message = txt[:lcd_columns]  # Show the first part of the text on the LCD
        sleep(0.2)  # Delay to control the speed of the scroll
        txt = txt[1:] + txt[0]  # Rotate text to the left
        
        if n >= len(txt): 
            break
        n += 1
        
    lcd.clear()

# Example usage:
message = "Hello World! How are you doing these days? I hope you are doing well. I am doing well too. I am just testing this code to see if it works. I hope it does."
DisplayScrollingLeft(message)
