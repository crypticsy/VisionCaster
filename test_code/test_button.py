import RPi.GPIO as GPIO
import time

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
                print("A short press is detected")

        # Save the last state
        prev_button_state = button_state

except KeyboardInterrupt:
    print("\nExiting the program.")
    GPIO.cleanup()
