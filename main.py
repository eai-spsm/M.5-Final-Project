import RPi.GPIO as GPIO
from time import sleep, time

# 1. Tell the Pi we are using BCM (GPIO labels), not physical board numbers
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


# Battery is North 
#R
IN1 = 4
IN2 = 17
IN3 = 27
IN4 = 22

#L
IN5 = 5
IN6 = 6
IN7 = 19
IN8 = 26

#ENA
ENA = 13 # R
ENB = 12 # L

# Start button (pulled up, wired to GND when pressed)
BTN_PIN = 21

# Ultrasonic (HC-SR04)
TRIG = 10
ECHO = 9

GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(IN5, GPIO.OUT)
GPIO.setup(IN6, GPIO.OUT)
GPIO.setup(IN7, GPIO.OUT)
GPIO.setup(IN8, GPIO.OUT)
GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, GPIO.LOW)


def get_distance():
    # Returns distance in cm, or None if the sensor timed out
    try:
        GPIO.output(TRIG, GPIO.HIGH)
        sleep(0.00001)
        GPIO.output(TRIG, GPIO.LOW)

        timeout = time() + 0.04
        while GPIO.input(ECHO) == GPIO.LOW:
            pulse_start = time()
            if pulse_start > timeout:
                return None

        timeout = time() + 0.04
        while GPIO.input(ECHO) == GPIO.HIGH:
            pulse_end = time()
            if pulse_end > timeout:
                return None

        return (pulse_end - pulse_start) * 34300 / 2
    except Exception as e:
        print(f"Ultrasonic Read Error: {e}")
        return None


# Wait for the start button, then hand off to the rest of the program
try:
    print("Ready. Press the button to start...")
    while True:
        if GPIO.input(BTN_PIN) == GPIO.LOW:
            print("Button pressed - starting...")
            break
        sleep(0.05)

    # TODO: put your startup logic here (e.g. call into default_control.main())

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    print("Cleaning up GPIO resources...")
    GPIO.cleanup()
    print("Done!")
