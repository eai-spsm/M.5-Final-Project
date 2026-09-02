import RPi.GPIO as GPIO
from time import sleep

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
ENA = 13 # L 
ENB = 12 # R

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


