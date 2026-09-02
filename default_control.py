import RPi.GPIO as GPIO
import sys
import termios
import tty
import select
import time

# 1. Tell the Pi we are using BCM (GPIO labels), not physical board numbers
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Orientation: battery = North (front), Raspberry Pi = South (back)

# Board 1 - Left side
ENA_L = 12  # enables IN1/IN2
ENB_L = 18  # enables IN3/IN4
IN1 = 4
IN2 = 17
IN3 = 27
IN4 = 22

# Board 2 - Right side
ENA_R = 13  # enables IN5/IN6
ENB_R = 23  # enables IN7/IN8
IN5 = 5
IN6 = 6
IN7 = 19
IN8 = 26

# Ultrasonic (HC-SR04)
TRIG = 10
ECHO = 9

GPIO.setup(ENA_L, GPIO.OUT)
GPIO.setup(ENB_L, GPIO.OUT)
GPIO.setup(ENA_R, GPIO.OUT)
GPIO.setup(ENB_R, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(IN5, GPIO.OUT)
GPIO.setup(IN6, GPIO.OUT)
GPIO.setup(IN7, GPIO.OUT)
GPIO.setup(IN8, GPIO.OUT)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, GPIO.LOW)

pwm_left_a = GPIO.PWM(ENA_L, 1000)   # IN1/IN2
pwm_left_b = GPIO.PWM(ENB_L, 1000)   # IN3/IN4
pwm_right_a = GPIO.PWM(ENA_R, 1000)  # IN5/IN6
pwm_right_b = GPIO.PWM(ENB_R, 1000)  # IN7/IN8
pwm_left_a.start(0)
pwm_left_b.start(0)
pwm_right_a.start(0)
pwm_right_b.start(0)

SPEED = 60  # duty cycle %
OBSTACLE_CM = 15  # minimum clearance before forward drive is blocked


def get_distance():
    # Returns distance in cm, or None if the sensor timed out
    try:
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG, GPIO.LOW)

        timeout = time.time() + 0.04
        while GPIO.input(ECHO) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start > timeout:
                return None

        timeout = time.time() + 0.04
        while GPIO.input(ECHO) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end > timeout:
                return None

        return (pulse_end - pulse_start) * 34300 / 2
    except Exception as e:
        print(f"Ultrasonic Read Error: {e}")
        return None


def left_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)


def left_backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)


def left_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)


def right_forward():
    GPIO.output(IN5, GPIO.HIGH)
    GPIO.output(IN6, GPIO.LOW)
    GPIO.output(IN7, GPIO.HIGH)
    GPIO.output(IN8, GPIO.LOW)


def right_backward():
    GPIO.output(IN5, GPIO.LOW)
    GPIO.output(IN6, GPIO.HIGH)
    GPIO.output(IN7, GPIO.LOW)
    GPIO.output(IN8, GPIO.HIGH)


def right_stop():
    GPIO.output(IN5, GPIO.LOW)
    GPIO.output(IN6, GPIO.LOW)
    GPIO.output(IN7, GPIO.LOW)
    GPIO.output(IN8, GPIO.LOW)


def stop_all():
    left_stop()
    right_stop()
    pwm_left_a.ChangeDutyCycle(0)
    pwm_left_b.ChangeDutyCycle(0)
    pwm_right_a.ChangeDutyCycle(0)
    pwm_right_b.ChangeDutyCycle(0)


def set_left_speed(duty):
    pwm_left_a.ChangeDutyCycle(duty)
    pwm_left_b.ChangeDutyCycle(duty)


def set_right_speed(duty):
    pwm_right_a.ChangeDutyCycle(duty)
    pwm_right_b.ChangeDutyCycle(duty)


def drive_forward():
    distance = get_distance()
    if distance is not None and distance < OBSTACLE_CM:
        print(f"Obstacle at {distance:.1f}cm - forward blocked")
        stop_all()
        return
    left_forward()
    right_forward()
    set_left_speed(SPEED)
    set_right_speed(SPEED)


def drive_backward():
    left_backward()
    right_backward()
    set_left_speed(SPEED)
    set_right_speed(SPEED)


def turn_left():
    left_backward()
    right_forward()
    set_left_speed(SPEED)
    set_right_speed(SPEED)


def turn_right():
    left_forward()
    right_backward()
    set_left_speed(SPEED)
    set_right_speed(SPEED)


def get_key(timeout=0.1):
    # Non-blocking single keypress read from the terminal (no Enter needed)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    print("WASD to drive, SPACE to stop, Q to quit.")
    try:
        while True:
            key = get_key(0.1)
            if key is None:
                continue
            key = key.lower()
            if key == "w":
                print("Forward")
                drive_forward()
            elif key == "s":
                print("Backward")
                drive_backward()
            elif key == "a":
                print("Left")
                turn_left()
            elif key == "d":
                print("Right")
                turn_right()
            elif key == " ":
                print("Stop")
                stop_all()
            elif key == "q":
                print("Quitting...")
                break
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        print("Cleaning up GPIO resources...")
        stop_all()
        pwm_left_a.stop()
        pwm_left_b.stop()
        pwm_right_a.stop()
        pwm_right_b.stop()
        GPIO.cleanup()
        print("Done!")


if __name__ == "__main__":
    main()
