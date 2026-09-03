import RPi.GPIO as GPIO
import time

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

DEFAULT_SPEED = 60  # duty cycle %
# Strafing runs each board's two channels in opposite directions at once,
# which on cheap L298N-style boards can couple electrical noise between
# channels and glitch the H-bridge. Lower speed = lower current/back-EMF.
DEFAULT_STRAFE_SPEED = 55  # duty cycle %
OBSTACLE_CM = 15  # minimum clearance before forward drive is blocked

# If a wheel drives backward when told to go forward, it's physically wired
# backward. Use MecanumDrive.test_wheel("fl"/"fr"/"rl"/"rr") to find which
# one, then flip its flag here.
WHEEL_INVERT = {"fl": False, "fr": False, "rl": False, "rr": True}

# If strafing drifts at an angle or spins instead of going straight sideways,
# the wheels aren't spinning at exactly matched speeds even at the same duty
# cycle - lower the trim on whichever wheel is "winning" until it cancels
# out. Keep effective duty (speed * trim) within roughly 45-60: lower stalls
# some motors, higher tends to reintroduce the same-board electrical glitch.
WHEEL_TRIM = {"fl": 0.9, "fr": 0.9, "rl": 1.0, "rr": 1.0}

# Each wheel is driven independently so we can strafe/rotate, not just go
# left/right. FL = IN1/IN2 (ENA_L), RL = IN3/IN4 (ENB_L), FR = IN7/IN8
# (ENB_R), RR = IN5/IN6 (ENA_R). Left and right boards each have their own,
# DIFFERENT front/rear channel order (confirmed with test_wheel() - do not
# assume they match; re-check after any wiring change).


class MecanumDrive:
    def __init__(self, speed=DEFAULT_SPEED, strafe_speed=DEFAULT_STRAFE_SPEED):
        self.speed = speed
        self.strafe_speed = strafe_speed

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

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

        self._pwm = {
            "fl": GPIO.PWM(ENA_L, 1000),
            "rl": GPIO.PWM(ENB_L, 1000),
            "rr": GPIO.PWM(ENA_R, 1000),
            "fr": GPIO.PWM(ENB_R, 1000),
        }
        for pwm in self._pwm.values():
            pwm.start(0)

        self._pins = {
            "fl": (IN1, IN2),
            "fr": (IN7, IN8),
            "rl": (IN3, IN4),
            "rr": (IN5, IN6),
        }

    # -- low-level -----------------------------------------------------

    def _wheel(self, pin_a, pin_b, forward):
        GPIO.output(pin_a, GPIO.LOW if forward else GPIO.HIGH)
        GPIO.output(pin_b, GPIO.HIGH if forward else GPIO.LOW)

    def _wheel_stop(self, pin_a, pin_b):
        GPIO.output(pin_a, GPIO.LOW)
        GPIO.output(pin_b, GPIO.LOW)

    def set_wheels(self, fl, fr, rl, rr, speed=None):
        # Each arg is 1 (forward), -1 (backward), or 0 (stop).
        # speed: custom duty cycle % for this call (defaults to self.speed).
        if speed is None:
            speed = self.speed
        wheels = {"fl": fl, "fr": fr, "rl": rl, "rr": rr}

        for name, direction in wheels.items():
            pin_a, pin_b = self._pins[name]
            if direction == 0:
                self._wheel_stop(pin_a, pin_b)
                self._pwm[name].ChangeDutyCycle(0)
            else:
                if WHEEL_INVERT[name]:
                    direction = -direction
                self._wheel(pin_a, pin_b, forward=direction > 0)
                duty = min(100, max(0, speed * WHEEL_TRIM[name]))
                self._pwm[name].ChangeDutyCycle(duty)

    def test_wheel(self, name):
        # Spins one wheel forward in isolation, for calibrating WHEEL_INVERT
        directions = {"fl": 0, "fr": 0, "rl": 0, "rr": 0}
        directions[name] = 1
        self.set_wheels(**directions)

    # -- movement (all accept an optional custom speed) -----------------

    def stop(self):
        self.set_wheels(0, 0, 0, 0)

    def forward(self, speed=None):
        # Ultrasonic obstacle check disabled for now - see get_distance()
        # if you want to re-enable a distance gate here.
        self.set_wheels(1, 1, 1, 1, speed=speed)

    def backward(self, speed=None):
        self.set_wheels(-1, -1, -1, -1, speed=speed)

    def strafe_left(self, speed=None):
        self.set_wheels(-1, 1, 1, -1, speed=speed or self.strafe_speed)

    def strafe_right(self, speed=None):
        self.set_wheels(1, -1, -1, 1, speed=speed or self.strafe_speed)

    def rotate_left(self, speed=None):
        self.set_wheels(-1, 1, -1, 1, speed=speed)

    def rotate_right(self, speed=None):
        self.set_wheels(1, -1, 1, -1, speed=speed)

    # -- ultrasonic ------------------------------------------------------

    def get_distance(self):
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

    # -- cleanup -----------------------------------------------------------

    def cleanup(self):
        self.stop()
        for pwm in self._pwm.values():
            pwm.stop()
        GPIO.cleanup()
