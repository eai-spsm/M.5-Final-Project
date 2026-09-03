import time

from movement import MecanumDrive
from movement.movement import DEFAULT_SPEED, DEFAULT_STRAFE_SPEED

from .navigator import Navigator

# PLACEHOLDER speed constants - measure these on the real robot and update.
# These are calibrated for DEFAULT_SPEED / DEFAULT_STRAFE_SPEED; if the
# actual commanded duty cycle differs (a custom speed= per call, or
# MecanumDrive.speed/strafe_speed adjusted at runtime), GuidedDrive scales
# these proportionally so the tracked position stays consistent instead of
# silently assuming the original default speed was used.
# How to calibrate: drive forward for a fixed time (e.g. 3s) at the default
# speed, measure the distance traveled with a tape measure, divide by time
# for LINEAR_SPEED_CM_S. Same idea for STRAFE_SPEED_CM_S (strafe instead of
# forward) and ROTATE_SPEED_DEG_S (rotate for a fixed time, measure the
# angle turned with a protractor/known angle mark instead of distance).
LINEAR_SPEED_CM_S = 20.0
STRAFE_SPEED_CM_S = 15.0
ROTATE_SPEED_DEG_S = 90.0

MIN_SPEED = 20   # below this, motors risk stalling (see docs/CALIBRATION_REPORT.md)
MAX_SPEED = 100


class GuidedDrive:
    """MecanumDrive + Navigator: driving the robot also tracks its
    estimated position/heading, starting at (0, 0) facing heading 0.
    """

    def __init__(self):
        self.drive = MecanumDrive()
        self.nav = Navigator()

    def forward(self, speed=None):
        actual = speed if speed is not None else self.drive.speed
        self.drive.forward(speed=speed)
        self.nav.set_velocity(vy=LINEAR_SPEED_CM_S * actual / DEFAULT_SPEED)

    def backward(self, speed=None):
        actual = speed if speed is not None else self.drive.speed
        self.drive.backward(speed=speed)
        self.nav.set_velocity(vy=-LINEAR_SPEED_CM_S * actual / DEFAULT_SPEED)

    def strafe_left(self, speed=None):
        actual = speed if speed is not None else self.drive.strafe_speed
        self.drive.strafe_left(speed=speed)
        self.nav.set_velocity(vx=-STRAFE_SPEED_CM_S * actual / DEFAULT_STRAFE_SPEED)

    def strafe_right(self, speed=None):
        actual = speed if speed is not None else self.drive.strafe_speed
        self.drive.strafe_right(speed=speed)
        self.nav.set_velocity(vx=STRAFE_SPEED_CM_S * actual / DEFAULT_STRAFE_SPEED)

    def rotate_left(self, speed=None):
        actual = speed if speed is not None else self.drive.speed
        self.drive.rotate_left(speed=speed)
        self.nav.set_velocity(omega=-ROTATE_SPEED_DEG_S * actual / DEFAULT_SPEED)

    def rotate_right(self, speed=None):
        actual = speed if speed is not None else self.drive.speed
        self.drive.rotate_right(speed=speed)
        self.nav.set_velocity(omega=ROTATE_SPEED_DEG_S * actual / DEFAULT_SPEED)

    def stop(self):
        self.drive.stop()
        self.nav.set_velocity(0, 0, 0)

    def rotate_to(self, target_heading_deg, speed=None, tolerance_deg=5, timeout=5.0):
        # Rotates toward target_heading_deg (0-360, clockwise, same
        # convention as Navigator) the short way, blocking until within
        # tolerance_deg or timeout seconds pass. Open-loop like everything
        # else here - relies on the same ROTATE_SPEED_DEG_S estimate, so it
        # can overshoot; the timeout is a safety net against never
        # converging, not a normal way for this to end.
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, _, heading = self.pose()
            diff = (target_heading_deg - heading + 180) % 360 - 180  # -180..180
            if abs(diff) <= tolerance_deg:
                break
            if diff > 0:
                self.rotate_right(speed=speed)
            else:
                self.rotate_left(speed=speed)
            time.sleep(0.05)
        self.stop()

    def test_wheel(self, name):
        self.drive.test_wheel(name)

    def pose(self):
        return self.nav.pose()

    def reset_position(self, x=0.0, y=0.0, heading_deg=0.0):
        # Re-zero the dead-reckoning estimate - use this at a known
        # reference point (e.g. touching a wall) since it drifts over time.
        self.nav.reset(x, y, heading_deg)

    def adjust_speed(self, delta):
        # Nudges both SPEED and STRAFE_SPEED by delta (%), clamped to
        # [MIN_SPEED, MAX_SPEED]. Returns the new (speed, strafe_speed).
        self.drive.speed = min(MAX_SPEED, max(MIN_SPEED, self.drive.speed + delta))
        self.drive.strafe_speed = min(MAX_SPEED, max(MIN_SPEED, self.drive.strafe_speed + delta))
        return self.drive.speed, self.drive.strafe_speed

    def cleanup(self):
        self.drive.cleanup()
