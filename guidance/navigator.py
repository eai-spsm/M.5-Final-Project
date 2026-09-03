import math
import time


class Navigator:
    """Dead-reckoning position/heading estimate.

    No encoders on this robot, so this is open-loop: it just integrates
    "how fast we told the motors to go, for how long" into an (x, y)
    position in cm and a heading in degrees. It will drift from the real
    position over time (wheel slip, uneven floor, imprecise speed
    constants) - good enough for "roughly where am I / which way am I
    facing", not for anything that needs to be exact. Re-zero it (new
    Navigator, or call reset()) whenever you have a known reference point
    (e.g. touching a wall, or a vision-based fix from perception/).

    Coordinate system: starts at (0, 0) facing heading 0. Heading is in
    degrees, compass-style: 0 = the direction the robot started facing
    ("north" / forward), increasing clockwise (90 = right of start, 180 =
    behind start, 270 = left of start). X = distance to the right of the
    start heading, Y = distance ahead of the start heading.
    """

    def __init__(self, x=0.0, y=0.0, heading_deg=0.0):
        self.x = x
        self.y = y
        self.heading_deg = heading_deg % 360
        self._vx = 0.0       # current commanded body-frame sideways speed (cm/s, + = right)
        self._vy = 0.0       # current commanded body-frame forward speed (cm/s, + = forward)
        self._omega = 0.0    # current commanded turn rate (deg/s, + = clockwise)
        self._last_t = time.time()

    def reset(self, x=0.0, y=0.0, heading_deg=0.0):
        self.x = x
        self.y = y
        self.heading_deg = heading_deg % 360
        self._vx = self._vy = self._omega = 0.0
        self._last_t = time.time()

    def _integrate(self):
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0:
            return

        heading_rad = math.radians(self.heading_deg)
        # Rotate body-frame (right, forward) velocity into world-frame (x, y)
        world_dx = self._vx * math.cos(heading_rad) + self._vy * math.sin(heading_rad)
        world_dy = -self._vx * math.sin(heading_rad) + self._vy * math.cos(heading_rad)

        self.x += world_dx * dt
        self.y += world_dy * dt
        self.heading_deg = (self.heading_deg + self._omega * dt) % 360

    def set_velocity(self, vx=0.0, vy=0.0, omega=0.0):
        # Flush movement under the OLD velocity before switching to the new one
        self._integrate()
        self._vx, self._vy, self._omega = vx, vy, omega

    def pose(self):
        # Returns (x_cm, y_cm, heading_deg) as of right now
        self._integrate()
        return (self.x, self.y, self.heading_deg)

    def __str__(self):
        x, y, heading = self.pose()
        return f"x={x:.1f}cm y={y:.1f}cm heading={heading:.0f}°"
