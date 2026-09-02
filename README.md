# M.5 Final Project — Setup Guide

Raspberry Pi 4 robot: 4-motor drive (2x L298N-style drivers) + YOLO camera detection.

## Hardware

- Raspberry Pi 4
- 2x motor driver boards (L298N-style), one for each side
- 4x DC motors (2 left, 2 right)
- Push button (start button)
- HC-SR04 ultrasonic distance sensor
- USB webcam
- Battery pack

Orientation: battery = North (front), Raspberry Pi = South (back).

## Wiring (BCM pin numbers)

| Signal | BCM Pin | Notes |
| --- | --- | --- |
| IN1 | 4  | Board 1 (Left) |
| IN2 | 17 | Board 1 (Left) |
| IN3 | 27 | Board 1 (Left) |
| IN4 | 22 | Board 1 (Left) |
| ENA_L | 12 | Board 1 enable A (IN1/IN2, PWM) |
| ENB_L | 18 | Board 1 enable B (IN3/IN4, PWM) |
| IN5 | 5  | Board 2 (Right) |
| IN6 | 6  | Board 2 (Right) |
| IN7 | 19 | Board 2 (Right) |
| IN8 | 26 | Board 2 (Right) |
| ENA_R | 13 | Board 2 enable A (IN5/IN6, PWM) |
| ENB_R | 23 | Board 2 enable B (IN7/IN8, PWM) |
| BTN_PIN | 21 | Start button (other leg to GND, internal pull-up) |
| TRIG | 10 | Ultrasonic trigger (output) |
| ECHO | 9  | Ultrasonic echo (input) |

## Software setup

1. Flash Raspberry Pi OS and enable SSH/camera as needed.
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-opencv
   pip3 install RPi.GPIO ultralytics
   ```
3. Copy this `final/` folder onto the Pi.
4. Drop your trained weights into `data/best.pt` (and `data/data.yaml` if you plan to retrain).

## Running

- `python3 main.py` — sets up all GPIO pins, waits for the start button, then hands off to the rest of the program.
- `python3 default_control.py` — keyboard control of the mecanum drive over the terminal.
- `python3 test.py` — runs the YOLO model on the webcam feed (Pi-optimized: smaller inference size, frame skipping).
- `python3 train.py` — trains a YOLO model from `data/data.yaml` (run on a PC with a GPU, not on the Pi).

## Controls

### Physical

| Button | Pin | Action |
| --- | --- | --- |
| Start button | BTN_PIN (BCM 21) | `main.py` waits for this press once, then hands off to the rest of the program. |

### Keyboard (`default_control.py`)

| Key | Action |
| --- | --- |
| W | Drive forward |
| S | Drive backward |
| A | Strafe left |
| D | Strafe right |
| Q | Rotate left (CCW) |
| E | Rotate right (CW) |
| Space | Stop all wheels |
| 1 | Spin front-left wheel alone (calibration) |
| 2 | Spin front-right wheel alone (calibration) |
| 3 | Spin rear-left wheel alone (calibration) |
| 4 | Spin rear-right wheel alone (calibration) |
| X | Quit (also cleans up GPIO) |
| Ctrl+C | Quit (also cleans up GPIO) |

Forward drive auto-blocks and stops if the ultrasonic sensor reads closer than `OBSTACLE_CM` (15cm default).

### Calibrating wheel direction

If W drives diagonally instead of straight, one wheel is physically wired backward (a motor lead or IN1/IN2 pair swapped). Since all four wheels spinning the same rotational direction always gives straight motion for any wheel type, a diagonal drift means one wheel isn't actually going the way the code thinks:

1. Jack the robot up (wheels off the ground) or watch it closely.
2. Press **1**, **2**, **3**, **4** one at a time to spin FL, FR, RL, RR in isolation.
3. Every wheel's roller pattern should push the robot generally forward when spun "forward" — find the one that pushes backward instead.
4. In `default_control.py`, flip that wheel's entry in `WHEEL_INVERT` to `True` and re-test with W.

### Calibrating strafe (A/D)

Once W/S/Q/E drive straight and rotate cleanly, A/D can still drift at an angle or rotate slightly instead of going purely sideways — this is normal, since strafing needs much tighter matching between wheel speeds than straight driving does, and no two motors spin at exactly the same RPM at the same duty cycle. Fix it with `WHEEL_TRIM` (a per-wheel speed multiplier, default `1.0` for all four): press A, see which side the robot rotates toward, and slightly lower the trim (e.g. `0.95`) on the wheel(s) "winning" that rotation. Re-test and nudge again until A/D go straight sideways.

**Keep every wheel's effective duty cycle (`STRAFE_SPEED × WHEEL_TRIM`) inside roughly 45–60.** Below ~45 some motors don't have enough torque to move at all (a "dead zone" — see the stall note below); above ~60 the electrical glitch below tends to come back. Make small trim adjustments (±0.05–0.1) and retest each one — big jumps (e.g. straight from `1.0` to `0.85` or `1.15`) tend to overshoot past one edge of that safe range into the other problem.

**If A/D visibly reverses direction mid-hold** (spins one way then flips to the other while the key is still held down, not just an angled drift): this is not a code issue — the direction command sent to the wheels doesn't change while a key is held, so a live flip means something electrical is glitching. Strafing is the only command that drives both channels on the *same* driver board in opposite directions at once (forward and rotate always keep a board's two channels in sync), and that combination can couple back-EMF/current-spike noise between channels on cheap L298N-style boards, occasionally flipping an H-bridge's state.

- Software mitigation already applied: `STRAFE_SPEED` (default `55`, slightly below `SPEED`) runs strafe at a lower duty cycle than straight driving to cut current/back-EMF. Don't drop it too far though — some motors won't have enough torque to actually turn below a certain duty cycle (a "dead zone"), which shows up as some wheels just not moving at all. Nudge it up/down and retest until you find the range that's low enough to avoid the reversal but high enough that all four wheels actually spin.
- Hardware checks worth doing: add a decoupling capacitor (100–470µF) across each board's motor power input, confirm the Pi and both driver boards share a solid common ground, and check the battery can supply both channels' peak current at once without sagging.
