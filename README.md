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

Once W/S/Q/E drive straight and rotate cleanly, A/D can still drift at an angle or rotate slightly instead of going purely sideways — this is normal, since strafing needs much tighter matching between wheel speeds than straight driving does, and no two motors spin at exactly the same RPM at the same duty cycle. Fix it with `WHEEL_TRIM` (a per-wheel speed multiplier, default `1.0` for all four): press A, see which side the robot rotates toward, and slightly lower the trim (e.g. `0.9`) on the wheel(s) "winning" that rotation. Re-test and nudge again until A/D go straight sideways.
