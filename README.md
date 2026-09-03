# M.5 Final Project — Setup Guide

Raspberry Pi 4 robot: 4-motor drive (2x L298N-style drivers) + YOLO camera detection.

See [docs/CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md) for the full mecanum wiring map, current tuning constants, and the debugging history behind them.

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
4. Drop your trained weights into `perception/data/best.pt` (and `perception/data/data.yaml` if you plan to retrain).

## Layout

```
final/
├── main.py              # entry point: start button -> hands off to match logic
├── default_control.py   # keyboard control (testing/manual driving)
├── movement/
│   ├── __init__.py
│   └── movement.py       # MecanumDrive class - all motor/GPIO logic lives here
├── guidance/
│   ├── navigator.py       # Navigator - dead-reckoning (x, y, heading) estimate
│   └── guided_drive.py    # GuidedDrive - MecanumDrive + Navigator combined
├── perception/
│   ├── test.py            # YOLO detection on the webcam feed
│   ├── train.py           # trains a YOLO model (run on a PC with a GPU)
│   └── data/               # best.pt / data.yaml go here
└── docs/
    ├── CALIBRATION_REPORT.md
    └── กติกาการแข่งขันหุ่นยนต์แตะบอล.md / .pdf   # competition rules
```

- **`movement/`** — the `MecanumDrive` class: all motor pin setup, calibration constants (`WHEEL_INVERT`, `WHEEL_TRIM`, speeds), and movement methods (`forward`, `backward`, `strafe_left/right`, `rotate_left/right`, `stop`, `test_wheel`, `get_distance`). Every movement method takes an optional `speed=` to override the default duty cycle for that call. Import with `from movement import MecanumDrive` from anything that needs to drive the robot — keyboard control, autonomous ball-chasing logic, etc. — instead of duplicating motor code.
- **`guidance/`** — position/heading tracking. `Navigator` is pure dead-reckoning math (no GPIO, works starts at `(0, 0)` facing heading `0`, X = right of start, Y = ahead of start, heading in degrees clockwise). `GuidedDrive` wraps `MecanumDrive` + `Navigator` so calling a movement method (`forward()`, `rotate_right()`, etc.) both drives the motors and updates the estimated pose — call `.pose()` any time to get `(x_cm, y_cm, heading_deg)`. **No wheel encoders on this robot**, so this is open-loop and will drift over time (wheel slip, uneven floor, the speed constants being approximate) — good for "roughly where am I", not precision navigation. See the calibration note in `guidance/guided_drive.py` for measuring the real speed constants.
- **`default_control.py`** — keyboard control: reads WASD/etc. from the terminal and calls into `GuidedDrive`, printing the tracked position after every move. No motor logic of its own.
- **`main.py`** — waits for the start button, then hands off to the rest of the program (currently a TODO — wire in perception + `MecanumDrive`/`GuidedDrive` here for the autonomous match code).
- **`perception/`** — camera/YOLO code: `test.py` runs detection on the webcam feed (Pi-optimized: smaller inference size, frame skipping), `train.py` trains a model from `perception/data/data.yaml`.
- **`docs/`** — the calibration report and the competition rules.

## Running

- `python3 main.py`
- `python3 default_control.py`
- `python3 perception/test.py`
- `python3 perception/train.py`

## Controls

### Physical

| Button | Pin | Action |
| --- | --- | --- |
| Start button | BTN_PIN (BCM 21) | `main.py` waits for this press once, then hands off to the rest of the program. |

### Keyboard (`default_control.py`)

The current tracked position/heading is always shown live on one updating status line — no key needed to see it. A drive timer also starts automatically on the first movement command (W/S/A/D/Q/E/B), pauses while HALTed, resumes when un-HALTed, and stops for good on quit (X or Ctrl+C) — total drive time is printed on exit.

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
| B | About-face (rotate 180° from current heading) |
| R | Reset tracked position to (0, 0), heading 0 |
| + / - | Adjust speed by 5 (clamped 20-100) |
| H | HALT — stops and locks out every other key until H is pressed again (X still works) |
| X | Quit (also cleans up GPIO) |
| Ctrl+C | Quit (also cleans up GPIO) |

The ultrasonic obstacle check on forward drive is currently disabled (see `MecanumDrive.forward()` in `movement/movement.py`) — `get_distance()` still works if you want to re-enable it.

### Calibrating wheel direction

If W drives diagonally instead of straight, one wheel is physically wired backward (a motor lead or IN1/IN2 pair swapped). Since all four wheels spinning the same rotational direction always gives straight motion for any wheel type, a diagonal drift means one wheel isn't actually going the way the code thinks:

1. Jack the robot up (wheels off the ground) or watch it closely.
2. Press **1**, **2**, **3**, **4** one at a time to spin FL, FR, RL, RR in isolation.
3. Every wheel's roller pattern should push the robot generally forward when spun "forward" — find the one that pushes backward instead.
4. In `movement/movement.py`, flip that wheel's entry in `WHEEL_INVERT` to `True` and re-test with W.

### Calibrating strafe (A/D)

Once W/S/Q/E drive straight and rotate cleanly, A/D can still drift at an angle or rotate slightly instead of going purely sideways — this is normal, since strafing needs much tighter matching between wheel speeds than straight driving does, and no two motors spin at exactly the same RPM at the same duty cycle. Fix it with `WHEEL_TRIM` (a per-wheel speed multiplier, default `1.0` for all four): press A, see which side the robot rotates toward, and slightly lower the trim (e.g. `0.95`) on the wheel(s) "winning" that rotation. Re-test and nudge again until A/D go straight sideways.

**Keep every wheel's effective duty cycle (`STRAFE_SPEED × WHEEL_TRIM`) inside roughly 45–60.** Below ~45 some motors don't have enough torque to move at all (a "dead zone" — see the stall note below); above ~60 the electrical glitch below tends to come back. Make small trim adjustments (±0.05–0.1) and retest each one — big jumps (e.g. straight from `1.0` to `0.85` or `1.15`) tend to overshoot past one edge of that safe range into the other problem.

**If A/D visibly reverses direction mid-hold** (spins one way then flips to the other while the key is still held down, not just an angled drift): this is not a code issue — the direction command sent to the wheels doesn't change while a key is held, so a live flip means something electrical is glitching. Strafing is the only command that drives both channels on the *same* driver board in opposite directions at once (forward and rotate always keep a board's two channels in sync), and that combination can couple back-EMF/current-spike noise between channels on cheap L298N-style boards, occasionally flipping an H-bridge's state.

- Software mitigation already applied: `STRAFE_SPEED` (default `55`, slightly below `SPEED`) runs strafe at a lower duty cycle than straight driving to cut current/back-EMF. Don't drop it too far though — some motors won't have enough torque to actually turn below a certain duty cycle (a "dead zone"), which shows up as some wheels just not moving at all. Nudge it up/down and retest until you find the range that's low enough to avoid the reversal but high enough that all four wheels actually spin.
- Hardware checks worth doing: add a decoupling capacitor (100–470µF) across each board's motor power input, confirm the Pi and both driver boards share a solid common ground, and check the battery can supply both channels' peak current at once without sagging.
