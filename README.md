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
├── cam_control.py        # live camera view over HTTP (no HDMI needed)
├── ball_chase.py          # demo: turns toward + drives at the ball using perception + guidance
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
- **`cam_control.py`** — live camera view, and the only thing that actually opens the webcam (one process can hold it open at a time, so this is the single source for every view rather than each perception script grabbing its own capture). Since there's no HDMI monitor on the Pi, `cv2.imshow()` (a desktop window) won't work — this instead serves color, grayscale, and the `perception/color_segment.py` debug view as MJPEG streams over HTTP, viewable from a browser anywhere on the same network, including VS Code's own Simple Browser. See "Viewing the camera" below.
- **`main.py`** — the actual match entry point: waits for the start button (currently skipped, see `WAIT_FOR_BUTTON`), then opens the camera and runs the same ball-chase loop as `ball_chase.py` (imports `open_camera`/`chase_loop` from it rather than duplicating the logic) until Ctrl+C. Also serves a live view of what it sees at `http://<pi-ip>:8080/` (`SERVE_LIVE_VIEW = True`) — reuses the exact frame the chase loop already reads via an `on_frame` hook, so it doesn't open a second camera connection (`cam_control.py` can't run at the same time as `main.py`, but you don't need it to — this is that same view, built in). If port 8080 is already taken (e.g. a previous `main.py` still running — it survives a dropped SSH session on purpose, see below, so it's easy to lose track of one), tries the next few ports instead of crashing the whole run over a debugging aid; watch the printed URL for which one it actually landed on. If none of the first 5 are free, it just runs without the live view rather than failing outright. No wall/goal awareness yet — just ball-seeking. Survives losing the SSH connection (ignores `SIGHUP`, and status printing is best-effort so a dead terminal pipe can't crash the driving loop) and survives the webcam disconnecting (stops the motors immediately on a failed read instead of driving blind on a stale command, then auto-reopens the camera after `CAMERA_RECONNECT_AFTER` consecutive failures and resumes).
- **`ball_chase.py`** — the ball-chase logic itself, runnable standalone (no button/GPIO wait, just Ctrl+C to stop) for faster iteration while testing. Finds the ball with `perception.ball_center()`, converts its pixel offset to a real angle with `perception.ball_angle_offset()` (needs `CAMERA_HFOV_DEG` in `perception/color_segment.py` calibrated for the real camera — placeholder for now), and turns toward it with `GuidedDrive.rotate_to()` when off-center by more than `CENTERED_TOLERANCE_DEG`, or drives forward when roughly centered. When the ball isn't found at all, spins in place at `SEARCH_SPEED`, tracking the actual rotation swept — if it completes a full `SEARCH_FULL_SWEEP_DEG` (350°) lap without finding it, nudges forward once before continuing the search, instead of spinning in the same spot forever. If the wall mask covers more than `WALL_COVERAGE_THRESHOLD` (80%) of the frame — facing straight into it — backs up instead of chasing whatever the ball logic thinks it sees. **Evade mode**: if the ultrasonic reads closer than `EVADE_DISTANCE_CM` (15cm) and it's *not* the wall (that's the vision check above — this one doesn't know or care what the obstruction actually is, opponent robot or otherwise), strafes around it instead of driving into it or spinning in place — toward whichever side the ball is currently on, or was last seen on if it's hidden behind whatever's blocking the way right now, so it goes around toward the ball rather than a fixed/blind direction. Normal chase (turn-then-approach) resumes on its own once the path is clear, so it naturally races back to intercept the ball once past the obstruction. **Goal dwell limit**: holds position immediately on entering a detected goal opening (found via `perception.find_goal_gap()` — a bitwise column-height dip in the wall mask, not a color, since a goal is the same wall color just shorter there) within `GOAL_AREA_DISTANCE_CM` (30cm), OR when the gap simply looks wide in-frame (`GOAL_GAP_CLOSE_WIDTH_FRACTION`, 50% of frame width) — vision alone is enough to trigger this, it doesn't solely depend on the ultrasonic reading correctly. Doesn't keep chasing the ball in. `GOAL_DWELL_LIMIT_S` (5s) is a safety net for forcing a retreat if it ends up stuck there anyway, not permission to enter during those first few seconds. Doesn't need to know which goal (see `guess_goal_ownership()` for telling them apart, used for not shooting into our own — not yet wired into an actual aim/shoot behavior since there's no shooting mechanism in code yet). **Pinned-ball recovery**: tracks how long the ball's been found and engaged with (shared across both the "turning" and "approaching" sub-states, so a jittering angle right at the `CENTERED_TOLERANCE_DEG` boundary can't reset the clock by bouncing between them) — past `PIN_STUCK_TIME_S` (3s) without ever completing the approach, assumes it's pinned against the wall, backs off for `UNPIN_BACKUP_S`, then strafes for `UNPIN_STRAFE_S` to re-approach from an angle instead of pushing uselessly straight into whatever's behind it, then resumes normal chase. Not gated on the ultrasonic (unlike evade) since a small ball flush against a flat wall may not reliably register as "close" on its own. Opens its own camera capture — can't run at the same time as `cam_control.py` or `main.py` (all three want the webcam).
- **`perception/`** — camera code: `test.py` runs YOLO detection on the webcam feed (Pi-optimized: smaller inference size, frame skipping), `train.py` trains a model from `perception/data/data.yaml`. `detect.py` is a lightweight standalone detector using both approaches together — no motor/GPIO code at all, detection only, served as an MJPEG view (same no-HDMI pattern) at `http://<pi-ip>:8082/`. Every frame first tries the cheap HSV/bitwise color check (`color_segment.ball_center()`); only when that finds nothing does it fall back to YOLO (`perception/data/best2.pt`) on that frame, so the expensive model runs only when it actually needs to, not on every frame by default. `color_segment.py` is a color-based approach for the ball (bright green in testing — not the orange/yellow the rulebook describes, so its HSV range is calibrated against the real ball, not the rules), wall (black), and floor (gray) using HSV masks + bitwise ops — logic only, no camera/server of its own; its debug view is served by `cam_control.py` (see below) since only one process can hold the webcam open at a time. `aruco_goal.py` detects ArUco tags marking each goal (`OUR_GOAL_ID`/`OPPONENT_GOAL_ID` — placeholders, set to whatever IDs are actually printed on the real tags) via `find_goal(frame, target_id)`, returning the tag's pixel center and area; feed the center into `ball_angle_offset()` (works for any pixel x-position, not just the ball) for a bearing, and use `area` as a rough "how close" proxy since there's no calibrated camera for true metric distance. Logic only, no camera/server of its own, same as `color_segment.py`. `color_segment.find_goal_gap()` is the non-tag alternative for finding a goal opening — it doesn't detect a color, it looks for a dip in the wall mask's per-column height (a goal is the same wall color, just a shorter barrier there with more open background above), returning the gap's pixel center/width. `ball_chase.guess_goal_ownership()` reconciles which goal a detected gap or tag actually is using the tracked heading (Navigator's world frame) against `OPPONENT_GOAL_HEADING_DEG` (placeholder — confirm against the real starting orientation) — prefer an ArUco tag ID when one's visible (ground truth), fall back to this heading guess when it's not.
- **`docs/`** — the calibration report and the competition rules.

## Running

- `sudo python3 main.py` — waits for the start button, then chases the ball
- `python3 default_control.py`
- `python3 cam_control.py`
- `sudo python3 ball_chase.py` (needs GPIO access like `default_control.py`; stop it with Ctrl+C — don't run alongside `cam_control.py`, they'll fight over the webcam)
- `python3 perception/test.py`
- `python3 perception/train.py`
- `python3 perception/detect.py` — detection-only view at `http://<pi-ip-address>:8082/` (no motor control, no GPIO)

## Viewing the camera (no HDMI)

1. On the Pi: `python3 cam_control.py` — it prints a URL like `http://<pi-ip-address>:8080/`. Find the Pi's actual IP with `hostname -I` if you don't already have it (it's whatever address you SSH to).
2. In VS Code (connected to the Pi over Remote-SSH): press `Ctrl+Shift+P` (or `Cmd+Shift+P`), run **"Simple Browser: Show"**, and paste that URL (with the Pi's real IP, not `<pi-ip-address>`) — the live feed opens in a tab inside VS Code, no HDMI or extra software needed.
   - Alternatively just open that URL in any regular browser on a device on the same network (phone, laptop).
3. Ctrl+C on the Pi to stop the stream.

The page shows three feeds: color and grayscale side by side (`/stream`, `/stream_gray`) — useful for previewing what a grayscale-based detector (e.g. the black wall/tape) would actually see — plus the `perception/color_segment.py` ball/wall/floor segmentation debug view (`/stream_segment`) below them, for tuning its HSV ranges.

The color/grayscale feeds are view-only. The segmentation view does run real detection logic (`get_masks`/`ball_center` from `perception/color_segment.py`) so you can see it working live, but `cam_control.py` itself doesn't act on it — it's for aiming the camera and tuning HSV ranges, not the autonomous match code.

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
| B | About-face — turns exactly 180° from current heading (tracked value, not just "close enough") |
| T | Turn by a typed angle (prompts for degrees; + = right/CW, - = left/CCW) |
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
