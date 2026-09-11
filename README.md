# M.5 Final Project — Setup Guide

Raspberry Pi 4 robot: 4-motor mecanum drive (2x L298N-style drivers) + plain HSV color-segmentation camera perception (no YOLO/ML — see "Why no YOLO" below).

See [docs/CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md) for the full mecanum wiring map, current tuning constants, and the debugging history behind them (including the power-supply/brownout issue below).

## Which file is which mode

| Mode | File | Run it with | What it does |
| --- | --- | --- | --- |
| **Berserk attacker** | `main.py` | `python3 main.py` | Rotates fast to face the ball and charges once roughly pointed at it. Recovers from being pinned (wall or stuck-detection) by backing off then strafing in from an angle. Terminal-only, no live view. |
| **Defender** | `goalkeeper.py` | `python3 goalkeeper.py` | Holds a fixed heading, strafes L/R to stay in front of the ball (or the opponent's blue chassis), rams forward until actual contact then returns to its starting spot. Terminal-only, saves periodic detection snapshots to `snapshots/`. |
| **Alternate attacker** | `attacker.py` | `python3 attacker.py` | Same strafe-to-center-then-ram strategy as `goalkeeper.py`, but searches by rotating a full circle (stop-and-look, no motion blur) instead of goalkeeper's strafe-based search. |
| Manual control | `default_control.py` | `python3 default_control.py` | Keyboard-driven (WASD/etc.), no perception at all - for wiring/calibration testing. |
| Camera tuning view | `cam_control.py` | `python3 cam_control.py` | Live HSV segmentation view over HTTP, no motor control - for aiming the camera and tuning HSV ranges. |

Only one of `main.py`/`goalkeeper.py`/`attacker.py`/`cam_control.py` can run at a time (they all want the webcam) - see "Camera conflicts" below.

## ⚠️ Power supply — needs a second supply, not just the 12V lithium pack

The Pi currently shares its power with the drive motors off the same lithium pack. Under load (any motor starting from a stop, or several reversing) that shared rail sags enough to trip the Pi's own undervoltage detection — confirmed directly via `vcgencmd get_throttled` (bit 16, "under-voltage has occurred") and `dmesg` (`Undervoltage detected!`). In practice this drops the Wi-Fi connection (looks exactly like a dropped SSH session) and can happen on the very first motor command of a run, not just after prolonged use.

`movement/movement.py` now soft-starts the motors (ramps duty cycle up over ~150ms instead of jumping straight to it) to reduce the inrush spike, but that only softens the problem — it doesn't fix a shared rail that's genuinely marginal. **The real fix is hardware**: power the Pi from its own separate supply (a USB power bank, or its own regulator off a different battery) instead of drawing 5V off the same pack the motors pull high current from. If a second supply genuinely isn't available, the next-best options are a bulk capacitor (1000-4700µF) across the motor driver boards' power input, and wiring the Pi's 5V regulator input directly to the battery terminals (not downstream of the driver boards) to reduce shared-wire voltage coupling.

## Hardware

- Raspberry Pi 4
- 2x motor driver boards (L298N-style), one for each side
- 4x DC motors (2 left, 2 right)
- Push button (start button)
- HC-SR04 ultrasonic distance sensor (mounted at the REAR — watches for something closing in from behind, not a front obstruction)
- USB webcam
- 2S lithium-ion battery pack — currently shared between the Pi and the motors (see the power-supply warning above)

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
   pip3 install RPi.GPIO
   ```
   No `ultralytics`/`torch` needed — perception is plain OpenCV color segmentation, see "Why no YOLO" below.
3. Copy this `final/` folder onto the Pi.

## Layout

```
final/
├── main.py              # berserk attacker: rotates fast to face the ball, charges, pin-recovery
├── goalkeeper.py         # defender: strafes L/R to track ball/opponent-chassis, ram-until-contact
├── attacker.py           # alternate attacker: same as goalkeeper.py but rotate-based search
├── default_control.py   # keyboard control (testing/manual driving)
├── cam_control.py        # live camera view over HTTP (no HDMI needed)
├── snapshots/            # gitignored - periodic detection debug images from goalkeeper.py/attacker.py
├── movement/
│   ├── __init__.py
│   └── movement.py       # MecanumDrive class - all motor/GPIO logic + soft-start ramp
├── guidance/
│   ├── navigator.py       # Navigator - dead-reckoning (x, y, heading) estimate
│   └── guided_drive.py    # GuidedDrive - MecanumDrive + Navigator combined
├── perception/
│   ├── __init__.py
│   ├── color_segment.py   # the ONLY shared perception module - HSV masks, ball_center, find_goal_gap
│   └── data/               # legacy YOLO weights, unused by any current script
└── docs/
    ├── CALIBRATION_REPORT.md
    ├── PERCEPTION_PLAN.md    # historical - the original YOLO plan and why it was dropped
    └── กติกาการแข่งขันหุ่นยนต์แตะบอล.md / .pdf   # competition rules
```

- **`movement/`** — the `MecanumDrive` class: all motor pin setup, calibration constants (`WHEEL_INVERT`, `WHEEL_TRIM`, speeds), and movement methods (`forward`, `backward`, `strafe_left/right`, `rotate_left/right`, `stop`, `test_wheel`, `get_distance`). Every movement method takes an optional `speed=` to override the default duty cycle for that call. **Soft-start**: `set_wheels()` tracks each wheel's last commanded direction and, whenever a wheel is starting from stopped or reversing, ramps its duty cycle up over `SOFT_START_STEPS` steps (~150ms) instead of jumping straight to it — tapers the motor's inrush current, which was enough to brown out the shared battery rail (see the power-supply warning above). A wheel already spinning the same direction just gets its duty updated instantly, so this doesn't add latency to normal continuous driving. Import with `from movement import MecanumDrive` from anything that needs to drive the robot instead of duplicating motor code.
- **`guidance/`** — position/heading tracking. `Navigator` is pure dead-reckoning math (no GPIO, starts at `(0, 0)` facing heading `0`, X = right of start, Y = ahead of start, heading in degrees clockwise). `GuidedDrive` wraps `MecanumDrive` + `Navigator` so calling a movement method (`forward()`, `rotate_right()`, etc.) both drives the motors and updates the estimated pose — call `.pose()` any time to get `(x_cm, y_cm, heading_deg)`. **No wheel encoders on this robot**, so this is open-loop and drifts (wheel slip, uneven floor, approximate speed constants) — good for "roughly where am I", not precision navigation. Because of that drift, both `main.py` and `goalkeeper.py` deliberately avoid relying on tracked position/heading for any safety-critical decision — vision (the camera) is the only thing either script trusts for wall/goal proximity.
- **`default_control.py`** — keyboard control: reads WASD/etc. from the terminal and calls into `GuidedDrive`, printing the tracked position after every move. No motor logic of its own.
- **`cam_control.py`** — live camera view, and the only thing that actually opens the webcam (one process can hold it open at a time — see "Camera conflicts" below). Since there's no HDMI monitor on the Pi, `cv2.imshow()` won't work — this instead serves color, grayscale, and the `perception/color_segment.py` debug view as MJPEG streams over HTTP, viewable from a browser anywhere on the same network, including VS Code's own Simple Browser. See "Viewing the camera" below.
- **`main.py`** — **the berserk attacker**: single-file entry point (camera + perception + drive state machine + GPIO button, all in one; no live view - that HTTP server was removed as unnecessary per-frame overhead). Priority order every step: (1) recover from being pinned against a wall (overrides everything); (2) don't go into the goal — `find_goal_gap()` (wall-mask structure, not color) plus checking if the ball itself sits inside the detected gap; (3) evade an incoming rear threat — the ultrasonic is mounted at the REAR, so this pushes forward (the only direction that actually increases distance from something behind) rather than picking a strafe side; (4) find/charge the ball — HSV `ball_center()` restricted to the lower part of the frame (`BALL_ROI_Y_START`), snapping toward it with continuous fast rotation (`ROTATE_SPEED`) once it's more than `CENTERED_TOLERANCE_DEG` off (looser tolerance than a cautious approach — charges the moment it's roughly pointed at it rather than centering precisely first), then driving forward at `CHARGE_SPEED` once aligned. **Pin recovery**: triggered either by wall-mask coverage or by a whole-frame motion-stall check (`_is_stuck()`, off by default via `STUCK_DETECT_ENABLED` — commanding full charge speed but the frame barely changing means the wheels are spinning against something even if it doesn't look like "wall"); backs off briefly then strafes toward whichever side the ball was last seen on, so the next charge comes at it from an angle instead of straight back into whatever it was stuck on. Prints per-step timing (`ms/step`). Survives losing the SSH connection (ignores `SIGHUP`) and the webcam disconnecting (auto-reopens after `CAMERA_RECONNECT_AFTER` failures).
- **`goalkeeper.py`** — **the defender**: alternate strategy, same HSV perception, different decision layer. Instead of turning to face the ball, it holds a fixed heading (`HOME_HEADING_DEG`) and strafes left/right to keep the ball (or the opponent's blue chassis — see below) centered — the frame is split into L/C/R zones (`_ball_zone()`) rather than using a continuous angle, since strafing is a fixed-speed burst, not a variable-rate turn. Once centered *and* close (`CLOSE_Y_THRESHOLD`), it **rams forward until actual contact** — the target's mask covering `CONTACT_AREA_FRACTION` of the frame, not a fixed time burst — then drives backward for exactly as long as the ram took (time-symmetric, since there's no reliable position tracking) to return to its starting spot instead of drifting forward on every hit. `LAUNCH_SPEED` is maxed out (95) for the ram itself — forward/backward doesn't have strafe's same-board-opposite-channel glitch risk, so it can go much harder than `STRAFE_SPEED` (pushed to 65, deliberately past the ~60 ceiling documented below - watch for the glitch reappearing). **Target detection**: checks for the ball first, falls back to the robot chassis's distinctive blue color (`CHASSIS_BLUE_LOW/HIGH`, `_find_target()`) if no ball is seen — the same fallback applies inside every search variant, not just the main per-frame check, so the search hunts for either one. Four selectable rotate/strafe search variants (`SEARCH_MODE`) when nothing's visible - default is `"cord"` (strafe-based, no heading change needed since strafing doesn't touch heading at all). A near-field motion check (`_near_field_motion()`, off by default via `NEAR_FIELD_MOTION_ENABLED`) exists for catching something closing in fast while stationary, but was disabled after it proved to intercept genuine ball tracking during brief HSV flicker with no recovery action. Terminal-only status output; saves a snapshot of the target mask + detected contour to `snapshots/` every `SNAPSHOT_INTERVAL_S` for reviewing detection quality after a run.
- **`attacker.py`** — a copy of `goalkeeper.py` with the search switched to rotate-based (`SEARCH_MODE = "rotate_continuous"`): a full ~360° turn in stop-and-look steps (rotate a small step, come to a full stop, wait briefly, then capture) rather than continuous rotation or goalkeeper's strafe-based cord search - motion blur only happens while actually moving, so stopping before capturing keeps every checked frame sharp regardless of rotation speed. Also has `_find_ball()`, a stricter local ball detector with a circularity/area filter (rejects thin curved noise contours that plain HSV can mistake for the ball, but skips the circularity check above `BALL_CIRCULARITY_BYPASS_AREA` since a real ball cropped by a frame edge at close range also reads as non-circular). The ram has no time limit at all (an attacker doesn't need to hold a defensive spot) and returns for a fixed short `RETURN_ADJUST_DURATION_S` rather than goalkeeper's time-symmetric return. Has its own `LOST_TRACK_GRACE_FRAMES` grace period so a single missed frame during strafing doesn't immediately dump into a full search.
- **`perception/`** — `color_segment.py` is the *only* shared perception module: `get_masks()` (HSV thresholds for ball/wall/floor), `ball_center()`, `ball_angle_offset()`, `find_goal_gap()` (the wall's actual physical structure — a normal wall section has two black bands per column, lower rail + gap + upper rail; a goal only has the crossbar with open floor beneath it, so this counts vertical black *runs* per column and looks for a stretch with just 1 instead of 2, plus a density check that rejects a beveled-corner joint that also happens to read as "1 run" locally), and `build_debug_view()` (used by `cam_control.py`). `goalkeeper.py`/`attacker.py` also do their own local, file-specific HSV/contour checks (blue chassis, stricter ball shape filtering) rather than adding those to the shared module. `perception/data/` still has old YOLO weights on disk but nothing imports them anymore — see "Why no YOLO" below for what changed and why.
- **`docs/`** — the calibration report, the (now historical) original perception plan, and the competition rules.

## Why no YOLO

`docs/PERCEPTION_PLAN.md` originally called for YOLO on the ball (color alone isn't guaranteed reliable per the rulebook) plus HSV color for everything else. That was built and tried — including running it in a background thread so it wouldn't block the control loop — but in practice on this Pi's CPU, plain HSV color detection for the ball turned out both faster *and* more reliable than YOLO, once the ball mask was restricted to the lower part of the frame to cut out background noise. Combined with the power-supply issue above (sustained CPU load from running a model draws more current, which is actively bad on an already-marginal shared rail), the whole YOLO/ArUco detection stack (`perception/yolo_perception.py`, `aruco_goal.py`, `zone_edges.py`, `detect.py`, `train.py`, `test.py`) was removed rather than kept as unused dead weight. See `docs/PERCEPTION_PLAN.md` for the full original reasoning (kept for historical context, not current behavior) and the git history around this cleanup for how the YOLO-hybrid version actually behaved before it was dropped.

## Camera conflicts

Only one process can hold the webcam open at a time. `cam_control.py`, `main.py`, `goalkeeper.py`, and `attacker.py` all want it (`default_control.py` doesn't touch the camera) — don't run two camera-using scripts at once; the second one will fail to open the device.

## Running

- `python3 main.py` — the berserk attacker: rotates fast to face the ball and charges
- `python3 goalkeeper.py` — the defender: strafes to track the ball/opponent chassis, rams until contact
- `python3 attacker.py` — alternate attacker: same as `goalkeeper.py` but searches by rotating instead of strafing
- `python3 default_control.py` — manual keyboard control
- `python3 cam_control.py` — live HSV segmentation view for tuning/aiming, at `http://<pi-ip-address>:8080/`

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

`goalkeeper.py` currently overrides its own `STRAFE_SPEED` to `65` — deliberately past this ceiling, on request, to get more responsive tracking. If A/D-style direction-flip glitches show up while running it, that confirms 65 is too far and it should come back down toward 60; see `docs/CALIBRATION_REPORT.md` for the full history.

**If A/D visibly reverses direction mid-hold** (spins one way then flips to the other while the key is still held down, not just an angled drift): this is not a code issue — the direction command sent to the wheels doesn't change while a key is held, so a live flip means something electrical is glitching. Strafing is the only command that drives both channels on the *same* driver board in opposite directions at once (forward and rotate always keep a board's two channels in sync), and that combination can couple back-EMF/current-spike noise between channels on cheap L298N-style boards, occasionally flipping an H-bridge's state.

- Software mitigation already applied: `STRAFE_SPEED` (default `55`, slightly below `SPEED`) runs strafe at a lower duty cycle than straight driving to cut current/back-EMF. Don't drop it too far though — some motors won't have enough torque to actually turn below a certain duty cycle (a "dead zone"), which shows up as some wheels just not moving at all. Nudge it up/down and retest until you find the range that's low enough to avoid the reversal but high enough that all four wheels actually spin.
- Hardware checks worth doing: add a decoupling capacitor (100–470µF) across each board's motor power input, confirm the Pi and both driver boards share a solid common ground, and check the battery can supply both channels' peak current at once without sagging.
