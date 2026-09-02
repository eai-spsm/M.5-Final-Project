# Mecanum Drive Calibration Report

Summary of building and calibrating the mecanum drive (`movement/movement.py`),
kept for reference if the robot is rewired or a motor/board is swapped
later.

## What the robot is

4-wheel mecanum drive, 2 L298N-style driver boards:

- **Board 1 (Left)**: drives the two left wheels, `ENA_L`=BCM12, `ENB_L`=BCM18, `IN1-4`
- **Board 2 (Right)**: drives the two right wheels, `ENA_R`=BCM13, `ENB_R`=BCM23, `IN5-8`

Plus a start button (BCM21) and an HC-SR04 ultrasonic sensor (TRIG=BCM10, ECHO=BCM9).

## Confirmed-correct final wiring map

| Wheel | Pins | PWM (speed) channel |
| --- | --- | --- |
| Front-Left (FL) | IN1/IN2 | `pwm_left_a` (ENA_L) |
| Rear-Left (RL) | IN3/IN4 | `pwm_left_b` (ENB_L) |
| Front-Right (FR) | IN7/IN8 | `pwm_right_b` (ENB_R) |
| Rear-Right (RR) | IN5/IN6 | `pwm_right_a` (ENA_R) |

**Important asymmetry**: the left board's first channel pair (IN1/IN2) is
the *front* wheel, but the right board's first channel pair (IN5/IN6) is
the *rear* wheel. The two boards were not wired with the same front/rear
convention. This was only found by testing each wheel in isolation with
`MecanumDrive.test_wheel()` (the `1`/`2`/`3`/`4` keys in
`default_control.py` call it) — don't assume symmetry if this ever needs
re-wiring.

## Calibration constants (current values)

In `movement/movement.py`:

```python
DEFAULT_SPEED = 60            # straight driving duty cycle %
DEFAULT_STRAFE_SPEED = 55      # strafe duty cycle % (lower than SPEED, see below)
WHEEL_INVERT = {"fl": False, "fr": False, "rl": False, "rr": True}
WHEEL_TRIM   = {"fl": 0.9,   "fr": 0.9,   "rl": 1.0,   "rr": 1.0}
```

- **`WHEEL_INVERT`**: RR's motor leads are physically reversed relative to
  the other three (confirmed via straight-line drive testing). `True`
  flips its GPIO signal so a "forward" request produces actual forward
  rotation, matching the others.
- **`WHEEL_TRIM`**: FL/FR run slightly under-power (`0.9`) relative to
  RL/RR — no two DC motors spin at exactly the same RPM at the same duty
  cycle, and strafing needs much tighter matching than straight driving to
  avoid drifting at an angle or rotating.
- **`DEFAULT_STRAFE_SPEED` (55) vs `DEFAULT_SPEED` (60)**: strafing is the only command
  that drives both channels on the *same* board in opposite directions at
  once (straight driving and rotation always keep a board's two channels
  in sync). That combination can couple back-EMF/current-spike noise
  between channels on cheap L298N-style boards, which showed up as the
  commanded direction visibly flipping mid-hold. Running strafe a bit
  slower reduces the current/back-EMF enough to avoid it.
- **Safe range**: keep every wheel's effective duty (`STRAFE_SPEED × WHEEL_TRIM`)
  between roughly **45 and 60**. Below ~45 some motors don't have enough
  torque to move at all (a "dead zone" — showed up as 2 wheels just not
  turning at `STRAFE_SPEED=40`). Above ~60 the electrical glitch above
  tends to reappear.

## Debugging path (for context on why these values, not others)

1. **W drove diagonally** → one wheel (RR) had reversed motor leads →
   fixed with `WHEEL_INVERT`.
2. **W then drove straight but backward** → the whole robot's
   forward/backward sense was flipped → fixed by swapping the HIGH/LOW
   convention in the shared `_wheel()` primitive (affects all commands
   uniformly, so strafe/rotate stayed consistent with drive).
3. **A/D strafed diagonally, not sideways** → assumed a single global
   front/rear pin mislabeling → swapped it → made things worse (pure
   rotation, no translation).
4. **Root cause found via the `1`/`2`/`3`/`4` calibration keys**: the left
   and right boards each have their *own*, independent front/rear channel
   order — they don't match each other. Re-mapped each board separately
   using the calibration keys as ground truth (not guesses).
5. **A/D still drifted at an angle after correct wiring** → real motor
   speed mismatch between wheels, not a wiring bug → tuned with
   `WHEEL_TRIM`, in small (~0.05) steps after two earlier big jumps
   (0.85, 1.15) overshot into the stall zone and the glitch zone
   respectively.
6. **A/D occasionally flipped direction mid-hold with an unchanging
   command** → traced to the same-board opposite-channel electrical
   coupling described above → mitigated with `STRAFE_SPEED`.

## If you rewire or swap hardware

Re-run the `1`/`2`/`3`/`4` calibration keys first, before touching
`WHEEL_INVERT` or `WHEEL_TRIM` — confirm each key spins the wheel it
claims to. Don't assume the two boards match each other's channel order.

## Code structure

All of the above (pin setup, calibration constants, movement logic) now
lives in `movement/movement.py` as a `MecanumDrive` class, not in
`default_control.py`. This is so the same calibrated motor code can be
reused by both keyboard testing (`default_control.py`) and the
autonomous match code (perception → `MecanumDrive` calls), instead of
being duplicated or copy-pasted between them. Every movement method
(`forward`, `strafe_left`, etc.) accepts an optional `speed=` argument to
override the default duty cycle for that call — e.g. `drive.forward(speed=30)`
for a slow final approach to the ball vs. the default speed for
repositioning.
