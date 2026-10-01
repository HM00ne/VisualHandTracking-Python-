# Hypermage

Weave, grab, spin and erase real higher-dimensional objects — 3D, 4D, 5D and 6D — with your bare
hands through your webcam. The scene starts empty.

**Create:** touch your fingertips together to stretch a 4D string figure between your hands, then
make a peace sign with **both** hands — the strings spiral into a vortex and weave a new object.
**Erase:** close **both** hands into fists — every object bursts and is gone.

## Install

You need Python 3.10–3.12 (64-bit) and a webcam.

**Windows**
```
cd path\to\hypermage
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**macOS / Linux**
```
cd path/to/hypermage
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The first run downloads the MediaPipe hand model (`hand_landmarker.task`, about 8 MB) into the
folder. If the download is blocked, get it from the URL printed in the error and put it next to
`main.py`.

Permissions: allow camera access for your terminal (Windows: Settings → Privacy & security →
Camera → "Let desktop apps access your camera"; macOS: System Settings → Privacy & Security →
Camera). On Intel Macs, use `pip install "mediapipe==0.10.21"` first.

## Gesture cheat sheet

Your **dominant** hand is the right hand by default (`DOMINANT_HAND` in `config.py`).

### Dominant hand
| Gesture | Effect |
|---|---|
| Pinch on an object | Grab it; it follows your pinch on a spring |
| Rotate your wrist while grabbing | Normal 3D rotation (XY, XZ, YZ); keeps spinning after you let go |
| Release the pinch while moving | The object keeps drifting with your hand's momentum |
| Open palm facing the camera, swipe fast | Fling nearby objects; they bounce off the screen edges |
| Point (index finger only) | Draw glowing light trails that fade over 2 s |

### Either hand
| Gesture | Effect |
|---|---|
| Three fingers up (index + middle + ring) | The nearest object steps to the next dimension: 3D → 4D → 5D → 6D → 3D. Lower your fingers and raise them again for the next step |
| Peace sign (V) — one hand, no strings up | Telekinesis: the nearest object locks on (a shimmering tether appears) and follows your two fingers from anywhere on screen, amplified 1.6×. Drop the pose and it keeps its momentum. Both hands can each hold a different object |

### Off hand (the "4D hand")
| Gesture | Effect |
|---|---|
| Open hand, roll it left/right | Spin through the XW plane |
| Open hand, tilt palm left/right / up/down | Spin through the YW / ZW planes |
| Open hand, raise / lower (slice mode) | Move the slicing hyperplane through w |

### Both hands
| Gesture | Effect |
|---|---|
| Touch only your fingertips together (palms apart) and hold still for a moment | A 4D string figure (cat's cradle) stretches between all ten fingertips; pull apart to stretch it, move to make it vibrate. Separate, then touch fingertips again to dissolve it |
| **While the strings are up: peace sign with both hands** | **Create.** The strings tear loose, spiral into a vortex around a glowing core and weave themselves into a new object (1.2 s), which lands with a flash. Each creation makes the next shape type (up to 6 objects; a 7th replaces the oldest) |
| **Both fists** | **Erase.** Every object bursts into particles and is gone (strings too) |
| Both hands pinch the same object, pull apart / push together | Scale it (twisting your hands rotates it) |
| Thumbs up held for 1 s (either hand) | Toggle slice mode / projection mode |

### Keyboard
| Key | Action |
|---|---|
| 1–7 | Select the shape your next creation weaves: Tesseract, 16-cell, 24-cell, 600-cell, 6×6 duoprism, Clifford torus, Hopf fibration |
| S | Slice mode on/off |
| B | Void background (pure black) |
| H | Hand "energy vein" skeleton on/off |
| R | Reset the scene |
| D | Debug overlay: landmark numbers and live gesture values |
| P | Screenshot (saved in `captures/`) |
| V | Start/stop MP4 recording (saved in `captures/`) |
| Q / Esc | Quit |

## Reading the picture

Every shape exists in four dimensions:

| Shape (key) | 3D | 4D | 5D | 6D |
|---|---|---|---|---|
| 1 Cube family | Cube | Tesseract | Penteract (5-cube) | Hexeract (6-cube) |
| 2 Cross-polytope family | Octahedron | 16-cell | 5-orthoplex | 6-orthoplex |
| 3–7 24-cell, 600-cell, duoprism, Clifford torus, Hopf fibration | its 3D shadow | the shape | its prism (5D) | its double prism (6D) |

5D and 6D objects are rotated in all 10 / 15 of their rotation planes and projected down one
dimension at a time (6D → 5D → 4D → 3D → 2D), so you see nested, doubled structure.
3D objects have no w, so they are drawn in a single colour.

Edge colour shows each edge's position in the fourth dimension: **cyan** edges have w < 0,
**magenta** edges have w > 0. Spin through a W plane and you'll see the colours trade places as
the object turns inside out. Thinner, dimmer edges are farther away.

In **slice mode** the bright white shape is the true 3D cross-section of the 4D object at the
current w — what a 3D being would see as the object passes through its world. The faint wireframe
behind it is the full projection, for reference.

## Lighting and camera tips

- Light your hands from the front (a window or lamp behind the camera). Avoid strong light behind you.
- A plain, uncluttered background helps tracking.
- Sit 50–80 cm from the camera so both hands fit with room to move.
- Keep your palms roughly facing the camera for the open-hand gestures.
- For the string figure, touch fingertips lightly and hold still for a moment; then pull your hands
  apart before making the double peace sign, so both hands are clearly visible.
- Close other apps that use the camera.
- If FPS is low, lower `CAPTURE_WIDTH`/`CAPTURE_HEIGHT` to 960×540, or press B for void mode.

## Tuning gestures

All thresholds are in `config.py`. Press **D** to turn on the debug overlay first: it prints, for
each hand, the current pose, which fingers count as extended (`ext(TIMRP)`, thumb→pinky,
1 = extended), the straightness value of each finger, the pinch ratio, the palm normal and the
roll angle. Make the gesture, read the numbers, and set the threshold between the value you see
when you do the gesture and the value when you don't.

| Problem | Setting in `config.py` |
|---|---|
| Pinch fires when you didn't pinch | Lower `PINCH_ON` (e.g. 0.25) |
| Pinch hard to trigger | Raise `PINCH_ON` (e.g. 0.36); keep `PINCH_OFF` about 0.15 higher |
| Grabbed object drops mid-move | Raise `PINCH_OFF` or `PINCH_EXIT_S` |
| A fist sometimes counts as a pinch | Raise `PINCH_MIN_REACH` |
| A finger reads "curled" when straight | Lower `FINGER_EXTEND_ON` (e.g. 0.7) |
| A bent finger reads "extended" | Raise `FINGER_EXTEND_OFF` (e.g. 0.65) |
| Thumbs-up not recognised | Lower `THUMB_EXTEND_ON` or `THUMBS_UP_MIN_UP` |
| Poses flicker between two names | Raise `POSE_HOLD_S` (e.g. 0.25) |
| Open-palm swipe doesn't fling | Lower `FLING_MIN_SPEED`, or `FACING_CAMERA_Z` towards -0.4 |
| Fling fires by accident | Raise `FLING_MIN_SPEED` |
| Telekinesis moves objects too far / not far enough | Change `TELEKINESIS_GAIN` |
| Three fingers not recognised (shows "open") | Tuck your pinky; or raise `FINGER_EXTEND_OFF` slightly |
| Dimension steps twice | Raise `DIMENSION_COOLDOWN_S` or `POSE_HOLD_S` |
| New objects should start in another dimension | Set `START_DIMENSION` (3–6) |
| Both fists don't erase | Check both hands show `fist` in the HUD; lower `FINGER_EXTEND_OFF` if a finger reads as extended |
| Erasing happens too easily | Raise `ERASE_HOLD_S` |
| Double peace doesn't create | Both hands must show `peace` in the HUD while the strings are up; keep ring and pinky folded |
| Weave animation too fast / slow | Change `WEAVE_S` (and `WEAVE_SPIN` for the swirl) |
| Fingertip touch not detected | Raise `FINGER_TOUCH_DIST` (e.g. 0.5) or lower `FINGERS_ONLY_MIN_GAP` |
| String figure appears during other two-hand moves | Raise `FINGER_TOUCH_HOLD_S` or lower `FINGER_TOUCH_MAX_SPEED` |
| Strings too wild / too calm | Change `STRING_AMPLITUDE` and `STRING_SPIN` |
| Low FPS with many 5D/6D objects | Lower `HEAVY_EDGE_COUNT` (thinner lines sooner) or `MAX_OBJECTS` |
| W-rotation drifts when your hand is still | Raise `W_ROTATION_DEADZONE` |
| Landmarks jittery | Lower `ONE_EURO_MIN_CUTOFF` |
| Hands feel laggy | Raise `ONE_EURO_BETA` |
| Left and right hands swapped in the HUD | Set `SWAP_HANDEDNESS = True` |
| You're left-handed | Set `DOMINANT_HAND = "Left"` |
