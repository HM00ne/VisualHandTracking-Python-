"""
config.py — every tunable number, colour and key binding in Hypermage lives here.

Units used below:
  * "palm" = palm size = distance from the wrist to the base of the middle finger, in pixels.
    Gesture distances are divided by it, so they work near or far from the camera.
  * "px" = screen pixels of the output window, "s" = seconds.
  * Colours are BGR (OpenCV order), 0..255.
"""

# ======================================================================================
# Camera & window
# ======================================================================================
CAMERA_INDEX = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
MIRROR = True                  # mirror the feed so it behaves like a mirror
WINDOW_NAME = "Hypermage"
FULLSCREEN = False

# ======================================================================================
# MediaPipe hand tracking
# ======================================================================================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# MediaPipe labels hands assuming a mirrored (selfie) image, which is what we feed it.
# If your left and right hands come out swapped in the HUD, set this to True.
SWAP_HANDEDNESS = False
DOMINANT_HAND = "Right"        # "Right" or "Left": the hand that rotates grabbed objects, flings and draws
HAND_LOST_GRACE_S = 0.25       # a hand missing for less than this keeps its smoothing state

# One Euro filter on all 21 landmarks (pixel units).
#   min_cutoff lower  -> steadier when still (more lag on slow moves)
#   beta higher       -> less lag when moving fast (more jitter)
ONE_EURO_MIN_CUTOFF = 1.7
ONE_EURO_BETA = 0.012
ONE_EURO_D_CUTOFF = 1.0

# ======================================================================================
# Gesture recognition (see README "Tuning gestures")
# ======================================================================================
# Finger straightness = cosine of the bend at the middle joint (1 = straight, 0 = 90° bent).
FINGER_EXTEND_ON = 0.80        # becomes "extended" above this
FINGER_EXTEND_OFF = 0.55       # becomes "curled" below this (gap = hysteresis)
THUMB_EXTEND_ON = 0.80
THUMB_EXTEND_OFF = 0.60
THUMB_MIN_REACH = 0.55         # thumb tip must also be this many palms from the index knuckle

PINCH_ON = 0.30                # thumb-tip to index-tip distance (palms) to start a pinch
PINCH_OFF = 0.45               # ...distance to end it
PINCH_MIN_REACH = 0.55         # pinch point must be this far from palm centre (stops fists pinching)
PINCH_ENTER_S = 0.05           # pinch must be seen this long before it counts
PINCH_EXIT_S = 0.08            # and released this long before it ends

POSE_HOLD_S = 0.15             # open/fist/point/peace/thumbs-up must be stable this long
FACING_CAMERA_Z = -0.55        # palm normal z below this = palm faces the camera
THUMBS_UP_MIN_UP = 0.6         # thumb direction must point this much upwards (0..1)
THUMBS_UP_HOLD_S = 1.0         # hold thumbs-up this long to toggle slice/projection
THUMBS_UP_EITHER_HAND = True   # False = both hands must show thumbs-up

# ======================================================================================
# Objects & physics
# ======================================================================================
OBJECT_SCALE = 115             # default object radius, px
OBJECT_SCALE_MIN = 40
OBJECT_SCALE_MAX = 420
MAX_OBJECTS = 6
GRAB_RADIUS = 1.25             # pinch within this × object radius grabs it
MATERIALIZE_S = 0.6

PROJ_DIST_4D = 2.4             # 4D camera distance on w (smaller = stronger inside-out effect)
PROJ_DIST_3D = 3.4             # 3D camera distance on z

SPRING_K = 170.0               # grab spring stiffness (1/s²)
SPRING_DAMPING = 18.0          # grab spring damping (1/s)
LINEAR_FRICTION = 0.45         # free-flying objects slow down by this rate (1/s)
BOUNCE_RESTITUTION = 0.85      # speed kept after bouncing off a screen edge
MAX_SPEED = 3500.0             # px/s
ANGULAR_DAMPING = 0.30         # spinning slows down by this rate (1/s)
IDLE_SPIN = 0.25               # rad/s random spin given to new objects

WRIST_ROTATION_GAIN = 1.6      # grabbed object turns this many × your wrist rotation
W_ROTATION_GAIN = 2.6          # off-hand tilt -> W-plane spin speed (rad/s at full tilt)
W_ROTATION_DEADZONE = 0.15     # ignore small off-hand tilts
W_ROLL_FULL = 0.8              # off-hand roll (rad) that counts as "full tilt"
SLICE_W_RANGE = 1.2            # raising/lowering the off-hand moves the slice between ±this
SLICE_W_SMOOTH = 8.0

FLING_MIN_SPEED = 1100.0       # open-palm swipe speed (px/s) needed to fling
FLING_GAIN = 1.1
FLING_REACH = 90               # px beyond an object's radius the palm can fling it from
FLING_COOLDOWN_S = 0.35

# ======================================================================================
# Two-hand magic
# ======================================================================================
# Create: make a string figure (fingertips touch), then a peace sign with BOTH hands
WEAVE_S = 1.2                  # how long the strings take to spiral into the new object
WEAVE_SPIN = 5.0               # how hard the vortex swirls (radians at the midpoint)
# Erase: close BOTH hands into fists
ERASE_HOLD_S = 0.2             # hold both fists this long (after they are recognised)

# Peace sign (either hand): telekinesis — the nearest object follows your two fingers
TELEKINESIS_GAIN = 1.6         # object moves this many × your hand movement
TELEKINESIS_SPRING_K = 90.0    # softer spring than a direct grab, so it feels "remote"
TELEKINESIS_DAMPING = 13.0

# Three fingers up (index + middle + ring) -> nearest object steps 3D -> 4D -> 5D -> 6D -> 3D
START_DIMENSION = 4            # dimension of newly created objects (3..6)
DIMENSION_COOLDOWN_S = 0.5

# Only fingertips of both hands touching -> 4D string figure between all ten fingertips
FINGER_TOUCH_DIST = 0.35       # a left and a right fingertip closer than this (palms) = touching
FINGERS_ONLY_MIN_GAP = 1.6     # ...while the palms stay at least this far apart (palms)
FINGER_TOUCH_HOLD_S = 0.35     # fingertips must rest together this long
FINGER_TOUCH_MAX_SPEED = 2.5   # ...with both hands moving slower than this (palms/s)
FINGER_TOUCH_REARM = 0.8       # fingertips must separate this far before the next touch counts
STRING_SAMPLES = 28            # points per string (more = smoother curves)
STRING_AMPLITUDE = 0.09        # 4D wobble size as a fraction of string length
STRING_SPIN = (0.0, 0.25, 0.7, 0.4, 0.5, 0.35)   # rad/s in XY, XZ, XW, YZ, YW, ZW
STRING_LOST_S = 0.35           # a hand missing this long dissolves the figure

# ======================================================================================
# Visuals
# ======================================================================================
BACKGROUND_DIM = 0.6           # webcam brightness (0.6 = darkened 40%)
BLOOM_DOWNSCALE = 4
BLOOM_SIGMA = 6.0
BLOOM_STRENGTH = 1.3
OBJECT_TRAIL_DECAY = 0.72      # ghost frames: fraction kept each frame
LIGHT_TRAIL_SECONDS = 2.0      # pointing trails fade over this long
COLOR_BINS = 10                # w-colour quantisation steps (more = smoother, slower)
HEAVY_EDGE_COUNT = 900         # objects with more edges than this draw thinner lines (keeps FPS up)

COLOR_W_NEG = (255, 255, 0)    # cyan for w < 0
COLOR_W_POS = (255, 0, 255)    # magenta for w > 0
COLOR_SLICE = (255, 255, 255)
COLOR_LEFT_HAND = (255, 80, 150)    # violet/blue sparks
COLOR_RIGHT_HAND = (40, 170, 255)   # gold/orange sparks
COLOR_TRAIL = (60, 200, 255)        # light-trail gold
COLOR_HUD = (230, 230, 230)

MAX_PARTICLES = 9000
SPARKS_BASE = 0.6              # sparks per fingertip per frame when still
SPARKS_PER_SPEED = 0.004       # extra sparks per px/s of fingertip speed
SPARKS_MAX = 6

SHAKE_DECAY = 6.0              # screen shake fades at this rate (1/s)
SHAKE_MAX_PX = 22
ABERRATION_MAX_PX = 10

# ======================================================================================
# Keyboard (backup controls)
# ======================================================================================
KEY_QUIT = (ord("q"), ord("Q"), 27)           # Q or Esc
KEY_SLICE = (ord("s"), ord("S"))
KEY_VOID = (ord("b"), ord("B"))
KEY_SKELETON = (ord("h"), ord("H"))
KEY_RESET = (ord("r"), ord("R"))
KEY_DEBUG = (ord("d"), ord("D"))
KEY_SCREENSHOT = (ord("p"), ord("P"))
KEY_RECORD = (ord("v"), ord("V"))
# 1..7 select the object type (see geometry4d.SHAPE_LIST for the order)

CAPTURE_DIR = "captures"       # screenshots and recordings go here
