"""
tracking.py — webcam capture + MediaPipe HandLandmarker in a background thread, then
landmark smoothing (One Euro filter), handedness clean-up and per-hand measurements
(palm centre, palm size, palm normal, orientation, velocities).
"""

from __future__ import annotations

import math
import os
import platform
import threading
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

import config as C

# Landmark indices (MediaPipe 21-point hand)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

HAND_CONNECTIONS = np.array([
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
], dtype=np.int32)


# ======================================================================================
# One Euro filter (vectorised: filters a whole array of values at once)
# ======================================================================================
class OneEuroFilter:
    """Speed-adaptive low-pass filter: heavy smoothing when slow, light when fast (Casiez 2012)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy()
        dt = max(t - self.t, 1e-4)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


# ======================================================================================
# Model download
# ======================================================================================
def ensure_model() -> str:
    path = C.MODEL_PATH
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(path):
        return path
    print(f"Downloading MediaPipe hand model to {path} ...")
    try:
        urllib.request.urlretrieve(C.MODEL_URL, path)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(
            f"Could not download the hand model ({exc}).\n"
            f"Download it manually from\n  {C.MODEL_URL}\nand save it as\n  {path}"
        ) from exc
    print("Model downloaded.")
    return path


# ======================================================================================
# Capture + detection thread
# ======================================================================================
@dataclass
class RawHand:
    label: str                 # "Left" / "Right" (the person's own hand)
    score: float
    norm: np.ndarray           # (21, 3) MediaPipe normalised coordinates


@dataclass
class TrackerFrame:
    frame: np.ndarray          # mirrored BGR frame
    hands: list[RawHand]
    timestamp: float           # time.monotonic() at capture
    seq: int


class HandTracker:
    """
    Runs camera capture and hand detection in a background thread so rendering never waits on
    MediaPipe. The main loop calls `latest()` to get the newest frame + hands.
    """

    def __init__(self):
        self.cap = self._open_camera()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=C.NUM_HANDS,
            min_hand_detection_confidence=C.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=C.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=C.MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        ok, first = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError("The camera opened but returned no image. Check camera permissions.")
        self.height, self.width = first.shape[:2]
        # Reused buffers: one for the raw camera image, a ring of three for mirrored output.
        self._raw = first
        self._ring = [np.empty_like(first) for _ in range(3)]
        self._rgb = np.empty_like(first)
        self._ring_i = 0

        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._latest: TrackerFrame | None = None
        self._seq = 0
        self._last_ts_ms = -1
        self._running = True
        self.error: Exception | None = None
        self.detect_ms = 0.0
        self._thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self._thread.start()

    @staticmethod
    def _open_camera() -> cv2.VideoCapture:
        backend = {"Windows": cv2.CAP_DSHOW, "Darwin": cv2.CAP_AVFOUNDATION}.get(platform.system(), cv2.CAP_ANY)
        cap = cv2.VideoCapture(C.CAMERA_INDEX, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(C.CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera #{C.CAMERA_INDEX}.\n"
                "  • Close other apps using the camera (Zoom, Teams, browser tabs).\n"
                "  • Windows: Settings → Privacy & security → Camera → allow desktop apps.\n"
                "  • macOS: System Settings → Privacy & Security → Camera → allow your terminal.\n"
                "  • If you have several cameras, change CAMERA_INDEX in config.py."
            )
        # MJPG lets most webcams deliver 1280x720 at 30 fps over USB.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, C.CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        failures = 0
        try:
            while self._running:
                ok, raw = self.cap.read(self._raw)
                if not ok or raw is None:
                    failures += 1
                    if failures > 60:
                        raise RuntimeError("Lost the camera feed.")
                    time.sleep(0.01)
                    continue
                failures = 0
                self._raw = raw
                ts = time.monotonic()

                out = self._ring[self._ring_i]
                self._ring_i = (self._ring_i + 1) % len(self._ring)
                if C.MIRROR:
                    cv2.flip(raw, 1, dst=out)
                else:
                    np.copyto(out, raw)

                hands = self._detect(out, ts)
                with self._lock:
                    self._seq += 1
                    self._latest = TrackerFrame(out, hands, ts, self._seq)
                    self._new.notify_all()
        except Exception as exc:  # noqa: BLE001 - reported to the main thread
            self.error = exc
            with self._lock:
                self._new.notify_all()

    def _detect(self, frame: np.ndarray, ts: float) -> list[RawHand]:
        t0 = time.perf_counter()
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._rgb)
        ts_ms = int(ts * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        result = self.landmarker.detect_for_video(image, ts_ms)
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            label = handed[0].category_name
            if C.SWAP_HANDEDNESS:
                label = "Left" if label == "Right" else "Right"
            # MediaPipe assumes a mirrored input. If we feed an unmirrored image, swap labels.
            if not C.MIRROR:
                label = "Left" if label == "Right" else "Right"
            norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            hands.append(RawHand(label, float(handed[0].score), norm))
        # Two hands with the same label: trust screen position instead
        # (in the mirrored view your left hand is on the left).
        if len(hands) == 2 and hands[0].label == hands[1].label:
            hands.sort(key=lambda h: float(h.norm[:, 0].mean()))
            hands[0].label, hands[1].label = "Left", "Right"
        self.detect_ms = 0.9 * self.detect_ms + 0.1 * (time.perf_counter() - t0) * 1000
        return hands

    def latest(self, last_seq: int, timeout: float = 0.5) -> TrackerFrame | None:
        """Wait for a frame newer than `last_seq` and return it (None on timeout)."""
        with self._lock:
            if self._latest is None or self._latest.seq == last_seq:
                self._new.wait(timeout)
            if self.error:
                raise self.error
            if self._latest is None or self._latest.seq == last_seq:
                return None
            return self._latest

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
        self.landmarker.close()


# ======================================================================================
# Smoothed, measured hands
# ======================================================================================
@dataclass
class Hand:
    label: str
    lm: np.ndarray              # (21, 3) smoothed landmarks in pixels (x, y, z scaled like x)
    palm_center: np.ndarray     # (3,)
    palm_size: float            # px, wrist -> middle knuckle
    normal: np.ndarray          # (3,) unit vector out of the palm; z < 0 = towards the camera
    roll: float                 # rad, hand rotation in the image plane (0 = fingers up)
    yaw: float                  # rad, palm turning left/right
    pitch: float                # rad, palm tilting up/down
    velocity: np.ndarray        # (2,) palm centre velocity, px/s
    tip_velocity: np.ndarray    # (5, 2) fingertip velocities, px/s
    score: float = 1.0
    raw_palm_center: np.ndarray | None = None   # unsmoothed palm centre (for fast gestures)

    @property
    def tips2d(self) -> np.ndarray:
        return self.lm[TIPS, :2]

    def point2d(self, idx: int) -> np.ndarray:
        return self.lm[idx, :2]


@dataclass
class _HandMemory:
    filt: OneEuroFilter
    last_seen: float
    prev_center: np.ndarray | None = None
    prev_tips: np.ndarray | None = None
    prev_t: float | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    tip_velocity: np.ndarray = field(default_factory=lambda: np.zeros((5, 2)))


class HandProcessor:
    """Turns raw MediaPipe hands into smoothed `Hand` objects keyed by "Left"/"Right"."""

    def __init__(self):
        self.memory: dict[str, _HandMemory] = {}

    def process(self, raw_hands: list[RawHand], width: int, height: int, t: float) -> dict[str, Hand]:
        scale = np.array([width, height, width], dtype=np.float32)
        out: dict[str, Hand] = {}
        for raw in raw_hands:
            mem = self.memory.get(raw.label)
            if mem is None or t - mem.last_seen > C.HAND_LOST_GRACE_S:
                mem = _HandMemory(OneEuroFilter(C.ONE_EURO_MIN_CUTOFF, C.ONE_EURO_BETA, C.ONE_EURO_D_CUTOFF), t)
                self.memory[raw.label] = mem
            mem.last_seen = t
            raw_px = raw.norm * scale
            lm = mem.filt(raw_px, t)
            hand = self._measure(raw.label, lm, mem, t, raw.score)
            hand.raw_palm_center = raw_px[PALM_POINTS].mean(axis=0)
            out[raw.label] = hand
        for label in list(self.memory):
            if t - self.memory[label].last_seen > C.HAND_LOST_GRACE_S:
                del self.memory[label]
        return out

    @staticmethod
    def _measure(label: str, lm: np.ndarray, mem: _HandMemory, t: float, score: float) -> Hand:
        palm_center = lm[PALM_POINTS].mean(axis=0)
        palm_size = float(max(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]), 1.0))

        # Palm normal from the cross product (index knuckle - wrist) x (pinky knuckle - wrist).
        # In the mirrored image this points *into* the palm for a right hand, so flip it for
        # right hands; the result points out of the palm for both (z < 0 = facing the camera).
        a = lm[INDEX_MCP] - lm[WRIST]
        b = lm[PINKY_MCP] - lm[WRIST]
        n = np.cross(a, b)
        if label == "Right":
            n = -n
        n = n / (np.linalg.norm(n) + 1e-6)

        up = lm[MIDDLE_MCP] - lm[WRIST]
        roll = math.atan2(float(up[0]), float(-up[1]))
        yaw = math.atan2(float(n[0]), float(-n[2]))
        pitch = math.atan2(float(n[1]), float(-n[2]))

        tips = lm[TIPS, :2]
        if mem.prev_t is not None:
            dt = max(t - mem.prev_t, 1e-3)
            v = (palm_center[:2] - mem.prev_center) / dt
            tv = (tips - mem.prev_tips) / dt
            mem.velocity = 0.5 * mem.velocity + 0.5 * v          # light smoothing of velocities
            mem.tip_velocity = 0.5 * mem.tip_velocity + 0.5 * tv
        mem.prev_center, mem.prev_tips, mem.prev_t = palm_center[:2].copy(), tips.copy(), t

        return Hand(label, lm, palm_center, palm_size, n, roll, yaw, pitch,
                    mem.velocity.copy(), mem.tip_velocity.copy(), score)
