"""
gestures.py — per-hand gesture recognition with hysteresis and minimum hold times.

Features measured on each hand:
  * finger extension from joint angles: cosine of the bend at the middle joint
    (MCP→PIP vs PIP→TIP), 1 = straight, ≤0 = bent 90° or more
  * pinch: thumb-tip to index-tip distance ÷ palm size
  * palm normal (from tracking.py): which way the palm faces

Poses produced: "open", "three", "fist", "point", "peace", "thumbs_up", "pinch", "none".
Pinch is tracked separately too (it has start/end events for grabbing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config as C
from tracking import (Hand, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
                      RING_MCP, RING_PIP, RING_TIP, PINKY_MCP, PINKY_PIP, PINKY_TIP,
                      THUMB_MCP, THUMB_IP, THUMB_TIP)

FINGERS = [  # (MCP, PIP, TIP) for index..pinky
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


@dataclass
class GestureState:
    """What one hand is doing this frame (read by scene.py and the HUD)."""
    present: bool = False
    pose: str = "none"                  # debounced pose
    pose_time: float = 0.0              # seconds the current pose has been held
    pose_entered: str | None = None     # set on the frame a new pose becomes active
    pinch: bool = False
    pinch_raw: bool = False             # fingers touching this very frame (no hold time)
    pinch_started: bool = False
    pinch_ended: bool = False
    pinch_point: np.ndarray = field(default_factory=lambda: np.zeros(2))
    facing_camera: bool = False
    extended: list[bool] = field(default_factory=lambda: [False] * 5)   # thumb..pinky
    # raw numbers for the debug overlay
    straightness: list[float] = field(default_factory=lambda: [0.0] * 5)
    pinch_ratio: float = 9.9
    normal_z: float = 0.0


class HandGestures:
    """Gesture state machine for one hand ("Left" or "Right")."""

    def __init__(self, label: str):
        self.label = label
        self.extended = [False] * 5
        self.pinch_raw = False
        self.pinch_on = False
        self.pinch_change_since: float | None = None
        self.pose = "none"
        self.pose_since = 0.0
        self.candidate = "none"
        self.candidate_since = 0.0

    def reset(self) -> None:
        self.__init__(self.label)

    # ---------------- features ----------------
    def _finger_states(self, lm: np.ndarray, palm: float) -> tuple[list[bool], list[float]]:
        straight = []
        # Thumb: bend at the IP joint, and the tip must reach away from the index knuckle.
        t_cos = _cos(lm[THUMB_IP] - lm[THUMB_MCP], lm[THUMB_TIP] - lm[THUMB_IP])
        reach = np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_MCP]) / palm
        straight.append(t_cos)
        thumb_on = t_cos > C.THUMB_EXTEND_ON and reach > C.THUMB_MIN_REACH
        thumb_keep = t_cos > C.THUMB_EXTEND_OFF and reach > C.THUMB_MIN_REACH * 0.8
        ext = [thumb_on or (self.extended[0] and thumb_keep)]
        for k, (mcp, pip, tip) in enumerate(FINGERS, start=1):
            c = _cos(lm[pip] - lm[mcp], lm[tip] - lm[pip])
            straight.append(c)
            # hysteresis: easy to stay in the current state, harder to switch
            ext.append(c > C.FINGER_EXTEND_ON or (self.extended[k] and c > C.FINGER_EXTEND_OFF))
        return ext, straight

    def _classify(self, hand: Hand, ext: list[bool]) -> str:
        thumb, index, middle, ring, pinky = ext
        if index and middle and ring and pinky:
            return "open"
        if index and middle and ring and not pinky:
            return "three"
        if index and middle and not ring and not pinky:
            return "peace"
        if index and not middle and not ring and not pinky:
            return "point"
        if not (index or middle or ring or pinky):
            if thumb:
                d = hand.lm[THUMB_TIP, :2] - hand.lm[THUMB_MCP, :2]
                up = -d[1] / (np.linalg.norm(d) + 1e-6)
                if up > C.THUMBS_UP_MIN_UP:
                    return "thumbs_up"
            return "fist"
        return "none"

    # ---------------- per-frame update ----------------
    def update(self, hand: Hand | None, now: float) -> GestureState:
        s = GestureState()
        if hand is None:
            if self.pinch_on:
                s.pinch_ended = True
            self.reset()
            return s

        lm, palm = hand.lm, hand.palm_size
        s.present = True
        s.normal_z = float(hand.normal[2])
        s.facing_camera = s.normal_z < C.FACING_CAMERA_Z

        # Finger extension
        self.extended, s.straightness = self._finger_states(lm, palm)
        s.extended = list(self.extended)

        # Pinch with hysteresis on distance + minimum enter/exit times
        ratio = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]) / palm)
        s.pinch_ratio = ratio
        s.pinch_point = (lm[THUMB_TIP, :2] + lm[INDEX_TIP, :2]) / 2
        reach = np.linalg.norm(((lm[THUMB_TIP] + lm[INDEX_TIP]) / 2) - hand.palm_center) / palm
        raw = ratio < (C.PINCH_OFF if self.pinch_raw else C.PINCH_ON) and reach > C.PINCH_MIN_REACH
        self.pinch_raw = raw
        if raw != self.pinch_on:
            if self.pinch_change_since is None:
                self.pinch_change_since = now
            need = C.PINCH_ENTER_S if raw else C.PINCH_EXIT_S
            if now - self.pinch_change_since >= need:
                self.pinch_on = raw
                self.pinch_change_since = None
                if raw:
                    s.pinch_started = True
                else:
                    s.pinch_ended = True
        else:
            self.pinch_change_since = None
        s.pinch = self.pinch_on
        s.pinch_raw = raw

        # Pose with debouncing
        cand = "pinch" if (self.pinch_on or self.pinch_raw) else self._classify(hand, self.extended)
        if cand != self.candidate:
            self.candidate, self.candidate_since = cand, now
        hold = 0.0 if cand == "pinch" else C.POSE_HOLD_S
        if self.candidate != self.pose and now - self.candidate_since >= hold:
            self.pose, self.pose_since = self.candidate, now
            if self.pose != "none":
                s.pose_entered = self.pose
        s.pose = self.pose
        s.pose_time = now - self.pose_since
        return s
