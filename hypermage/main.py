"""
main.py — Hypermage: conjure and manipulate 4D objects with your bare hands.

Run:  python main.py
Keys: 1-7 object, S slice mode, B void, H skeleton, R reset, D debug, P screenshot,
      V record MP4, Q / Esc quit.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

import config as C
import geometry4d as g4
from effects import Effects
from gestures import GestureState, HandGestures
from scene import Scene
from tracking import Hand, HandProcessor, HandTracker, RawHand


class App:
    """Everything except the camera: gestures -> scene -> effects -> final image."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.fx = Effects()
        self.fx.ensure_buffers(height, width)
        self.scene = Scene(width, height)
        self.processor = HandProcessor()
        self.gestures = {"Left": HandGestures("Left"), "Right": HandGestures("Right")}
        self.out = np.zeros((height, width, 3), np.uint8)
        self.void = False
        self.skeleton = True
        self.debug = False
        self.writer: cv2.VideoWriter | None = None
        self.fps = 0.0
        self.last_t: float | None = None
        self.detect_ms = 0.0
        self.quit = False
        self.hands: dict[str, Hand] = {}
        self.gs: dict[str, GestureState] = {}
        self.scene.reset(self.fx, 0.0)

    # ---------------- one frame ----------------
    def step(self, frame: np.ndarray, raw_hands: list[RawHand], now: float) -> np.ndarray:
        dt = 1 / 30 if self.last_t is None else float(np.clip(now - self.last_t, 1e-3, 0.1))
        self.last_t = now
        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt

        self.hands = self.processor.process(raw_hands, self.w, self.h, now)
        self.gs = {k: g.update(self.hands.get(k), now) for k, g in self.gestures.items()}
        for k, g in self.gs.items():
            # routine pose names never overwrite an event message (CREATED, ERASED, ...) still showing
            if g.pose_entered and g.pose_entered != "pinch" and now - self.fx.label_t > 1.5:
                self.fx.announce(f"{k[0]}: {g.pose_entered.replace('_', ' ').upper()}", now)

        fx = self.fx
        fx.begin_frame(dt)
        self.scene.update(self.hands, self.gs, now, dt, fx)

        for label, hand in self.hands.items():
            color = C.COLOR_LEFT_HAND if label == "Left" else C.COLOR_RIGHT_HAND
            fx.sparks(hand, color)
            if self.skeleton:
                fx.draw_veins(hand, color, now)

        self.scene.draw(fx, now)
        fx.particles.draw(fx.fx)
        fx.compose(frame, self.out, self.void)
        fx.post(self.out)
        self._hud(now)
        if self.debug:
            self._debug_overlay()
        if self.writer is not None:
            self.writer.write(self.out)
        return self.out

    # ---------------- overlays ----------------
    def _hud(self, now: float) -> None:
        def pose(label):
            g = self.gs.get(label)
            return g.pose if g and g.present else "-"
        lines = [f"FPS: {self.fps:4.1f}   detect {self.detect_ms:4.1f} ms"]
        lines += self.scene.hud_lines()
        lines.append(f"L: {pose('Left'):10s} R: {pose('Right')}")
        if self.void:
            lines.append("VOID")
        if self.writer is not None:
            lines.append("REC")
        self.fx.draw_hud(self.out, lines, now)
        if self.writer is not None and int(now * 2) % 2 == 0:
            cv2.circle(self.out, (self.w - 30, 30), 10, (0, 0, 255), -1)

    def _debug_overlay(self) -> None:
        y = self.h - 12
        for label, hand in self.hands.items():
            for i, p in enumerate(hand.lm[:, :2]):
                cv2.putText(self.out, str(i), (int(p[0]) + 4, int(p[1]) - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
            g = self.gs[label]
            straight = " ".join(f"{s:+.2f}" for s in g.straightness)
            ext = "".join("1" if e else "0" for e in g.extended)
            txt = (f"{label[0]} pose={g.pose:9s} ext(TIMRP)={ext} straight=[{straight}] "
                   f"pinch={g.pinch_ratio:.2f} normal_z={g.normal_z:+.2f} roll={hand.roll:+.2f}")
            cv2.putText(self.out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
            y -= 20
        cv2.putText(self.out, f"objects={len(self.scene.objects)} weaving={len(self.scene.weaves)} "
                              f"strings={'on' if self.scene.strings else 'off'} particles={self.fx.particles.count}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

    # ---------------- keyboard ----------------
    def handle_key(self, key: int, now: float) -> None:
        if key < 0:
            return
        key &= 0xFF                              # drop modifier bits some platforms add
        if key in C.KEY_QUIT:
            self.quit = True
        elif ord("1") <= key <= ord(str(len(g4.SHAPE_BUILDERS))):
            self.scene.select(key - ord("1"), self.fx, now)
        elif key in C.KEY_SLICE:
            self.scene.toggle_slice(self.fx, now)
        elif key in C.KEY_VOID:
            self.void = not self.void
        elif key in C.KEY_SKELETON:
            self.skeleton = not self.skeleton
        elif key in C.KEY_RESET:
            self.scene.reset(self.fx, now)
            self.fx.announce("RESET", now)
        elif key in C.KEY_DEBUG:
            self.debug = not self.debug
        elif key in C.KEY_SCREENSHOT:
            path = self._capture_path("png")
            cv2.imwrite(path, self.out)
            self.fx.announce("SCREENSHOT SAVED", now)
            print("Saved", path)
        elif key in C.KEY_RECORD:
            self.toggle_recording(now)

    def toggle_recording(self, now: float) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.fx.announce("RECORDING SAVED", now)
            return
        path = self._capture_path("mp4")
        fps = float(np.clip(self.fps, 10, 60)) if self.fps else 30.0
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.w, self.h))
        if not writer.isOpened():
            self.fx.announce("CANNOT RECORD", now)
            return
        self.writer = writer
        self.fx.announce("RECORDING", now)
        print("Recording to", path)

    @staticmethod
    def _capture_path(ext: str) -> str:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.CAPTURE_DIR)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, time.strftime(f"hypermage_%Y%m%d_%H%M%S.{ext}"))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def main() -> int:
    try:
        tracker = HandTracker()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1

    app = App(tracker.width, tracker.height)
    cv2.namedWindow(C.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(C.WINDOW_NAME, tracker.width, tracker.height)
    if C.FULLSCREEN:
        cv2.setWindowProperty(C.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print(f"Camera {tracker.width}x{tracker.height}. Show your hands! Q or Esc to quit.")

    seq = 0
    try:
        while not app.quit:
            item = tracker.latest(seq)
            if item is None:                    # no new frame yet; keep the window responsive
                app.handle_key(cv2.waitKey(1), time.monotonic())
                continue
            seq = item.seq
            app.detect_ms = tracker.detect_ms
            out = app.step(item.frame, item.hands, item.timestamp)
            cv2.imshow(C.WINDOW_NAME, out)
            app.handle_key(cv2.waitKey(1), item.timestamp)
            if cv2.getWindowProperty(C.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    finally:
        app.close()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
