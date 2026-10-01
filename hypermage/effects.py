"""
effects.py — everything that glows: neon edge drawing, particles, light trails, object ghost
trails, bloom, lightning tethers, screen shake, chromatic
aberration, hand "energy veins" and the HUD.

Rendering model: several black uint8 layers are drawn into with OpenCV and combined
additively over the darkened webcam image. A blurred, downscaled copy of the glowing layers
is added on top as bloom. All buffers are allocated once and reused every frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import config as C
from tracking import HAND_CONNECTIONS, Hand

AA = cv2.LINE_AA

# Colour table for w-depth: COLOR_BINS steps from cyan (w<0) to magenta (w>0)
_K = C.COLOR_BINS
W_COLORS = np.array([np.array(C.COLOR_W_NEG) * (1 - t) + np.array(C.COLOR_W_POS) * t
                     for t in np.linspace(0, 1, _K)], dtype=np.float32)


def _fade_lut(factor: float, minus: int) -> np.ndarray:
    """Lookup table that multiplies by `factor` and subtracts `minus`, so faint pixels reach 0
    (plain multiplication rounds 1 back up to 1 forever and leaves a haze)."""
    return np.clip(np.rint(np.arange(256) * factor) - minus, 0, 255).astype(np.uint8)


def w_to_color(w: np.ndarray) -> np.ndarray:
    """(N,) w in about [-1, 1] -> (N, 3) float colours."""
    idx = np.clip(np.rint((w + 1) * 0.5 * (_K - 1)), 0, _K - 1).astype(np.int32)
    return W_COLORS[idx]


def draw_segments(img: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray, near: np.ndarray,
                  alpha: float = 1.0, thick: float = 1.0, color=None) -> None:
    """
    Draw many 2D segments at once, coloured by w and dimmed/thinned when far away.
    Segments are grouped into a few colour/thickness buckets and each bucket is one
    cv2.polylines call, so there is no per-edge Python loop.
    """
    if len(a) == 0 or alpha <= 0.01:
        return
    if color is None:
        cbin = np.clip(np.rint((w + 1) * 0.5 * (_K - 1)), 0, _K - 1).astype(np.int32)
    else:
        cbin = np.zeros(len(a), np.int32)
    close = (near >= 0.5).astype(np.int32)
    key = cbin * 2 + close
    pts = np.stack([a, b], axis=1)
    finite = np.isfinite(pts).all(axis=(1, 2)) & (np.abs(pts) < 1e5).all(axis=(1, 2))
    pts = np.rint(pts).astype(np.int32)
    for k in np.unique(key[finite]):
        m = (key == k) & finite
        base = np.array(color, np.float32) if color is not None else W_COLORS[k // 2]
        bright = 1.0 if k % 2 else 0.5
        col = tuple(float(c) for c in base * bright * alpha)
        th = (2 if k % 2 else 1) * thick
        # Thick anti-aliased lines cost ~4x thin ones. Very heavy objects (5D/6D prisms) draw
        # every edge 1 px thinner; bloom keeps them glowing.
        if len(a) >= C.HEAVY_EDGE_COUNT:
            th -= 1
        cv2.polylines(img, pts[m], False, col, max(1, int(round(th))), AA)


def splat(img: np.ndarray, xy: np.ndarray, colors: np.ndarray) -> None:
    """Draw many small glowing dots (a bright centre pixel plus a dimmer cross) at once."""
    if len(xy) == 0:
        return
    h, w = img.shape[:2]
    xi = np.rint(xy[:, 0]).astype(np.int32)
    yi = np.rint(xy[:, 1]).astype(np.int32)
    cols = np.clip(colors, 0, 255).astype(np.uint8)
    half = (cols // 2)
    for (dx, dy), c in (((0, 0), cols), ((1, 0), half), ((-1, 0), half), ((0, 1), half), ((0, -1), half)):
        x, y = xi + dx, yi + dy
        m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        img[y[m], x[m]] = np.maximum(img[y[m], x[m]], c[m])


# ======================================================================================
# Particles
# ======================================================================================
class Particles:
    """Fixed-size particle pool (ring buffer) updated and drawn with NumPy."""

    def __init__(self, capacity: int = C.MAX_PARTICLES):
        self.n = capacity
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.max_life = np.ones(capacity, np.float32)
        self.color = np.zeros((capacity, 3), np.float32)
        self.drag = np.zeros(capacity, np.float32)
        self.head = 0

    def emit(self, pos, vel, life, color, drag=0.0) -> None:
        pos = np.asarray(pos, np.float32).reshape(-1, 2)
        m = len(pos)
        if m == 0:
            return
        m = min(m, self.n)
        idx = (self.head + np.arange(m)) % self.n
        self.head = int((self.head + m) % self.n)
        self.pos[idx] = pos[:m]
        self.vel[idx] = np.broadcast_to(np.asarray(vel, np.float32), (len(pos), 2))[:m]
        lf = np.broadcast_to(np.asarray(life, np.float32), (len(pos),))[:m]
        self.life[idx] = lf
        self.max_life[idx] = np.maximum(lf, 1e-3)
        self.color[idx] = np.broadcast_to(np.asarray(color, np.float32), (len(pos), 3))[:m]
        self.drag[idx] = np.broadcast_to(np.asarray(drag, np.float32), (len(pos),))[:m]

    def update(self, dt: float) -> None:
        alive = self.life > 0
        if not alive.any():
            return
        self.pos[alive] += self.vel[alive] * dt
        self.vel[alive] *= np.exp(-self.drag[alive] * dt)[:, None]
        self.life[alive] -= dt

    def draw(self, img: np.ndarray) -> None:
        alive = self.life > 0
        if not alive.any():
            return
        fade = (self.life[alive] / self.max_life[alive])[:, None]
        splat(img, self.pos[alive], self.color[alive] * np.sqrt(fade))

    def clear(self) -> None:
        self.life[:] = 0

    @property
    def count(self) -> int:
        return int((self.life > 0).sum())


# ======================================================================================
# The effects manager
# ======================================================================================
class Effects:
    def __init__(self):
        self.size = None
        self.particles = Particles()
        self.shake = 0.0
        self.aberration = 0.0
        self.label_text = ""
        self.label_t = -10.0
        self.rng = np.random.default_rng()

    # ---------------- buffers ----------------
    def ensure_buffers(self, h: int, w: int) -> None:
        if self.size == (h, w):
            return
        self.size = (h, w)
        z = lambda: np.zeros((h, w, 3), np.uint8)                    # noqa: E731
        self.fx = z()          # this frame's effects (particles, veins, strings, tethers, weaves)
        self.obj = z()         # this frame's 4D objects
        self.ghost = z()       # decaying copy of `obj` -> motion trails
        self.light = z()       # pointing light trails (fade over ~2 s)
        self.glow = z()        # fx + ghost + light
        self.tmp = z()
        ds = C.BLOOM_DOWNSCALE
        self.small = np.zeros((h // ds, w // ds, 3), np.uint8)
        self.bloom = z()

    def begin_frame(self, dt: float) -> None:
        self.fx.fill(0)
        self.obj.fill(0)
        # light trails fade to ~4 % over LIGHT_TRAIL_SECONDS
        cv2.LUT(self.light, _fade_lut(0.04 ** (dt / C.LIGHT_TRAIL_SECONDS), 1), dst=self.light)
        self.particles.update(dt)
        self.shake *= math.exp(-C.SHAKE_DECAY * dt)
        self.aberration *= math.exp(-C.SHAKE_DECAY * dt)

    # ---------------- spawners ----------------
    def announce(self, text: str, now: float) -> None:
        self.label_text, self.label_t = text, now

    def sparks(self, hand: Hand, color) -> None:
        speed = np.linalg.norm(hand.tip_velocity, axis=1)
        rate = np.minimum(C.SPARKS_BASE + speed * C.SPARKS_PER_SPEED, C.SPARKS_MAX)
        counts = (rate + self.rng.random(5)).astype(np.int32)          # stochastic rounding
        total = int(counts.sum())
        if total == 0:
            return
        src = np.repeat(hand.tips2d, counts, axis=0)
        tv = np.repeat(hand.tip_velocity, counts, axis=0)
        vel = self.rng.normal(0, 70, (total, 2)) + tv * 0.15
        life = self.rng.uniform(0.25, 0.65, total)
        col = np.array(color, np.float32) * self.rng.uniform(0.6, 1.0, (total, 1))
        self.particles.emit(src + self.rng.normal(0, 2, (total, 2)), vel, life, col, drag=3.0)

    def converge(self, targets: np.ndarray, colors: np.ndarray, duration: float) -> None:
        """Particles flying in from all around to land exactly on `targets` (materialize)."""
        n = len(targets)
        if n == 0:
            return
        ang = self.rng.uniform(0, 2 * np.pi, n)
        dist = self.rng.uniform(60, 220, n)
        start = targets + np.stack([np.cos(ang), np.sin(ang)], 1) * dist[:, None]
        t = self.rng.uniform(0.6, 1.0, n) * duration
        self.particles.emit(start, (targets - start) / t[:, None], t, colors, drag=0.0)

    def explode(self, points: np.ndarray, colors: np.ndarray, center: np.ndarray, power: float = 1.0) -> None:
        n = len(points)
        if n == 0:
            return
        out = points - center
        vel = out * self.rng.uniform(1.5, 4.5, (n, 1)) * power + self.rng.normal(0, 90, (n, 2)) * power
        self.particles.emit(points, vel, self.rng.uniform(0.7, 1.6, n), colors, drag=1.6)

    def kick(self, strength: float) -> None:
        self.shake = max(self.shake, C.SHAKE_MAX_PX * strength)
        self.aberration = max(self.aberration, C.ABERRATION_MAX_PX * strength)

    def trail(self, p0: np.ndarray, p1: np.ndarray) -> None:
        a = tuple(int(v) for v in np.rint(p0))
        b = tuple(int(v) for v in np.rint(p1))
        cv2.line(self.light, a, b, C.COLOR_TRAIL, 7, AA)
        cv2.line(self.light, a, b, (200, 245, 255), 2, AA)

    # ---------------- drawing into the fx layer ----------------
    def draw_veins(self, hand: Hand, color, now: float) -> None:
        """Hand skeleton as faint flickering energy veins."""
        flick = 0.55 + 0.25 * math.sin(now * 9.0) + 0.2 * self.rng.random()
        p = hand.lm[:, :2]
        segs = np.rint(np.stack([p[HAND_CONNECTIONS[:, 0]], p[HAND_CONNECTIONS[:, 1]]], 1)).astype(np.int32)
        col = tuple(float(c) * 0.45 * flick for c in color)
        cv2.polylines(self.fx, segs, False, col, 3, AA)
        core = tuple(min(255.0, float(c) * 0.5 + 90) * flick for c in color)
        cv2.polylines(self.fx, segs, False, core, 1, AA)
        splat(self.fx, p, np.tile(np.array(core, np.float32), (21, 1)))

    def bolt(self, p0: np.ndarray, p1: np.ndarray, color, jitter: float = 14.0, segments: int = 10,
             thickness: int = 1) -> None:
        """Jagged lightning between two points (fresh random shape every frame)."""
        t = np.linspace(0, 1, segments + 1)[:, None]
        d = p1 - p0
        length = float(np.linalg.norm(d)) + 1e-6
        perp = np.array([-d[1], d[0]]) / length
        off = self.rng.normal(0, jitter, (segments + 1, 1)) * np.sin(np.pi * t)
        pts = p0 + d * t + perp * off
        cv2.polylines(self.fx, [np.rint(pts).astype(np.int32)], False, color, thickness, AA)

    # ---------------- final composition ----------------
    def compose(self, frame: np.ndarray, out: np.ndarray, void: bool) -> None:
        # 1) background
        if void:
            out.fill(0)
        else:
            cv2.convertScaleAbs(frame, dst=out, alpha=C.BACKGROUND_DIM)

        # 2) objects leave decaying ghost frames
        cv2.LUT(self.ghost, _fade_lut(C.OBJECT_TRAIL_DECAY, 2), dst=self.ghost)
        cv2.max(self.ghost, self.obj, dst=self.ghost)

        # 3) everything that glows
        cv2.add(self.fx, self.ghost, dst=self.glow)
        cv2.add(self.glow, self.light, dst=self.glow)

        # 4) bloom: blur a small copy and add it back
        h, w = out.shape[:2]
        cv2.resize(self.glow, (self.small.shape[1], self.small.shape[0]), dst=self.small,
                   interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(self.small, (0, 0), C.BLOOM_SIGMA, dst=self.small)
        cv2.resize(self.small, (w, h), dst=self.bloom, interpolation=cv2.INTER_LINEAR)
        cv2.add(out, self.glow, dst=out)
        cv2.addWeighted(out, 1.0, self.bloom, C.BLOOM_STRENGTH, 0, dst=out)

    def post(self, out: np.ndarray) -> None:
        """Screen shake and chromatic aberration (on creation, erasing and dimension jumps)."""
        h, w = out.shape[:2]
        if self.shake > 0.5:
            dx, dy = self.rng.normal(0, self.shake, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            cv2.warpAffine(out, M, (w, h), dst=self.tmp, borderMode=cv2.BORDER_REFLECT)
            np.copyto(out, self.tmp)
        k = int(round(self.aberration))
        if k >= 1:
            np.copyto(self.tmp, out)
            out[:, k:, 0] = self.tmp[:, :-k, 0]       # blue shifted right
            out[:, :-k, 2] = self.tmp[:, k:, 2]       # red shifted left

    # ---------------- HUD ----------------
    def draw_hud(self, out: np.ndarray, lines: list[str], now: float) -> None:
        x, y = 14, 26
        box_h = 10 + 22 * len(lines)
        roi = out[0:box_h, 0:330]
        roi[:] = (roi * 0.45).astype(np.uint8)
        for line in lines:
            cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C.COLOR_HUD, 1, AA)
            y += 22
        # recognised-gesture label: fade in 0.15 s, hold, fade out
        age = now - self.label_t
        if self.label_text and age < 1.8:
            a = min(1.0, age / 0.15) * (1.0 if age < 1.2 else max(0.0, 1 - (age - 1.2) / 0.6))
            text = self.label_text
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
            px = (out.shape[1] - size[0]) // 2
            col = tuple(v * a for v in (255, 230, 250))
            cv2.putText(out, text, (px, 60), cv2.FONT_HERSHEY_DUPLEX, 1.1, col, 2, AA)
