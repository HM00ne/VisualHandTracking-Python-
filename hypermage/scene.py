"""
scene.py — the 4D objects, their physics, and how gestures act on them
(string figure + double peace sign to create, two fists to erase, grab, rotate, scale, fling,
W-twist, slice, telekinesis, three fingers to change dimension).
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

import config as C
import geometry4d as g4
from effects import Effects, draw_segments, splat, w_to_color
from gestures import GestureState
from tracking import Hand, INDEX_TIP


def _wrap(a: float) -> float:
    """Wrap an angle difference into [-π, π)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _deadzone(x: float, dz: float) -> float:
    if abs(x) < dz:
        return 0.0
    return math.copysign((abs(x) - dz) / (1 - dz), x)


class Obj4D:
    """
    One floating object of dimension 3-6: its shape, an angle + angular velocity for each of the
    15 rotation planes of 6D space (only the planes inside its own dimension are used), 2D position.
    """

    def __init__(self, shape_index: int, pos, now: float, rng: np.random.Generator, dim: int = 4):
        self.shape_index = shape_index
        self.dim = dim
        self.shape = g4.get_shape(shape_index, dim)
        self.angles = rng.uniform(-0.6, 0.6, g4.N_PLANES)
        self.angvel = rng.normal(0, C.IDLE_SPIN, g4.N_PLANES)
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2)
        self.scale = float(C.OBJECT_SCALE)
        self.target_scale = float(C.OBJECT_SCALE)
        self.spawn_t = now
        self.dead = False
        # filled by project(): screen-space edges, reused for hit tests and explosions
        self.seg_a = self.seg_b = None
        self.seg_w = self.seg_near = None

    def set_dim(self, dim: int, now: float) -> None:
        self.dim = dim
        self.shape = g4.get_shape(self.shape_index, dim)
        self.spawn_t = now                                     # replay the materialize animation

    @property
    def radius(self) -> float:
        return self.scale * 1.15

    def visible(self, now: float) -> bool:
        return not self.dead and now >= self.spawn_t           # hidden while it's being woven

    def rotated(self) -> np.ndarray:
        return g4.rotate(self.shape.verts, self.angles)

    def to_screen(self, p3: np.ndarray):
        """3D points -> screen pixels (y down) and a 0..1 'nearness' per point."""
        p2, z = g4.project_3d_to_2d(p3, C.PROJ_DIST_3D)
        scr = self.pos + p2 * np.array([1.0, -1.0]) * self.scale
        return scr, z

    def project(self) -> None:
        """Full 4D -> 3D -> 2D projection of all edges (stored on the object)."""
        v = self.rotated()
        p3, w = g4.project_nd_to_3d(v, C.PROJ_DIST_4D)
        scr, z = self.to_screen(p3)
        near = np.clip(0.5 + 0.3 * z + 0.2 * w, 0, 1)        # closer in z and w = brighter/thicker
        e = self.shape.edges
        self.seg_a, self.seg_b = scr[e[:, 0]], scr[e[:, 1]]
        self.seg_w = (w[e[:, 0]] + w[e[:, 1]]) * 0.5
        self.seg_near = (near[e[:, 0]] + near[e[:, 1]]) * 0.5

    def sample_points(self, per_edge: int, rng: np.random.Generator, cap: int = 900):
        """Random points along the projected edges, with their colours (for particle effects)."""
        if self.seg_a is None:
            self.project()
        n = len(self.seg_a)
        idx = np.repeat(np.arange(n), per_edge)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        t = rng.random(len(idx))[:, None]
        pts = self.seg_a[idx] + (self.seg_b[idx] - self.seg_a[idx]) * t
        return pts, w_to_color(self.seg_w[idx])


class StringFigure:
    """
    A cat's-cradle of glowing strings stretched between the ten fingertips of both hands.
    The ends are pinned to your fingertips; everything in between winds through 4D (each string
    follows a torus knot on the Clifford torus), is rotated in all six planes and projected
    like the other objects, so the strings fold and change colour as they pass through w.
    """
    # fingertip indices: 0-4 = left thumb..pinky, 5-9 = right thumb..pinky
    PAIRS = np.array([(i, 5 + i) for i in range(5)]                 # straight across
                     + [(i, 6 + i) for i in range(4)]               # crossing one way
                     + [(i + 1, 5 + i) for i in range(4)]           # crossing the other way
                     + [(i, i + 1) for i in range(4)]               # loops over the left fingers
                     + [(5 + i, 6 + i) for i in range(4)])          # loops over the right fingers

    def __init__(self, now: float, rng: np.random.Generator):
        self.t0 = now
        self.last_seen = now
        self.angles = np.zeros(g4.N_PLANES)
        self.angles[[g4.XY, g4.XZ, g4.XW, g4.YZ, g4.YW, g4.ZW]] = rng.uniform(-0.5, 0.5, 6)
        self.spin = np.zeros(g4.N_PLANES)
        self.spin[[g4.XY, g4.XZ, g4.XW, g4.YZ, g4.YW, g4.ZW]] = C.STRING_SPIN
        self.phase = 0.0
        self.energy = 0.0
        self.anchors: np.ndarray | None = None
        self.u = np.linspace(0, 1, C.STRING_SAMPLES).astype(np.float32)
        self.envelope = np.sin(np.pi * self.u)                       # 0 at the fingertips

    def update(self, left: Hand | None, right: Hand | None, now: float, dt: float) -> None:
        if left is not None and right is not None:
            self.anchors = np.vstack([left.tips2d, right.tips2d])
            self.last_seen = now
            speed = float(np.mean(np.linalg.norm(np.vstack([left.tip_velocity, right.tip_velocity]), axis=1)))
            self.energy += (min(speed / 1500.0, 1.0) - self.energy) * min(1.0, dt * 4)
        # moving your hands makes the strings vibrate and spin faster
        self.angles += self.spin * dt * (1 + 2 * self.energy)
        self.phase += dt * (1.2 + 3 * self.energy)

    def lost_for(self, now: float) -> float:
        return now - self.last_seen

    def geometry(self):
        """Returns points (S, K, 2), w (S, K) and nearness (S, K) for all strings."""
        A, B = self.anchors[self.PAIRS[:, 0]], self.anchors[self.PAIRS[:, 1]]
        S, K = len(A), len(self.u)
        base = A[:, None, :] + (B - A)[:, None, :] * self.u[None, :, None]
        lift = g4.rotate(g4.string_lift(S, self.u, self.phase).reshape(-1, 4), self.angles)
        p3, w = g4.project_nd_to_3d(lift, C.PROJ_DIST_4D)
        p2, z = g4.project_3d_to_2d(p3, C.PROJ_DIST_3D)
        length = np.linalg.norm(B - A, axis=1)
        amp = (C.STRING_AMPLITUDE * length * (1 + self.energy) + 6)[:, None, None]
        disp = (p2 * np.array([1.0, -1.0])).reshape(S, K, 2) * self.envelope[None, :, None] * amp
        near = np.clip(0.5 + 0.3 * z + 0.2 * w, 0, 1).reshape(S, K)
        return base + disp, (w * 2.2).reshape(S, K), near             # w scaled up: full colour range

    def draw(self, fx: Effects, now: float) -> None:
        if self.anchors is None:
            return
        pts, w, near = self.geometry()
        a, b = pts[:, :-1].reshape(-1, 2), pts[:, 1:].reshape(-1, 2)
        ws = ((w[:, :-1] + w[:, 1:]) / 2).ravel()
        ns = ((near[:, :-1] + near[:, 1:]) / 2).ravel()
        fade_in = min(1.0, (now - self.t0) / 0.4)
        fade_out = max(0.0, 1 - max(0.0, self.lost_for(now) - 0.05) / C.STRING_LOST_S)
        # drawn on the effects layer (no ghost trails), so the fast-moving strings stay crisp
        draw_segments(fx.fx, a, b, ws, ns, alpha=fade_in * fade_out, thick=1.0)
        splat(fx.fx, self.anchors, np.full((10, 3), 255.0 * fade_in))

    def sample_points(self):
        pts, w, _ = self.geometry()
        return pts.reshape(-1, 2), w_to_color(w.ravel())


class Scene:
    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.rng = np.random.default_rng()
        self.dom = C.DOMINANT_HAND
        self.off = "Left" if self.dom == "Right" else "Right"
        self.reset_state()

    # ======================================================================================
    # state
    # ======================================================================================
    def reset_state(self, now: float = 0.0) -> None:
        self.objects: list[Obj4D] = []
        self.selected = 0
        self.slice_mode = False
        self.slice_w = 0.0
        self.last_obj: Obj4D | None = None
        self.grabs: dict[str, dict] = {}
        self.pair: dict | None = None             # two-hand scaling
        self.prev_orient: dict[str, tuple] = {}
        self.prev_tip: np.ndarray | None = None   # pointing trail
        self.thumbs_latched = {"Left": False, "Right": False}
        self.tele: dict[str, dict] = {}             # peace-sign telekinesis, per hand
        self.ftouch = {"since": None, "armed": True}                # fingertips touching -> strings
        self.strings: StringFigure | None = None
        self.last_fling = self.last_dim = -10.0
        self.erase = {"since": None, "armed": True}                  # both fists -> erase everything
        self.weaves: list[dict] = []                                 # strings weaving into new objects

    def reset(self, fx: Effects, now: float) -> None:
        self.reset_state(now)
        fx.particles.clear()
        fx.light.fill(0)
        fx.ghost.fill(0)      # the scene starts empty: strings + double peace sign create objects

    def select(self, index: int, fx: Effects, now: float) -> None:
        self.selected = index % len(g4.SHAPE_BUILDERS)
        fx.announce(g4.SHAPE_NAMES[self.selected], now)

    def toggle_slice(self, fx: Effects, now: float) -> None:
        self.slice_mode = not self.slice_mode
        fx.announce("SLICE MODE" if self.slice_mode else "PROJECTION MODE", now)

    # ======================================================================================
    # spawning / destruction
    # ======================================================================================
    def spawn(self, pos: np.ndarray, fx: Effects, now: float, materialize: bool = True) -> Obj4D:
        live = [o for o in self.objects if not o.dead]
        if len(live) >= C.MAX_OBJECTS:                          # make room: oldest one bursts
            self.explode(live[0], fx, power=0.7)
        obj = Obj4D(self.selected, pos, now, self.rng, dim=C.START_DIMENSION)
        self.objects.append(obj)
        self.last_obj = obj
        if materialize:
            self._materialize_particles(obj, fx)
        return obj

    def _materialize_particles(self, obj: Obj4D, fx: Effects) -> None:
        obj.project()
        pts, cols = obj.sample_points(3, self.rng, cap=700)
        fx.converge(pts, cols, C.MATERIALIZE_S * 0.85)

    def explode(self, obj: Obj4D, fx: Effects, power: float = 1.0) -> None:
        pts, cols = obj.sample_points(4, self.rng, cap=1100)
        fx.explode(pts, cols, obj.pos, power)
        for label in [k for k, gr in self.grabs.items() if gr["obj"] is obj]:
            del self.grabs[label]
        if self.pair and self.pair["obj"] is obj:
            self.pair = None
        obj.dead = True

    def hit_test(self, p: np.ndarray, now: float) -> Obj4D | None:
        best, best_d = None, 1e9
        for o in self.objects:
            if not o.visible(now):
                continue
            d = float(np.linalg.norm(o.pos - p))
            if d < o.scale * C.GRAB_RADIUS and d < best_d:
                best, best_d = o, d
        return best

    # ======================================================================================
    # per-frame update
    # ======================================================================================
    def update(self, hands: dict[str, Hand], gs: dict[str, GestureState], now: float, dt: float,
               fx: Effects) -> None:
        dom_h, off_h = hands.get(self.dom), hands.get(self.off)
        dom_g, off_g = gs[self.dom], gs[self.off]

        self._thumbs_up(gs, fx, now)
        self._pinches(hands, gs, fx, now)
        targets = self._grabs(hands, gs, now, dt)
        self._weave_create(hands, gs, fx, now)     # strings + peace sign on both hands -> new object
        if self.strings is None:                   # hands holding strings aren't doing telekinesis
            self._telekinesis(hands, gs, targets, fx, now)
        self._springs(targets, dt)
        self._finger_touch(dom_h, off_h, dom_g, off_g, fx, now)
        self._fist_erase(hands, gs, fx, now)
        if off_h is not None and off_g.pose == "open":
            self._w_control(off_h, dt)
        self._pointing(dom_h, dom_g, fx)
        if dom_h is not None and dom_g.pose == "open" and dom_g.facing_camera:
            self._fling(dom_h, fx, now)
        self._dimension(hands, gs, fx, now)
        if self.strings is not None:
            self.strings.update(hands.get("Left"), hands.get("Right"), now, dt)
            if self.strings.lost_for(now) > C.STRING_LOST_S:
                self._dissolve_strings(fx, now)
        self._physics(now, dt, fx)

    # ---------------- gestures -> actions ----------------
    def _thumbs_up(self, gs, fx, now) -> None:
        held = {k: g.pose == "thumbs_up" and g.pose_time >= C.THUMBS_UP_HOLD_S for k, g in gs.items()}
        for k, g in gs.items():
            if g.pose != "thumbs_up":
                self.thumbs_latched[k] = False
        ready = any(held.values()) if C.THUMBS_UP_EITHER_HAND else all(held.values())
        if ready and not any(self.thumbs_latched[k] for k in held if held[k]):
            self.toggle_slice(fx, now)
            for k in held:
                if held[k]:
                    self.thumbs_latched[k] = True

    def _pinches(self, hands, gs, fx, now) -> None:
        for label, g in gs.items():
            if g.pinch_ended and label in self.grabs:
                obj = self.grabs.pop(label)["obj"]
                if self.pair and self.pair["obj"] is obj:
                    self.pair = None
                    # the hand still holding it keeps it without a jump
                    for other, gr in self.grabs.items():
                        if gr["obj"] is obj and other in hands:
                            gr["offset"] = obj.pos - gs[other].pinch_point
            if not g.pinch_started:
                continue
            p = g.pinch_point.copy()
            obj = self.hit_test(p, now)
            if obj is not None:
                self.grabs[label] = {"obj": obj, "offset": obj.pos - p}
                self.last_obj = obj
                self.prev_orient.pop(label, None)
                holders = [k for k, gr in self.grabs.items() if gr["obj"] is obj]
                if len(holders) == 2:
                    pa, pb = gs[holders[0]].pinch_point, gs[holders[1]].pinch_point
                    mid = (pa + pb) / 2
                    d = pb - pa
                    self.pair = {"obj": obj, "d0": max(float(np.linalg.norm(d)), 20.0),
                                 "s0": obj.target_scale, "offset": obj.pos - mid,
                                 "ang0": math.atan2(d[1], d[0]), "xy0": obj.angles[g4.XY],
                                 "labels": holders}
                    fx.announce("SCALE", now)
                else:
                    fx.announce("GRAB", now)

    def _grabs(self, hands: dict[str, Hand], gs: dict[str, GestureState], now: float, dt: float) -> dict:
        """Work out where every held object should go. Returns {id(obj): (obj, target, k, damping)}."""
        # drop grabs whose hand is gone (pinch_ended already handled most of these)
        for label in [k for k in self.grabs if k not in hands]:
            del self.grabs[label]
        if self.pair and any(k not in self.grabs for k in self.pair["labels"]):
            self.pair = None

        # While the fingers are opening (release grace period) the point between them jumps, so
        # each grab follows its last *touching* pinch point instead -> no yank on release.
        held_at = {}
        for label, gr in self.grabs.items():
            if gs[label].pinch_raw or "last_p" not in gr:
                gr["last_p"] = self._pinch_point(hands[label])
            held_at[label] = gr["last_p"]

        targets: dict[int, np.ndarray] = {}
        if self.pair:
            obj = self.pair["obj"]
            pa = held_at[self.pair["labels"][0]]
            pb = held_at[self.pair["labels"][1]]
            d = pb - pa
            dist = max(float(np.linalg.norm(d)), 1.0)
            obj.target_scale = float(np.clip(self.pair["s0"] * dist / self.pair["d0"],
                                             C.OBJECT_SCALE_MIN, C.OBJECT_SCALE_MAX))
            twist = _wrap(math.atan2(d[1], d[0]) - self.pair["ang0"])
            obj.angles[g4.XY] = self.pair["xy0"] - twist         # turning the two-hand "steering wheel"
            targets[id(obj)] = (pa + pb) / 2 + self.pair["offset"]

        for label, gr in self.grabs.items():
            obj = gr["obj"]
            hand = hands[label]
            if id(obj) not in targets:
                targets[id(obj)] = held_at[label] + gr["offset"]
            # wrist rotation -> normal 3D rotation (only the dominant hand, only single-hand grabs)
            if label == self.dom and not (self.pair and self.pair["obj"] is obj):
                cur = (hand.roll, hand.yaw, hand.pitch)
                prev = self.prev_orient.get(label)
                self.prev_orient[label] = cur
                if prev is not None and dt > 0:
                    deltas = [_wrap(c - p) for c, p in zip(cur, prev)]
                    for plane, d, sign in ((g4.XY, deltas[0], -1), (g4.XZ, deltas[1], 1), (g4.YZ, deltas[2], 1)):
                        step = sign * d * C.WRIST_ROTATION_GAIN
                        obj.angles[plane] += step
                        # remember the spin rate so the object keeps turning when released
                        obj.angvel[plane] = 0.7 * obj.angvel[plane] + 0.3 * step / dt

        out = {}
        for gr in self.grabs.values():
            out[id(gr["obj"])] = (gr["obj"], targets[id(gr["obj"])], C.SPRING_K, C.SPRING_DAMPING)
        return out

    @staticmethod
    def _springs(targets: dict, dt: float) -> None:
        """Spring each held object towards its target (its velocity carries over on release)."""
        for obj, target, k, damping in targets.values():
            acc = k * (target - obj.pos) - damping * obj.vel
            obj.vel += acc * dt
            obj.pos += obj.vel * dt

    # ---------------- peace sign: telekinesis ----------------
    @staticmethod
    def _v_point(hand: Hand) -> np.ndarray:
        return (hand.lm[8, :2] + hand.lm[12, :2]) / 2         # between index and middle fingertips

    def _telekinesis(self, hands, gs, targets: dict, fx: Effects, now: float) -> None:
        """
        Peace sign (either hand) locks onto the nearest free object, wherever it is on screen.
        Moving your hand moves it (amplified by TELEKINESIS_GAIN); drop the pose and it keeps
        its momentum.
        """
        for label in list(self.tele):
            t = self.tele[label]
            if gs[label].pose != "peace" or label not in hands or not t["obj"].visible(now) \
                    or id(t["obj"]) in targets:
                del self.tele[label]
        for label, g in gs.items():
            if g.pose_entered == "peace" and label in hands and label not in self.tele:
                p = self._v_point(hands[label])
                taken = set(targets) | {id(t["obj"]) for t in self.tele.values()}
                free = [o for o in self.objects if o.visible(now) and id(o) not in taken]
                if free:
                    obj = min(free, key=lambda o: float(np.linalg.norm(o.pos - p)))
                    self.tele[label] = {"obj": obj, "anchor": p.copy(), "start": obj.pos.copy()}
                    self.last_obj = obj
                    fx.announce("TELEKINESIS", now)
        for label, t in self.tele.items():
            p = self._v_point(hands[label])
            obj = t["obj"]
            target = t["start"] + (p - t["anchor"]) * C.TELEKINESIS_GAIN
            target = np.clip(target, [0, 0], [self.w, self.h])
            targets[id(obj)] = (obj, target, C.TELEKINESIS_SPRING_K, C.TELEKINESIS_DAMPING)
            # a shimmering tether from your fingers to the object
            col = C.COLOR_LEFT_HAND if label == "Left" else C.COLOR_RIGHT_HAND
            fx.bolt(p, obj.pos, tuple(c * 0.8 for c in col), jitter=5, segments=16)

    @staticmethod
    def _pinch_point(hand: Hand) -> np.ndarray:
        return (hand.lm[4, :2] + hand.lm[8, :2]) / 2

    def _w_control(self, hand: Hand, dt: float) -> None:
        """Off hand open: tilting/rolling it spins objects through the 4th dimension."""
        roll = _deadzone(float(np.clip(hand.roll / C.W_ROLL_FULL, -1, 1)), C.W_ROTATION_DEADZONE)
        nx = _deadzone(float(np.clip(hand.normal[0] / 0.8, -1, 1)), C.W_ROTATION_DEADZONE)
        ny = _deadzone(float(np.clip(hand.normal[1] / 0.8, -1, 1)), C.W_ROTATION_DEADZONE)
        held = [gr["obj"] for gr in self.grabs.values()]
        targets = held if held else [o for o in self.objects if not o.dead]
        # roll -> XW (+XV, XU), side tilt -> YW (+YV, YU), forward tilt -> ZW (+ZV, ZU):
        # the V and U planes only matter for 5D and 6D objects.
        for o in targets:
            for base, amount in ((0, roll), (1, nx), (2, ny)):
                if amount:
                    for extra in range(3, o.dim):
                        o.angvel[g4.PLANE_INDEX[(base, extra)]] = amount * C.W_ROTATION_GAIN
        if self.slice_mode:
            # raise the hand -> slice moves towards +w, lower it -> towards -w
            yn = float(np.clip(hand.palm_center[1] / self.h, 0.1, 0.9))
            target = C.SLICE_W_RANGE * (1 - 2 * (yn - 0.1) / 0.8)
            self.slice_w += (target - self.slice_w) * min(1.0, dt * C.SLICE_W_SMOOTH)

    def _pointing(self, hand: Hand | None, g: GestureState, fx: Effects) -> None:
        if hand is not None and g.pose == "point":
            tip = hand.lm[INDEX_TIP, :2].copy()
            if self.prev_tip is not None:
                fx.trail(self.prev_tip, tip)
            self.prev_tip = tip
        else:
            self.prev_tip = None

    def _fling(self, hand: Hand, fx: Effects, now: float) -> None:
        speed = float(np.linalg.norm(hand.velocity))
        if speed < C.FLING_MIN_SPEED or now - self.last_fling < C.FLING_COOLDOWN_S:
            return
        held = {id(gr["obj"]) for gr in self.grabs.values()}
        hit = False
        for o in self.objects:
            if not o.visible(now) or id(o) in held:
                continue
            if np.linalg.norm(o.pos - hand.palm_center[:2]) < o.radius + C.FLING_REACH:
                o.vel = hand.velocity * C.FLING_GAIN
                o.angvel[g4.XY] += self.rng.choice([-1, 1]) * speed / 600
                o.angvel[g4.XW] += self.rng.normal(0, 1.5)
                self.last_obj = o
                hit = True
        if hit:
            self.last_fling = now
            fx.announce("FLING", now)

    # ---------------- two hands touching ----------------
    @staticmethod
    def _gap(a: Hand, b: Hand) -> float:
        """Distance between the palm centres, in palm sizes."""
        return float(np.linalg.norm(a.palm_center[:2] - b.palm_center[:2])) / ((a.palm_size + b.palm_size) / 2)

    def _finger_touch(self, a: Hand | None, b: Hand | None, ga: GestureState, gb: GestureState,
                      fx: Effects, now: float) -> bool:
        """
        Only the fingertips of both hands touching (palms apart) -> a 4D string figure stretches
        between all ten fingertips. Touch fingertips again to dissolve it.
        Returns True while fingertips are touching.
        """
        t = self.ftouch
        if a is None or b is None or ga.pose == "pinch" or gb.pose == "pinch":
            t["since"] = None
            return False
        size = (a.palm_size + b.palm_size) / 2
        tip_d = float(np.linalg.norm(a.tips2d[:, None, :] - b.tips2d[None, :, :], axis=2).min()) / size
        # Fingertips brushing past each other don't count: the hands must be (nearly) still.
        calm = max(np.linalg.norm(a.velocity), np.linalg.norm(b.velocity)) / size < C.FINGER_TOUCH_MAX_SPEED
        touching = tip_d < C.FINGER_TOUCH_DIST and self._gap(a, b) > C.FINGERS_ONLY_MIN_GAP and calm
        if tip_d > C.FINGER_TOUCH_REARM:
            t["armed"] = True
        if touching and t["armed"]:
            t["since"] = t["since"] or now
            if now - t["since"] >= C.FINGER_TOUCH_HOLD_S:
                t["armed"], t["since"] = False, None
                if self.strings is None:
                    self.strings = StringFigure(now, self.rng)
                    left, right = (a, b) if a.label == "Left" else (b, a)
                    self.strings.update(left, right, now, 0.0)
                    fx.announce("STRING FIGURE", now)
                elif now - self.strings.t0 > 1.0:
                    self._dissolve_strings(fx, now)
        elif not touching:
            t["since"] = None
        return touching

    def _dissolve_strings(self, fx: Effects, now: float) -> None:
        s = self.strings
        self.strings = None
        if s is None or s.anchors is None:
            return
        pts, cols = s.sample_points()
        fx.explode(pts, cols, pts.mean(axis=0), 0.6)
        fx.announce("STRINGS DISSOLVE", now)

    # ---------------- creation: string figure + peace sign on both hands ----------------
    def _weave_create(self, hands, gs, fx: Effects, now: float) -> None:
        """
        While a string figure is stretched between your hands, make a peace sign with BOTH hands:
        the strings tear loose, spiral into a vortex and weave themselves into a new object.
        """
        if self.strings is None or self.strings.anchors is None:
            return
        if not all(k in hands and gs[k].pose == "peace" for k in ("Left", "Right")):
            return
        pts, cols = self.strings.sample_points()
        center = self.strings.anchors.mean(axis=0)
        margin = C.OBJECT_SCALE
        center = np.clip(center, [margin, margin], [self.w - margin, self.h - margin])
        self.strings = None                                   # the strings are used up
        obj = self.spawn(center, fx, now, materialize=False)
        obj.spawn_t = now + C.WEAVE_S                          # hidden until the weave lands
        obj.project()
        n = len(pts)
        self.weaves.append({
            "obj": obj, "t0": now, "src": pts, "prev": pts.copy(), "cols": cols,
            "center": center.copy(),
            "edge": self.rng.integers(0, len(obj.shape.edges), n),   # where each strand lands
            "u": self.rng.random(n),
            "spin": self.rng.choice([-1.0, 1.0]) * C.WEAVE_SPIN,
        })
        self.selected = (self.selected + 1) % len(g4.SHAPE_BUILDERS)   # next creation = next shape
        fx.announce("WEAVING...", now)

    def _draw_weaves(self, fx: Effects, now: float) -> None:
        """Every strand point spirals from where the string was to its spot on the new object."""
        alive = []
        for wv in self.weaves:
            obj = wv["obj"]
            if obj.dead:
                continue
            p = (now - wv["t0"]) / C.WEAVE_S
            if p >= 1.0:
                # landing: flash + spark burst, then the object's edges grow in (materialize)
                c = tuple(int(v) for v in obj.pos)
                cv2.circle(fx.fx, c, int(obj.scale * 1.3), (255, 255, 255), 3, cv2.LINE_AA)
                pts, cols = obj.sample_points(2, self.rng, cap=500)
                fx.explode(pts, cols, obj.pos, 0.35)
                fx.kick(0.3)
                fx.announce(f"CREATED {obj.shape.name}", now)
                continue
            alive.append(wv)
            e = p * p * (3 - 2 * p)                                   # smooth ease in/out
            obj.project()
            a, b = obj.seg_a[wv["edge"]], obj.seg_b[wv["edge"]]
            target = a + (b - a) * wv["u"][:, None]
            center = wv["center"] + (obj.pos - wv["center"]) * e
            rel = (1 - e) * (wv["src"] - wv["center"]) + e * (target - obj.pos)
            ang = wv["spin"] * math.sin(math.pi * e)                  # swirl hardest mid-flight
            cs, sn = math.cos(ang), math.sin(ang)
            rel = rel @ np.array([[cs, sn], [-sn, cs]])
            cur = center + rel
            glow = 0.6 + 0.4 * math.sin(math.pi * p)
            segs = np.rint(np.stack([wv["prev"], cur], 1)).astype(np.int32)
            col = tuple(float(v) for v in np.mean(wv["cols"], axis=0) * glow)
            cv2.polylines(fx.fx, segs, False, col, 1, cv2.LINE_AA)    # short streaks = motion
            splat(fx.fx, cur, wv["cols"] * glow)
            wv["prev"] = cur
            # a bright core gathering at the centre
            r = max(2, int(6 + 18 * e))
            cv2.circle(fx.fx, tuple(int(v) for v in center), r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(fx.fx, tuple(int(v) for v in center), int(obj.scale * (1.6 - 0.6 * e)),
                       tuple(v * 0.5 * glow for v in C.COLOR_W_POS), 2, cv2.LINE_AA)
        self.weaves = alive

    # ---------------- erase: both fists ----------------
    def _fist_erase(self, hands, gs, fx: Effects, now: float) -> None:
        """Close both hands into fists (and hold briefly) -> every object bursts and is gone."""
        er = self.erase
        both = all(k in hands and gs[k].pose == "fist" for k in ("Left", "Right"))
        if not both:
            er["since"], er["armed"] = None, True
            return
        er["since"] = er["since"] or now
        if er["armed"] and now - er["since"] >= C.ERASE_HOLD_S:
            er["armed"] = False
            live = [o for o in self.objects if not o.dead]
            for o in live:
                self.explode(o, fx, power=1.4)
            self.weaves.clear()
            if self.strings is not None:
                self._dissolve_strings(fx, now)
            fx.kick(0.5 if live else 0.15)
            fx.announce(f"ERASED {len(live)}" if live else "NOTHING TO ERASE", now)

    # ---------------- three fingers: change dimension ----------------
    def _dimension(self, hands, gs, fx: Effects, now: float) -> None:
        """
        Three fingers up (index, middle, ring) -> the nearest object steps to the next dimension:
        3D -> 4D -> 5D -> 6D -> 3D. Lower the fingers and raise them again for another step.
        """
        for label, g in gs.items():
            if g.pose_entered != "three" or label not in hands or now - self.last_dim < C.DIMENSION_COOLDOWN_S:
                continue
            hand = hands[label]
            p = hand.lm[[8, 12, 16], :2].mean(axis=0)
            live = [o for o in self.objects if o.visible(now)]
            if not live:
                fx.announce("NO OBJECT - WEAVE ONE FIRST", now)
                continue
            obj = min(live, key=lambda o: float(np.linalg.norm(o.pos - p)))
            dims = list(g4.DIMENSIONS)
            new = dims[(dims.index(obj.dim) + 1) % len(dims)]
            obj.set_dim(new, now)
            self._materialize_particles(obj, fx)
            fx.kick(0.2)
            fx.bolt(p, obj.pos, (255, 255, 255), jitter=10, segments=14, thickness=2)
            fx.announce(f"{new}D: {obj.shape.name}", now)
            self.last_obj = obj
            self.last_dim = now

    # ---------------- physics ----------------
    def _physics(self, now: float, dt: float, fx: Effects) -> None:
        held = {id(gr["obj"]) for gr in self.grabs.values()} | {id(t["obj"]) for t in self.tele.values()}
        for o in self.objects:
            if o.dead:
                continue
            if not o.visible(now):
                continue
            o.scale += (o.target_scale - o.scale) * min(1.0, dt * 10)
            o.angles += o.angvel * dt
            o.angvel *= math.exp(-C.ANGULAR_DAMPING * dt)
            if id(o) in held:
                continue
            o.vel *= math.exp(-C.LINEAR_FRICTION * dt)
            sp = float(np.linalg.norm(o.vel))
            if sp > C.MAX_SPEED:
                o.vel *= C.MAX_SPEED / sp
            o.pos += o.vel * dt
            r = o.radius * 0.85
            for axis, hi in ((0, self.w), (1, self.h)):          # bounce off the screen edges
                if o.pos[axis] < r and o.vel[axis] < 0:
                    o.pos[axis] = r
                    o.vel[axis] *= -C.BOUNCE_RESTITUTION
                elif o.pos[axis] > hi - r and o.vel[axis] > 0:
                    o.pos[axis] = hi - r
                    o.vel[axis] *= -C.BOUNCE_RESTITUTION
        self.objects = [o for o in self.objects if not o.dead]

    # ======================================================================================
    # drawing
    # ======================================================================================
    def draw(self, fx: Effects, now: float) -> None:
        for o in self.objects:
            if not o.visible(now) or now < o.spawn_t:          # still being woven
                continue
            o.project()
            p = min(1.0, (now - o.spawn_t) / C.MATERIALIZE_S)
            ease = 1 - (1 - p) ** 3
            a, b = o.seg_a, o.seg_b
            if p < 1:                                            # edges grow in from their start
                b = a + (b - a) * ease
            if self.slice_mode and o.dim >= 4:                  # a 3D object has no w to slice
                draw_segments(fx.obj, a, b, o.seg_w, o.seg_near, alpha=0.18 * ease)
                self._draw_slice(o, fx, ease)
            else:
                draw_segments(fx.obj, a, b, o.seg_w, o.seg_near, alpha=0.3 + 0.7 * ease,
                              thick=o.scale / C.OBJECT_SCALE)

        if self.strings is not None:
            self.strings.draw(fx, now)
        self._draw_weaves(fx, now)

    def _draw_slice(self, o: Obj4D, fx: Effects, alpha: float) -> None:
        sa, sb, pts = g4.slice_at_w(o.shape, o.rotated(), self.slice_w)
        # the cut of an N-D object is (N-1)-dimensional: project any extra axes down to 3D
        if sa.shape[1] > 3:
            sa = g4.project_nd_to_3d(sa, C.PROJ_DIST_4D)[0]
            sb = g4.project_nd_to_3d(sb, C.PROJ_DIST_4D)[0]
        if pts.shape[1] > 3:
            pts = g4.project_nd_to_3d(pts, C.PROJ_DIST_4D)[0]
        color = w_to_color(np.array([self.slice_w]))[0] * 0.5 + np.array(C.COLOR_SLICE) * 0.5
        if len(sa):
            a, za = o.to_screen(sa)
            b, zb = o.to_screen(sb)
            near = np.clip(0.5 + 0.35 * (za + zb) / 2, 0, 1)
            draw_segments(fx.obj, a, b, np.zeros(len(a)), near, alpha=alpha,
                          thick=1.4 * o.scale / C.OBJECT_SCALE, color=tuple(color))
        if len(pts):
            p2, _ = o.to_screen(pts)
            splat(fx.obj, p2, np.tile(color * alpha, (len(p2), 1)))

    def hud_lines(self) -> list[str]:
        mode = f"Slice  w={self.slice_w:+.2f}" if self.slice_mode else "Projection"
        lines = [f"Next object: {g4.SHAPE_NAMES[self.selected]}", f"Mode:   {mode}",
                 f"Objects: {len([o for o in self.objects if not o.dead])}"]
        if self.strings is not None:
            lines.append("String figure: active")
        return lines
