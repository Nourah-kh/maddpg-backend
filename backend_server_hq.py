import os
import io
import time
import threading
import random
import math
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw

# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════

W, H = 1280, 720

NUM_DRONES = 4
NUM_OBSTACLES = 4

GOAL_RADIUS = 85          # BIGGER GOAL (FIXED)
DRONE_RADIUS = 18
OBSTACLE_RADIUS = 35

DRONE_SPEED = 2.8

# ═══════════════════════════════════════
# STATE
# ═══════════════════════════════════════

class State:
    def __init__(self):
        self.running = False
        self.lock = threading.Lock()

        self.episode = 0
        self.step = 0

        self.reset_world()

    # ─────────────────────────────
    # SAFE DISTANCE
    # ─────────────────────────────

    def dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def far_enough(self, p, others, min_d):
        return all(self.dist(p, o) > min_d for o in others)

    def random_safe(self, avoid, min_d):
        for _ in range(300):
            x = random.randint(80, W-80)
            y = random.randint(80, H-80)
            if self.far_enough((x, y), avoid, min_d):
                return (x, y)
        return (200, 200)

    # ─────────────────────────────
    # RESET WORLD (NO OVERLAP)
    # ─────────────────────────────

    def reset_world(self):
        self.obstacles = []

        raw = [
            (2.5, 0.0),
            (-2.5, 0.0),
            (0.0, 1.5),
            (1.5, 2.5),
        ]

        for i in range(NUM_OBSTACLES):
            x, y = raw[i]
            cx = int((x + 5) * W / 10)
            cy = int((5 - y) * H / 10)
            self.obstacles.append((cx, cy))

        # ── SAFE GOAL ──
        self.goal = self.random_safe(
            self.obstacles,
            GOAL_RADIUS + 60
        )

        # ── SAFE DRONES ──
        self.drones = []
        for _ in range(NUM_DRONES):
            self.drones.append(
                self.random_safe(
                    self.obstacles + [self.goal] + self.drones,
                    DRONE_RADIUS + 50
                )
            )

    # ─────────────────────────────
    # SWARM MOTION (KEEP FEEL)
    # ─────────────────────────────

    def step_sim(self):
        gx, gy = self.goal

        new = []
        all_in_goal = True

        for i, (x, y) in enumerate(self.drones):

            # attraction to goal
            dx = gx - x
            dy = gy - y
            d = math.hypot(dx, dy) + 1e-6

            vx = (dx / d) * DRONE_SPEED
            vy = (dy / d) * DRONE_SPEED

            # swarm cohesion (IMPORTANT for "swarm feel")
            for j, (ox, oy) in enumerate(self.drones):
                if i == j:
                    continue
                ddx = x - ox
                ddy = y - oy
                dist = math.hypot(ddx, ddy) + 1e-6

                if dist < 90:
                    vx += (ddx / dist) * 0.6
                    vy += (ddy / dist) * 0.6

            # obstacle repulsion
            for ox, oy in self.obstacles:
                ddx = x - ox
                ddy = y - oy
                dist = math.hypot(ddx, ddy) + 1e-6

                if dist < 120:
                    f = (120 - dist) / 120
                    vx += (ddx / dist) * f * 5
                    vy += (ddy / dist) * f * 5

            nx = x + vx
            ny = y + vy

            new.append((nx, ny))

            # check goal
            if math.hypot(nx - gx, ny - gy) > GOAL_RADIUS:
                all_in_goal = False

        self.drones = new

        # ── EPISODE RESET (FIXED) ──
        if all_in_goal:
            self.episode += 1
            self.reset_world()

        self.step += 1

    # ─────────────────────────────
    # RENDER
    # ─────────────────────────────

    def frame(self):
        img = Image.new("RGB", (W, H), (15, 20, 25))
        d = ImageDraw.Draw(img)

        # obstacles
        for x, y in self.obstacles:
            d.ellipse([x-OBSTACLE_RADIUS, y-OBSTACLE_RADIUS,
                       x+OBSTACLE_RADIUS, y+OBSTACLE_RADIUS],
                      fill=(180, 50, 50))

        # goal (BIG + CLEAR)
        gx, gy = self.goal
        d.ellipse([gx-GOAL_RADIUS, gy-GOAL_RADIUS,
                   gx+GOAL_RADIUS, gy+GOAL_RADIUS],
                  outline=(255, 220, 80), width=4)

        # drones
        for i, (x, y) in enumerate(self.drones):
            d.ellipse([x-DRONE_RADIUS, y-DRONE_RADIUS,
                       x+DRONE_RADIUS, y+DRONE_RADIUS],
                      fill=(80, 220, 120))
            d.text((x+10, y+10), f"UAV-{i+1}", fill=(220, 255, 220))

        # HUD
        d.text((10, 10),
               f"EP: {self.episode}  STEP: {self.step}",
               fill=(255, 255, 255))

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


# ═══════════════════════════════════════
# APP
# ═══════════════════════════════════════

state = State()
app = Flask(__name__)
CORS(app)

def loop():
    while True:
        if state.running:
            state.step_sim()

        with state.lock:
            state.latest = state.frame()

        time.sleep(0.03)


@app.route("/video_feed")
def video():
    def gen():
        while True:
            with state.lock:
                f = state.latest
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
            time.sleep(0.03)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/start", methods=["POST"])
def start():
    state.running = True
    return {"status": "started"}


@app.route("/stop", methods=["POST"])
def stop():
    state.running = False
    return {"status": "stopped"}


@app.route("/")
def home():
    return {"status": "ok"}


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
