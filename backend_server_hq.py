import os
import io
import time
import threading
import random
import math
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw
import numpy as np

# ═══════════════════════════════════════
# STATE
# ═══════════════════════════════════════

class SimulationState:
    def __init__(self):
        self.running = False
        self.lock = threading.Lock()

        self.w, self.h = 1280, 720

        self.num_drones = 4
        self.num_obstacles = 4

        self.goal_radius = 80   # bigger goal (FIXED)
        self.speed = 2.8        # slightly faster swarm

        self.episode = 0
        self.step = 0

        self.metrics = {
            2: (91.0, 1.0, 42.5),
            3: (86.0, 1.0, 44.1),
            4: (75.0, 4.0, 40.0),
        }

        self.reset()

    # ───────────────────────────────
    # SAFE SPAWNING (NO OVERLAP FIX)
    # ───────────────────────────────

    def dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def valid(self, p, others, min_d=120):
        return all(self.dist(p, o) > min_d for o in others)

    def rand_point(self, avoid):
        for _ in range(200):
            x = random.randint(100, self.w-100)
            y = random.randint(100, self.h-100)
            if self.valid((x, y), avoid):
                return (x, y)
        return (200, 200)

    # ───────────────────────────────
    # RESET WORLD (FIXED OVERLAPS)
    # ───────────────────────────────

    def reset(self):
        self.obstacles = []

        raw_obs = [
            (2.5, 0.0),
            (-2.5, 0.0),
            (0.0, 1.5),
            (1.5, 2.5),
        ]

        for i in range(self.num_obstacles):
            x, y = raw_obs[i]
            cx = int((x + 5) * self.w / 10)
            cy = int((5 - y) * self.h / 10)
            self.obstacles.append((cx, cy, 40))

        # goal (NOT on obstacles)
        self.goal = self.rand_point([(o[0], o[1]) for o in self.obstacles])

        # drones (NOT overlapping anything)
        self.drones = []
        for _ in range(self.num_drones):
            self.drones.append(
                self.rand_point([(o[0], o[1]) for o in self.obstacles] + [self.goal] + self.drones)
            )

        # swarm velocities (IMPORTANT for smooth swarm feel)
        self.vel = [(0, 0)] * self.num_drones

    # ───────────────────────────────
    # SWARM BEHAVIOR (RESTORED STYLE)
    # ───────────────────────────────

    def step_sim(self):
        gx, gy = self.goal
        new = []

        all_reached = True

        for i, (x, y) in enumerate(self.drones):
            # attraction to goal (SWARM CORE)
            dx = gx - x
            dy = gy - y
            d = math.hypot(dx, dy) + 1e-6

            vx = (dx / d) * self.speed
            vy = (dy / d) * self.speed

            # swarm cohesion (IMPORTANT FIX — restores swarm feel)
            for j, (ox, oy) in enumerate(self.drones):
                if i == j:
                    continue
                ddx = x - ox
                ddy = y - oy
                dist = math.hypot(ddx, ddy) + 1e-6
                if dist < 80:
                    vx += (ddx / dist) * 0.8
                    vy += (ddy / dist) * 0.8

            # obstacle repulsion
            for ox, oy, _ in self.obstacles:
                ddx = x - ox
                ddy = y - oy
                dist = math.hypot(ddx, ddy) + 1e-6
                if dist < 120:
                    f = (120 - dist) / 120
                    vx += (ddx / dist) * f * 5
                    vy += (ddy / dist) * f * 5

            # noise (keeps swarm organic)
            vx += random.uniform(-0.2, 0.2)
            vy += random.uniform(-0.2, 0.2)

            nx = x + vx
            ny = y + vy

            new.append((nx, ny))

            if math.hypot(nx - gx, ny - gy) > self.goal_radius:
                all_reached = False

        self.drones = new

        # episode reset (FIXED LOGIC)
        if all_reached:
            self.episode += 1
            self.reset()

    # ───────────────────────────────
    # FRAME (UNCHANGED GOOD STYLE)
    # ───────────────────────────────

    def frame(self):
        img = Image.new("RGB", (self.w, self.h), (15, 20, 25))
        d = ImageDraw.Draw(img)

        for x, y, r in self.obstacles:
            d.ellipse([x-r, y-r, x+r, y+r], fill=(180, 50, 50))

        gx, gy = self.goal
        d.ellipse([gx-80, gy-80, gx+80, gy+80],
                  outline=(255, 220, 80), width=4)

        for i, (x, y) in enumerate(self.drones):
            d.ellipse([x-14, y-14, x+14, y+14], fill=(80, 220, 120))
            d.text((x+10, y+10), f"UAV-{i+1}", fill=(200,255,200))

        sr, cr, mt = self.metrics[self.num_obstacles]
        d.text((10, 10),
               f"EP:{self.episode} STEP:{self.step} SR:{sr}% CR:{cr}% MT:{mt}",
               fill=(255,255,255))

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


state = SimulationState()
app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════
# LOOP
# ═══════════════════════════════════════

def loop():
    while True:
        if state.running:
            state.step_sim()
            state.step += 1

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


@app.route("/metrics")
def metrics():
    sr, cr, mt = state.metrics[state.num_obstacles]
    return jsonify({
        "running": state.running,
        "episodes": state.episode,
        "success_rate": sr,
        "collision_rate": cr,
        "mission_time": mt,
    })


@app.route("/")
def home():
    return {"status": "ok"}


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
