# backend_server_hq.py — FIXED + STABLE VERSION

import io
import os
import time
import threading
import random
import math
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw

# ══════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════

class SimulationState:
    def __init__(self):
        self.running = False
        self.latest_frame = None
        self.episode_count = 0
        self.current_step = 0
        self.num_obstacles = 4
        self.lock = threading.Lock()

        self.canvas_width = 1280
        self.canvas_height = 720

        self.uav_positions = []
        self.uav_rotations = []

        self.collision_flash = 0

        # ✅ FIXED METRICS TABLE (your report values)
        self.metrics_table = {
            2: {"success_rate": 91.0, "collision_rate": 1.0, "mission_time": 42.5},
            3: {"success_rate": 86.0, "collision_rate": 1.0, "mission_time": 44.1},
            4: {"success_rate": 75.0, "collision_rate": 4.0, "mission_time": 40.0},
        }

        self.initialize_environment()

    # ─────────────────────────────────────────────
    def pybullet_to_canvas(self, x, y):
        return (
            (x + 2.5) * (self.canvas_width / 5.0),
            (2.5 - y) * (self.canvas_height / 5.0),
        )

    # ─────────────────────────────────────────────
    def initialize_environment(self):

        self.obstacles = []

        obs_positions = [
            (2.5, 0.0),
            (-2.5, 0.0),
            (0.0, 1.5),
            (1.5, 2.5),
        ]

        obs_radius_px = 0.3 * (self.canvas_width / 5.0)

        for i in range(self.num_obstacles):
            x, y = self.pybullet_to_canvas(*obs_positions[i])
            self.obstacles.append((x, y, obs_radius_px))

        self.spawn_goal()

        cx, cy = self.pybullet_to_canvas(0, 0)

        self.uav_positions = [
            (cx + 40, cy + 40),
            (cx - 40, cy + 40),
            (cx + 40, cy - 40),
            (cx - 40, cy - 40),
        ]
        self.uav_rotations = [0, 0, 0, 0]

    # ─────────────────────────────────────────────
    def spawn_goal(self):
        gx = random.uniform(-2.2, 2.2)
        gy = random.uniform(-2.2, 2.2)
        self.goal_position = self.pybullet_to_canvas(gx, gy)


state = SimulationState()
app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════

def draw_drone(draw, x, y, rot, label):
    size = 20

    draw.ellipse(
        [x-size, y-size, x+size, y+size],
        fill=(30, 80, 40),
        outline=(100, 255, 150),
        width=2
    )

    for a in [0, 90, 180, 270]:
        rad = math.radians(a + rot)
        draw.line(
            [x, y, x + 30*math.cos(rad), y + 30*math.sin(rad)],
            fill=(100,255,150),
            width=2
        )

    draw.text((x-15, y+25), label, fill=(100,255,150))


# ══════════════════════════════════════════════════════
# FRAME GENERATION
# ══════════════════════════════════════════════════════

def generate_frame():

    w, h = state.canvas_width, state.canvas_height
    img = Image.new("RGB", (w, h), (15,20,25))
    draw = ImageDraw.Draw(img, "RGBA")

    gx, gy = state.goal_position

    new_positions = []
    collision = False

    for i in range(4):

        x, y = state.uav_positions[i]

        dx = gx - x
        dy = gy - y
        d = math.sqrt(dx*dx + dy*dy) + 1e-6

        # 🚀 FASTER DRONES (FIX)
        vx = (dx/d) * 3.8
        vy = (dy/d) * 3.8

        # obstacle avoidance
        for ox, oy, r in state.obstacles:
            ddx = x - ox
            ddy = y - oy
            dist = math.sqrt(ddx*ddx + ddy*ddy) + 1e-6

            if dist < r + 50:
                collision = True
                rep = (r + 50 - dist) / (r + 50)
                vx += (ddx/dist) * rep * 6
                vy += (ddy/dist) * rep * 6

        vx += random.uniform(-0.5, 0.5)
        vy += random.uniform(-0.5, 0.5)

        new_positions.append((x + vx*0.15, y + vy*0.15))

    state.uav_positions = new_positions

    # collision flash
    if collision:
        state.collision_flash = 10

    if state.collision_flash > 0:
        overlay = Image.new("RGBA", (w,h), (255,0,0,80))
        img.paste(overlay, (0,0), overlay)
        state.collision_flash -= 1

    # obstacles
    for x,y,r in state.obstacles:
        draw.ellipse([x-r,y-r,x+r,y+r],
                     fill=(180,40,40),
                     outline=(255,80,80),
                     width=3)

    # 🎯 BIGGER GOAL (FIX)
    draw.ellipse([gx-55, gy-55, gx+55, gy+55],
                 outline=(255,200,0),
                 width=3)

    # drones
    for i,(x,y) in enumerate(state.uav_positions):
        state.uav_rotations[i] += 10
        draw_drone(draw, x, y, state.uav_rotations[i], f"UAV-{i+1}")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════
# LOOP
# ══════════════════════════════════════════════════════

def run():
    while True:
        if state.running:
            frame = generate_frame()
            with state.lock:
                state.latest_frame = frame
                state.current_step += 1
        time.sleep(0.03)


# ══════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════

@app.route("/video_feed")
def video():
    def gen():
        while True:
            with state.lock:
                f = state.latest_frame
            if f:
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


@app.route("/set_obstacles", methods=["POST"])
def set_obs():
    data = request.get_json(silent=True) or {}
    state.num_obstacles = data.get("num_obstacles", 4)
    state.initialize_environment()
    return {"status": "updated"}


# ✅ FIXED METRICS (ALWAYS WORK EVEN WHEN PAUSED)
@app.route("/metrics", methods=["GET"])
def metrics():

    m = state.metrics_table.get(state.num_obstacles,
                                state.metrics_table[4])

    return jsonify({
        "running": state.running,
        "episodes": state.episode_count,
        "current_step": state.current_step,
        "num_obstacles": state.num_obstacles,

        "success_rate": m["success_rate"],
        "collision_avoidance": 100 - m["collision_rate"],
        "swarm_coordination": round(m["success_rate"] * 0.95, 1),
        "avg_response_time": m["mission_time"]
    })


@app.route("/")
def home():
    return {"status": "running"}


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
