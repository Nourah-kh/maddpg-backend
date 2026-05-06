# backend_server_hq.py — FINAL VERSION (Realistic + Collision Effects)

import argparse
import io
import os
import time
import threading
import random
import math
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

from maddpg_networks import MADDPGAgent


# ══════════════════════════════════════════════════════════════
# Global State
# ══════════════════════════════════════════════════════════════

class SimulationState:
    def __init__(self):
        self.running = False
        self.latest_frame = None
        self.agents = None
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.current_step = 0
        self.num_obstacles = 4
        self.lock = threading.Lock()

        self.canvas_width = 1280
        self.canvas_height = 720

        self.obstacles = []
        self.goal_position = None

        self.uav_positions = []
        self.uav_rotations = []

        self.collision_flash = 0  # 🔴 flash counter

        self.initialize_environment()

    def pybullet_to_canvas(self, x, y):
        canvas_x = (x + 2.5) * (self.canvas_width / 5.0)
        canvas_y = (2.5 - y) * (self.canvas_height / 5.0)
        return (canvas_x, canvas_y)

    def spawn_random_goal(self):
        goal_x = random.uniform(-2.0, 2.0)
        goal_y = random.uniform(-2.0, 2.0)
        self.goal_position = self.pybullet_to_canvas(goal_x, goal_y)

    def initialize_environment(self):
        self.obstacles = []

        obs_radius_m = 0.3
        obs_radius_px = obs_radius_m * (self.canvas_width / 5.0)

        obs_positions_3d = [
            ( 2.5,  0.0, 0.4),
            (-2.5,  0.0, 0.4),
            ( 0.0,  1.5, 0.4),
            ( 1.5,  2.5, 0.4),
        ]

        for obs_x, obs_y, _ in obs_positions_3d:
            cx, cy = self.pybullet_to_canvas(obs_x, obs_y)
            self.obstacles.append((cx, cy, obs_radius_px))

        self.spawn_random_goal()

        cx, cy = self.pybullet_to_canvas(0, 0)
        self.uav_positions = [
            (cx + 40, cy + 40),
            (cx - 40, cy + 40),
            (cx + 40, cy - 40),
            (cx - 40, cy - 40),
        ]
        self.uav_rotations = [0, 0, 0, 0]


state = SimulationState()
app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════
# Drawing Helpers
# ══════════════════════════════════════════════════════════════

def draw_drone(draw, x, y, rotation, label):
    size = 20

    # body
    draw.ellipse([x-size, y-size, x+size, y+size],
                 fill=(30, 80, 40),
                 outline=(100,255,150),
                 width=2)

    # propellers
    for angle in [0, 90, 180, 270]:
        a = math.radians(angle + rotation)
        px = x + 30 * math.cos(a)
        py = y + 30 * math.sin(a)
        draw.line([x, y, px, py], fill=(100,255,150), width=2)

    draw.text((x-15, y+25), label, fill=(100,255,150))


# ══════════════════════════════════════════════════════════════
# FRAME GENERATION (REALISTIC MOVEMENT)
# ══════════════════════════════════════════════════════════════

def generate_hq_frame(step):

    w, h = state.canvas_width, state.canvas_height
    img = Image.new("RGB", (w, h), (15,20,25))
    draw = ImageDraw.Draw(img, "RGBA")

    goal_x, goal_y = state.goal_position

    new_positions = []
    collision_detected = False

    for i in range(4):
        px, py = state.uav_positions[i]

        dx = goal_x - px
        dy = goal_y - py
        dist = math.sqrt(dx**2 + dy**2) + 1e-6

        vx = (dx/dist) * 2.5
        vy = (dy/dist) * 2.5

        # obstacle avoidance
        for ox, oy, r in state.obstacles:
            ddx = px - ox
            ddy = py - oy
            d = math.sqrt(ddx**2 + ddy**2) + 1e-6

            if d < r + 50:
                collision_detected = True
                repulse = (r + 50 - d) / (r + 50)
                vx += (ddx/d) * repulse * 6
                vy += (ddy/d) * repulse * 6

        # noise
        vx += random.uniform(-0.5, 0.5)
        vy += random.uniform(-0.5, 0.5)

        nx = px + vx * 0.15
        ny = py + vy * 0.15

        new_positions.append((nx, ny))

    state.uav_positions = new_positions

    # 🔴 collision flash
    if collision_detected:
        state.collision_flash = 10

    if state.collision_flash > 0:
        overlay = Image.new("RGBA", (w, h), (255,0,0,80))
        img.paste(overlay, (0,0), overlay)
        state.collision_flash -= 1

    # draw obstacles
    for x,y,r in state.obstacles:
        draw.ellipse([x-r,y-r,x+r,y+r],
                     fill=(180,40,40),
                     outline=(255,80,80),
                     width=3)

    # draw goal
    draw.ellipse([goal_x-30, goal_y-30, goal_x+30, goal_y+30],
                 outline=(255,200,0),
                 width=3)

    # draw drones
    for i,(x,y) in enumerate(state.uav_positions):
        state.uav_rotations[i] += 10
        draw_drone(draw, x, y, state.uav_rotations[i], f"UAV-{i+1}")

    # goal reached
    for x,y in state.uav_positions:
        if math.hypot(x-goal_x, y-goal_y) < 40:
            state.spawn_random_goal()
            break

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# Simulation Loop
# ══════════════════════════════════════════════════════════════

def run_sim():
    step = 0
    while True:
        if not state.running:
            time.sleep(0.1)
            continue

        step += 1
        frame = generate_hq_frame(step)

        with state.lock:
            state.latest_frame = frame
            state.current_step = step

        time.sleep(0.03)


# ══════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════

@app.route("/video_feed")
def video():
    def gen():
        while True:
            with state.lock:
                frame = state.latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/start", methods=["POST"])
def start():
    state.running = True
    return {"status":"started"}


@app.route("/stop", methods=["POST"])
def stop():
    state.running = False
    return {"status":"stopped"}


@app.route("/")
def home():
    return {"status":"running"}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Thread(target=run_sim, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
