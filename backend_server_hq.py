import io
import os
import time
import threading
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

from custom_aviary_standalone import CustomAviaryMADDPG


# ================= MODEL =================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# ================= STATE =================
class State:
    def __init__(self):
        self.env = None
        self.agents = {}
        self.running = True

        self.frame = None
        self.lock = threading.Lock()

state = State()

app = Flask(__name__)
CORS(app)


# ================= LOAD =================
def load_model(path):
    ckpt = torch.load(path, map_location="cpu")

    for i in range(4):
        actor = Actor()
        actor.load_state_dict(ckpt[f"actor_{i}"])
        actor.eval()
        state.agents[f"drone_{i}"] = actor


# ================= RENDER =================
def render():
    img = Image.new("RGB", (800, 600), (20, 20, 30))
    draw = ImageDraw.Draw(img)

    import pybullet as p

    for i, drone in enumerate(state.env.DRONE_IDS):
        pos, _ = p.getBasePositionAndOrientation(drone)

        x = int((pos[0] + 5) * 80)
        y = int((5 - pos[1]) * 60)

        draw.ellipse([x-10, y-10, x+10, y+10], fill=(0,255,100))

    gx, gy = state.env.goal_pos[:2]
    gx = int((gx + 5) * 80)
    gy = int((5 - gy) * 60)

    draw.ellipse([gx-15, gy-15, gx+15, gy+15], outline=(255,200,0), width=3)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ================= LOOP =================
def loop():
    obs = None

    while True:
        if not state.running:
            time.sleep(0.1)
            continue

        if obs is None:
            obs, _ = state.env.reset()

        actions = {}

        for i in range(4):
            o = torch.FloatTensor(obs[f"drone_{i}"]).unsqueeze(0)
            a = state.agents[f"drone_{i}"](o).detach().numpy()[0]
            actions[f"drone_{i}"] = a

        obs, _, terminated, truncated, _ = state.env.step(actions)

        if any(terminated.values()) or all(truncated.values()):
            obs = None

        frame = render()

        with state.lock:
            state.frame = frame

        time.sleep(0.03)


# ================= ROUTES =================
@app.route("/video_feed")
def video():
    def gen():
        while True:
            with state.lock:
                frame = state.frame
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/metrics")
def metrics():
    return jsonify({"running": state.running})


# ================= MAIN =================
if __name__ == "__main__":
    state.env = CustomAviaryMADDPG()
    load_model("checkpoints/maddpg_final-Ep17-o4-.pt")

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000)
