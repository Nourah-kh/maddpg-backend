"""
backend_server_hq.py — PRODUCTION BACKEND
==========================================================================
Integrates custom_aviary_standalone.py with MADDPG inference and MJPEG streaming.

COMMUNICATION FLOW:
  1. Flask receives /start request
  2. Loads MADDPG trained checkpoints
  3. Initializes custom_aviary_standalone.py environment
  4. Main loop:
     - Get observations from environment
     - Run MADDPG inference (actor networks)
     - Apply actions to environment via env.step()
     - Render 2D top-down view with PIL
     - Stream MJPEG to frontend
"""

import argparse
import io
import os
import time
import threading
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import torch
import math

from custom_aviary_standalone import CustomAviaryMADDPG
from maddpg_networks import MADDPGAgent


# ══════════════════════════════════════════════════════════════════════
# Global Backend State
# ══════════════════════════════════════════════════════════════════════

class BackendState:
    """Manages environment, agents, and visualization state"""
    
    def __init__(self):
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # Environment state
        self.env = None
        self.agents = {}  # Dictionary of MADDPGAgent objects
        self.num_drones = 4
        self.num_obstacles = 4
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.success_count = 0
        self.collision_count = 0
        
        # Visualization
        self.canvas_width = 1280
        self.canvas_height = 720
        self.last_obs = None
        self.last_actions = None
        self.last_crashed = []
        
        # Device
        self.device = "cpu"
    
    def load_checkpoint(self, checkpoint_dir, num_agents=4):
        """Load trained MADDPG checkpoints"""
        print(f"[BACKEND] Loading checkpoints from {checkpoint_dir}")
        
        self.agents = {}
        for i in range(num_agents):
            agent = MADDPGAgent(
                obs_dim=13,
                act_dim=4,
                num_agents=num_agents,
                hidden_dim=256,
                device=self.device
            )
            
            # Load actor checkpoint
            actor_path = os.path.join(checkpoint_dir, f"actor_{i}.pt")
            if os.path.exists(actor_path):
                agent.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                print(f"✅ Loaded actor_{i}.pt")
            else:
                print(f"⚠️  No checkpoint found at {actor_path}, using random initialization")
            
            agent.actor.eval()  # Set to evaluation mode
            self.agents[f"drone_{i}"] = agent
    
    def init_environment(self, num_drones=4, num_obstacles=4):
        """Initialize or reset the environment"""
        if self.env is not None:
            self.env.close()
        
        self.env = CustomAviaryMADDPG(
            num_drones=num_drones,
            num_obstacles=num_obstacles,
            gui=False  # Headless mode
        )
        
        self.num_drones = num_drones
        self.num_obstacles = num_obstacles
        self.step_count = 0
        self.episode_reward = 0.0
        
        print(f"[ENV] Initialized with {num_drones} drones, {num_obstacles} obstacles")
    
    def pybullet_to_canvas(self, x, y):
        """Convert PyBullet coordinates to canvas pixels"""
        canvas_x = (x + 2.5) * (self.canvas_width / 5.0)
        canvas_y = (2.5 - y) * (self.canvas_height / 5.0)
        return (canvas_x, canvas_y)
    
    def get_maddpg_actions(self, observations):
        """Run MADDPG inference to get actions"""
        actions = {}
        
        with torch.no_grad():
            for i in range(self.num_drones):
                drone_key = f"drone_{i}"
                
                if drone_key not in observations:
                    continue
                
                obs_np = observations[drone_key]  # 13D observation
                obs_tensor = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
                
                # Actor network inference
                action_tensor = self.agents[drone_key].actor(obs_tensor)
                action_np = action_tensor.squeeze(0).cpu().numpy()
                
                actions[drone_key] = action_np
        
        return actions


state = BackendState()
app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════════════
# Visualization Helpers
# ══════════════════════════════════════════════════════════════════════

def draw_drone(draw, x, y, rotation, label, size=20):
    """Draw a drone as circle with propellers"""
    # Body
    draw.ellipse(
        [x-size, y-size, x+size, y+size],
        fill=(30, 80, 40),
        outline=(100, 255, 150),
        width=2
    )
    
    # Propellers
    for angle in [0, 90, 180, 270]:
        a = math.radians(angle + rotation)
        px = x + 30 * math.cos(a)
        py = y + 30 * math.sin(a)
        draw.line([x, y, px, py], fill=(100, 255, 150), width=2)
    
    # Label
    draw.text((x-15, y+25), label, fill=(100, 255, 150), font=None)


def render_frame():
    """Render 2D top-down view of simulation"""
    if state.env is None or state.last_obs is None:
        return None
    
    w, h = state.canvas_width, state.canvas_height
    img = Image.new("RGB", (w, h), (15, 20, 25))
    draw = ImageDraw.Draw(img, "RGBA")
    
    # Get drone positions from PyBullet
    import pybullet as p
    
    # ═══ Draw obstacles ═══
    obs_radius_m = 0.3
    obs_radius_px = obs_radius_m * (w / 5.0)
    
    for obs_id in state.env.obstacle_ids:
        obs_pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=state.env.client)
        cx, cy = state.pybullet_to_canvas(obs_pos[0], obs_pos[1])
        
        draw.ellipse(
            [cx-obs_radius_px, cy-obs_radius_px, cx+obs_radius_px, cy+obs_radius_px],
            fill=(180, 40, 40),
            outline=(255, 80, 80),
            width=3
        )
    
    # ═══ Draw goal ═══
    goal_canvas = state.pybullet_to_canvas(state.env.goal_position[0], state.env.goal_position[1])
    draw.ellipse(
        [goal_canvas[0]-30, goal_canvas[1]-30, goal_canvas[0]+30, goal_canvas[1]+30],
        outline=(255, 200, 0),
        width=3
    )
    
    # ═══ Draw drones ═══
    drone_rotations = [0] * state.num_drones
    for i in range(state.num_drones):
        pos, _ = p.getBasePositionAndOrientation(state.env.drone_ids[i], physicsClientId=state.env.client)
        cx, cy = state.pybullet_to_canvas(pos[0], pos[1])
        
        # Increment rotation for animation
        drone_rotations[i] += 10
        
        # Draw
        draw_drone(draw, cx, cy, drone_rotations[i], f"UAV-{i+1}", size=20)
    
    # ═══ Draw crash indicator ═══
    if any(state.last_crashed):
        overlay = Image.new("RGBA", (w, h), (255, 0, 0, 80))
        img.paste(overlay, (0, 0), overlay)
    
    # ═══ Draw HUD (info text) ═══
    hud_text = (
        f"Episode: {state.episode_count} | "
        f"Step: {state.step_count} | "
        f"Reward: {state.episode_reward:.2f} | "
        f"Collisions: {state.collision_count}"
    )
    draw.text((10, 10), hud_text, fill=(100, 255, 150), font=None)
    
    # Encode to JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════
# Main Simulation Loop
# ══════════════════════════════════════════════════════════════════════

def run_simulation():
    """Main loop: run environment with MADDPG inference"""
    
    while True:
        if not state.running or state.env is None:
            time.sleep(0.1)
            continue
        
        # ═══ RESET episode if needed ═══
        if state.step_count == 0 or state.step_count >= 300:
            obs, info = state.env.reset()
            state.step_count = 0
            state.episode_reward = 0.0
            state.episode_count += 1
            print(f"[EPISODE {state.episode_count}] Starting new episode")
        
        state.last_obs = obs
        
        # ═══ GET MADDPG ACTIONS ═══
        actions = state.get_maddpg_actions(obs)
        state.last_actions = actions
        
        # ═══ STEP ENVIRONMENT ═══
        try:
            obs, rewards, terminated, truncated, info = state.env.step(actions)
            
            # Track reward
            for reward in rewards.values():
                state.episode_reward += reward
            
            # Track collisions
            state.last_crashed = state.env.crashed.copy()
            if any(state.last_crashed):
                state.collision_count += 1
            
            state.step_count += 1
            
        except Exception as e:
            print(f"❌ Error in step: {e}")
            state.running = False
            continue
        
        # ═══ RENDER FRAME ═══
        frame = render_frame()
        
        with state.lock:
            state.latest_frame = frame
        
        # ~33 FPS (30ms per frame)
        time.sleep(0.03)


# ══════════════════════════════════════════════════════════════════════
# Flask API Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """Health check"""
    return jsonify({
        "status": "running",
        "episode": state.episode_count,
        "step": state.step_count,
        "drones": state.num_drones,
        "obstacles": state.num_obstacles
    })


@app.route("/config", methods=["GET", "POST"])
def config():
    """Get or set environment configuration"""
    if request.method == "POST":
        data = request.json or {}
        num_drones = data.get("num_drones", state.num_drones)
        num_obstacles = data.get("num_obstacles", state.num_obstacles)
        
        state.init_environment(num_drones, num_obstacles)
        
        return jsonify({
            "status": "configured",
            "num_drones": num_drones,
            "num_obstacles": num_obstacles
        })
    
    return jsonify({
        "num_drones": state.num_drones,
        "num_obstacles": state.num_obstacles,
        "running": state.running
    })


@app.route("/start", methods=["POST"])
def start():
    """Start simulation"""
    data = request.json or {}
    checkpoint_dir = data.get("checkpoint_dir", "checkpoints")
    num_drones = data.get("num_drones", 4)
    num_obstacles = data.get("num_obstacles", 4)
    
    # Load environment
    state.init_environment(num_drones, num_obstacles)
    
    # Load trained agents
    state.load_checkpoint(checkpoint_dir, num_drones)
    
    state.running = True
    state.episode_count = 0
    state.step_count = 0
    
    print("[BACKEND] ✅ Simulation started")
    return jsonify({"status": "started", "drones": num_drones, "obstacles": num_obstacles})


@app.route("/stop", methods=["POST"])
def stop():
    """Stop simulation"""
    state.running = False
    
    if state.env:
        state.env.close()
    
    print("[BACKEND] ⏹️  Simulation stopped")
    return jsonify({"status": "stopped"})


@app.route("/video_feed")
def video_feed():
    """Stream MJPEG video"""
    def generate():
        while True:
            with state.lock:
                frame = state.latest_frame
            
            if frame:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                    + frame + b'\r\n'
                )
            
            time.sleep(0.03)  # ~33 FPS
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/stats", methods=["GET"])
def stats():
    """Get current simulation stats"""
    return jsonify({
        "episode": state.episode_count,
        "step": state.step_count,
        "episode_reward": state.episode_reward,
        "collisions": state.collision_count,
        "success_count": state.success_count,
        "drones_crashed": int(sum(state.last_crashed)) if state.last_crashed else 0
    })


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Start simulation thread
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    # Start Flask
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)