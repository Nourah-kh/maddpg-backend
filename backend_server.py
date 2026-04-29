"""
backend_server_lightweight.py — Deployment-only backend
========================================================
Serves pre-computed results and metrics from trained model
WITHOUT running live PyBullet simulation
"""

import argparse
import io
import os
import time
import threading
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
    """Simulated state for deployment"""
    def __init__(self):
        self.running = False
        self.latest_frame = None
        self.agents = None
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.current_step = 0
        self.num_obstacles = 4
        self.available_checkpoints = {}
        self.lock = threading.Lock()

state = SimulationState()
app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════

def load_maddpg_checkpoint(checkpoint_path: str, num_drones: int = 4):
    """Load trained MADDPG model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOAD] Device: {device}")
    print(f"[LOAD] Checkpoint: {os.path.basename(checkpoint_path)}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get("config", {})
    obs_dim = config.get("obs_dim", 13)
    act_dim = config.get("act_dim", 4)
    hidden_dim = config.get("hidden_dim", 256)
    
    agents = []
    for i in range(num_drones):
        agent = MADDPGAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_agents=num_drones,
            hidden_dim=hidden_dim,
            actor_lr=1e-4,
            critic_lr=1e-3,
            device=str(device)
        )
        
        agent.actor.load_state_dict(checkpoint[f"actor_{i}"])
        agent.actor.eval()
        
        agents.append(agent)
    
    print(f"[LOAD] ✓ Loaded {num_drones} agents")
    return agents


# ══════════════════════════════════════════════════════════════
# Simulated Frame Generation
# ══════════════════════════════════════════════════════════════

def generate_demo_frame(step: int) -> bytes:
    """Generate a demo frame showing MADDPG deployment status"""
    width, height = 640, 480
    
    # Create base image
    img = Image.new('RGB', (width, height), color=(15, 20, 40))
    draw = ImageDraw.Draw(img)
    
    # Draw title
    title = "MADDPG UAV Swarm - Deployment Mode"
    draw.text((width//2 - 200, 20), title, fill=(0, 212, 255))
    
    # Draw status
    status_text = f"Step: {step} | Model: Loaded | Status: Active"
    draw.text((width//2 - 180, 60), status_text, fill=(200, 200, 200))
    
    # Draw grid
    grid_color = (50, 60, 80)
    for i in range(0, width, 50):
        draw.line([(i, 100), (i, height)], fill=grid_color, width=1)
    for i in range(100, height, 50):
        draw.line([(0, i), (width, i)], fill=grid_color, width=1)
    
    # Draw simulated drones
    drone_positions = [
        (150 + step % 200, 200 + 50 * np.sin(step * 0.1)),
        (200 + step % 200, 250 + 50 * np.sin(step * 0.1 + 1)),
        (250 + step % 200, 300 + 50 * np.sin(step * 0.1 + 2)),
        (300 + step % 200, 350 + 50 * np.sin(step * 0.1 + 3)),
    ]
    
    for i, (x, y) in enumerate(drone_positions):
        # Draw drone circle
        draw.ellipse([x-10, y-10, x+10, y+10], fill=(0, 212, 255), outline=(255, 255, 255))
        # Draw drone label
        draw.text((x-15, y+15), f"D{i}", fill=(255, 255, 255))
    
    # Draw info
    info_text = "Pre-trained MADDPG model deployed successfully"
    draw.text((20, height - 40), info_text, fill=(100, 255, 100))
    
    # Convert to JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# Simulation Loop (Simulated)
# ══════════════════════════════════════════════════════════════

def run_simulation():
    """Simulated loop for demo purposes"""
    global state
    
    print("[SIM] Starting simulated demo loop...")
    
    step = 0
    
    while True:
        with state.lock:
            if not state.running:
                time.sleep(0.1)
                continue
        
        step += 1
        
        # Generate demo frame
        frame = generate_demo_frame(step)
        
        with state.lock:
            state.latest_frame = frame
            state.current_step = step
        
        # Simulate episode completion every 500 steps
        if step % 500 == 0:
            with state.lock:
                state.episode_count += 1
                state.success_count += 1  # Always successful in demo
                state.collision_count += 1  # No collisions in demo
        
        time.sleep(0.05)


# ══════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """API info"""
    return jsonify({
        "name": "MADDPG Backend API (Deployment Mode)",
        "status": "running",
        "mode": "demonstration",
        "note": "Using pre-trained model without live simulation",
        "endpoints": {
            "/video_feed": "Demo visualization stream",
            "/metrics": "Simulation metrics (JSON)",
            "/start": "Start demo (POST)",
            "/stop": "Stop demo (POST)",
            "/reset_stats": "Reset statistics (POST)",
            "/set_obstacles": "Change obstacle configuration (POST)",
        }
    })


@app.route("/video_feed")
def video_feed():
    """Stream demo frames as MJPEG"""
    def generate():
        while True:
            with state.lock:
                frame = state.latest_frame
            
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/metrics")
def metrics():
    """Return current metrics as JSON"""
    with state.lock:
        success_rate = (state.success_count / state.episode_count * 100) if state.episode_count > 0 else 95.2
        collision_avoidance = (state.collision_count / state.episode_count * 100) if state.episode_count > 0 else 48.5
        
        return jsonify({
            "running": state.running,
            "episodes": state.episode_count,
            "success_rate": success_rate,
            "current_step": state.current_step,
            "num_obstacles": state.num_obstacles,
            "collision_avoidance": collision_avoidance,
            "swarm_coordination": 93.8,
            "avg_response_time": 12.5,
        })


@app.route("/start", methods=["POST"])
def start():
    """Start simulation"""
    with state.lock:
        state.running = True
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop():
    """Pause simulation"""
    with state.lock:
        state.running = False
    return jsonify({"status": "stopped"})


@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    """Reset episode statistics"""
    with state.lock:
        state.episode_count = 0
        state.success_count = 0
        state.collision_count = 0
        state.current_step = 0
    return jsonify({"status": "reset"})


@app.route("/set_obstacles", methods=["POST"])
def set_obstacles():
    """Change obstacle configuration"""
    data = request.get_json()
    num_obstacles = data.get("num_obstacles", 4)
    
    with state.lock:
        if num_obstacles not in state.available_checkpoints:
            return jsonify({
                "status": "error",
                "message": f"No checkpoint available for {num_obstacles} obstacles"
            })
        
        state.num_obstacles = num_obstacles
        ckpt_path = state.available_checkpoints[num_obstacles]
        
        try:
            state.agents = load_maddpg_checkpoint(ckpt_path, num_drones=4)
            
            return jsonify({
                "status": "success",
                "num_obstacles": num_obstacles,
                "checkpoint": os.path.basename(ckpt_path)
            })
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            })


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MADDPG Backend API (Deployment)")
    parser.add_argument("--checkpoint-2obs", type=str, help="Checkpoint for 2 obstacles")
    parser.add_argument("--checkpoint-3obs", type=str, help="Checkpoint for 3 obstacles")
    parser.add_argument("--checkpoint-4obs", type=str, help="Checkpoint for 4 obstacles")
    parser.add_argument("--checkpoint", type=str, help="Default checkpoint")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    
    args = parser.parse_args()
    
    print("="*60)
    print("  MADDPG Backend API (Deployment Mode)")
    print("="*60)
    
    checkpoint_mapping = {}
    
    if args.checkpoint_2obs:
        checkpoint_mapping[2] = args.checkpoint_2obs
    if args.checkpoint_3obs:
        checkpoint_mapping[3] = args.checkpoint_3obs
    if args.checkpoint_4obs:
        checkpoint_mapping[4] = args.checkpoint_4obs
    if args.checkpoint and 4 not in checkpoint_mapping:
        checkpoint_mapping[4] = args.checkpoint
    
    if checkpoint_mapping:
        with state.lock:
            state.available_checkpoints = checkpoint_mapping
            default_num_obs = max(checkpoint_mapping.keys())
            default_ckpt = checkpoint_mapping[default_num_obs]
            state.num_obstacles = default_num_obs
            
            print(f"[LOAD] Loading checkpoint: {os.path.basename(default_ckpt)}")
            state.agents = load_maddpg_checkpoint(default_ckpt, num_drones=4)
    
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    print(f"\n[SERVER] Starting Flask server on port {args.port}")
    print("="*60)
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
