"""
MADDPG Deployment Server
========================
Flask server for deploying trained MADDPG UAV swarm model.

Features:
- Load checkpoint from disk
- Run simulation with trained policy
- Stream live video from PyBullet
- REST API for control (start/stop/reset)

Usage:
    python maddpg_deployment_server.py --checkpoint ./checkpoints/maddpg_best.pt
"""

import argparse
import io
import time
import threading
from datetime import datetime
from flask import Flask, Response, jsonify, render_template_string
from PIL import Image
import numpy as np
import torch
import pybullet as p

# ══════════════════════════════════════════════════════════════
# Import MADDPG components
# ══════════════════════════════════════════════════════════════
# NOTE: You need to place these files in the same directory:
# - maddpg_networks.py
# - custom_aviary_maddpg.py

from maddpg_networks import MADDPGAgent
from custom_aviary_maddpg import CustomAviaryMADDPG


# ══════════════════════════════════════════════════════════════
# Global State
# ══════════════════════════════════════════════════════════════

class SimulationState:
    """Thread-safe simulation state"""
    def __init__(self):
        self.running = False
        self.latest_frame = None
        self.agents = None
        self.env = None
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.current_step = 0
        self.lock = threading.Lock()

state = SimulationState()
app = Flask(__name__)


# ══════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════

def load_maddpg_checkpoint(checkpoint_path: str, num_drones: int = 4):
    """
    Load trained MADDPG model from checkpoint.
    
    Returns:
        List[MADDPGAgent]: one agent per drone
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOAD] Device: {device}")
    print(f"[LOAD] Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint.get("config", {})
    obs_dim = config.get("obs_dim", 13)
    act_dim = config.get("act_dim", 4)
    hidden_dim = config.get("hidden_dim", 256)
    
    # Create agents
    agents = []
    for i in range(num_drones):
        agent = MADDPGAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_agents=num_drones,
            hidden_dim=hidden_dim,
            actor_lr=1e-4,  # not used during inference
            critic_lr=1e-3,  # not used during inference
            device=str(device)
        )
        
        # Load weights
        agent.actor.load_state_dict(checkpoint[f"actor_{i}"])
        agent.actor.eval()  # Set to evaluation mode
        
        agents.append(agent)
    
    print(f"[LOAD] ✓ Loaded {num_drones} agents")
    return agents


# ══════════════════════════════════════════════════════════════
# Simulation Loop
# ══════════════════════════════════════════════════════════════

def run_simulation():
    """
    Main simulation loop - runs in background thread.
    """
    global state
    
    print("[SIM] Starting simulation loop...")
    
    # Create environment
    env = CustomAviaryMADDPG({
        "num_drones": 4,
        "gui": False  # headless mode for server
    })
    
    with state.lock:
        state.env = env
    
    # Reset environment
    obs_dict = env.reset()[0]  # RLlib returns (obs, info)
    
    while True:
        with state.lock:
            if not state.running:
                time.sleep(0.1)
                continue
        
        # ══════════════════════════════════════════════════════════
        # Compute actions from MADDPG policy
        # ══════════════════════════════════════════════════════════
        actions = {}
        for agent_id, obs in obs_dict.items():
            agent_idx = int(agent_id.split("_")[1])
            
            # Use trained actor (no exploration noise)
            action = state.agents[agent_idx].act(
                obs=obs,
                noise_scale=0.0,  # deterministic policy
                act_bounds=(-1.0, 1.0)
            )
            actions[agent_id] = action
        
        # Step environment
        obs_dict, rewards, dones, truncs, infos = env.step(actions)
        
        with state.lock:
            state.current_step += 1
        
        # ══════════════════════════════════════════════════════════
        # Capture frame from PyBullet
        # ══════════════════════════════════════════════════════════
        width, height = 640, 480
        
        # Camera position (bird's eye view)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 8],     # top-down view
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[1, 0, 0]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100
        )
        
        _, _, rgb, _, _ = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER  # faster for server
        )
        
        # Convert to JPEG
        img = Image.fromarray(rgb[:, :, :3], mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        
        with state.lock:
            state.latest_frame = buf.getvalue()
        
        # ══════════════════════════════════════════════════════════
        # Handle episode termination
        # ══════════════════════════════════════════════════════════
        if dones["__all__"] or truncs["__all__"]:
            with state.lock:
                state.episode_count += 1
                
                # Track success/collision
                if env.is_success:
                    state.success_count += 1
                if env.is_collision:
                    state.collision_count += 1
                
                state.current_step = 0
            
            print(f"[SIM] Episode {state.episode_count} done | "
                  f"Success: {env.is_success} | "
                  f"Collision: {env.is_collision} | "
                  f"Steps: {env.step_counter}")
            
            # Reset for next episode
            obs_dict = env.reset()[0]
        
        # Control frame rate
        time.sleep(0.02)  # ~50 FPS


# ══════════════════════════════════════════════════════════════
# Flask Routes
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Main page with video stream and controls"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MADDPG UAV Swarm</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #0a0e27;
                color: #e0e0e0;
            }
            h1 {
                color: #00d4ff;
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #888;
                margin-bottom: 30px;
            }
            .container {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
            }
            .video-panel {
                background: #1a1f3a;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0,212,255,0.1);
            }
            #stream {
                width: 100%;
                border-radius: 8px;
                display: block;
            }
            .controls-panel {
                background: #1a1f3a;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0,212,255,0.1);
            }
            .metrics {
                background: #0f1428;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .metric-row {
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                padding: 8px;
                background: #1a1f3a;
                border-radius: 4px;
            }
            .metric-label {
                color: #888;
            }
            .metric-value {
                color: #00d4ff;
                font-weight: bold;
            }
            button {
                width: 100%;
                padding: 12px;
                margin: 8px 0;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .btn-start {
                background: #00d4ff;
                color: #0a0e27;
            }
            .btn-start:hover {
                background: #00a8cc;
            }
            .btn-stop {
                background: #ff4757;
                color: white;
            }
            .btn-stop:hover {
                background: #cc3a47;
            }
            .btn-reset {
                background: #ffa502;
                color: white;
            }
            .btn-reset:hover {
                background: #cc8400;
            }
            .status {
                padding: 10px;
                border-radius: 6px;
                text-align: center;
                margin-bottom: 15px;
                font-weight: bold;
            }
            .status.running {
                background: rgba(0, 212, 255, 0.2);
                border: 2px solid #00d4ff;
                color: #00d4ff;
            }
            .status.stopped {
                background: rgba(255, 71, 87, 0.2);
                border: 2px solid #ff4757;
                color: #ff4757;
            }
        </style>
    </head>
    <body>
        <h1>🚁 MADDPG UAV Swarm Deployment</h1>
        <p class="subtitle">Multi-Agent Deep Deterministic Policy Gradient</p>
        
        <div class="container">
            <div class="video-panel">
                <img id="stream" src="/video_feed" alt="Simulation Stream">
            </div>
            
            <div class="controls-panel">
                <div id="status" class="status stopped">● STOPPED</div>
                
                <div class="metrics">
                    <h3 style="margin-top: 0; color: #00d4ff;">📊 Metrics</h3>
                    <div class="metric-row">
                        <span class="metric-label">Episodes</span>
                        <span class="metric-value" id="episodes">0</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value" id="success">0.0%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Collision Rate</span>
                        <span class="metric-value" id="collision">0.0%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Current Step</span>
                        <span class="metric-value" id="step">0</span>
                    </div>
                </div>
                
                <button class="btn-start" onclick="startSim()">▶ Start Simulation</button>
                <button class="btn-stop" onclick="stopSim()">⏸ Pause Simulation</button>
                <button class="btn-reset" onclick="resetStats()">🔄 Reset Statistics</button>
            </div>
        </div>
        
        <script>
            function updateMetrics() {
                fetch('/metrics')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('episodes').textContent = data.episodes;
                        document.getElementById('success').textContent = data.success_rate.toFixed(1) + '%';
                        document.getElementById('collision').textContent = data.collision_rate.toFixed(1) + '%';
                        document.getElementById('step').textContent = data.current_step;
                        
                        const statusEl = document.getElementById('status');
                        if (data.running) {
                            statusEl.className = 'status running';
                            statusEl.textContent = '● RUNNING';
                        } else {
                            statusEl.className = 'status stopped';
                            statusEl.textContent = '● STOPPED';
                        }
                    });
            }
            
            function startSim() {
                fetch('/start', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => console.log(data));
            }
            
            function stopSim() {
                fetch('/stop', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => console.log(data));
            }
            
            function resetStats() {
                fetch('/reset_stats', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => console.log(data));
            }
            
            // Update metrics every 500ms
            setInterval(updateMetrics, 500);
            updateMetrics();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/video_feed")
def video_feed():
    """Stream video frames as MJPEG"""
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
        success_rate = (state.success_count / state.episode_count * 100) if state.episode_count > 0 else 0.0
        collision_rate = (state.collision_count / state.episode_count * 100) if state.episode_count > 0 else 0.0
        
        return jsonify({
            "running": state.running,
            "episodes": state.episode_count,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "current_step": state.current_step,
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


# ══════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MADDPG Deployment Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to MADDPG checkpoint (.pt file)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("  MADDPG Deployment Server")
    print("="*60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Host:       {args.host}")
    print(f"  Port:       {args.port}")
    print("="*60)
    
    # Load trained model
    agents = load_maddpg_checkpoint(args.checkpoint, num_drones=4)
    
    with state.lock:
        state.agents = agents
    
    # Start simulation thread
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    print(f"\n[SERVER] Starting Flask server...")
    print(f"[SERVER] Open browser: http://localhost:{args.port}")
    print("="*60)
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
