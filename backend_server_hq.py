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
import torch.nn as nn

# Actor network — reconstructed from checkpoint weight shapes:
# Linear(13→256) → LayerNorm(256) → ReLU → Linear(256→256) → LayerNorm(256) → ReLU → Linear(256→4) → Tanh
class ActorNet(nn.Module):
    def __init__(self, obs_dim=13, act_dim=4, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


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

        # ✅ Persistent drone rotation
        self.drone_rotations = [0.0] * 4
        
        # Device
        self.device = "cpu"
        self.has_ever_started = False  # stays False until user presses Start
        self.mission_time_avg = 0.0    # rolling average of steps per episode
    
    def load_checkpoint(self, checkpoint_path, num_agents=4):
        """Load actor weights directly from checkpoint into ActorNet instances."""
        print(f"[BACKEND] Loading checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path} — using random weights")
            checkpoint = {}
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        config     = checkpoint.get("config", {})
        obs_dim    = config.get("obs_dim",    13)
        act_dim    = config.get("act_dim",     4)
        hidden_dim = config.get("hidden_dim", 256)

        self.agents = {}
        for i in range(num_agents):
            actor = ActorNet(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
            key   = f"actor_{i}"
            if key in checkpoint:
                actor.load_state_dict(checkpoint[key])
                print(f"✅ Loaded {key} from checkpoint")
            else:
                print(f"⚠️  Key '{key}' not found — using random weights")
            actor.eval()
            self.agents[f"drone_{i}"] = actor

        print(f"[BACKEND] ✅ {num_agents} actors ready")
    
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
        """Run actor networks to get actions for all drones."""
        actions = {}
        with torch.no_grad():
            for i in range(self.num_drones):
                key    = f"drone_{i}"
                actor  = self.agents.get(key)
                if actor is None or key not in observations:
                    continue
                obs_t  = torch.FloatTensor(observations[key]).unsqueeze(0)
                act_t  = actor(obs_t)
                actions[key] = act_t.squeeze(0).numpy()
        return actions


state = BackendState()
app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════════════
# Visualization Helpers
# ══════════════════════════════════════════════════════════════════════

def draw_drone(draw, x, y, rotation, label, size=20, color=(100, 255, 150)):
    """Draw a drone as circle with propeller arms"""
    draw.ellipse(
        [x-size, y-size, x+size, y+size],
        fill=(30, 80, 40), outline=color, width=2
    )
    for angle in [0, 90, 180, 270]:
        a  = math.radians(angle + rotation)
        px = x + (size + 12) * math.cos(a)
        py = y + (size + 12) * math.sin(a)
        draw.line([x, y, px, py], fill=color, width=2)
    draw.text((x - 14, y + size + 4), label, fill=color)


# ══════════════════════════════════════════════════════════════════════
# FIXED render_frame()
# ══════════════════════════════════════════════════════════════════════

def render_frame():
    """Render 2D top-down view of simulation"""
    if state.env is None or not state.env.DRONE_IDS:
        return None

    w, h = state.canvas_width, state.canvas_height
    img  = Image.new("RGB", (w, h), (15, 20, 25))
    draw = ImageDraw.Draw(img, "RGBA")

    import pybullet as p

    # ── Obstacles ──
    obs_radius_px = 0.4 * (w / 10.0)  # visual radius in pixels
    for obs_id in state.env.obstacle_ids:
        try:
            obs_pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=state.env.client)
            cx, cy = state.pybullet_to_canvas(obs_pos[0], obs_pos[1])
            draw.ellipse(
                [cx-obs_radius_px, cy-obs_radius_px, cx+obs_radius_px, cy+obs_radius_px],
                fill=(180, 40, 40), outline=(255, 80, 80), width=3
            )
        except Exception:
            pass

    # ── Goal (pulsing ring) ──
    try:
        gx, gy = state.pybullet_to_canvas(state.env.goal_pos[0], state.env.goal_pos[1])
        # Goal radius is 3.0m — show it as a large circle
        goal_r_px = 3.0 * (w / 10.0)
        pulse = 5 * math.sin(time.time() * 3)
        draw.ellipse(
            [gx - goal_r_px - pulse, gy - goal_r_px - pulse,
             gx + goal_r_px + pulse, gy + goal_r_px + pulse],
            outline=(255, 200, 0, 120), width=2
        )
        # Inner dot
        draw.ellipse([gx-8, gy-8, gx+8, gy+8], fill=(255, 200, 0))
    except Exception:
        pass

    # ── Drones ──
    crashed = state.env.crashed  # numpy bool array
    for i in range(state.num_drones):
        try:
            pos, _ = p.getBasePositionAndOrientation(
                state.env.DRONE_IDS[i], physicsClientId=state.env.client
            )
            cx, cy = state.pybullet_to_canvas(pos[0], pos[1])
            state.drone_rotations[i] = (state.drone_rotations[i] + 10) % 360

            color = (255, 80, 80) if crashed[i] else (100, 255, 150)
            draw_drone(draw, cx, cy, state.drone_rotations[i], f"UAV-{i+1}", size=18, color=color)
        except Exception:
            pass

    # ── Crash flash ──
    if any(crashed):
        overlay = Image.new("RGBA", (w, h), (255, 0, 0, 60))
        img.paste(overlay, (0, 0), overlay)

    # ── HUD ──
    hud = (f"Episode: {state.episode_count}  |  "
           f"Step: {state.step_count}  |  "
           f"Obstacles: {state.num_obstacles}")
    draw.text((10, 10), hud, fill=(100, 255, 150))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════
# FIXED run_simulation()
# ══════════════════════════════════════════════════════════════════════

def run_simulation():
    """Main simulation loop. Runs many episodes continuously until stopped."""
    obs = None
    episode_had_collision = False
    episode_goal_reached = False

    while True:
        # ── Pause if stopped or env not ready ──
        if not state.running or state.env is None:
            obs = None  # force reset when we resume
            time.sleep(0.1)
            continue

        # ── Start a new episode ──
        if obs is None:
            try:
                obs, info = state.env.reset()
            except Exception as e:
                print(f"❌ Reset error: {e}")
                time.sleep(0.5)
                continue

            state.step_count = 0
            state.episode_reward = 0.0
            state.episode_count += 1
            episode_had_collision = False
            episode_goal_reached = False
            print(f"[EPISODE {state.episode_count}] starting...")

        state.last_obs = obs

        # ── MADDPG inference ──
        actions = state.get_maddpg_actions(obs)
        state.last_actions = actions

        # ── Step environment ──
        try:
            obs, rewards, terminated, truncated, info = state.env.step(actions)

            state.episode_reward += sum(rewards.values())
            state.step_count += 1

            state.last_crashed = state.env.crashed.copy()
            if any(state.last_crashed):
                episode_had_collision = True

            # Check if episode should end
            goal_reached = info.get("goal_reached", False)
            episode_done = goal_reached or all(truncated.values())

            if episode_done:
                if goal_reached:
                    state.success_count += 1
                    episode_goal_reached = True
                if episode_had_collision:
                    state.collision_count += 1

                # Rolling average mission time
                n = state.episode_count
                state.mission_time_avg = (
                    state.mission_time_avg * (n - 1) + state.step_count
                ) / n if n > 0 else state.step_count

                print(f"[EPISODE {state.episode_count}] done — "
                      f"steps={state.step_count} "
                      f"goal={'✅' if goal_reached else '❌'} "
                      f"collision={'💥' if episode_had_collision else 'none'}")

                obs = None  # triggers new episode on next iteration

        except Exception as e:
            print(f"❌ Step error: {e}")
            obs = None
            time.sleep(0.2)
            continue

        # ── Render ──
        frame = render_frame()
        if frame:
            with state.lock:
                state.latest_frame = frame

        time.sleep(0.03)


# ══════════════════════════════════════════════════════════════════════
# Flask API Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "episode": state.episode_count,
        "step": state.step_count,
        "drones": state.num_drones,
        "obstacles": state.num_obstacles
    })


@app.route("/config", methods=["GET", "POST"])
def config():
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
    """Start or resume the simulation"""
    data = request.get_json(silent=True) or {}  # silent=True: never 415, just returns {}
    num_obstacles = data.get("num_obstacles", state.num_obstacles)

    checkpoint_map = {
        2: "checkpoints/maddpg_final-Ep17.pt",
        3: "checkpoints/maddpg_final-Ep17-o3-v2.pt",
        4: "checkpoints/maddpg_final-Ep17-o4-.pt",
    }

    # Only init if env doesn't exist yet (first boot edge case)
    if state.env is None:
        state.init_environment(num_drones=4, num_obstacles=num_obstacles)
        state.load_checkpoint(checkpoint_map.get(num_obstacles, checkpoint_map[4]), num_agents=4)
        state.episode_count = 0
        state.step_count = 0

    state.running = True
    state.has_ever_started = True
    print("[BACKEND] ▶️  Simulation started")
    return jsonify({"status": "started", "obstacles": state.num_obstacles})


@app.route("/stop", methods=["POST"])
def stop():
    """Pause the simulation — environment stays alive, just stops stepping"""
    state.running = False
    print("[BACKEND] ⏹️  Simulation paused")
    return jsonify({"status": "stopped"})


@app.route("/video_feed")
def video_feed():
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
            
            time.sleep(0.03)
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Return metrics.
    - Before first Start: all zeros.
    - While running: zeros (results shown only when paused).
    - While paused after episodes: real computed values.
    """
    # Not started yet — all zeros
    if not state.has_ever_started:
        return jsonify({
            "running": False,
            "episodes": 0,
            "success_rate": 0.0,
            "current_step": 0,
            "num_obstacles": state.num_obstacles,
            "collision_avoidance": 0.0,
            "swarm_coordination": 0.0,
            "avg_response_time": 0.0,
        })

    episodes = state.episode_count

    # While running — show live step/episode count but not final stats yet
    if state.running:
        return jsonify({
            "running": True,
            "episodes": episodes,
            "success_rate": None,          # frontend should show "—" while running
            "current_step": state.step_count,
            "num_obstacles": state.num_obstacles,
            "collision_avoidance": None,
            "swarm_coordination": None,
            "avg_response_time": None,
        })

    # Paused — show real results
    if episodes == 0:
        success_rate       = 0.0
        collision_avoidance = 0.0
    else:
        success_rate        = round(state.success_count   / episodes * 100.0, 1)
        collision_avoidance = round(
            max(0.0, 100.0 - state.collision_count / episodes * 100.0), 1
        )

    return jsonify({
        "running": False,
        "episodes": episodes,
        "success_rate": success_rate,
        "current_step": state.step_count,
        "num_obstacles": state.num_obstacles,
        "collision_avoidance": collision_avoidance,
        "swarm_coordination": round(
            (state.success_count / max(episodes, 1)) * 100.0, 1
        ),
        "avg_response_time": round(state.mission_time_avg, 1),
    })


@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    """Reset episode statistics — sim continues but counters start from zero"""
    state.episode_count   = 0
    state.success_count   = 0
    state.collision_count = 0
    state.step_count      = 0
    state.episode_reward  = 0.0
    state.mission_time_avg = 0.0
    state.last_obs        = None
    return jsonify({"status": "reset"})


@app.route("/set_obstacles", methods=["POST"])
def set_obstacles():
    """Switch obstacle config — pauses sim, swaps obstacles in existing PyBullet session, resumes"""
    data = request.get_json(silent=True) or {}
    num_obstacles = data.get("num_obstacles", 4)

    checkpoint_map = {
        2: "checkpoints/maddpg_final-Ep17.pt",
        3: "checkpoints/maddpg_final-Ep17-o3-v2.pt",
        4: "checkpoints/maddpg_final-Ep17-o4-.pt",
    }

    if num_obstacles not in checkpoint_map:
        return jsonify({"status": "error", "message": f"Invalid num_obstacles: {num_obstacles}"}), 400

    was_running = state.running
    state.running = False
    time.sleep(0.35)  # wait for sim thread to finish current step

    try:
        # Change num_obstacles on existing env — reset() will respawn with new obstacle count
        if state.env is not None:
            state.env.num_obstacles = num_obstacles
        state.num_obstacles = num_obstacles

        # Reload the matching checkpoint
        state.load_checkpoint(checkpoint_map[num_obstacles], num_agents=4)

        # Reset all counters
        state.episode_count    = 0
        state.success_count    = 0
        state.collision_count  = 0
        state.step_count       = 0
        state.episode_reward   = 0.0
        state.mission_time_avg = 0.0
        state.last_crashed     = []
        state.last_obs         = None
        state.has_ever_started = False

        if was_running:
            state.running = True

        return jsonify({"status": "success", "num_obstacles": num_obstacles})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500




# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pre-load environment and checkpoint on boot so /start is instant.
    # simulation does NOT run until user presses Start Simulation.
    print("[BOOT] Pre-loading environment (simulation will NOT run until /start is called)...")
    try:
        state.init_environment(num_drones=4, num_obstacles=4)
        state.load_checkpoint("checkpoints/maddpg_final-Ep17-o4-.pt", num_agents=4)
        state.running = False  # explicitly paused — user must press Start
        print("[BOOT] ✅ Ready. Waiting for /start request.")
    except Exception as e:
        print(f"[BOOT] ⚠️  Pre-load failed: {e}")

    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
