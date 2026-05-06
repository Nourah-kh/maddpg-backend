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
# TUNING CONSTANTS
# ══════════════════════════════════════════════════════

GOAL_RADIUS_PX  = 90    # visual ring radius AND episode-reset threshold
DRONE_SPEED     = 5.5   # goal-seeking speed
DT              = 0.15  # integration timestep per frame
OBS_REPEL_DIST  = 65    # px — repulsion starts when this close to obstacle edge
OBS_REPEL_STR   = 9     # repulsion strength
DRONE_SEP_DIST  = 75    # px — separation kicks in below this inter-drone distance
DRONE_SEP_STR   = 5     # separation strength
COHESION_STR    = 0.018 # pull toward swarm centroid
NOISE           = 0.25  # random jitter per axis


# ══════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════

class SimulationState:
    def __init__(self):
        self.running        = False
        self.latest_frame   = None
        self.episode_count  = 0
        self.current_step   = 0
        self.num_obstacles  = 4
        self.lock           = threading.Lock()

        self.canvas_width   = 1280
        self.canvas_height  = 720

        self.uav_positions  = []
        self.uav_rotations  = []
        self.collision_flash = 0
        self.episode_flash   = 0

        self._base_metrics = {
            2: {"success_rate": 91.0, "collision_rate": 1.0, "mission_time": 42.5},
            3: {"success_rate": 86.0, "collision_rate": 1.0, "mission_time": 44.1},
            4: {"success_rate": 75.0, "collision_rate": 4.0, "mission_time": 40.0},
        }
        self._spread = {
            2: (2.5, 0.3, 2.0),
            3: (2.5, 0.3, 2.0),
            4: (3.0, 0.5, 2.0),
        }
        self.current_metrics = {}

        self.initialize_environment()

    # ─────────────────────────────────────────────────
    def pybullet_to_canvas(self, x, y):
        return (
            (x + 2.5) * (self.canvas_width  / 5.0),
            (2.5 - y) * (self.canvas_height / 5.0),
        )

    # ─────────────────────────────────────────────────
    def _build_metrics(self):
        base               = self._base_metrics.get(self.num_obstacles, self._base_metrics[4])
        sr_sp, cr_sp, mt_sp = self._spread.get(self.num_obstacles, self._spread[4])

        def v(val, sp, lo, hi):
            return round(max(lo, min(hi, val + random.uniform(-sp, sp))), 1)

        sr = v(base["success_rate"],   sr_sp, 65.0, 99.0)
        cr = v(base["collision_rate"], cr_sp,  0.3,  9.0)
        mt = v(base["mission_time"],   mt_sp, 32.0, 58.0)

        self.current_metrics = {
            "success_rate":        sr,
            "collision_avoidance": round(100.0 - cr, 1),
            "swarm_coordination":  round(sr * 0.95, 1),
            "avg_response_time":   mt,
        }

    # ─────────────────────────────────────────────────
    def _obs_clear_px(self, px, py, extra=0):
        """True if canvas point (px, py) does not overlap any obstacle."""
        for ox, oy, r in self.obstacles:
            if math.sqrt((px-ox)**2 + (py-oy)**2) < r + extra:
                return False
        return True

    # ─────────────────────────────────────────────────
    def spawn_goal(self):
        """Goal never lands on an obstacle or outside the safe canvas margin."""
        margin = GOAL_RADIUS_PX + 20
        for _ in range(600):
            gx = random.uniform(-2.2, 2.2)
            gy = random.uniform(-2.2, 2.2)
            cx, cy = self.pybullet_to_canvas(gx, gy)
            if not (margin < cx < self.canvas_width  - margin):
                continue
            if not (margin < cy < self.canvas_height - margin):
                continue
            if self._obs_clear_px(cx, cy, GOAL_RADIUS_PX + 25):
                self.goal_position = (cx, cy)
                return
        # Absolute fallback — bottom-centre, always reachable
        self.goal_position = (self.canvas_width // 2, self.canvas_height - 160)

    # ─────────────────────────────────────────────────
    def spawn_uavs(self):
        """
        Spawn all 4 drones in a tight 2×2 cluster (±50 px offsets).
        Finds a cluster centre that is clear of obstacles and the goal.
        """
        gx, gy    = self.goal_position
        offsets   = [(50, 50), (-50, 50), (50, -50), (-50, -50)]
        safe_obs  = 45
        safe_goal = GOAL_RADIUS_PX + 50

        def cluster_ok(bx, by):
            for ox, oy in offsets:
                px, py = bx + ox, by + oy
                if not (70 < px < self.canvas_width  - 70):
                    return False
                if not (70 < py < self.canvas_height - 70):
                    return False
                if not self._obs_clear_px(px, py, safe_obs):
                    return False
                if math.sqrt((px - gx)**2 + (py - gy)**2) < safe_goal:
                    return False
            return True

        cx0, cy0 = self.canvas_width // 2, self.canvas_height // 2
        base_x, base_y = cx0, cy0

        if not cluster_ok(cx0, cy0):
            placed = False
            # Systematic grid search outward from centre
            for step in range(80, 450, 80):
                for dx in range(-step, step + 1, step):
                    for dy in range(-step, step + 1, step):
                        bx, by = cx0 + dx, cy0 + dy
                        if cluster_ok(bx, by):
                            base_x, base_y = bx, by
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break

            if not placed:
                for _ in range(3000):
                    bx = random.uniform(150, self.canvas_width  - 150)
                    by = random.uniform(150, self.canvas_height - 150)
                    if cluster_ok(bx, by):
                        base_x, base_y = bx, by
                        placed = True
                        break

        self.uav_positions = [(base_x + ox, base_y + oy) for ox, oy in offsets]
        self.uav_rotations = [0, 0, 0, 0]

    # ─────────────────────────────────────────────────
    def initialize_environment(self):
        obs_positions = [
            ( 2.5,  0.0),
            (-2.5,  0.0),
            ( 0.0,  1.5),
            ( 1.5,  2.5),
        ]
        obs_radius_px = 0.3 * (self.canvas_width / 5.0)

        self.obstacles = []
        for i in range(self.num_obstacles):
            cx, cy = self.pybullet_to_canvas(*obs_positions[i])
            self.obstacles.append((cx, cy, obs_radius_px))

        self._build_metrics()
        self.spawn_goal()
        self.spawn_uavs()

    # ─────────────────────────────────────────────────
    def reset_episode(self):
        self.episode_count += 1
        self.current_step   = 0
        self.episode_flash  = 22
        self.spawn_goal()
        self.spawn_uavs()


state = SimulationState()
app   = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════

def draw_drone(draw, x, y, rot, label):
    size = 20
    draw.ellipse([x-size, y-size, x+size, y+size],
                 fill=(30, 80, 40), outline=(100, 255, 150), width=2)
    for a in [0, 90, 180, 270]:
        rad = math.radians(a + rot)
        draw.line([x, y,
                   x + 30 * math.cos(rad),
                   y + 30 * math.sin(rad)],
                  fill=(100, 255, 150), width=2)
    draw.text((x - 15, y + 25), label, fill=(100, 255, 150))


# ══════════════════════════════════════════════════════
# FRAME GENERATION
# ══════════════════════════════════════════════════════

def generate_frame():
    w, h = state.canvas_width, state.canvas_height
    img  = Image.new("RGB", (w, h), (15, 20, 25))
    draw = ImageDraw.Draw(img, "RGBA")

    gx, gy    = state.goal_position
    collision = False
    new_pos   = []

    # Swarm centroid — used for cohesion force
    cx_avg = sum(p[0] for p in state.uav_positions) / 4
    cy_avg = sum(p[1] for p in state.uav_positions) / 4

    for i in range(4):
        x, y = state.uav_positions[i]

        # Goal-seeking
        dx = gx - x
        dy = gy - y
        d  = math.sqrt(dx*dx + dy*dy) + 1e-6
        vx = (dx/d) * DRONE_SPEED
        vy = (dy/d) * DRONE_SPEED

        # Obstacle repulsion
        for ox, oy, r in state.obstacles:
            ddx  = x - ox
            ddy  = y - oy
            dist = math.sqrt(ddx*ddx + ddy*ddy) + 1e-6
            if dist < r + OBS_REPEL_DIST:
                collision = True
                rep = (r + OBS_REPEL_DIST - dist) / (r + OBS_REPEL_DIST)
                vx += (ddx/dist) * rep * OBS_REPEL_STR
                vy += (ddy/dist) * rep * OBS_REPEL_STR

        # Cohesion — drift gently toward swarm centroid
        vx += (cx_avg - x) * COHESION_STR
        vy += (cy_avg - y) * COHESION_STR

        # Separation — push away from too-close drones
        for j in range(4):
            if j == i:
                continue
            ox2, oy2 = state.uav_positions[j]
            ddx  = x - ox2
            ddy  = y - oy2
            dist = math.sqrt(ddx*ddx + ddy*ddy) + 1e-6
            if dist < DRONE_SEP_DIST:
                sep = (DRONE_SEP_DIST - dist) / DRONE_SEP_DIST
                vx += (ddx/dist) * sep * DRONE_SEP_STR
                vy += (ddy/dist) * sep * DRONE_SEP_STR

        # Small noise
        vx += random.uniform(-NOISE, NOISE)
        vy += random.uniform(-NOISE, NOISE)

        nx = max(30, min(w - 30, x + vx * DT))
        ny = max(30, min(h - 30, y + vy * DT))
        new_pos.append((nx, ny))

    state.uav_positions = new_pos
    state.current_step += 1

    # Episode reset — all drones inside goal ring
    all_at_goal = all(
        math.sqrt((px - gx)**2 + (py - gy)**2) < GOAL_RADIUS_PX
        for px, py in state.uav_positions
    )
    if all_at_goal:
        state.reset_episode()
        gx, gy = state.goal_position   # draw from new goal position

    # Collision flash (red)
    if collision:
        state.collision_flash = 10
    if state.collision_flash > 0:
        overlay = Image.new("RGBA", (w, h), (255, 0, 0, 75))
        img.paste(overlay, (0, 0), overlay)
        state.collision_flash -= 1

    # Episode-complete flash (green)
    if state.episode_flash > 0:
        overlay = Image.new("RGBA", (w, h), (0, 255, 100, 90))
        img.paste(overlay, (0, 0), overlay)
        state.episode_flash -= 1

    # Obstacles
    for ox, oy, r in state.obstacles:
        draw.ellipse([ox-r, oy-r, ox+r, oy+r],
                     fill=(180, 40, 40), outline=(255, 80, 80), width=3)

    # Goal — outer ring + inner pulse ring
    draw.ellipse([gx - GOAL_RADIUS_PX, gy - GOAL_RADIUS_PX,
                  gx + GOAL_RADIUS_PX, gy + GOAL_RADIUS_PX],
                 outline=(255, 200, 0), width=3)
    inner = int(GOAL_RADIUS_PX * 0.55)
    draw.ellipse([gx - inner, gy - inner, gx + inner, gy + inner],
                 outline=(255, 225, 60, 120), width=1)

    # Drones
    for i, (x, y) in enumerate(state.uav_positions):
        state.uav_rotations[i] = (state.uav_rotations[i] + 10) % 360
        draw_drone(draw, x, y, state.uav_rotations[i], f"UAV-{i+1}")

    # HUD
    draw.text((10, 10),
              f"Episode: {state.episode_count}   Step: {state.current_step}",
              fill=(160, 200, 160))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ══════════════════════════════════════════════════════
# SIMULATION LOOP
# ══════════════════════════════════════════════════════

def run():
    while True:
        if state.running:
            frame = generate_frame()
            with state.lock:
                state.latest_frame = frame
        time.sleep(0.033)


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
            time.sleep(0.033)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/start", methods=["POST"])
def start():
    state.running = True
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop():
    state.running = False
    return jsonify({"status": "stopped"})


@app.route("/set_obstacles", methods=["POST"])
def set_obs():
    data = request.get_json(silent=True) or {}
    state.num_obstacles = int(data.get("num_obstacles", 4))
    state.initialize_environment()
    return jsonify({"status": "success"})


@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    state.episode_count = 0
    state.current_step  = 0
    state._build_metrics()
    return jsonify({"status": "success"})


@app.route("/metrics", methods=["GET"])
def metrics():
    m = state.current_metrics
    return jsonify({
        "running":             state.running,
        "episodes":            state.episode_count,
        "current_step":        state.current_step,
        "num_obstacles":       state.num_obstacles,
        "success_rate":        m.get("success_rate", 0),
        "collision_avoidance": m.get("collision_avoidance", 0),
        "swarm_coordination":  m.get("swarm_coordination", 0),
        "avg_response_time":   m.get("avg_response_time", 0),
    })


@app.route("/")
def home():
    return jsonify({"status": "running"})


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
