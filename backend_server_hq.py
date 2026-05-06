from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading
import time
import torch
import numpy as np

from custom_aviary_standalone import CustomAviaryMADDPG

app = Flask(__name__)
CORS(app)

env = CustomAviaryMADDPG()
obs, _ = env.reset()

running = True

def loop():
    global obs
    while True:
        if running:
            actions = {
                f"drone_{i}": np.random.uniform(-1, 1, 4)
                for i in range(4)
            }
            obs, _, term, trunc, _ = env.step(actions)

            if all(term.values()) or all(trunc.values()):
                obs, _ = env.reset()

        time.sleep(0.03)

threading.Thread(target=loop, daemon=True).start()

@app.route("/")
def home():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    app.run(port=5000)
