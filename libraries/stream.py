"""Web streaming server for CartPole-v1 simulation running indefinitely.

Features:
    - Real-time MJPEG video stream (multipart/x-mixed-replace)
    - Real-time Server-Sent Events (SSE) telemetry data stream
    - Infinite continuous balancing mode or continuous multi-episode mode
    - Safe algorithm switching (DQN, REINFORCE, Heuristic, Random, Manual)
    - Interactive perturbations (push cart, tilt pole, random impulse)
    - Real-time HTML5 canvas telemetry charts and smooth policy bias gauges

Execution:
    python -m libraries.stream --port 8000 --algo dqn --mode infinite
"""
import argparse
from collections import deque
import glob
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from .dqn import QNetwork
    from .policies import heuristic_policy, random_policy
    from .reinforce import PolicyNetwork
    from .utils import ensure_dir
except (ImportError, ValueError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from libraries.dqn import QNetwork
    from libraries.policies import heuristic_policy, random_policy
    from libraries.reinforce import PolicyNetwork
    from libraries.utils import ensure_dir

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_FPS = 30
DEFAULT_ALGO = "dqn"
DEFAULT_MODE = "infinite"
DEFAULT_SEED = 42
DEFAULT_SIGMA = 0.0
DEFAULT_WEB_DIR = "public/web"
DEFAULT_WEB_INDEX = f"{DEFAULT_WEB_DIR}/index.html"
DEFAULT_WEB_CSS = f"{DEFAULT_WEB_DIR}/css/styles.css"
DEFAULT_WEB_JS = f"{DEFAULT_WEB_DIR}/js/app.js"
THETA_LIMIT_RAD = 12 * 2 * np.pi / 360
X_LIMIT = 2.4
STATIC_MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
}


class PolicyHandler:
    """Wraps inference for DQN, REINFORCE, heuristic, random, and manual policies.

    Input (init): algo       -- algorithm name ("dqn", "reinforce", "heuristic", "random", "manual");
                  model_path -- path to PyTorch checkpoint file (optional);
                  seed       -- integer seed for random policy (default 0).
    Output: PolicyHandler instance ready for step inference.
    Flow:   initialize network architecture if neural checkpoint provided,
            set evaluation mode with torch.no_grad().
    """

    def __init__(self, algo: str = DEFAULT_ALGO, model_path: Optional[str] = None, seed: int = 0):
        self.algo = algo
        self.model_path = model_path
        self.seed = seed
        self.net = None
        self.rng = np.random.default_rng(seed)
        self.load_model(algo, model_path)

    def load_model(self, algo: str, model_path: Optional[str]) -> None:
        """Load neural network weights or configure rule-based policy safely.

        Input:  algo       -- algorithm name;
                model_path -- path to .pt file or None.
        Output: None.
        Flow:   branch on algo; if dqn or reinforce, inspect checkpoint metadata,
                auto-detect actual algo from checkpoint, load state_dict safely.
        """
        self.algo = algo
        self.model_path = model_path
        self.net = None

        if algo in ("heuristic", "random", "manual"):
            return

        if algo in ("dqn", "reinforce") and model_path and os.path.exists(model_path):
            try:
                ck = torch.load(model_path, map_location="cpu", weights_only=False)
                ck_algo = ck.get("algo", algo)
                hidden = ck.get("hidden", 128)
                if ck_algo == "dqn":
                    net = QNetwork(hidden=hidden)
                elif ck_algo == "reinforce":
                    net = PolicyNetwork(hidden=hidden)
                else:
                    raise ValueError(f"Unsupported algorithm in checkpoint: {ck_algo}")

                net.load_state_dict(ck["state_dict"])
                net.eval()
                self.net = net
                self.algo = ck_algo
                self.model_path = model_path
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint {model_path}: {e}")
                self.net = None

    def predict(self, obs: np.ndarray, manual_action: int = 0) -> Tuple[int, Optional[List[float]], Optional[List[float]]]:
        """Compute action, Q-values, and action probabilities for observation.

        Input:  obs           -- observation ndarray of shape (4,);
                manual_action -- fallback action for manual control mode (0 or 1).
        Output: tuple of (action: int, q_values: list or None, probs: list or None).
        Flow:   branch on algo; compute forward tensor pass with torch.no_grad()
                or apply deterministic rule / RNG.
        """
        if self.algo == "dqn":
            if self.net is not None:
                with torch.no_grad():
                    q = self.net(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy().flatten()
                    action = int(np.argmax(q))
                    q_vals = [round(float(q[0]), 3), round(float(q[1]), 3)]
                    e_q = np.exp(q - np.max(q))
                    probs = (e_q / np.sum(e_q)).tolist()
                    probs = [round(float(p), 3) for p in probs]
                    return action, q_vals, probs
            pol = heuristic_policy()
            act = int(pol(obs))
            return act, None, [1.0, 0.0] if act == 0 else [0.0, 1.0]

        if self.algo == "reinforce":
            if self.net is not None:
                with torch.no_grad():
                    logits = self.net(torch.as_tensor(obs, dtype=torch.float32))
                    probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
                    action = int(np.argmax(probs))
                    probs_list = [round(float(probs[0]), 3), round(float(probs[1]), 3)]
                    return action, None, probs_list
            pol = heuristic_policy()
            act = int(pol(obs))
            return act, None, [1.0, 0.0] if act == 0 else [0.0, 1.0]

        if self.algo == "heuristic":
            pol = heuristic_policy()
            act = int(pol(obs))
            probs = [1.0, 0.0] if act == 0 else [0.0, 1.0]
            return act, None, probs

        if self.algo == "random":
            act = int(self.rng.integers(0, 2))
            return act, None, [0.5, 0.5]

        act = 1 if manual_action == 1 else 0
        return act, None, [1.0, 0.0] if act == 0 else [0.0, 1.0]


class CartPoleStreamEngine:
    """Thread-safe simulation engine running CartPole-v1 indefinitely with live streaming.

    Input (init): algo  -- initial algorithm name (default DEFAULT_ALGO = "dqn");
                  model -- initial checkpoint path (default auto-detected best DQN);
                  mode  -- "infinite" (continuous balance) or "standard" (500-step episodes);
                  fps   -- target simulation ticks per second (default DEFAULT_FPS = 30);
                  sigma -- Gaussian noise standard deviation added to observation (default 0.0);
                  seed  -- base environment seed (default DEFAULT_SEED = 42).
    Output: CartPoleStreamEngine instance.
    Flow:   initialize gymnasium environment, load policy handler, manage background
            stepping thread, encode JPEG frames, and notify connected clients.
    """

    def __init__(
        self,
        algo: str = DEFAULT_ALGO,
        model: Optional[str] = None,
        mode: str = DEFAULT_MODE,
        fps: int = DEFAULT_FPS,
        sigma: float = DEFAULT_SIGMA,
        seed: int = DEFAULT_SEED,
    ):
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition(threading.Lock())

        self.algo = algo
        self.mode = mode
        self.target_fps = max(1, int(fps))
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.is_paused = False
        self.is_running = False
        self.manual_action = 0
        self.pending_perturbation: Optional[Tuple[float, float]] = None

        self.start_time = time.time()
        self.total_steps = 0
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.total_episodes = 0
        self.best_steps = 0
        self.episode_history: List[Dict[str, Any]] = []
        self.current_fps = 0.0
        self.recent_actions = deque(maxlen=30)

        self.latest_jpeg: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = {}

        if model is None:
            model = self._find_default_model(algo)
        self.model_path = model
        self.policy_handler = PolicyHandler(algo, self.model_path, seed=self.seed)

        self._init_env()
        self.step_thread: Optional[threading.Thread] = None

    def _find_default_model(self, algo: str) -> Optional[str]:
        """Auto-detect default checkpoint model path from models directory.

        Input:  algo -- algorithm string ('dqn' or 'reinforce').
        Output: string path to best model file or None.
        Flow:   check for canonical seed0 best model, then fall back to glob match.
        """
        canonical = f"models/{algo}_base_seed0_best.pt"
        if os.path.exists(canonical):
            return canonical
        matches = sorted(glob.glob(f"models/{algo}_*_best.pt"))
        if matches:
            return matches[0]
        any_pt = sorted(glob.glob(f"models/{algo}_*.pt"))
        return any_pt[0] if any_pt else None

    def _init_env(self) -> None:
        """Initialize or reset Gymnasium CartPole environment.

        Input:  None.
        Output: None.
        Flow:   create gym.make with rgb_array render mode and 10 million step limit
                to allow infinite horizon balancing, reset with seed.
        """
        max_steps = 10_000_000 if self.mode == "infinite" else 500
        self.env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=max_steps)
        obs, _ = self.env.reset(seed=self.seed)
        self.obs = obs
        self._record_frame_and_telemetry(action=0, q_vals=None, probs=None, step_reward=0.0)

    def _encode_jpeg(self, frame_arr: np.ndarray, quality: int = 80) -> bytes:
        """Encode RGB numpy image array into JPEG byte buffer.

        Input:  frame_arr -- ndarray uint8 with shape (H, W, 3);
                quality   -- JPEG compression quality 1..100 (default 80).
        Output: bytes containing JPEG image data.
        Flow:   instantiate PIL Image from array, save to BytesIO buffer in JPEG format.
        """
        img = Image.fromarray(frame_arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def _record_frame_and_telemetry(
        self,
        action: int,
        q_vals: Optional[List[float]],
        probs: Optional[List[float]],
        step_reward: float,
    ) -> None:
        """Capture render frame, encode JPEG, format telemetry dict, and notify listeners.

        Input:  action      -- discrete action chosen;
                q_vals      -- estimated Q-values or None;
                probs       -- estimated action probabilities or None;
                step_reward -- reward received in current step.
        Output: None.
        Flow:   render RGB frame, convert to JPEG bytes, build telemetry dictionary,
                calculate rolling action distribution ratio, update buffers under lock.
        """
        frame_rgb = self.env.render()
        jpeg_bytes = self._encode_jpeg(frame_rgb)

        raw_state = list(self.env.unwrapped.state)
        x, x_dot, theta, theta_dot = raw_state
        theta_deg = round(float(theta * 180.0 / np.pi), 2)
        x_pos = round(float(x), 3)

        left_ratio = 50.0
        right_ratio = 50.0
        if len(self.recent_actions) > 0:
            left_count = sum(1 for a in self.recent_actions if a == 0)
            left_ratio = round((left_count / len(self.recent_actions)) * 100.0, 1)
            right_ratio = round(100.0 - left_ratio, 1)

        telemetry = {
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "episode_steps": self.episode_steps,
            "episode_reward": round(self.episode_reward, 1),
            "best_steps": self.best_steps,
            "algo": self.algo,
            "model_path": self.model_path or "N/A",
            "mode": self.mode,
            "fps_target": self.target_fps,
            "fps_real": round(self.current_fps, 1),
            "is_paused": self.is_paused,
            "sigma": round(self.sigma, 3),
            "action": int(action),
            "action_name": "RIGHT" if action == 1 else "LEFT",
            "action_ratio": {
                "left": left_ratio,
                "right": right_ratio,
            },
            "q_values": q_vals,
            "probabilities": probs,
            "state": {
                "x": x_pos,
                "x_dot": round(float(x_dot), 3),
                "theta": round(float(theta), 4),
                "theta_deg": theta_deg,
                "theta_dot": round(float(theta_dot), 3),
            },
            "limits": {
                "x_threshold": X_LIMIT,
                "theta_deg_threshold": round(THETA_LIMIT_RAD * 180.0 / np.pi, 2),
            },
            "recent_episodes": self.episode_history[-5:],
        }

        with self.frame_condition:
            self.latest_jpeg = jpeg_bytes
            self.latest_telemetry = telemetry
            self.frame_condition.notify_all()

    def step(self) -> None:
        """Perform one simulation step, handle perturbations, check boundaries, and auto-reset.

        Input:  None.
        Output: None.
        Flow:   apply pending perturbations to env state, apply observation noise,
                query policy handler, advance environment step, check termination,
                auto-reset if episode completed, and update analytics.
        """
        with self.lock:
            if self.is_paused:
                return

            if self.pending_perturbation is not None:
                dx, dtheta = self.pending_perturbation
                self.env.unwrapped.state[1] += dx
                self.env.unwrapped.state[3] += dtheta
                self.pending_perturbation = None

            active_obs = self.obs.copy()
            if self.sigma > 0.0:
                noise = np.random.normal(0.0, self.sigma, size=active_obs.shape)
                active_obs = (active_obs + noise).astype(np.float32)

            action, q_vals, probs = self.policy_handler.predict(active_obs, self.manual_action)
            self.recent_actions.append(int(action))

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            raw_r = float(info.get("raw_reward", reward))

            self.total_steps += 1
            self.episode_steps += 1
            self.episode_reward += raw_r
            self.obs = next_obs

            if self.episode_steps > self.best_steps:
                self.best_steps = self.episode_steps

            is_standard_truncation = (self.mode == "standard" and self.episode_steps >= 500)
            is_done = terminated or truncated or is_standard_truncation

            if is_done:
                self.total_episodes += 1
                reason = "fall" if terminated else "timeout"
                self.episode_history.append({
                    "episode": self.total_episodes,
                    "steps": self.episode_steps,
                    "reward": round(self.episode_reward, 1),
                    "reason": reason,
                })
                if len(self.episode_history) > 50:
                    self.episode_history.pop(0)

                self.seed += 1
                self.obs, _ = self.env.reset(seed=self.seed)
                self.episode_steps = 0
                self.episode_reward = 0.0

            self._record_frame_and_telemetry(action, q_vals, probs, raw_r)

    def run_loop(self) -> None:
        """Main loop executed in background simulation thread.

        Input:  None.
        Output: None.
        Flow:   loop while self.is_running, calculate frame delta, invoke self.step(),
                sleep for remaining frame budget, measure rolling FPS.
        """
        fps_frames = 0
        fps_timer = time.time()

        while self.is_running:
            target_period = 1.0 / max(1, self.target_fps)
            t0 = time.time()

            self.step()

            fps_frames += 1
            elapsed_fps = time.time() - fps_timer
            if elapsed_fps >= 1.0:
                self.current_fps = fps_frames / elapsed_fps
                fps_frames = 0
                fps_timer = time.time()

            t1 = time.time()
            sleep_time = target_period - (t1 - t0)
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def start(self) -> None:
        """Start the background simulation stepping thread.

        Input:  None.
        Output: None.
        Flow:   set is_running=True, spawn daemon thread for run_loop.
        """
        if self.is_running:
            return
        self.is_running = True
        self.step_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.step_thread.start()

    def stop(self) -> None:
        """Stop the background simulation thread and close environment.

        Input:  None.
        Output: None.
        Flow:   set is_running=False, join thread, close gymnasium environment.
        """
        self.is_running = False
        if self.step_thread is not None:
            self.step_thread.join(timeout=2.0)
        with self.lock:
            self.env.close()

    def set_algo(self, algo: str, model_path: Optional[str] = None) -> None:
        """Switch active policy algorithm and model safely.

        Input:  algo       -- algorithm string ('dqn', 'reinforce', 'heuristic', 'random', 'manual');
                model_path -- path to .pt file or None.
        Output: None.
        Flow:   acquire lock, validate model path against requested algo,
                update policy handler, and sync algorithm state.
        """
        with self.lock:
            if algo in ("heuristic", "random", "manual"):
                model_path = None
            elif algo in ("dqn", "reinforce"):
                if not model_path or (algo == "dqn" and "reinforce" in model_path) or (algo == "reinforce" and "dqn" in model_path):
                    model_path = self._find_default_model(algo)
            self.algo = algo
            self.model_path = model_path
            self.policy_handler.load_model(algo, model_path)
            self.algo = self.policy_handler.algo
            self.model_path = self.policy_handler.model_path

    def set_mode(self, mode: str) -> None:
        """Switch between infinite horizon and standard 500-step episode modes.

        Input:  mode -- "infinite" or "standard".
        Output: None.
        Flow:   acquire lock, set mode, re-initialize environment with matching max_steps.
        """
        with self.lock:
            if mode not in ("infinite", "standard"):
                return
            self.mode = mode
            max_steps = 10_000_000 if mode == "infinite" else 500
            self.env.close()
            self.env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=max_steps)
            self.obs, _ = self.env.reset(seed=self.seed)

    def set_fps(self, fps: int) -> None:
        """Update target simulation ticks per second.

        Input:  fps -- integer frame rate between 1 and 120.
        Output: None.
        Flow:   clamp fps to valid range, update target_fps under lock.
        """
        with self.lock:
            self.target_fps = max(1, min(120, int(fps)))

    def set_sigma(self, sigma: float) -> None:
        """Update observation noise standard deviation.

        Input:  sigma -- standard deviation float >= 0.0.
        Output: None.
        Flow:   update sigma under lock.
        """
        with self.lock:
            self.sigma = max(0.0, float(sigma))

    def toggle_pause(self) -> bool:
        """Toggle paused state of simulation.

        Input:  None.
        Output: boolean new paused status.
        Flow:   invert is_paused under lock, return new status.
        """
        with self.lock:
            self.is_paused = not self.is_paused
            return self.is_paused

    def reset_env(self) -> None:
        """Force manual reset of the environment.

        Input:  None.
        Output: None.
        Flow:   acquire lock, reset env with incremented seed, zero episode counters.
        """
        with self.lock:
            self.seed += 1
            self.obs, _ = self.env.reset(seed=self.seed)
            self.episode_steps = 0
            self.episode_reward = 0.0
            self._record_frame_and_telemetry(0, None, None, 0.0)

    def perturb(self, p_type: str) -> None:
        """Inject physical impulse disturbance into the cart-pole system.

        Input:  p_type -- perturbation type: "push_left", "push_right", "tip_left", "tip_right", "shock".
        Output: None.
        Flow:   map p_type to velocity offsets (dx, dtheta), queue into pending_perturbation.
        """
        with self.lock:
            if p_type == "push_left":
                self.pending_perturbation = (-1.0, 0.0)
            elif p_type == "push_right":
                self.pending_perturbation = (1.0, 0.0)
            elif p_type == "tip_left":
                self.pending_perturbation = (0.0, -0.3)
            elif p_type == "tip_right":
                self.pending_perturbation = (0.0, 0.3)
            elif p_type == "shock":
                dx = float(np.random.choice([-1.2, 1.2]))
                dtheta = float(np.random.choice([-0.25, 0.25]))
                self.pending_perturbation = (dx, dtheta)

    def set_manual_action(self, action: int) -> None:
        """Set manual control action from user keyboard or button press.

        Input:  action -- 0 (left) or 1 (right).
        Output: None.
        Flow:   store in manual_action under lock.
        """
        with self.lock:
            self.manual_action = 1 if action == 1 else 0

    def get_telemetry(self) -> Dict[str, Any]:
        """Retrieve copy of current telemetry data.

        Input:  None.
        Output: dictionary with full state, analytics, and status metrics.
        Flow:   acquire frame_condition lock, return copy of latest_telemetry.
        """
        with self.frame_condition:
            return dict(self.latest_telemetry)

    def get_available_models(self) -> Dict[str, Any]:
        """Scan models directory and return categorized dictionary of checkpoint files.

        Input:  None.
        Output: dict with keys "dqn", "reinforce", and "all".
        Flow:   invoke glob.glob on models/*.pt, sort, and partition into categories.
        """
        files = sorted([f.replace("\\", "/") for f in glob.glob("models/*.pt")])
        dqn_models = [f for f in files if "dqn" in f]
        reinforce_models = [f for f in files if "reinforce" in f]
        return {
            "dqn": dqn_models,
            "reinforce": reinforce_models,
            "all": files,
        }


class StreamHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler supporting video streaming, SSE telemetry, and REST controls.

    Input (init): standard BaseHTTPRequestHandler initialization parameters.
    Output: StreamHTTPRequestHandler instance.
    Flow:   route GET and POST requests by path to streaming loops, API handlers,
            or static HTML delivery.
    """

    engine: CartPoleStreamEngine

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server access logs to prevent console flooding.

        Input:  format -- log format string; args -- formatting arguments.
        Output: None.
        Flow:   do nothing (silent server logging).
        """
        return

    def handle(self) -> None:
        """Handle incoming requests and catch client disconnection exceptions cleanly.

        Input:  None.
        Output: None.
        Flow:   invoke super().handle(), suppress normal client disconnections on browser close.
        """
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError, OSError):
            pass

    def do_GET(self) -> None:
        """Handle HTTP GET requests: root HTML, MJPEG video feed, SSE events, or JSON stats.

        Input:  self.path -- requested URL path.
        Output: HTTP response sent to client.
        Flow:   branch on path; delegate to handle_index, handle_stream, handle_events,
                handle_stats, handle_models, or handle_static for /css and /js assets.
        """
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.handle_index()
        elif path in ("/stream", "/video_feed"):
            self.handle_stream()
        elif path == "/events":
            self.handle_events()
        elif path == "/api/stats":
            self.handle_stats()
        elif path == "/api/models":
            self.handle_models()
        elif path.startswith("/css/") or path.startswith("/js/"):
            self.handle_static(path)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        """Handle HTTP POST requests for control commands safely.

        Input:  self.path -- requested URL path; request body containing JSON command.
        Output: JSON response status.
        Flow:   verify path == '/api/control', parse JSON payload, execute command
                inside try-except block, return JSON result or error.
        """
        parsed = urlparse(self.path)
        if parsed.path != "/api/control":
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            data = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            self.send_json({"status": "error", "message": f"Invalid JSON payload: {e}"}, status_code=400)
            return

        cmd = data.get("action", "")
        engine = self.engine

        try:
            if cmd == "set_algo":
                algo = data.get("algo", "dqn")
                model = data.get("model")
                engine.set_algo(algo, model)
            elif cmd == "set_mode":
                mode = data.get("mode", "infinite")
                engine.set_mode(mode)
            elif cmd == "set_fps":
                fps = data.get("fps", 30)
                engine.set_fps(int(fps))
            elif cmd == "set_sigma":
                sigma = data.get("sigma", 0.0)
                engine.set_sigma(float(sigma))
            elif cmd == "toggle_pause":
                engine.toggle_pause()
            elif cmd == "reset":
                engine.reset_env()
            elif cmd == "perturb":
                p_type = data.get("type", "shock")
                engine.perturb(p_type)
            elif cmd == "manual_action":
                act = data.get("value", 0)
                engine.set_manual_action(int(act))
            else:
                self.send_json({"status": "error", "message": f"Unknown action: {cmd}"}, status_code=400)
                return

            self.send_json({"status": "ok", "telemetry": engine.get_telemetry()})
        except Exception as e:
            self.send_json({"status": "error", "message": str(e)}, status_code=500)

    def send_json(self, data: Any, status_code: int = 200) -> None:
        """Send JSON response with appropriate headers.

        Input:  data        -- JSON-serializable object;
                status_code -- HTTP status integer (default 200).
        Output: None.
        Flow:   serialize data to JSON bytes, send Content-Type application/json, write payload.
        """
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def handle_stats(self) -> None:
        """Serve JSON snapshot of current telemetry.

        Input:  None.
        Output: JSON telemetry object.
        Flow:   query engine.get_telemetry(), send via send_json.
        """
        self.send_json(self.engine.get_telemetry())

    def handle_models(self) -> None:
        """Serve JSON dictionary of available checkpoint models by algorithm.

        Input:  None.
        Output: JSON object containing categorized model lists.
        Flow:   query engine.get_available_models(), send via send_json.
        """
        self.send_json(self.engine.get_available_models())

    def handle_stream(self) -> None:
        """Stream real-time MJPEG video using multipart/x-mixed-replace boundary.

        Input:  None.
        Output: continuous HTTP multipart image stream.
        Flow:   write multipart header, loop waiting for engine.frame_condition notification,
                write JPEG boundary frame, catch client disconnect exceptions cleanly.
        """
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        engine = self.engine
        try:
            while engine.is_running:
                with engine.frame_condition:
                    engine.frame_condition.wait(timeout=1.0)
                    jpeg = engine.latest_jpeg

                if jpeg is not None:
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                    )
                    self.wfile.write(header)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def handle_events(self) -> None:
        """Stream real-time Server-Sent Events (SSE) telemetry data.

        Input:  None.
        Output: continuous text/event-stream data events.
        Flow:   write event-stream headers, loop waiting for frame_condition,
                write formatted JSON payload as data event, handle client disconnects.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        engine = self.engine
        try:
            while engine.is_running:
                with engine.frame_condition:
                    engine.frame_condition.wait(timeout=1.0)
                    telem = engine.latest_telemetry

                if telem:
                    msg = f"data: {json.dumps(telem)}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def handle_index(self) -> None:
        """Serve responsive HTML dashboard page.

        Input:  None.
        Output: HTML5 document.
        Flow:   load DEFAULT_WEB_INDEX from disk when present, otherwise fall back
                to get_embedded_html(), then send text/html response headers.
        """
        if os.path.exists(DEFAULT_WEB_INDEX):
            try:
                with open(DEFAULT_WEB_INDEX, "r", encoding="utf-8") as f:
                    content = f.read().encode("utf-8")
            except OSError:
                content = get_embedded_html().encode("utf-8")
        else:
            content = get_embedded_html().encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_static(self, path: str) -> None:
        """Serve CSS/JS assets from DEFAULT_WEB_DIR with path-traversal protection.

        Input:  path -- URL path beginning with /css/ or /js/.
        Output: static file bytes or HTTP 404.
        Flow:   resolve relative path under DEFAULT_WEB_DIR, reject escapes,
                detect MIME type, stream file contents.
        """
        relative = path.lstrip("/")
        candidate = os.path.abspath(os.path.join(DEFAULT_WEB_DIR, relative))
        root = os.path.abspath(DEFAULT_WEB_DIR)
        if not candidate.startswith(root + os.sep) or not os.path.isfile(candidate):
            self.send_error(404, "Not Found")
            return
        ext = os.path.splitext(candidate)[1].lower()
        content_type = STATIC_MIME_TYPES.get(ext, "application/octet-stream")
        try:
            with open(candidate, "rb") as f:
                content = f.read()
        except OSError:
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class StreamingServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that suppresses harmless socket reset and abort tracebacks.

    Input (init): server_address, RequestHandlerClass.
    Output: StreamingServer instance.
    Flow:   daemonize worker threads and override handle_error to ignore normal browser drops.
    """

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Silently ignore normal client disconnects (tab closed, reloaded, socket aborted).

        Input:  request, client_address.
        Output: None.
        Flow:   check exception type, suppress ConnectionResetError and ConnectionAbortedError.
        """
        ex_type, _, _ = sys.exc_info()
        if ex_type in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError, OSError):
            return
        super().handle_error(request, client_address)


def get_embedded_html() -> str:
    """Assemble a self-contained HTML fallback from modular public/web assets.

    Input:  None.
    Output: complete HTML document string with inlined CSS and JS when files exist.
    Flow:   read index.html, styles.css, and app.js from DEFAULT_WEB_DIR; inline
            stylesheet and script contents so the fallback works without static routes;
            if files are missing, return a minimal document with required element IDs.
    """
    if (
        os.path.isfile(DEFAULT_WEB_INDEX)
        and os.path.isfile(DEFAULT_WEB_CSS)
        and os.path.isfile(DEFAULT_WEB_JS)
    ):
        with open(DEFAULT_WEB_INDEX, "r", encoding="utf-8") as f:
            html = f.read()
        with open(DEFAULT_WEB_CSS, "r", encoding="utf-8") as f:
            css = f.read()
        with open(DEFAULT_WEB_JS, "r", encoding="utf-8") as f:
            js = f.read()
        html = html.replace(
            '<link rel="stylesheet" href="/css/styles.css">',
            "<style>\n" + css + "\n</style>",
            1,
        )
        html = html.replace(
            '<script src="/js/app.js"></script>',
            "<script>\n" + js + "\n</script>",
            1,
        )
        return html
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        "<title>CartPole Stream</title></head><body>"
        '<img id="video-stream" src="/stream" alt="stream">'
        '<canvas id="waveform-canvas"></canvas>'
        '<div id="prob-bar-left"></div><div id="prob-bar-right"></div>'
        '<script>new EventSource("/events");'
        'fetch("/api/control",{method:"POST"});</script>'
        "</body></html>"
    )


def start_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    algo: str = DEFAULT_ALGO,
    model: Optional[str] = None,
    mode: str = DEFAULT_MODE,
    fps: int = DEFAULT_FPS,
    sigma: float = DEFAULT_SIGMA,
) -> None:
    """Instantiate simulation engine and launch ThreadingHTTPServer.

    Input:  host  -- bind host IP address (default '0.0.0.0');
            port  -- bind port integer (default 8000);
            algo  -- initial algorithm name;
            model -- path to checkpoint file;
            mode  -- "infinite" or "standard";
            fps   -- simulation tick rate;
            sigma -- initial observation noise.
    Output: None (runs indefinitely until KeyboardInterrupt).
    Flow:   instantiate CartPoleStreamEngine, start simulation thread,
            bind ThreadingHTTPServer, serve_forever, catch KeyboardInterrupt,
            and stop engine cleanly.
    """
    engine = CartPoleStreamEngine(
        algo=algo,
        model=model,
        mode=mode,
        fps=fps,
        sigma=sigma,
    )
    engine.start()

    StreamHTTPRequestHandler.engine = engine
    server_address = (host, port)
    httpd = StreamingServer(server_address, StreamHTTPRequestHandler)

    url = f"http://localhost:{port}" if host in ("0.0.0.0", "127.0.0.1") else f"http://{host}:{port}"
    print("\n" + "=" * 60)
    print(f" CartPole-v1 Infinite Live Stream Server Active!")
    print(f" Web UI Dashboard:   {url}")
    print(f" Direct MJPEG Feed:  {url}/stream")
    print(f" Real-Time Telemetry:{url}/events")
    print(f" Algorithm:          {algo.upper()} ({engine.model_path or 'Rule-based'})")
    print(f" Mode:               {mode.upper()} (Runs indefinitely)")
    print(f" Target FPS:         {fps}")
    print(" Press Ctrl+C to terminate.")
    print("=" * 60 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping streaming server...")
    finally:
        engine.stop()
        httpd.server_close()
        print("Server stopped cleanly.")


def main() -> None:
    """CLI entry point for CartPole web streaming.

    Input:  command line arguments (--host, --port, --algo, --model, --mode, --fps, --sigma).
    Output: None (runs streaming web server).
    Flow:   parse CLI options via argparse, invoke start_server.
    """
    ap = argparse.ArgumentParser(description="CartPole-v1 Infinite Web Streaming Server")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Bind IP address (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on (default: 8000)")
    ap.add_argument("--algo", choices=["dqn", "reinforce", "heuristic", "random", "manual"],
                    default=DEFAULT_ALGO, help="Initial algorithm")
    ap.add_argument("--model", help="Path to checkpoint model .pt file")
    ap.add_argument("--mode", choices=["infinite", "standard"], default=DEFAULT_MODE,
                    help="Infinite horizon (no 500 step cutoff) or standard 500-step episodes")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Target simulation FPS (default: 30)")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Gaussian observation noise sigma")
    args = ap.parse_args()

    start_server(
        host=args.host,
        port=args.port,
        algo=args.algo,
        model=args.model,
        mode=args.mode,
        fps=args.fps,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
