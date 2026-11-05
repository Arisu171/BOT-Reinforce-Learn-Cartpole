"""CartPole-v1 environment factory with variants V0 through V3 and customizable physical properties.

V0: Default 500-step horizon       V1: Truncated 200-step horizon
V2: Gaussian observation noise (sigma)
V3: Minor reward shaping (alpha), raw unshaped reward stored in info["raw_reward"]

Rule: All evaluation and reported returns must track raw unshaped reward: info.get("raw_reward", r).
"""
import gymnasium as gym
import numpy as np

DEFAULT_MASSPOLE = 0.1
DEFAULT_MASSCART = 1.0
DEFAULT_LENGTH = 0.5
DEFAULT_FORCE_MAG = 10.0
DEFAULT_SIGMA = 0.05
DEFAULT_ALPHA = 0.1
THETA_LIMIT = 12 * 2 * np.pi / 360


class NoisyObs(gym.ObservationWrapper):
    """Observation wrapper adding zero-mean Gaussian noise to state observations at each step.

    Input (init): env   -- base Gymnasium environment;
                  sigma -- Gaussian noise standard deviation (default DEFAULT_SIGMA);
                  seed  -- integer seed for internal random generator (default 0).
    Output: noisy observation vector as float32 ndarray.
    Flow:   intercept observation at each step, sample Gaussian noise, add to state, cast to float32.
    """

    def __init__(self, env, sigma: float = DEFAULT_SIGMA, seed: int = 0):
        super().__init__(env)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def observation(self, obs):
        """Add Gaussian noise to the observation vector.

        Input:  obs -- input state ndarray with shape (4,).
        Output: noisy state ndarray with shape (4,) and float32 dtype.
        Flow:   sample from N(0, sigma), add to obs, return float32 array.
        """
        return (obs + self.rng.normal(0.0, self.sigma, size=obs.shape)).astype(np.float32)


class ShapedReward(gym.Wrapper):
    """Reward shaping wrapper adding an upright posture bonus inversely proportional to pole angle.

    Input (init): env   -- base Gymnasium environment;
                  alpha -- shaping coefficient scale (default DEFAULT_ALPHA).
    Output: modified reward scalar, preserving original raw reward in info["raw_reward"].
    Flow:   call step(), compute posture bonus alpha * (1 - |theta| / THETA_LIMIT),
            save raw reward to info dictionary, return modified reward.
    """

    def __init__(self, env, alpha: float = DEFAULT_ALPHA):
        super().__init__(env)
        self.alpha = float(alpha)

    def step(self, action):
        """Execute one environment step with shaped reward.

        Input:  action -- discrete action integer (0 or 1).
        Output: tuple of (obs, shaped_reward, terminated, truncated, info) where
                info["raw_reward"] contains the unshaped original reward.
        Flow:   delegate to self.env.step(), extract pole angle theta from unwrapped state,
                compute bonus, record raw_reward, return sum.
        """
        obs, r, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = r
        theta = float(self.env.unwrapped.state[2])
        bonus = self.alpha * (1.0 - min(1.0, abs(theta) / THETA_LIMIT))
        return obs, r + bonus, terminated, truncated, info


def make_env(variant: str = "v0", seed: int = 0, sigma: float = DEFAULT_SIGMA,
             alpha: float = DEFAULT_ALPHA, render_mode=None,
             masspole: float = DEFAULT_MASSPOLE, masscart: float = DEFAULT_MASSCART,
             length: float = DEFAULT_LENGTH, force_mag: float = DEFAULT_FORCE_MAG):
    """Create a CartPole-v1 environment with specified variant wrapper and physical parameters.

    Input:  variant     -- environment variant: "v0" (standard), "v1" (200 steps), "v2" (noise), "v3" (shaped);
            seed        -- seed passed to NoisyObs wrapper;
            sigma       -- noise standard deviation for V2 (default DEFAULT_SIGMA);
            alpha       -- reward shaping factor for V3 (default DEFAULT_ALPHA);
            render_mode -- rendering mode: None or "rgb_array";
            masspole    -- mass of the pole in kg (default DEFAULT_MASSPOLE = 0.1 kg);
            masscart    -- mass of the cart in kg (default DEFAULT_MASSCART = 1.0 kg);
            length      -- half-length of the pole in meters (default DEFAULT_LENGTH = 0.5 m);
            force_mag   -- magnitude of cart push force in Newtons (default DEFAULT_FORCE_MAG = 10.0 N).
    Output: configured gymnasium Env instance wrapped with variant logic.
    Flow:   instantiate gym.make("CartPole-v1"), update physical attributes on unwrapped env,
            apply corresponding wrapper, raise ValueError on unknown variant.
    """
    kwargs = {"render_mode": render_mode}
    if variant == "v1":
        kwargs["max_episode_steps"] = 200
    env = gym.make("CartPole-v1", **kwargs)

    env.unwrapped.masspole = float(masspole)
    env.unwrapped.masscart = float(masscart)
    env.unwrapped.length = float(length)
    env.unwrapped.force_mag = float(force_mag)
    env.unwrapped.total_mass = float(masscart) + float(masspole)
    env.unwrapped.polemass_length = float(masspole) * float(length)

    if variant == "v2":
        env = NoisyObs(env, sigma=sigma, seed=seed)
    elif variant == "v3":
        env = ShapedReward(env, alpha=alpha)
    elif variant not in ("v0", "v1"):
        raise ValueError(f"Invalid environment variant: {variant}")
    return env
