"""Non-learning baseline policies (random, heuristic) and policy checkpoint loaders.
Use get_policy() to retrieve a callable policy by algorithm name.
"""
import numpy as np


def random_policy(seed: int = 0):
    """Create a uniform random policy.

    Input:  seed -- seed for internal RNG (default 0).
    Output: lambda function obs -> int (0 or 1) using deterministic RNG.
    Flow:   initialize np.random.default_rng(seed), return lambda invoking rng.integers(0, 2).
    """
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(0, 2))


def heuristic_policy():
    """Create a hand-crafted heuristic policy: push cart in the direction of pole lean.

    Input:  None.
    Output: lambda function obs -> int; returns 1 if (theta + 0.5 * theta_dot) > 0, else 0.
    Flow:   deterministic rule based on obs[2] (theta) and obs[3] (theta_dot).
    """
    return lambda obs: 1 if (obs[2] + 0.5 * obs[3]) > 0 else 0


def load_policy(algo: str, path: str):
    """Load a trained PyTorch checkpoint (.pt) and return a greedy action selector.

    Input:  algo -- algorithm identifier: "dqn" or "reinforce";
            path -- checkpoint file path (.pt containing keys: algo, hidden, state_dict).
    Output: callable act(obs) -> int choosing argmax of network output.
    Flow:   call torch.load, instantiate network architecture, load weights, set eval mode,
            return greedy closure with torch.no_grad().
    """
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["algo"] == "dqn":
        from .dqn import QNetwork
        net = QNetwork(hidden=ck["hidden"])
    elif ck["algo"] == "reinforce":
        from .reinforce import PolicyNetwork
        net = PolicyNetwork(hidden=ck["hidden"])
    else:
        raise ValueError(f"Unsupported algorithm: {ck['algo']}")
    net.load_state_dict(ck["state_dict"])
    net.eval()

    def act(obs):
        """Select greedy action from loaded policy network.

        Input:  obs -- input observation array with shape (4,).
        Output: discrete action integer (0 or 1).
        Flow:   convert obs to float32 tensor, perform forward pass, return argmax item.
        """
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    return act


def get_policy(algo: str, model: str = None, seed: int = 0):
    """Return a policy action selector corresponding to requested algorithm.

    Input:  algo  -- algorithm name: "random", "heuristic", "dqn", or "reinforce";
            model -- checkpoint path (.pt), required for dqn and reinforce;
            seed  -- random seed for random_policy.
    Output: callable select_action(obs) -> int.
    Flow:   branch on algo; invoke load_policy(algo, model) when deep RL algorithm requested.
    """
    if algo == "random":
        return random_policy(seed)
    if algo == "heuristic":
        return heuristic_policy()
    if model is None:
        raise ValueError("Checkpoint --model is required for dqn and reinforce")
    return load_policy(algo, model)
