import numpy as np
from typing import List
import torch

def calculate_reward_for_path(
    path,
    query_emb,
    gold_paths=None,
    embeddings=None,
    node_to_idx=None,
    alpha_sim=1.0,
    alpha_coherence=0.1,
    alpha_length=0.05,
    alpha_rapl=0.5,
):
    """
    Compute reward for a reasoning path.

    Combines:
    - Query relevance
    - Path coherence
    - Length penalty
    - Alignment with pseudo-gold paths (RAPL)

    Parameters
    ----------
    path : List[str]
        Traversed path.
    query_emb : np.ndarray
        Query embedding.
    gold_paths : List[List[str]], optional
        Pseudo-gold reasoning paths.
    embeddings : np.ndarray
        Node embeddings.
    node_to_idx : Dict[str, int]
        Node index mapping.

    Returns
    -------
    float
        Reward score.
    """

    # 🔹 1. Query relevance (FINAL node)
    last_node = path[-1]
    last_emb = embeddings[node_to_idx[last_node]]
    sim = float(np.dot(last_emb, query_emb))

    # 🔹 2. Path coherence (pairwise similarity)
    coherence = 0
    for i in range(len(path) - 1):
        u = embeddings[node_to_idx[path[i]]]
        v = embeddings[node_to_idx[path[i + 1]]]
        coherence += np.dot(u, v)

    coherence /= (len(path) - 1 + 1e-6)

    # 🔹 3. Length penalty
    length_penalty = -len(path)

    # 🔹 4. RAPL alignment (structure-aware)
    rapl_score = 0
    if gold_paths:
        for gold in gold_paths:
            overlap = len(set(path) & set(gold))
            rapl_score = max(rapl_score, overlap / (len(gold) + 1e-6))

    # 🔹 Final reward
    reward = (
        alpha_sim * sim
        + alpha_coherence * coherence
        + alpha_length * length_penalty
        + alpha_rapl * rapl_score
    )

    return reward

def compute_supervised_loss(
    self,
    query_embedding: np.ndarray,
    gold_paths: List[List[str]]
) -> torch.Tensor:
    """
    Compute imitation learning loss from pseudo-gold paths.

    The model learns to select the correct next node at each step.

    Parameters
    ----------
    query_embedding : np.ndarray
        Query embedding.
    gold_paths : List[List[str]]
        Pseudo-gold reasoning paths.

    Returns
    -------
    torch.Tensor
        Supervised loss.
    """
    losses = []

    for path in gold_paths:
        for i in range(len(path) - 1):
            current = path[i]
            target = path[i + 1]

            neighbors = list(self.G.successors(current))
            if target not in neighbors:
                continue

            states = [
                self.build_state(query_embedding, current, n)
                for n in neighbors
            ]

            states_tensor = torch.stack(states).to(self.device)
            scores = self.PolicyNetwork(states_tensor).squeeze(-1)

            target_idx = neighbors.index(target)

            loss = torch.nn.functional.cross_entropy(
                scores.unsqueeze(0),
                torch.tensor([target_idx]).to(self.device)
            )

            losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True).to(self.device)

    return torch.stack(losses).mean()

def compute_rl_loss(
    log_probs: List[torch.Tensor],
    rewards: List[float],
    device='cpu'
) -> torch.Tensor:
    """
    Compute the REINFORCE loss for a batch of sampled paths.

    This function implements the policy gradient objective:
        L = - E[ log(pi(a|s)) * reward ]

    Rewards are normalized for training stability.

    Parameters
    ----------
    log_probs : List[torch.Tensor]
        Log-probabilities of sampled trajectories.
    rewards : List[float]
        Scalar rewards associated with each trajectory.
    device :
        "cpu" for first tries

    Returns
    -------
    torch.Tensor
        Scalar RL loss.
    """
    if len(log_probs) == 0:
        return torch.tensor(0.0, requires_grad=True)

    device = log_probs[0].device

    rewards_tensor = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    # 🔹 Normalize rewards (critical for stability)
    rewards_tensor = (
        rewards_tensor - rewards_tensor.mean()
    ) / (rewards_tensor.std() + 1e-6)

    # 🔹 REINFORCE loss
    losses = [
        -log_prob * reward
        for log_prob, reward in zip(log_probs, rewards_tensor)
    ]

    return torch.stack(losses).mean()