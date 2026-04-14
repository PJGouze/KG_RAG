import numpy as np
from typing import List
import torch

def compute_reward(
    self,
    path: List[str],
    query_emb: np.ndarray,
    gold_paths: List[List[str]] = None
) -> float:
    """
    Compute loss for a sampled path.

    The loss combines:
    - semantic similarity with the query
    - path coherence
    - length penalty
    - optional bonus if the path matches rational paths (RAPL-style)

    Parameters
    ----------
    path : List[str]
        Traversed path.
    query_emb : np.ndarray
        Query embedding.
    gold_paths : List[List[str]], optional
        Pseudo-gold reasoning paths.

    Returns
    -------
    float
        Loss value.
    """
    last_node = path[-1]
    node_emb = self.embeddings[self.node_to_idx[last_node]]

    # 🔹 1. Similarité avec la query
    sim_reward = float(np.dot(node_emb, query_emb))

    # 🔹 2. Pénalité longueur
    length_penalty = -0.05 * len(path)

    # 🔹 3. Cohérence interne du path
    coherence = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        coherence += np.dot(
            self.embeddings[self.node_to_idx[u]],
            self.embeddings[self.node_to_idx[v]]
        )
    coherence /= (len(path) - 1 + 1e-6)

    # 🔹 4. BONUS RAPL (matching partiel)
    rapl_bonus = 0
    if gold_paths is not None:
        for gold in gold_paths:
            overlap = len(set(path) & set(gold))
            rapl_bonus = max(rapl_bonus, overlap / len(gold))

    return sim_reward + length_penalty + 0.1 * coherence + 0.5 * rapl_bonus

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
