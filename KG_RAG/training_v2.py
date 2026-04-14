import torch
from typing import List, Callable
import numpy as np
from DeepRetrieval import *

# =================================
# Pseudo Ground truth
# =================================

def find_rational_paths(
    G,
    query_emb,
    embeddings,
    node_to_idx,
    max_hops=3,
    top_k=3,
    beam_width=5
):
    """
    Generate high-quality pseudo-gold reasoning paths using beam search.

    This function improves over naive DFS by:
    - scoring full paths (not only last node)
    - encouraging coherent paths
    - limiting exploration with beam search

    Returns paths as sequences of triples (h, r, t).
    """

    # =========================
    # 1. Start nodes (query similarity)
    # =========================
    sims = np.dot(embeddings, query_emb)
    start_indices = np.argsort(sims)[-beam_width:]
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    start_nodes = [idx_to_node[i] for i in start_indices]

    # =========================
    # 2. Beam search
    # =========================
    beams = []

    for start in start_nodes:
        beams.append(([], start))  # (path_triples, current_node)

    all_paths = []

    for _ in range(max_hops):
        new_beams = []

        for path, current in beams:
            neighbors = list(G.successors(current))

            for nbr in neighbors:
                if any(t[2] == nbr for t in path):
                    continue  # avoid cycles

                rel = G[current][nbr].get("relation", "related_to")
                triple = (current, rel, nbr)

                new_path = path + [triple]

                # =========================
                # 3. Path scoring
                # =========================
                nodes_in_path = [t[0] for t in new_path] + [new_path[-1][2]]

                # 🔹 Query relevance (mean)
                relevance = np.mean([
                    np.dot(embeddings[node_to_idx[n]], query_emb)
                    for n in nodes_in_path
                ])

                # 🔹 Path coherence
                coherence = 0
                for i in range(len(nodes_in_path) - 1):
                    u, v = nodes_in_path[i], nodes_in_path[i+1]
                    coherence += np.dot(
                        embeddings[node_to_idx[u]],
                        embeddings[node_to_idx[v]]
                    )
                coherence /= (len(nodes_in_path) - 1 + 1e-6)

                # 🔹 Length penalty
                length_penalty = -0.05 * len(new_path)

                score = relevance + 0.2 * coherence + length_penalty

                new_beams.append((new_path, nbr, score))

        # 🔹 Beam pruning
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = [(p, n) for p, n, _ in new_beams[:beam_width]]

        all_paths.extend([p for p, _, _ in new_beams])

    # =========================
    # 4. Final selection
    # =========================
    scored_paths = []

    for path in all_paths:
        nodes = [t[0] for t in path] + [path[-1][2]]

        score = np.mean([
            np.dot(embeddings[node_to_idx[n]], query_emb)
            for n in nodes
        ])

        scored_paths.append((path, score))

    scored_paths.sort(key=lambda x: x[1], reverse=True)

    return [p for p, _ in scored_paths[:top_k]]

# =================================
# Different Losses
# =================================

def compute_rl_loss(
    log_probs: List[torch.Tensor],
    rewards: List[float]
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

# =================================
# Training functions
# =================================

def training_step(
    retriever,
    query_embedding,
    optimizer,
    find_rational_paths_fn: Callable,
    reward_fn: Callable,
    supervised_loss_fn: Callable = None,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Perform one training step combining:
    - Reinforcement Learning (policy gradient)
    - Optional supervised imitation learning (RAPL-style)

    The training pipeline follows:
    1. Sample paths from the policy
    2. Generate pseudo-gold paths (rational paths)
    3. Compute rewards
    4. Compute RL loss
    5. Optionally compute supervised loss
    6. Backpropagate and update model

    Parameters
    ----------
    retriever : DeepRetriever
        Model containing GNN + policy network.
    query_embedding : np.ndarray
        Query embedding vector.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    find_rational_paths_fn : Callable
        Function generating pseudo-gold reasoning paths.
    reward_fn : Callable
        Function computing reward(path, query_embedding, gold_paths).
    supervised_loss_fn : Callable, optional
        Function computing supervised imitation loss.
    alpha : float, optional
        Weight of supervised loss in total loss.

    Returns
    -------
    torch.Tensor
        Total training loss.
    """

    # =========================
    # 1. Forward (sampling)
    # =========================
    # IMPORTANT: sample_paths must internally call encode_graph(force=True)
    paths, log_probs = retriever.sample_paths(query_embedding)

    # =========================
    # 2. Rational paths (RAPL)
    # =========================
    gold_paths = find_rational_paths_fn(
        retriever.G,
        query_embedding,
        retriever.embeddings,
        retriever.node_to_idx
    )

    # =========================
    # 3. Compute rewards
    # =========================
    rewards = [
        reward_fn(path, query_embedding, gold_paths)
        for path in paths
    ]

    rl_loss = compute_rl_loss(log_probs, rewards)

    # =========================
    # 4. Supervised loss (optional)
    # =========================
    if supervised_loss_fn is not None and len(gold_paths) > 0:
        sup_loss = supervised_loss_fn(
            retriever,
            query_embedding,
            gold_paths
        )
    else:
        sup_loss = torch.tensor(0.0, device=retriever.device)

    # =========================
    # 5. Total loss
    # =========================
    total_loss = rl_loss + alpha * sup_loss

    # =========================
    # 6. Backpropagation
    # =========================
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss

def train_loop(
    retriever,
    queries: List[str],
    embed_fn: Callable,
    optimizer,
    find_rational_paths_fn: Callable,
    reward_fn: Callable,
    supervised_loss_fn: Callable = None,
    epochs: int = 10,
    alpha: float = 0.5,
    verbose: bool = True
):
    """
    Main training loop for the DeepRetriever.

    This function iterates over queries for multiple epochs and applies
    the training_step function to update model parameters.

    Parameters
    ----------
    retriever : DeepRetriever
        Model to train.
    queries : List[str]
        List of natural language queries.
    embed_fn : Callable
        Function mapping a query string to an embedding vector.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.
    find_rational_paths_fn : Callable
        Function generating pseudo-gold reasoning paths.
    reward_fn : Callable
        Reward function.
    supervised_loss_fn : Callable, optional
        Supervised loss function.
    epochs : int, optional
        Number of training epochs.
    alpha : float, optional
        Weight for supervised loss.
    verbose : bool, optional
        Whether to print training progress.

    Returns
    -------
    None
    """

    retriever.PolicyNetwork.train()
    retriever.gnn_encoder.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for query in queries:
            query_embedding = embed_fn(query)

            loss = training_step(
                retriever,
                query_embedding,
                optimizer,
                find_rational_paths_fn,
                reward_fn,
                supervised_loss_fn,
                alpha
            )

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(queries)

        if verbose:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")