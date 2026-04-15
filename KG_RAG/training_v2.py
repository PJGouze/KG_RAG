import torch
from typing import List, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
from DeepRetrieval import *
from RAPL_main import *
from losses import calculate_reward_for_path, compute_rl_loss, compute_supervised_loss

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
# Training functions
# =================================

def check_gradients(model):
    total = 0.0
    count = 0

    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.abs().mean().item()
            count += 1

    print("avg grad signal:", total / max(count, 1))

def triples_to_nodes(path_triples):
    """
    A function to convert in the right format the path extracted

    Returns : 
        list of nodes
    """
    nodes = [path_triples[0][0]]
    for (_, _, t) in path_triples:
        nodes.append(t)
    return nodes

def training_step(
    retriever,
    query_embedding,
    optimizer,
    find_rational_paths_fn: Callable,
    calculate_reward_for_path:  Callable,
    compute_supervised_loss: Callable = None,
    alpha: float = 0.5,
    debug=False
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
    calculate_reward_for_path : Callable
        Function computing reward(path, query_embedding, gold_paths).
    compute_supervised_loss : Callable, optional
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
    gold_paths_triples = find_rational_paths_fn(
        retriever.G,
        query_embedding,
        retriever.embeddings,
        retriever.node_to_idx
    )

    gold_paths = [
        triples_to_nodes(p) for p in gold_paths_triples
    ]

    # =========================
    # 3. Compute rewards
    # =========================
    rewards = [
        calculate_reward_for_path(
            path,
            query_embedding,
            gold_paths,
            retriever.embeddings,
            retriever.node_to_idx
        )
        for path in paths
        ]

    rl_loss = compute_rl_loss(log_probs, rewards)

    # =========================
    # 4. Supervised loss (optional)
    # =========================
    if compute_supervised_loss is not None and len(gold_paths) > 0:
        sup_loss = compute_supervised_loss(
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
    if debug:
        print("===== GRADIENT CHECK GNN =====")
        for name, param in retriever.gnn_encoder.named_parameters():
            if param.grad is not None:
                print(name, param.grad.abs().mean().item())
            else:
                print(name, "NO GRAD")

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
                alpha,
                debug=True
            )

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(queries)

        if verbose:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

# =====================================================
# TRAIN PIPELINE
# =====================================================

class KGRAGTrainPipeline:
    """
    Training pipeline for Deep KG-RAG retriever.

    This class is ONLY responsible for:
    - building graph
    - building embeddings
    - initializing retriever
    - launching training loop
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.device = device

        # =========================
        # 1. Text encoder
        # =========================
        self.model = SentenceTransformer(model_name)

        # =========================
        # 2. Build graph
        # =========================
        self.graph = build_kg()

        # =========================
        # 3. Node embeddings (static)
        # =========================
        self.embeddings, self.node_to_idx, self.idx_to_node = build_node_embeddings(
            self.graph,
            self.model
        )

        # =========================
        # 4. Relation embeddings
        # =========================
        self.relation_embeddings = build_relation_embeddings(
            self.graph,
            self.model
        )

        dim = self.embeddings.shape[1]

        # =========================
        # 5. Policy Network
        # =========================
        self.policy_net = PolicyNetwork(
            input_dim=4 * dim,
            hidden_dim=128
        )

        # =========================
        # 6. GNN encoder
        # =========================
        self.gnn_encoder = GNNEncoder(
            dim=dim,
            num_layers=2
        )

        # =========================
        # 7. Deep Retriever
        # =========================
        self.retriever = DeepRetriever(
            G=self.graph,
            embeddings=self.embeddings,
            node_to_idx=self.node_to_idx,
            idx_to_node=self.idx_to_node,
            relation_embeddings=self.relation_embeddings,
            PolicyNetwork=self.policy_net,
            gnn_encoder=self.gnn_encoder,
            device=self.device
        )

        # =========================
        # 8. Optimizer
        # =========================
        self.optimizer = torch.optim.Adam(
            list(self.retriever.PolicyNetwork.parameters()) +
            list(self.retriever.gnn_encoder.parameters()),
            lr=1e-4
        )

    # =====================================================
    # EMBEDDING FUNCTION (used in training loop)
    # =====================================================
    def embed_fn(self, query: str):
        emb = self.model.encode([query], convert_to_numpy=True)
        return normalize(emb)[0]

    # =====================================================
    # TRAIN ENTRY POINT
    # =====================================================
    def train(
        self,
        queries,
        epochs: int = 10,
        alpha: float = 0.5,
        verbose: bool = True
    ):
        """
        Run full training loop.

        Parameters
        ----------
        queries : List[str]
            Training queries
        epochs : int
            Number of epochs
        alpha : float
            Weight of supervised loss
        """

        # =========================
        # Train mode
        # =========================
        self.retriever.PolicyNetwork.train()
        self.retriever.gnn_encoder.train()

        # =========================
        # Loop
        # =========================
        for epoch in range(epochs):
            total_loss = 0.0

            for query in queries:

                query_emb = self.embed_fn(query)

                loss = training_step(
                    retriever=self.retriever,
                    query_embedding=query_emb,
                    optimizer=self.optimizer,
                    find_rational_paths_fn=find_rational_paths,
                    calculate_reward_for_path=calculate_reward_for_path,
                    compute_supervised_loss=compute_supervised_loss,
                    alpha=alpha,
                    debug=True
                )

                total_loss += loss.item()

            avg_loss = total_loss / len(queries)

            if verbose:
                print(f"[Epoch {epoch+1}/{epochs}] Loss = {avg_loss:.4f}")

    # =====================================================
    # SAVE / LOAD (important for training)
    # =====================================================
    def save(self, path="retriever.pt"):
        """
        Save trained model weights.

        Saves both:
        - Policy Network
        - GNN encoder
        """
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "gnn_state_dict": self.gnn_encoder.state_dict(),
        }, path)

    def load(self, path="retriever.pt"):
        """
        Load trained model weights.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.gnn_encoder.load_state_dict(checkpoint["gnn_state_dict"])

# =====================================================
# MAIN (ENTRY POINT)
# =====================================================

if __name__ == "__main__": 

    # =========================
    # 1. Init training pipeline
    # =========================
    pipeline = KGRAGTrainPipeline(
        model_name="all-MiniLM-L6-v2",
        device="cpu"   # mets "cuda" si GPU
    )

    # =========================
    # 2. Training dataset
    # =========================
    queries = [
        "What causes sepsis?",
        "What are symptoms of sepsis?",
        "How is sepsis treated?",
        "What causes infection?",
        "What are symptoms of infection?",
        "What is hypotension?",
        "What is tachycardia?"
    ]

    # =========================
    # 3. Launch training
    # =========================
    print("🚀 Starting training...\n")

    pipeline.train(
        queries=queries,
        epochs=200,
        alpha=0.5,
        verbose=True
    )

    # =========================
    # 4. Save model
    # =========================
    pipeline.save("deep_retriever.pt")
    print("\n✅ Model saved")

    # =========================
    # 5. Quick evaluation
    # =========================
    print("\n🔎 Testing trained retriever...")

    test_query = "What causes sepsis?"
    query_emb = pipeline.embed_fn(test_query)

    paths = pipeline.retriever.retrieve_paths(query_emb)

    print("\nRetrieved paths:")
    for p in paths:
        print(p)