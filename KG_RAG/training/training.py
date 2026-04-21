from typing import List
import torch
import random
from main_RAPL_v2 import *

def train_deep_retriever(
    retriever,
    queries: List[str],
    model,  # SentenceTransformer
    epochs: int = 100,
    lr: float = 1e-3,
    print_every: int = 10
):
    """
    Train the DeepRetriever using REINFORCE.

    Parameters
    ----------
    retriever : DeepRetriever
        The retriever to train.
    queries : List[str]
        List of training queries.
    model : SentenceTransformer
        Model used to encode queries.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    print_every : int
        Logging frequency.

    Returns
    -------
    None
    """
    optimizer = torch.optim.Adam(
        retriever.PolicyNetwork.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
        total_loss = 0

        for query in queries:
            # 🔹 Encode query
            query_embedding = model.encode([query], convert_to_numpy=True)[0]

            # 🔹 Train step
            loss = retriever.train_step(query_embedding, optimizer)

            total_loss += loss.item()

        avg_loss = total_loss / len(queries)

        if epoch % print_every == 0:
            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

def train_deep_retriever_v2(
    retriever,
    queries: List[str],
    model,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-3
):
    """
    Train the DeepRetriever using REINFORCE with mini batch
    Parameters
    ----------
    retriever : DeepRetriever
        The retriever to train.
    queries : List[str]
        List of training queries.
    model : SentenceTransformer
        Model used to encode queries.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    print_every : int
        Logging frequency.

    Returns
    -------
    None
    """
    optimizer = torch.optim.Adam(
        retriever.PolicyNetwork.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
        #random.shuffle(queries)
        total_loss = 0

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]

            batch_loss = 0

            for query in batch:
                query_embedding = model.encode([query], convert_to_numpy=True)[0]
                loss = retriever.train_step(query_embedding, optimizer)
                batch_loss += loss

            batch_loss = batch_loss / len(batch)
            total_loss += batch_loss.item()

        print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")

# ====================================
# pipeline d'entrainement + evaluation
# ====================================

def train_one_epoch(
    retriever,
    queries_embeddings,
    optimizer,
    find_rational_paths_fn,
    loss_fn,
    device="cpu"
):
    retriever.PolicyNetwork.train()
    retriever.gnn.train()

    total_loss = 0

    for query_emb in queries_embeddings:
        loss = retriever.train_step(
            query_embedding=query_emb,
            optimizer=optimizer,
            find_rational_paths_fn=find_rational_paths_fn,
            reward_fn=None,
            supervised_loss_fn=loss_fn
        )

        total_loss += loss.item()

    return total_loss / len(queries_embeddings)

def evaluate(
    retriever,
    queries_embeddings,
    device="cpu"
):
    retriever.PolicyNetwork.eval()
    retriever.gnn.eval()

    total_reward = 0

    with torch.no_grad():
        for query_emb in queries_embeddings:
            paths = retriever.retrieve_paths(query_emb)

            rewards = [
                retriever.compute_reward(
                    [triple[0] for triple in path] + [path[-1][2]],
                    query_emb
                )
                for path in paths if len(path) > 0
            ]

            if rewards:
                total_reward += max(rewards)

    return total_reward / len(queries_embeddings)

def train_loop(
    retriever,
    train_queries_embeddings,
    val_queries_embeddings,
    optimizer,
    find_rational_paths_fn,
    loss_fn,
    epochs=10,
    device="cpu"
):
    history = {
        "train_loss": [],
        "val_reward": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # =========================
        # TRAIN
        # =========================
        train_loss = train_one_epoch(
            retriever,
            train_queries_embeddings,
            optimizer,
            find_rational_paths_fn,
            loss_fn,
            device
        )

        # =========================
        # VALIDATION
        # =========================
        val_reward = evaluate(
            retriever,
            val_queries_embeddings,
            device
        )

        history["train_loss"].append(train_loss)
        history["val_reward"].append(val_reward)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Reward: {val_reward:.4f}")

    return history

def find_rational_paths(G, query_emb, embeddings, node_to_idx, max_hops=3, top_k=3):
    """
    Generate pseudo-gold reasoning paths (RAPL-style) for training.

    This function explores the graph starting from nodes similar to the query,
    and builds candidate paths up to a maximum number of hops. Paths are then
    scored based on their relevance to the query.

    Parameters
    ----------
    G : nx.DiGraph
        Knowledge graph.
    query_emb : np.ndarray
        Query embedding vector.
    embeddings : np.ndarray
        Node embeddings matrix.
    node_to_idx : Dict[str, int]
        Mapping from node name to embedding index.
    max_hops : int, optional
        Maximum length of paths, by default 3.
    top_k : int, optional
        Number of top paths to return, by default 3.

    Returns
    -------
    List[List[str]]
        List of top-k reasoning paths.
    """
    sims = np.dot(embeddings, query_emb)
    start_indices = np.argsort(sims)[-top_k:]
    node_list = list(node_to_idx.keys())
    start_nodes = [node_list[i] for i in start_indices]

    paths = []

    for start in start_nodes:
        stack = [(start, [start])]

        while stack:
            current, path = stack.pop()

            if len(path) >= max_hops:
                paths.append(path)
                continue

            for neighbor in G.successors(current):
                if neighbor in path:
                    continue

                stack.append((neighbor, path + [neighbor]))

    scored = []
    for path in paths:
        last = path[-1]
        score = np.dot(embeddings[node_to_idx[last]], query_emb)
        score -= 0.1 * len(path)  # penalty for long paths
        scored.append((path, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [p for p, _ in scored[:top_k]]

def evaluate_retriever(retriever, queries, model):
    for query in queries:
        query_embedding = model.encode([query], convert_to_numpy=True)[0]

        paths = retriever.retrieve_paths(query_embedding)

        print("\nQuery:", query)
        for p in paths:
            print(" -> ".join(p))

# ==================================
# Main
# ==================================             

if __name__ == "__main__":
    # =========================
    # 1. Init pipeline
    # =========================
    pipeline = KGRAGPipeline(retriever_type="deep")

    # =========================
    # 2. Dataset (toy)
    # =========================
    queries = [
        "What causes sepsis?",
        "What are symptoms of sepsis?",
        "How is sepsis treated?",
        "How is Lactate liked to sepsis?",
        "How does sepsis affect organs?"
    ]

    # 🔹 Convert queries → embeddings
    train_embeddings = [
        pipeline.model.encode(q, convert_to_numpy=True)
        for q in queries
    ]

    # 🔹 Normalisation (important)
    train_embeddings = [
        emb / (np.linalg.norm(emb) + 1e-6)
        for emb in train_embeddings
    ]

    # split simple
    train_set = train_embeddings[:2]
    val_set = train_embeddings[2:]

    # =========================
    # 3. Optimizer
    # =========================
    optimizer = torch.optim.Adam(
        list(pipeline.retriever.PolicyNetwork.parameters()) +
        list(pipeline.retriever.gnn_encoder.parameters()),
        lr=1e-3
    )

    # =========================
    # 4. Loss function (simple)
    # =========================
    def loss_fn(path, query_emb, gold_paths):
        """
        Reward shaping simple :
        + similarité finale
        + bonus si overlap avec gold
        """
        last_node = path[-1]
        node_emb = pipeline.retriever.embeddings[
            pipeline.retriever.node_to_idx[last_node]
        ]

        sim = float(np.dot(node_emb, query_emb))

        # bonus imitation learning (overlap)
        overlap = 0
        for gold in gold_paths:
            overlap += len(set(path) & set(gold))

        return sim + 0.1 * overlap

    # =========================
    # 5. Training loop
    # =========================
    history = train_loop(
        retriever=pipeline.retriever,
        train_queries_embeddings=train_set,
        val_queries_embeddings=val_set,
        optimizer=optimizer,
        find_rational_paths_fn=find_rational_paths,
        loss_fn=loss_fn,
        epochs=20
    )

    # =========================
    # 6. Test inference
    # =========================
    query = "What causes sepsis?"
    answer, _ = pipeline.query(query)

    print("\nFinal Answer:")
    print(answer)