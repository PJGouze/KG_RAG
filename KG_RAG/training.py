from typing import List
import torch
from RAPL_main import *

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


import random

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
        random.shuffle(queries)
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
    pipeline = KGRAGPipeline(retriever_type='deep')

    queries = [
        "What causes sepsis?",
        "What are symptoms of sepsis?",
        "How is sepsis treated?"
    ]

    train_deep_retriever(
        pipeline.retriever,
        queries,
        pipeline.model,
        epochs=50
    )

    # Test
    answer = pipeline.query("What causes sepsis?")
    print(answer)