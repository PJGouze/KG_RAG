import faiss
import numpy as np
import networkx as nx
from typing import List, Dict

class BaseRetriever:
    def retrieve(self, query_embedding):
        raise NotImplementedError

## Heuristic method ##
def search_nodes(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 5
) -> List[int]:
    """
    Retrieve the top-k most similar nodes from the FAISS index.

    Parameters
    ----------
    index : faiss.Index
        FAISS index containing node embeddings.
    query_embedding : np.ndarray
        Normalized embedding vector of the query (shape: dim).
    k : int, optional
        Number of nearest neighbors to retrieve, by default 5.

    Returns
    -------
    List[int]
        Indices of the top-k most similar nodes.
    """
    query = query_embedding.reshape(1, -1).astype("float32")
    _, indices = index.search(query, k)
    return indices[0].tolist()

def get_neighbors(G: nx.DiGraph, node: str) -> List[str]:
    """
    Retrieve both incoming and outgoing neighbors of a node.

    Parameters
    ----------
    G : nx.DiGraph
        Directed knowledge graph.
    node : str
        Node for which neighbors are retrieved.

    Returns
    -------
    List[str]
        List of neighboring nodes (predecessors + successors).
    """
    successors = list(G.successors(node))
    predecessors = list(G.predecessors(node))
    return successors + predecessors


def multi_hop_retrieval(
    G: nx.DiGraph,
    query_embedding: np.ndarray,
    index: faiss.Index,
    embeddings: np.ndarray,
    node_to_idx: Dict[str, int],
    idx_to_node: Dict[int, str],
    hops: int = 2,
    k: int = 5
) -> List[str]:
    """
    Perform multi-hop retrieval over the Knowledge Graph.

    The algorithm:
    1. Retrieves initial nodes via FAISS similarity search
    2. Expands neighbors iteratively (multi-hop)
    3. Ranks candidates using embedding similarity
    4. Keeps top-k nodes per hop

    Parameters
    ----------
    G : nx.DiGraph
        Directed knowledge graph.
    query_embedding : np.ndarray
        Normalized embedding of the query.
    index : faiss.Index
        FAISS index for initial retrieval.
    embeddings : np.ndarray
        Node embeddings matrix.
    node_to_idx : Dict[str, int]
        Mapping node -> embedding index.
    idx_to_node : Dict[int, str]
        Mapping index -> node.
    hops : int, optional
        Number of expansion steps, by default 2.
    k : int, optional
        Number of nodes retained at each step, by default 5.

    Returns
    -------
    List[str]
        Set of retrieved nodes after multi-hop expansion.
    """
    start_indices = search_nodes(index, query_embedding, k)
    current_nodes = [idx_to_node[i] for i in start_indices]

    visited = set(current_nodes)

    for _ in range(hops):
        candidates = set()

        for node in current_nodes:
            neighbors = get_neighbors(G, node)
            candidates.update(neighbors)

        candidates = list(candidates - visited)

        if not candidates:
            break

        candidate_indices = [node_to_idx[n] for n in candidates]
        candidate_embeddings = embeddings[candidate_indices]

        scores = np.dot(candidate_embeddings, query_embedding)

        top_k_idx = np.argsort(scores)[-k:]
        current_nodes = [candidates[i] for i in top_k_idx]

        visited.update(current_nodes)

    return list(visited)


class HeuristicRetriever(BaseRetriever):
    def __init__(self, G, index, embeddings, node_to_idx, idx_to_node):
        self.G = G
        self.index = index
        self.embeddings = embeddings
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node

    def retrieve(self, query_embedding, hops=2, k=5):
        return multi_hop_retrieval(
            self.G,
            query_embedding,
            self.index,
            self.embeddings,
            self.node_to_idx,
            self.idx_to_node,
            hops=hops,
            k=k
        )