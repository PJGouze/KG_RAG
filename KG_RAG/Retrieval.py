import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import faiss
from typing import Dict, List, Tuple

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


class BaseRetriever:
    def retrieve(self, query_embedding):
        raise NotImplementedError


class DeepRetriever(BaseRetriever):
    """
    Deep Learning-based retriever for Knowledge Graph traversal.

    This retriever replaces heuristic multi-hop expansion with a learned
    policy network that selects which neighbor to explore at each step.

    It supports:
    - Inference mode (greedy traversal)
    - Training mode (stochastic sampling + log-prob tracking for RL)

    Parameters
    ----------
    G : nx.DiGraph
        Knowledge graph.
    embeddings : np.ndarray
        Node embeddings matrix of shape (num_nodes, dim).
    node_to_idx : Dict[str, int]
        Mapping from node name to embedding index.
    idx_to_node : Dict[int, str]
        Mapping from index to node name.
    relation_embeddings : Dict[str, np.ndarray]
        Embeddings for relations.
    PolicyNetwork : torch.nn.Module
        Neural network scoring actions.
    device : str, optional
        Device for computation ("cpu" or "cuda").
    """

    def __init__(
        self,
        G,
        embeddings,
        node_to_idx,
        idx_to_node,
        relation_embeddings,
        PolicyNetwork,
        gnn_encoder,
        device: str = "cpu"
    ):
        self.G = G
        self.embeddings = embeddings
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.relation_embeddings = relation_embeddings
        self.node_embeddings_torch = {
            node: torch.tensor(embeddings[idx], dtype=torch.float32).to(device)
            for node, idx in node_to_idx.items()
            }

        self.relation_embeddings = {
            k: torch.tensor(v, dtype=torch.float32).to(device)
            for k, v in relation_embeddings.items()
        }

        self.PolicyNetwork = PolicyNetwork.to(device)
        self.gnn = gnn_encoder.to(device)  
        self.device = device
        self._cached_gnn_embeddings = None
    
    def encode_graph(self):
        """
        Compute context-aware node embeddings using the GNN.

        Returns
        -------
        Dict[str, torch.Tensor]
            Updated node embeddings.
        """
        self._cached_gnn_embeddings = self.gnn(
            self.G,
            self.node_embeddings_torch,
            self.relation_embeddings
        )
        return self._cached_gnn_embeddings

    # =========================
    # State construction
    # =========================

    def build_state(
        self,
        query_emb: np.ndarray,
        current_node: str,
        neighbor: str
    ) -> torch.Tensor:
        """
        Build the state representation for a candidate action.

        The state encodes:
        - query embedding
        - current node embedding
        - candidate neighbor embedding
        - relation embedding

        Parameters
        ----------
        query_emb : np.ndarray
            Query embedding vector.
        current_node : str
            Current node in traversal.
        neighbor : str
            Candidate next node.

        Returns
        -------
        torch.Tensor
            Concatenated state vector.
        """
        # convertir query
        query_tensor = torch.tensor(query_emb, dtype=torch.float32).to(self.device)

        # embeddings GNN
        node_emb = self._cached_gnn_embeddings[current_node]
        neighbor_emb = self._cached_gnn_embeddings[neighbor]

        rel = self.G[current_node][neighbor]["relation"]
        rel_emb = self.relation_embeddings.get(
            rel,
            torch.zeros_like(node_emb)
        )

        return torch.cat([query_tensor, node_emb, neighbor_emb, rel_emb])

    # =========================
    # Action selection
    # =========================

    def select_next(
        self,
        states: List[torch.Tensor],
        candidates: List[str],
        training: bool = False
    ) -> Tuple[str, torch.Tensor]:
        """
        Select the next node among candidates using the policy network.

        Parameters
        ----------
        states : List[torch.Tensor]
            List of state vectors (one per candidate).
        candidates : List[str]
            Candidate neighbor nodes.
        training : bool, optional
            If True, samples from distribution (for RL).
            If False, selects argmax (greedy inference).

        Returns
        -------
        next_node : str
            Selected next node.
        log_prob : torch.Tensor
            Log-probability of the selected action (used for RL).
        """
        states_tensor = torch.stack(states).to(self.device)

        scores = self.PolicyNetwork(states_tensor).squeeze(-1)
        probs = torch.softmax(scores, dim=0)

        if training:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-10)

        return candidates[action.item()], log_prob

    # =========================
    # Path sampling
    # =========================

    def sample_paths(
        self,
        query_embedding: np.ndarray,
        start_k: int = 5,
        steps: int = 3
    ) -> Tuple[List[List[str]], List[torch.Tensor]]:
        """
        Sample paths from the graph using the GNN and the learned policy.

        This function is used during training and returns both:
        - sampled paths
        - log-probabilities for REINFORCE

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding vector.
        start_k : int, optional
            Number of starting nodes.
        steps : int, optional
            Number of traversal steps.

        Returns
        -------
        paths : List[List[str]]
            Sampled paths.
        log_probs : List[torch.Tensor]
            Log-probabilities of each path.
        """

        self.encode_graph()
        sims = np.dot(self.embeddings, query_embedding)
        start_indices = np.argsort(sims)[-start_k:]
        start_nodes = [self.idx_to_node[i] for i in start_indices]

        paths = []
        log_probs = []

        for start in start_nodes:
            current = start
            path = [current]
            path_log_prob = 0

            for _ in range(steps):
                neighbors = list(self.G.successors(current))

                if not neighbors:
                    break

                states = [
                    self.build_state(query_embedding, current, n)
                    for n in neighbors
                ]

                next_node, log_prob = self.select_next(
                    states,
                    neighbors,
                    training=True
                )

                path.append(next_node)
                path_log_prob += log_prob
                current = next_node

            paths.append(path)
            log_probs.append(path_log_prob)

        return paths, log_probs

    # =========================
    # Inference retrieval
    # =========================

    def retrieve(
        self,
        query_embedding: np.ndarray,
        start_k: int = 5,
        steps: int = 3
    ) -> List[str]:
        """
        Retrieve relevant nodes using greedy policy traversal.

        This function is used at inference time.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding vector.
        start_k : int, optional
            Number of starting nodes.
        steps : int, optional
            Number of traversal steps.

        Returns
        -------
        List[str]
            Set of visited nodes.
        """
        self.encode_graph()

        sims = np.dot(self.embeddings, query_embedding)
        start_indices = np.argsort(sims)[-start_k:]
        start_nodes = [self.idx_to_node[i] for i in start_indices]

        visited = set(start_nodes)

        for start in start_nodes:
            current = start

            for _ in range(steps):
                neighbors = list(self.G.successors(current))

                if not neighbors:
                    break

                states = [
                    self.build_state(query_embedding, current, n)
                    for n in neighbors
                ]

                next_node, _ = self.select_next(
                    states,
                    neighbors,
                    training=False
                )

                if next_node in visited:
                    break

                visited.add(next_node)
                current = next_node

        return list(visited)

    def compute_reward(
        self,
        path: List[str],
        query_emb: np.ndarray
    ) -> float:
        """
        Compute reward for a sampled path.

        The reward is based on the similarity between the query embedding
        and the embedding of the final node in the path.

        Parameters
        ----------
        path : List[str]
            Traversed path.
        query_emb : np.ndarray
            Query embedding.

        Returns
        -------
        float
            Reward value.
        """
        last_node = path[-1]
        node_emb = self.embeddings[self.node_to_idx[last_node]]

        reward = float(np.dot(node_emb, query_emb))

        return reward
    
    def train_step(
    self,
    query_embedding: np.ndarray,
    optimizer
    ):
        """
        Perform one REINFORCE training step.

        This function:
        1. Samples paths using the current policy
        2. Computes rewards for each path
        3. Updates the policy using REINFORCE

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding.
        optimizer : torch.optim.Optimizer
            Optimizer for policy network.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        # 🔹 Sample paths
        paths, log_probs = self.sample_paths(query_embedding)

        rewards = []
        for path in paths:
            r = self.compute_reward(path, query_embedding)
            rewards.append(r)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # 🔥 NORMALISATION (CRUCIAL pour stabilité)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # 🔹 Compute loss
        losses = []
        for log_prob, reward in zip(log_probs, rewards):
            losses.append(-log_prob * reward)

        loss = torch.stack(losses).mean()

        # 🔹 Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss