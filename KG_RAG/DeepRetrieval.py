import torch
import numpy as np
from typing import List, Tuple

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
        Compute context-aware node embeddings using the GNN if it has not been encoded before

        Returns
        -------
        Dict[str, torch.Tensor]
            Updated node embeddings.
        """
        if self._cached_gnn_embeddings is None:
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
        temperature = 0.5
        probs = torch.softmax(scores / temperature, dim=0)

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

                if next_node in path:
                    break
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

    def retrieve_paths(
    self,
    query_embedding: np.ndarray,
    start_k: int = 5,
    steps: int = 3
) -> List[List[str]]:
        """
    Retrieve reasoning paths from the knowledge graph using a learned policy.

    This function performs inference-time traversal of the knowledge graph,
    guided by the trained policy network. Unlike `retrieve`, which returns
    a set of visited nodes, this method explicitly returns structured paths,
    making it suitable for reasoning-based downstream tasks (e.g., RAPL-style
    linearization for LLM prompting).

    The traversal is deterministic (greedy) and operates as follows:
    - Select top-k starting nodes based on similarity to the query
    - Iteratively expand each path using the policy network
    - At each step, select the most relevant neighbor (argmax policy)
    - Stop when no valid expansion is possible or a cycle is detected

    Parameters
    ----------
    query_embedding : np.ndarray
        Normalized embedding vector representing the input query.
    start_k : int, optional
        Number of initial nodes selected via similarity search (default: 5).
    steps : int, optional
        Maximum number of expansion steps per path (default: 3).

    Returns
    -------
    List[List[str]]
        A list of reasoning paths, where each path is a sequence of nodes.

    Algorithm
    ---------
    1. Encode the graph using the GNN to obtain context-aware node embeddings.
    2. Compute similarity between the query and all node embeddings.
    3. Select the top-k most relevant nodes as starting points.
    4. For each starting node:
        a. Initialize a path with the starting node.
        b. For a fixed number of steps:
            - Retrieve outgoing neighbors of the current node.
            - Build state representations for each candidate neighbor.
            - Score candidates using the policy network.
            - Select the best candidate (greedy).
            - Append it to the path.
            - Stop if:
                * no neighbors exist
                * the next node creates a cycle
    5. Return all generated paths.

    Notes
    -----
    - This function is intended for inference (no stochastic sampling).
    - It preserves path structure, unlike `retrieve`.
    - Compatible with RAPL-style linearization.
    - The GNN embeddings are cached for efficiency.

    Limitations
    -----------
    - Greedy selection may miss globally optimal paths.
    - No beam search or reranking is applied.
    - Only forward traversal (successors) is considered.

    Possible Improvements
    ---------------------
    - Beam search instead of greedy decoding.
    - Path scoring and reranking.
    - Bidirectional traversal (add predecessors).
    - Early stopping based on confidence threshold.
        """

    # 🔹 Encode graph (cached)
        self.encode_graph()

    # 🔹 Select starting nodes via similarity
        sims = np.dot(self.embeddings, query_embedding)
        start_indices = np.argsort(sims)[-start_k:]
        start_nodes = [self.idx_to_node[i] for i in start_indices]

        paths = []

        for start in start_nodes:
            current = start
            path = [current]

            for _ in range(steps):
                neighbors = list(self.G.successors(current))

                if not neighbors:
                    break

            # 🔹 Build states
                states = [
                    self.build_state(query_embedding, current, n)
                    for n in neighbors
                ]

            # 🔹 Greedy selection
                next_node, _ = self.select_next(
                    states,
                    neighbors,
                    training=False
                )

            # 🔹 Stop if cycle
                if next_node in path:
                    break

            path.append(next_node)
            current = next_node

        paths.append(path)

        return paths

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

        sim_reward = float(np.dot(node_emb, query_emb))

        # pénalité longueur
        length_penalty = -0.05 * len(path)

        coherence = 0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            coherence += np.dot(
                self.embeddings[self.node_to_idx[u]],
                self.embeddings[self.node_to_idx[v]]
            )

        coherence /= (len(path) - 1 + 1e-6)

        return sim_reward + length_penalty + 0.1 * coherence
    
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
        self._cached_gnn_embeddings = None

        # 🔹 Sample paths
        paths, log_probs = self.sample_paths(query_embedding)

        rewards = []
        for path in paths:
            r = self.compute_reward(path, query_embedding)
            rewards.append(r)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # NORMALISATION (CRUCIAL pour stabilité)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # 🔹 Compute loss
        baseline = rewards.mean()

        losses = []
        for log_prob, reward in zip(log_probs, rewards):
            advantage = reward - baseline
            losses.append(-log_prob * advantage)
        
        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
