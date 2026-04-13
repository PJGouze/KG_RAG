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
) -> List[List[Tuple[str, str, str]]]:
        """
        Retrieve reasoning paths as sequences of triples.

        Each path is a list of triples (head, relation, tail),
        instead of raw nodes.

        Returns
        -------
        List[List[Tuple[str, str, str]]]
            List of paths, each path being a sequence of triples.
        """

        self.encode_graph()

        sims = np.dot(self.embeddings, query_embedding)
        start_indices = np.argsort(sims)[-start_k:]
        start_nodes = [self.idx_to_node[i] for i in start_indices]

        all_paths = []

        for start in start_nodes:
            current = start
            path = []

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

                if any(triple[2] == next_node for triple in path):
                    break

                rel = self.G[current][next_node].get("relation", "related_to")

                #  IMPORTANT : on stocke un TRIPLET
                triple = (current, rel, next_node)
                path.append(triple)

                current = next_node

            if path:
                all_paths.append(path)

        return all_paths

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

    def train_step(
        self,
        query_embedding: np.ndarray,
        optimizer,
        find_rational_paths_fn,
        reward_fn=None,
        supervised_loss_fn=None,
        alpha: float = 0.5
    ):
        """
        Perform one training step combining RL and optional supervised loss.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding.
        optimizer : torch.optim.Optimizer
            Optimizer.
        find_rational_paths_fn : callable
            Function generating pseudo-gold paths.
        reward_fn : callable, optional
            Function computing reward(path, query_embedding, gold_paths).
        supervised_loss_fn : callable, optional
            Function computing supervised loss.
        alpha : float
            Weight of supervised loss.

        Returns
        -------
        torch.Tensor
            Training loss.
        """

        # 🔹 Default functions
        if reward_fn is None:
            reward_fn = self.compute_reward

        if supervised_loss_fn is None:
            supervised_loss_fn = self.compute_supervised_loss

        # 🔹 Reset GNN (important pour backprop)
        self._cached_gnn_embeddings = None
        self.encode_graph()

        # =========================
        # 1. Rational paths
        # =========================
        gold_paths = find_rational_paths_fn(
            self.G,
            query_embedding,
            self.embeddings,
            self.node_to_idx
        )

        # =========================
        # 2. RL sampling
        # =========================
        paths, log_probs = self.sample_paths(query_embedding)

        rewards = []
        for path in paths:
            r = reward_fn(path, query_embedding, gold_paths)
            rewards.append(r)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # 🔹 Normalize
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        baseline = rewards.mean()

        rl_losses = []
        for log_prob, reward in zip(log_probs, rewards):
            advantage = reward - baseline
            rl_losses.append(-log_prob * advantage)

        rl_loss = torch.stack(rl_losses).mean()

        # =========================
        # 3. Supervised loss
        # =========================
        sup_loss = torch.tensor(0.0, device=self.device)

        if supervised_loss_fn is not None and len(gold_paths) > 0:
            sup_loss = supervised_loss_fn(query_embedding, gold_paths)

        # =========================
        # 4. Total loss
        # =========================
        total_loss = rl_loss + alpha * sup_loss

        # =========================
        # 5. Backprop
        # =========================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss
    

