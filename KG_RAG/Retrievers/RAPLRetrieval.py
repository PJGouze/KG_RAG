import numpy as np
import networkx as nx
from typing import List, Tuple, Any, Dict
from collections import defaultdict


class RAPLRetriever:
    """
    Retriever implementing the RAPL (Rationalized Annotator, Path-based reasoning,
    and Line graph transformation) framework for Knowledge Graph QA.

    This class performs path-based reasoning over a line graph representation
    of a knowledge graph, using a learned policy network to sequentially select
    relevant triplets.

    The retrieval pipeline follows these steps:
        1. Construct a line graph from a query-specific subgraph
        2. Select initial triplets (start points)
        3. Perform multiple rollout simulations using a learned policy
        4. Rank and deduplicate candidate reasoning paths
        5. Return the top-M most probable reasoning paths

    Notes
    -----
    - Nodes in the line graph correspond to triplets (h, r, t)
    - Edges represent valid reasoning transitions between triplets
    - A special STOP action enables dynamic path termination
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        triplet_embeddings: np.ndarray,
        node_to_idx: Dict[Tuple[str, str, str], int],
        idx_to_node: Dict[int, Tuple[str, str, str]],
        policy_network: Any,
        gnn_encoder: Any,
        device: str = "cpu",
        start_k: int = 5,
        n_rollouts: int = 3,
        max_steps: int = 4,
        top_m: int = 5
    ):
        """
        Initialize the RAPL retriever.

        Parameters
        ----------
        graph : nx.DiGraph
            Original knowledge graph.
        triplet_embeddings : np.ndarray
            Embeddings for triplet nodes in the line graph.
        node_to_idx : dict
            Mapping from triplet (h, r, t) to embedding index.
        idx_to_node : dict
            Reverse mapping from index to triplet.
        policy_network : Any
            Neural network used to score candidate transitions.
        gnn_encoder : Any
            Graph neural network encoder for triplet representations.
        device : str, optional
            Computation device ("cpu" or "cuda"), by default "cpu".
        start_k : int, optional
            Number of initial triplets sampled at the start, by default 5.
        n_rollouts : int, optional
            Number of rollout simulations per start node, by default 3.
        max_steps : int, optional
            Maximum path length (upper bound), by default 4.
        top_m : int, optional
            Number of final paths returned after ranking, by default 5.
        """

        self.graph = graph
        self.triplet_embeddings = triplet_embeddings
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node

        self.policy_network = policy_network
        self.gnn_encoder = gnn_encoder

        self.device = device

        # Hyperparameters
        self.start_k = start_k
        self.n_rollouts = n_rollouts
        self.max_steps = max_steps
        self.top_m = top_m

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def retrieve_paths(
        self,
        query_embedding: np.ndarray,
        subgraph: nx.DiGraph
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Execute the full RAPL retrieval pipeline.

        Parameters
        ----------
        query_embedding : np.ndarray
            Vector representation of the input query.
        subgraph : nx.DiGraph
            Query-specific subgraph extracted around the question entity.

        Returns
        -------
        List[List[Tuple[str, str, str]]]
            A list of reasoning paths, where each path is a sequence of triplets.
        """

        # Step 1: Build line graph
        G_line = self.build_line_graph(subgraph)

        # Step 2: Select starting triplets
        start_triplets = self.get_start_triplets(query_embedding, G_line)

        all_paths = []

        # Step 3: Perform rollout simulations
        for start in start_triplets:
            for _ in range(self.n_rollouts):
                path, score = self.rollout(start, query_embedding, G_line)
                if path:
                    all_paths.append((path, score))

        # Step 4: Rank paths
        ranked_paths = self.rank_paths(all_paths)

        # Step 5: Deduplicate paths
        unique_paths = self.deduplicate_paths(ranked_paths)

        # Step 6: Select top-M
        return [p for p, _ in unique_paths[:self.top_m]]

    # =========================================================
    # GRAPH PROCESSING
    # =========================================================

    def build_line_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Convert a subgraph into its corresponding directed line graph.

        In the line graph:
            - Nodes represent triplets (h, r, t)
            - Directed edges connect consecutive reasoning steps:
              (h1, r1, t1) -> (h2, r2, t2) if t1 == h2

        Parameters
        ----------
        G : nx.DiGraph
            Input subgraph.

        Returns
        -------
        nx.DiGraph
            Line graph representation.
        """

        LG = nx.DiGraph()

        # index: subject → list of triplets
        subject_index = defaultdict(list)

        # =========================
        # Create nodes + index
        # =========================

        for h, t, data in G.edges(data=True):
            r = data.get("relation", "related_to")
            triplet = (h, r, t)
            LG.add_node(triplet)
            subject_index[h].append(triplet)

        # =========================
        # Create edges efficiently
        # =========================
        for h, t, data in G.edges(data=True):
            r = data.get("relation", "related_to")
            triplet1 = (h, r, t)

            # find all triplets starting from t
            for triplet2 in subject_index.get(t, []):

                LG.add_edge(triplet1, triplet2)
         
        return LG
    
    # =========================================================
    # START SELECTION
    # =========================================================

    def get_start_triplets(
        self,
        query_embedding: np.ndarray,
        G_line: nx.DiGraph,
        R_star: List[str],
        eq: str
        ) -> List[Tuple[str, str, str]]:
        """
        Select initial triplets for reasoning based on:
            - connection to question entity (eq)
            - relation filtering (R*)
            - similarity with query

        Parameters
        ----------
        query_embedding : np.ndarray
            Query vector representation.
        G_line : nx.DiGraph
            Line graph (nodes = triplets).
        R_star : List[str]
            Relevant relations for the query.
        eq : str
            Question entity.

        Returns
        -------
        List[Tuple[str, str, str]]
            Selected starting triplets.
        """

        candidates = []

        # =========================
        # 1. Filter by entity + relations
        # =========================
        for triplet in G_line.nodes():
            h, r, t = triplet

            if h == eq and r in R_star:
                candidates.append(triplet)

        # fallback if nothing found
        if len(candidates) == 0:
            for triplet in G_line.nodes():
                h, r, t = triplet
                if h == eq:
                    candidates.append(triplet)

        # fallback ultime
        if len(candidates) == 0:
            candidates = list(G_line.nodes())

        # =========================
        # 2. Score with similarity
        # =========================
        scores = []
        for triplet in candidates:
            idx = self.node_to_idx.get(triplet)

            if idx is None:
                scores.append(-1e9)
                continue

            emb = self.triplet_embeddings[idx]
            score = np.dot(emb, query_embedding)

            scores.append(score)

        # =========================
        # 3. Select top-k
        # =========================
        scores = np.array(scores)
        top_k_idx = np.argsort(scores)[-self.start_k:]

        start_triplets = [candidates[i] for i in top_k_idx]

        return start_triplets

    # =========================================================
    # ROLLOUT
    # =========================================================

    def rollout(
        self,
        start_triplet: Tuple[str, str, str],
        query_embedding: np.ndarray,
        G_line: nx.DiGraph,
        training: bool = False,
        return_details: bool = False
    ) -> Tuple[List[Tuple[str, str, str]], float]:
        """
        Perform a RAPL-style rollout (sequential reasoning path generation).

        The rollout iteratively selects the next triplet using the policy network,
        until either:
            - STOP is selected
            - max_steps is reached
            - no valid neighbors exist

        Parameters
        ----------
        start_triplet : tuple
            Initial triplet (h, r, t).
        query_embedding : np.ndarray
            Query representation.
        G_line : nx.DiGraph
            Line graph.
        training : bool, optional
            If True, uses stochastic sampling (exploration).
            If False, uses greedy selection.
        return_details : bool, optional
            If True, also returns states/actions/log-probs (useful for training).

        Returns
        -------
        Tuple[path, score] or Tuple[path, score, details]
            path : list of triplets
            score : cumulative log-probability
            details (optional) : dict for training
        """

        current = start_triplet
        path = [start_triplet]

        total_log_prob = 0.0

        # For training (policy gradient / supervised)
        states_history = []
        actions_history = []
        log_probs_history = []

        for step in range(self.max_steps):

            # =========================
            # 1. Get neighbors
            # =========================
            neighbors = self.get_neighbors(current, G_line)

            # Dead-end → stop
            if not neighbors:
                break

            # =========================
            # 2. Add STOP action
            # =========================
            candidates = neighbors + ["STOP"]

            # =========================
            # 3. Build states
            # =========================
            states = [
                self.build_state(query_embedding, current, c, path)
                for c in candidates
            ]

            # =========================
            # 4. Select next action
            # =========================
            next_action, prob = self.select_next(
                states,
                candidates,
                training=training
            )

            # Convert prob → log-prob (more stable for training)
            log_prob = np.log(prob + 1e-12)

            total_log_prob += log_prob

            # Store training info
            if training:
                states_history.append(states)
                actions_history.append(next_action)
                log_probs_history.append(log_prob)

            # =========================
            # 5. STOP condition (RAPL key)
            # =========================
            if next_action == "STOP":
                break

            # =========================
            # 6. Avoid cycles
            # =========================
            if next_action in path:
                break

            # =========================
            # 7. Move forward
            # =========================
            path.append(next_action)
            current = next_action

        # =========================
        # 8. Return
        # =========================
        if return_details:
            return path, total_log_prob, {
                "states": states_history,
                "actions": actions_history,
                "log_probs": log_probs_history
            }

        return path, total_log_prob
    
    # =========================================================
    # NEIGHBORS
    # =========================================================

    def get_neighbors(
        self,
        current_triplet: Tuple[str, str, str],
        G_line: nx.DiGraph
    ) -> List[Tuple[str, str, str]]:
        """
        Retrieve valid next triplets from the line graph.

        Parameters
        ----------
        current_triplet : tuple
            Current node in the line graph.
        G_line : nx.DiGraph
            Line graph.

        Returns
        -------
        List[tuple]
            Neighboring triplets.
        """
        return list(G_line.successors(current_triplet))

    # =========================================================
    # STATE CONSTRUCTION
    # =========================================================

    def build_state(
        self,
        query_embedding: np.ndarray,
        current_triplet: Tuple[str, str, str],
        candidate_triplet: Any,
        path: List[Tuple[str, str, str]]
        ) -> np.ndarray:
        """
        Build the state representation for the RAPL policy network.

        This state encodes:
            - global query intent
            - current reasoning position
            - candidate next action
            - accumulated reasoning path

        Parameters
        ----------
        query_embedding : np.ndarray
            Embedding of the input query.
        current_triplet : tuple
            Current node in the line graph (h, r, t).
        candidate_triplet : tuple or "STOP"
            Candidate next action.
        path : list of tuples
            Current reasoning history.

        Returns
        -------
        np.ndarray
            State vector used by the policy network.
        """

        # =========================
        # 1. Query embedding
        # =========================
        q = query_embedding

        # =========================
        # 2. Current triplet embedding
        # =========================
        idx_curr = self.node_to_idx.get(current_triplet)
        if idx_curr is None:
            curr_emb = np.zeros_like(q)
        else:
            curr_emb = self.triplet_embeddings[idx_curr]

        # =========================
        # 3. Candidate embedding
        # =========================
        if candidate_triplet == "STOP":
            cand_emb = self.compute_stop_embedding(path)
        else:
            idx_cand = self.node_to_idx.get(candidate_triplet)
            if idx_cand is None:
                cand_emb = np.zeros_like(q)
            else:
                cand_emb = self.triplet_embeddings[idx_cand]

        # =========================
        # 4. Path embedding (memory)
        # =========================
        if len(path) == 0:
            path_emb = np.zeros_like(q)
        else:
            path_emb = np.mean(
                [self.triplet_embeddings[self.node_to_idx[t]] for t in path],
                axis=0
            )

        # =========================
        # 5. Concatenation (RAPL-style state)
        # =========================
        state = np.concatenate([
            q,
            curr_emb,
            cand_emb,
            path_emb
        ])

        return state

    # =========================================================
    # POLICY
    # =========================================================

    def select_next(
        self,
        states: List[np.ndarray],
        candidates: List[Any],
        training: bool = False
    ) -> Tuple[Any, float]:
        """
        RAPL-style action selection using a learned policy network.

        This function computes scores for all candidate actions
        (triplets + STOP) and selects the next step in the reasoning path.

        Parameters
        ----------
        states : List[np.ndarray]
            State representations for each candidate action.
            Each state encodes:
                (query, current_triplet, candidate_triplet, path_memory)

        candidates : List[Any]
            Candidate triplets + special "STOP" action.

        training : bool, optional
            If True, samples stochastically (exploration).
            If False, uses argmax (inference).

        Returns
        -------
        Tuple[Any, float]
            Selected action (triplet or "STOP") and its score.
        """

        # =========================
        # 1. Stack states
        # =========================
        state_tensor = np.stack(states)  # (n_candidates, dim)

        # =========================
        # 2. Policy network forward pass
        # =========================
        # output: raw logits (one per candidate)
        logits = self.policy_network(state_tensor)

        logits = logits.squeeze()

        # =========================
        # 3. Numerical stability
        # =========================
        logits = logits - np.max(logits)

        probs = np.exp(logits) / np.sum(np.exp(logits))

        # =========================
        # 4. Selection strategy
        # =========================
        if training:
            # stochastic sampling (exploration)
            idx = np.random.choice(len(candidates), p=probs)
        else:
            # greedy selection (inference)
            idx = int(np.argmax(probs))

        selected_action = candidates[idx]
        selected_score = float(probs[idx])

        # =========================
        # 5. STOP handling (RAPL key idea)
        # =========================
        if selected_action == "STOP":
            return "STOP", selected_score

        return selected_action, selected_score

    # =========================================================
    # STOP NODE
    # =========================================================

    def compute_stop_embedding(
        self,
        path: List[Tuple[str, str, str]]
    ) -> np.ndarray:
        """
        Compute the embedding of the STOP node.

        Typically based on:
            - aggregation of path embeddings
            - transformation via MLP

        Parameters
        ----------
        path : list
            Current reasoning path.

        Returns
        -------
        np.ndarray
            STOP embedding.
        """
        raise NotImplementedError

    # =========================================================
    # POST-PROCESSING
    # =========================================================

    def rank_paths(
        self,
        paths: List[Tuple[List[Tuple[str, str, str]], float]]
    ):
        """
        Rank paths by their cumulative score.

        Parameters
        ----------
        paths : list of (path, score)

        Returns
        -------
        list
            Sorted paths.
        """
        return sorted(paths, key=lambda x: x[1], reverse=True)

    def deduplicate_paths(
        self,
        paths: List[Tuple[List[Tuple[str, str, str]], float]]
    ):
        """
        Remove duplicate reasoning paths.

        Parameters
        ----------
        paths : list of (path, score)

        Returns
        -------
        list
            Unique paths.
        """
        seen = set()
        unique = []

        for path, score in paths:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique.append((path, score))

        return unique