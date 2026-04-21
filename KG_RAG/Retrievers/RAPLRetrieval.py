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
        G_line: nx.DiGraph
    ) -> Tuple[List[Tuple[str, str, str]], float]:
        """
        Perform a rollout (sequential path generation) starting from a triplet.

        At each step:
            - Retrieve neighboring triplets
            - Include STOP action
            - Select next action using policy network

        Parameters
        ----------
        start_triplet : tuple
            Initial triplet.
        query_embedding : np.ndarray
            Query representation.
        G_line : nx.DiGraph
            Line graph.

        Returns
        -------
        Tuple[List[Tuple[str, str, str]], float]
            Generated path and its cumulative score.
        """

        current = start_triplet
        path = []
        total_score = 0.0

        for _ in range(self.max_steps):

            neighbors = self.get_neighbors(current, G_line)

            if not neighbors:
                break

            candidates = neighbors + ["STOP"]

            states = [
                self.build_state(query_embedding, current, c, path)
                for c in candidates
            ]

            next_node, score = self.select_next(states, candidates)

            if next_node == "STOP":
                break

            path.append(next_node)
            total_score += score
            current = next_node

        return path, total_score

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
        Construct the state representation used by the policy network.

        Typically includes:
            - query embedding
            - current triplet embedding
            - candidate triplet embedding
            - path embedding (history)

        Parameters
        ----------
        query_embedding : np.ndarray
        current_triplet : tuple
        candidate_triplet : tuple or "STOP"
        path : list

        Returns
        -------
        np.ndarray
            State vector.
        """
        raise NotImplementedError

    # =========================================================
    # POLICY
    # =========================================================

    def select_next(
        self,
        states: List[np.ndarray],
        candidates: List[Any]
    ) -> Tuple[Any, float]:
        """
        Select the next action using the policy network.

        Parameters
        ----------
        states : list of np.ndarray
            State representations for each candidate.
        candidates : list
            Candidate actions (triplets + STOP).

        Returns
        -------
        Tuple[Any, float]
            Selected action and its score.
        """
        raise NotImplementedError

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