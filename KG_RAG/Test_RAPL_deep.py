import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from KG_utility import build_kg

# =========================
# 2. Embeddings
# =========================

def build_node_embeddings(
    G: nx.DiGraph,
    model: SentenceTransformer
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Compute embeddings for all nodes in the graph using enriched node attributes.

    Each node is represented by a textual description including:
    - its name
    - its description
    - its type
    - its synonyms

    The resulting embeddings are normalized for cosine similarity.

    Parameters
    ----------
    G : nx.DiGraph
        Input graph containing nodes with attributes:
        'description', 'type', 'synonyms'.
    model : SentenceTransformer
        Pretrained embedding model.

    Returns
    -------
    embeddings : np.ndarray
        Normalized embeddings of shape (num_nodes, dim).
    node_to_idx : Dict[str, int]
        Mapping from node name to embedding index.
    idx_to_node : Dict[int, str]
        Reverse mapping from index to node name.
    """

    node_list = list(G.nodes)

    # =========================
    # Build enriched text
    # =========================
    texts = []
    for node in node_list:
        data = G.nodes[node]

        description = data.get("description", "")
        node_type = data.get("type", "")
        synonyms = data.get("synonyms", [])

        synonyms_text = ", ".join(synonyms) if synonyms else ""

        text = f"{node}. {description}. Type: {node_type}. Synonyms: {synonyms_text}"
        texts.append(text)

    # =========================
    # Compute embeddings
    # =========================
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = normalize(embeddings)

    # =========================
    # Mappings
    # =========================
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    return embeddings, node_to_idx, idx_to_node

def build_relation_embeddings(G, model):
    """
    Compute embeddings for all relation types in the knowledge graph.

    Each unique relation label present in the graph edges is encoded
    using the same sentence transformer model as the nodes. This allows
    the model to incorporate semantic information about edge types
    during graph traversal.

    Parameters
    ----------
    G : nx.DiGraph
        Input knowledge graph. Each edge must have a "relation" attribute.
    model : SentenceTransformer
        Pretrained sentence transformer used to encode relation labels.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each relation label to its embedding vector.

    Notes
    -----
    - Relation embeddings are shared across all edges of the same type.
    - These embeddings are later converted to torch tensors inside the
        DeepRetriever.
    - If a relation is missing at inference time, a zero vector is used
    as fallback.
    """
    relations = list(set([data["relation"] for _, _, data in G.edges(data=True)]))
    rel_embeddings = model.encode(relations, convert_to_numpy=True)
    rel_embeddings = normalize(rel_embeddings)
    return dict(zip(relations, rel_embeddings))

# =========================
# 3. FAISS Index
# =========================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for fast similarity search.

    The index uses inner product similarity, which is equivalent
    to cosine similarity when embeddings are normalized.

    Parameters
    ----------
    embeddings : np.ndarray
        Normalized embeddings of shape (num_nodes, dim).

    Returns
    -------
    faiss.Index
        FAISS index containing all node embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


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

## Deep method ##
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
    
# =========================
# 4. Retrievers
# =========================

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

class RelationalGATLayer(nn.Module):
    """
    Relational Graph Attention Layer.

    This layer performs message passing over a directed graph while
    incorporating edge (relation) information into the attention mechanism.

    For each node, it aggregates messages from its predecessors using
    attention weights computed from:
    - source node embedding
    - target node embedding
    - relation embedding

    Parameters
    ----------
    dim : int
        Dimension of node and relation embeddings.
    """

    def __init__(self, dim):
        super().__init__()
        self.W_node = nn.Linear(dim, dim)
        self.W_rel = nn.Linear(dim, dim)
        self.attn = nn.Linear(3 * dim, 1)

    def forward(self, G, node_embeddings, relation_embeddings):
        """
        Forward pass of the relational GAT layer.

        Parameters
        ----------
        G : nx.DiGraph
            Input knowledge graph.
        node_embeddings : Dict[str, torch.Tensor]
            Dictionary mapping node names to embedding tensors.
        relation_embeddings : Dict[str, torch.Tensor]
            Dictionary mapping relation names to embedding tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Updated node embeddings after one message passing step.
        """
        new_embeddings = {}

        for node in G.nodes:
            neighbors = list(G.predecessors(node))

            if not neighbors:
                new_embeddings[node] = node_embeddings[node]
                continue

            messages = []
            attn_scores = []

            for nbr in neighbors:
                rel = G[nbr][node]["relation"]

                h_src = node_embeddings[nbr]
                h_tgt = node_embeddings[node]
                h_rel = relation_embeddings[rel]

                attn_input = torch.cat([h_src, h_tgt, h_rel])
                score = self.attn(attn_input)

                message = self.W_node(h_src) + self.W_rel(h_rel)

                messages.append(message)
                attn_scores.append(score)

            attn_scores = torch.softmax(torch.stack(attn_scores), dim=0)
            agg = sum(a * m for a, m in zip(attn_scores, messages))

            new_embeddings[node] = agg

        return new_embeddings


class GNNEncoder(nn.Module):
    """
    Multi-layer Graph Neural Network encoder with relational attention.

    This encoder stacks multiple RelationalGATLayer layers to compute
    context-aware node embeddings based on graph structure and relations.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    num_layers : int, optional
        Number of GNN layers, by default 2.
    """

    def __init__(self, dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            RelationalGATLayer(dim) for _ in range(num_layers)
        ])

    def forward(self, G, node_embeddings, relation_embeddings):
        """
        Forward pass of the GNN encoder.

        Parameters
        ----------
        G : nx.DiGraph
            Input knowledge graph.
        node_embeddings : Dict[str, torch.Tensor]
            Initial node embeddings.
        relation_embeddings : Dict[str, torch.Tensor]
            Relation embeddings.

        Returns
        -------
        Dict[str, torch.Tensor]
            Updated node embeddings after all GNN layers.
        """
        h = node_embeddings
        for layer in self.layers:
            h = layer(G, h, relation_embeddings)
        return h

def init_node_embeddings_tensor(embeddings, idx_to_node, device):
    """
    Convert numpy node embeddings into PyTorch tensors.

    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings matrix of shape (num_nodes, dim).
    idx_to_node : Dict[int, str]
        Mapping from index to node name.
    device : str
        Device to place tensors on ("cpu" or "cuda").

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary mapping node names to torch tensors.
    """
    return {
        node: torch.tensor(embeddings[idx], dtype=torch.float32).to(device)
        for idx, node in idx_to_node.items()
    }



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

        if self._cached_gnn_embeddings is None:
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
    
# =========================
# 5. Subgraph & Context
# =========================

def build_subgraph(G: nx.DiGraph, nodes: List[str]) -> nx.DiGraph:
    """
    Extract a subgraph induced by a set of nodes.

    Parameters
    ----------
    G : nx.DiGraph
        Original graph.
    nodes : List[str]
        Nodes to include in the subgraph.

    Returns
    -------
    nx.DiGraph
        Subgraph containing only the specified nodes.
    """
    return G.subgraph(nodes).copy()


def linearize_graph(G: nx.DiGraph, paths: List[List[str]]) -> str:
    """
    Convert reasoning paths into a textual representation.

    This function follows the RAPL paradigm: instead of representing
    the graph as independent triples, it encodes structured reasoning
    paths as sequences of nodes and relations.

    Parameters
    ----------
    G : nx.DiGraph
        Knowledge graph.
    paths : List[List[str]]
        List of paths, where each path is a sequence of nodes.

    Returns
    -------
    str
        Textual representation of reasoning paths.
    """
    path_texts = []

    for path in paths:
        if len(path) < 2:
            continue

        elements = [path[0]]

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            rel = G[u][v].get("relation", "related_to")

            elements.append(rel)
            elements.append(v)

        path_text = " -> ".join(elements)
        path_texts.append(path_text)

    return "\n".join(path_texts)


# =========================
# 6. Answer Generation
# =========================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_answer(context: str, query: str) -> str:
    """
    Generate a final answer using a TinyLlama LLM.

    This function formats a prompt combining the user query and
    the retrieved knowledge graph context, then uses a causal
    language model to generate an answer.

    Parameters
    ----------
    context : str
        Linearized knowledge graph context.
    query : str
        User query.

    Returns
    -------
    str
        Generated answer from the LLM.
    """

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Load tokenizer + model (idéalement à faire une seule fois en prod)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # ou float16 si GPU
        device_map="cpu"
    )

    # Prompt structuré (important pour la qualité)
    prompt = f"""You are a medical assistant using a knowledge graph.

    Answer the question ONLY using the information from the graph.
    Be concise and medically accurate.

    Question:
    {query}

    Knowledge Graph retrieval:
    {context}

    Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Nettoyage pour ne garder que la réponse
    answer = response.split("Answer:")[-1].strip()

    return answer, query

# =========================
# 7. Pipeline
# =========================

class KGRAGPipeline:
    """
    End-to-end pipeline for Knowledge Graph Retrieval-Augmented Generation.

    This class encapsulates:
    - Graph construction
    - Node embedding
    - FAISS indexing
    - KG retrieval
    - Answer generation
    """

    def __init__(self,
                retriever_type: str ="heuristic",
                model_name: str = "all-MiniLM-L6-v2",
                device: str = "cpu"
                ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        retriever_type : str, mandatory, defining the method of KG exploration
            by default "heuristic", can be "deep"
        model_name : str, optional
            SentenceTransformer model name, by default "all-MiniLM-L6-v2".
        """
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.graph = build_kg()

        self.embeddings, self.node_to_idx, self.idx_to_node = build_node_embeddings(
            self.graph, self.model
        )

        self.index = build_faiss_index(self.embeddings)
        # =========================
        # Heuristic retriever
        # =========================
        if retriever_type == "heuristic":
            self.retriever = HeuristicRetriever(
                self.graph,
                self.index,
                self.embeddings,
                self.node_to_idx,
                self.idx_to_node
            )

        # =========================
        # Deep retriever
        # =========================
        elif retriever_type == "deep":
            self.relation_embeddings = build_relation_embeddings(
                self.graph,
                self.model
            )

            dim = self.embeddings.shape[1]

            # Policy network
            self.policy_net = PolicyNetwork(
                input_dim=4 * dim,  # query + node + neighbor + relation
                hidden_dim=128
            )

            # GNN encoder
            self.gnn_encoder = GNNEncoder(
                dim=dim,
                num_layers=2
            )
            # Deep retriever
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

        else:
            raise ValueError("Unknown retriever type")

    def query(self, query: str) -> str:
        """
        Run the full KG-RAG pipeline for a given query.

        Parameters
        ----------
        query : str
            Input natural language query.

        Returns
        -------
        str
            Generated answer.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = normalize(query_embedding)[0]

        #nodes = self.retriever.retrieve(query_embedding)
        paths, _ = self.retriever.sample_paths(query_embedding)
        
        #subgraph = build_subgraph(self.graph, nodes)
        context = linearize_graph(self.graph,paths)

        return generate_answer(context, query)


# =========================
# 8. Main
# =========================

if __name__ == "__main__":
    pipeline = KGRAGPipeline(retriever_type='deep')
    query = "What causes sepsis?"
    
    answer = pipeline.query(query)

    print(answer)