import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


# =========================
# 1. Knowledge Graph
# =========================

def build_kg() -> nx.DiGraph:
    """
    Build a directed biomedical Knowledge Graph.

    The graph contains nodes representing biomedical entities
    (e.g., symptoms, diseases, biological processes) and directed
    edges representing semantic relationships.

    Returns
    -------
    nx.DiGraph
        A directed graph where:
        - nodes are entity names (str) and description
        - edges have a "relation" attribute (str)
    """

    G = nx.DiGraph()

    # =========================
    # 1. Nodes (STRUCTURED)
    # =========================
    nodes = {
        "Sepsis": {
            "description": "Life-threatening condition caused by infection leading to organ dysfunction",
            "type": "disease",
            "synonyms": ["septic condition", "systemic infection"]
        },
        "Infection": {
            "description": "Invasion of the body by pathogenic microorganisms",
            "type": "condition",
            "synonyms": ["pathogen invasion"]
        },
        "Bacteria": {
            "description": "Microscopic organisms that can cause infections",
            "type": "pathogen",
            "synonyms": ["bacterial agent"]
        },
        "Fever": {
            "description": "Elevated body temperature, often due to infection",
            "type": "symptom",
            "synonyms": ["high temperature"]
        },
        "Hypotension": {
            "description": "Low blood pressure, common in sepsis and septic shock",
            "type": "symptom",
            "synonyms": ["low blood pressure"]
        },
        "Tachycardia": {
            "description": "Abnormally fast heart rate, often seen in infection",
            "type": "symptom",
            "synonyms": ["high heart rate"]
        },
        "Organ Failure": {
            "description": "Loss of function of one or more organs",
            "type": "condition",
            "synonyms": ["organ dysfunction"]
        },
        "Septic Shock": {
            "description": "Severe sepsis with persistent hypotension and organ failure",
            "type": "disease",
            "synonyms": ["shock due to sepsis"]
        },
        "Severe Hypotension": {
            "description": "Critically low blood pressure requiring intervention",
            "type": "symptom",
            "synonyms": []
        },
        "Multi-Organ Failure": {
            "description": "Failure of multiple organ systems",
            "type": "condition",
            "synonyms": []
        },
        "Bloodstream": {
            "description": "Circulatory system transporting blood",
            "type": "anatomy",
            "synonyms": []
        },
        "Immune Response": {
            "description": "Body defense mechanism against pathogens",
            "type": "process",
            "synonyms": ["immune reaction"]
        },
        "Inflammation": {
            "description": "Biological response to harmful stimuli",
            "type": "process",
            "synonyms": []
        },
        "Organ Dysfunction": {
            "description": "Impaired organ function",
            "type": "condition",
            "synonyms": []
        },
        "Antibiotics": {
            "description": "Drugs used to treat bacterial infections",
            "type": "treatment",
            "synonyms": []
        },
        "Fluid Resuscitation": {
            "description": "Administration of fluids to restore blood volume",
            "type": "treatment",
            "synonyms": []
        },
        "ICU": {
            "description": "Intensive care unit for critically ill patients",
            "type": "location",
            "synonyms": ["intensive care"]
        },
        "Lactate": {
            "description": "Biomarker indicating severity of sepsis and tissue hypoxia",
            "type": "biomarker",
            "synonyms": []
        },
        "Blood Culture": {
            "description": "Test used to detect bacteria in blood",
            "type": "test",
            "synonyms": []
        },
        "SOFA Score": {
            "description": "Clinical score assessing organ failure in sepsis",
            "type": "score",
            "synonyms": []
        },
        "Sepsis Severity": {
            "description": "Degree of severity of sepsis",
            "type": "concept",
            "synonyms": []
        },
    }

    for node, attributes in nodes.items():
        G.add_node(node, **attributes)

    # =========================
    # 2. Edges
    # =========================
    edges = [
        ("Sepsis", "Infection", "caused_by"),
        ("Sepsis", "Bacteria", "often_caused_by"),
        ("Sepsis", "Fever", "has_symptom"),
        ("Sepsis", "Hypotension", "has_symptom"),
        ("Sepsis", "Tachycardia", "has_symptom"),
        ("Sepsis", "Organ Failure", "can_lead_to"),
        ("Sepsis", "Septic Shock", "can_progress_to"),

        ("Septic Shock", "Sepsis", "is_a"),
        ("Septic Shock", "Severe Hypotension", "characterized_by"),
        ("Septic Shock", "Multi-Organ Failure", "can_lead_to"),

        ("Bacteria", "Bloodstream", "can_infect"),
        ("Infection", "Immune Response", "triggers"),
        ("Immune Response", "Inflammation", "causes"),
        ("Inflammation", "Organ Dysfunction", "can_lead_to"),

        ("Sepsis", "Antibiotics", "treated_with"),
        ("Sepsis", "Fluid Resuscitation", "treated_with"),
        ("Sepsis", "ICU", "managed_in"),

        ("Lactate", "Sepsis", "biomarker_of"),
        ("Blood Culture", "Bacteria", "detects"),
        ("SOFA Score", "Sepsis Severity", "assesses"),
    ]

    for source, target, relation in edges:
        G.add_edge(source, target, relation=relation)

    return G

# =========================
# 2. Embeddings
# =========================

def build_embeddings(
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

import torch
import numpy as np
from typing import List, Tuple, Dict


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
        device: str = "cpu"
    ):
        self.G = G
        self.embeddings = embeddings
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.relation_embeddings = relation_embeddings
        self.PolicyNetwork = PolicyNetwork.to(device)
        self.device = device

    # =========================
    # State construction
    # =========================

    def build_state(
        self,
        query_emb: np.ndarray,
        current_node: str,
        neighbor: str
    ) -> np.ndarray:
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
        np.ndarray
            Concatenated state vector.
        """
        node_emb = self.embeddings[self.node_to_idx[current_node]]
        neighbor_emb = self.embeddings[self.node_to_idx[neighbor]]

        rel = self.G[current_node][neighbor]["relation"]
        rel_emb = self.relation_embeddings.get(rel, np.zeros_like(node_emb))

        return np.concatenate([query_emb, node_emb, neighbor_emb, rel_emb])

    # =========================
    # Action selection
    # =========================

    def select_next(
        self,
        states: List[np.ndarray],
        candidates: List[str],
        training: bool = False
    ) -> Tuple[str, torch.Tensor]:
        """
        Select the next node among candidates using the policy network.

        Parameters
        ----------
        states : List[np.ndarray]
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
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)

        scores = self.PolicyNetwork(states_tensor).squeeze()  # (num_candidates,)
        probs = torch.softmax(scores, dim=0)

        if training:
            # Sampling for exploration (RL)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # Greedy selection (inference)
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
        Sample paths from the graph using the learned policy.

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


def linearize_graph(G_sub: nx.DiGraph) -> str:
    """
    Convert a graph into a textual representation (triples).

    Parameters
    ----------
    G_sub : nx.DiGraph
        Subgraph to linearize.

    Returns
    -------
    str
        Text representation of edges as triples.
    """
    triples = []
    for u, v, data in G_sub.edges(data=True):
        rel = data.get("relation", "related_to")
        triples.append(f"{u} --{rel}--> {v}")
    return "\n".join(triples)


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
                model_name: str = "all-MiniLM-L6-v2"
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
        self.model = SentenceTransformer(model_name)
        self.graph = build_kg()

        self.embeddings, self.node_to_idx, self.idx_to_node = build_embeddings(
            self.graph, self.model
        )

        self.index = build_faiss_index(self.embeddings)

        if retriever_type == "heuristic":
            self.retriever = HeuristicRetriever(
                self.graph,
                self.index,
                self.embeddings,
                self.node_to_idx,
                self.idx_to_node
            )

        elif retriever_type == "deep":
            self.retriever = self._init_deep_retriever()

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

        nodes = self.retriever.retrieve(query_embedding)

        subgraph = build_subgraph(self.graph, nodes)
        context = linearize_graph(subgraph)

        return generate_answer(context, query)


# =========================
# 8. Main
# =========================

if __name__ == "__main__":
    pipeline = KGRAGPipeline(retriever_type='heuristic')
    query = input("What is your query?")
    
    answer = pipeline.query(query)

    print(answer)