import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple


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


# =========================
# 4. Multi-hop Retrieval
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
    - Multi-hop retrieval
    - Answer generation
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the pipeline.

        Parameters
        ----------
        model_name : str, optional
            SentenceTransformer model name, by default "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model_name)
        self.graph = build_kg()

        self.embeddings, self.node_to_idx, self.idx_to_node = build_embeddings(
            self.graph, self.model
        )

        self.index = build_faiss_index(self.embeddings)

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

        nodes = multi_hop_retrieval(
            self.graph,
            query_embedding,
            self.index,
            self.embeddings,
            self.node_to_idx,
            self.idx_to_node,
            hops=2,
            k=5
        )

        subgraph = build_subgraph(self.graph, nodes)
        context = linearize_graph(subgraph)

        return generate_answer(context, query)


# =========================
# 8. Main
# =========================

if __name__ == "__main__":
    pipeline = KGRAGPipeline()
    query = input("What is your query?")
    
    answer = pipeline.query(query)

    print(answer)