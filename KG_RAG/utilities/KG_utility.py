import networkx as nx
import numpy as np
import faiss
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict

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
# FAISS indexing
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

# ==============================
# Embeddings
# ==============================

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

def linearize_graph_v2(G: nx.DiGraph, paths):
    """
    Build a linearized graph of triples (RAPL-style).

    Each triple becomes a node, and edges are created between
    triples if:
        tail(triple_1) == head(triple_2)

    Parameters
    ----------
    G : nx.DiGraph
        Original knowledge graph (not directly used here but kept for compatibility).
    paths : List[List[Tuple[str, str, str]]]
        List of reasoning paths (triples).

    Returns
    -------
    str
        Textual representation of connected triples.
    """

    triples = []
    edges = []

    # 🔹 Flatten all triples
    for path in paths:
        for triple in path:
            triples.append(triple)

    # 🔹 Remove duplicates
    triples = list(set(triples))

    # 🔹 Build connections between triples
    for t1 in triples:
        for t2 in triples:
            if t1 != t2 and t1[2] == t2[0]:
                edges.append((t1, t2))

    # 🔹 Convert to text
    lines = []

    for t1, t2 in edges:
        line = f"({t1[0]}, {t1[1]}, {t1[2]}) -> ({t2[0]}, {t2[1]}, {t2[2]})"
        lines.append(line)

    return "\n".join(lines)

def linearize_graph_v3(G: nx.DiGraph) -> nx.DiGraph:
    """
    Convert a KG into its directed line graph.

    Nodes: triplets (h, r, t)
    Edges: (h1,r1,t1) -> (h2,r2,t2) if t1 == h2

    Parameters
    ----------
    G : nx.DiGraph
        Original knowledge graph (not directly used here but kept for compatibility).

    Returns
    -------
    nx.Digraph
        Representing a graph connecting triplets
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
        for triplet2 in subject_index[t]:
            LG.add_edge(triplet1, triplet2)

    return LG