import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from KG_utility import build_kg, build_subgraph, linearize_graph_v2, build_faiss_index, build_node_embeddings, build_relation_embeddings
from DeepRetrieval import DeepRetriever
from HeuristicRetrieval import search_nodes, get_neighbors, multi_hop_retrieval, HeuristicRetriever
from GNN_utility import PolicyNetwork, GNNEncoder, RelationalGATLayer

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
        paths = self.retriever.retrieve_paths_v2(query_embedding)
        
        #subgraph = build_subgraph(self.graph, nodes)
        context = linearize_graph_v2(self.graph,paths)

        return generate_answer(context, query)


# =========================
# 8. Main
# =========================

if __name__ == "__main__":
    pipeline = KGRAGPipeline(retriever_type='deep')
    query = "What causes sepsis?"
    
    answer = pipeline.query(query)

    print(answer)