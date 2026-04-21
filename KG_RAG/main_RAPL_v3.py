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

    Answer the question using ONLY the information from the graph.
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

class NewKGRAGPipeline:
    """
    Pipeline targeted 
    - Input Query
    - Extracting the relevant triples regarding the query
    - Generating path with dynamic stoping process 
    - Selecting the most relevant path and suppressing copies
    - Generating LLM answer with the context retrieval
    """

    def __init__(self,
                retriever_type: str ="Deep",
                model_name: str = "all-MiniLM-L6-v2", 
                checkpoint_path: str = None,
                device: str = "cpu",
                graph : str = None
                ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        retriever_type : str, mandatory, defining the method of KG exploration
            by default "heuristic", can be "deep"
        model_name : str, optional
            SentenceTransformer model name, by default "all-MiniLM-L6-v2".
        checkpoint_path : str, optional 
            None by default, used to load the weights of a pretrained model.
        device : str, optional
            Device for computation ("cpu" or "cuda").
        graph : str, optional
            Knowledge Graph used in the process, generic KG if not precised.
        """

        self.device = device
        self.model = SentenceTransformer(model_name)

        if not self. graph: 
            self.graph = build_kg()
        else: 
            self.graph = graph

        # First encoding of the KG
        self.embeddings, self.node_to_idx, self.idx_to_node = build_node_embeddings(
            self.graph, self.model
        )
        # Indexing the KG embeddings
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
            # Adding the relation embedding to the mix in order to create a better message passing with the GNN
            self.relation_embeddings = build_relation_embeddings(
                self.graph,
                self.model
            )

            dim = self.embeddings.shape[1]

            # =====================================================================
            #Creating the Neural Networks object, can be modified for experimenting
            # =====================================================================

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
            # Loading the trained weights for both of the Neural Networks
            if checkpoint_path is not None:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                print(checkpoint.keys())    

                self.retriever.PolicyNetwork.load_state_dict(
                    checkpoint["policy_state_dict"]
                )
                self.retriever.gnn_encoder.load_state_dict(
                    checkpoint["gnn_state_dict"]
                )

                self.retriever.PolicyNetwork.eval()
                self.retriever.gnn_encoder.eval()

        else:
            raise ValueError("Unknown retriever type")
        
        self.reasoner = self.build_reasoner()

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
        query_embedding = normalize(query_embedding)[0] # to verify if needed

        eq = self.entity_linking(query_embedding)

        Gq = self.extract_subgraph(eq)

        paths = self.retriever.retrieve_paths(
            query_embedding, 
            Gq
            )
        
        context = self.format_paths(paths)

        answer = self.generate_answer(query, context)

        return answer


# =========================
# 8. Main
# =========================

if __name__ == "__main__":
    pipeline = NewKGRAGPipeline(retriever_type='deep', checkpoint_path="deep_retriever.pt")
    query = "What causes sepsis?"
    
    answer = pipeline.query(query)

    print(answer)
    