import torch
import torch.nn as nn
import numpy as np

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
    
class DeepRetriever(BaseRetriever):
    def __init__(
        self,
        G,
        embeddings,
        node_to_idx,
        idx_to_node,
        policy_net,
        relation_embeddings
    ):
        self.G = G
        self.embeddings = embeddings
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.policy_net = policy_net
        self.relation_embeddings = relation_embeddings

    def build_state(self, query_emb, current_node, neighbor):
        node_emb = self.embeddings[self.node_to_idx[current_node]]
        neighbor_emb = self.embeddings[self.node_to_idx[neighbor]]

        rel = self.G[current_node][neighbor]["relation"]
        rel_emb = self.relation_embeddings.get(rel, np.zeros_like(node_emb))

        return np.concatenate([query_emb, node_emb, neighbor_emb, rel_emb])