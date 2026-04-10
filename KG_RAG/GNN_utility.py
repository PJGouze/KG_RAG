import torch.nn as nn
import torch

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