import faiss
import numpy as np
from typing import List

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