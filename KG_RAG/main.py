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