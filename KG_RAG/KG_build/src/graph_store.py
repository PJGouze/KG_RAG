from rdflib import Graph

from config import ACTIVE_ONTOLOGY

from ontology import EX
from ontology import ONTOLOGY_REGISTRY


class GraphStore:
    """
    RDF graph storage and querying manager.
    """

    def __init__(self):
        """
        Initialize RDF graph.
        """

        self.graph = Graph()

        self.active_ontology = ACTIVE_ONTOLOGY

        self.ontology_config = ONTOLOGY_REGISTRY[self.active_ontology]

        self._bind_namespaces()

    def _bind_namespaces(self):
        """
        Bind RDF namespaces.
        """

        # ---------------------------------------------------------------------
        # Base namespace
        # ---------------------------------------------------------------------

        self.graph.bind("ex", EX)

        # ---------------------------------------------------------------------
        # Active ontology namespace
        # ---------------------------------------------------------------------

        ontology_prefix = self.ontology_config["prefix"]

        ontology_namespace = self.ontology_config["namespace"]

        self.graph.bind(
            ontology_prefix,
            ontology_namespace
        )

    def add_triple(self, subject, predicate, object_):
        """
        Add RDF triple to graph.

        Parameters
        ----------
        subject : URIRef
            RDF subject.

        predicate : URIRef
            RDF predicate.

        object_ : URIRef | Literal
            RDF object.
        """

        self.graph.add((subject, predicate, object_))

    def serialize(self, output_path: str, rdf_format: str = "turtle"):
        """
        Serialize RDF graph.

        Parameters
        ----------
        output_path : str
            Output file path.

        rdf_format : str
            RDF serialization format.
        """

        self.graph.serialize(
            destination=output_path,
            format=rdf_format
        )

    def load_graph(self, input_path: str, rdf_format: str = "turtle"):
        """
        Load RDF graph from file.

        Parameters
        ----------
        input_path : str
            RDF file path.

        rdf_format : str
            RDF serialization format.
        """

        self.graph.parse(
            input_path,
            format=rdf_format
        )

    def query(self, sparql_query: str):
        """
        Execute SPARQL query.

        Parameters
        ----------
        sparql_query : str
            SPARQL query string.

        Returns
        -------
        Result
            SPARQL query result.
        """

        return self.graph.query(sparql_query)

    def graph_size(self) -> int:
        """
        Return number of RDF triples.

        Returns
        -------
        int
            Number of triples.
        """

        return len(self.graph)