from rdflib import Literal
from rdflib.namespace import RDF

from ontology import (
    EX,
    LABEL,
    RELATION_MAPPING,
    create_patient_uri,
    create_report_uri,
    create_concept_uri
)


class RDFBuilder:
    """
    Transforms ClinicalReport objects into RDF triples.

    This class is ontology-agnostic:
    - It does NOT hardcode SNOMED, DOID, or HPO
    - It relies entirely on ontology.py for URI resolution
    """

    def __init__(self, graph_store, entity_ontology_map=None):
        """
        Initialize RDF builder.

        Parameters
        ----------
        graph_store : GraphStore
            RDF storage backend.

        entity_ontology_map : dict | None
            Optional mapping from entity type to ontology name.
            Example:
                {
                    "DISEASE": "SNOMED",
                    "SYMPTOM": "HPO"
                }
        """

        self.graph_store = graph_store

        # Default mapping if not provided
        self.entity_ontology_map = entity_ontology_map or {
            "DISEASE": "SNOMED",
            "SYMPTOM": "HPO",
            "PROCEDURE": "SNOMED",
            "MEDICATION": "SNOMED"
        }

    # -------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------------------

    def add_report(self, report):
        """
        Convert a ClinicalReport into RDF triples.

        Parameters
        ----------
        report : ClinicalReport
            Parsed clinical report.
        """

        patient_uri = create_patient_uri(report.patient_id)

        report_uri = create_report_uri(report.report_id)

        entity_uri_map = {}

        # ---------------------------------------------------------------------
        # Link patient ↔ report
        # ---------------------------------------------------------------------

        self.graph_store.add_triple(
            patient_uri,
            EX.hasReport,
            report_uri
        )

        # ---------------------------------------------------------------------
        # ENTITIES
        # ---------------------------------------------------------------------

        for entity in report.entities:

            if entity.ontology_id is None:
                continue

            ontology_name = self._resolve_ontology(entity)

            entity_uri = create_concept_uri(
                ontology_name,
                entity.ontology_id
            )

            entity_uri_map[entity.entity_id] = entity_uri

            # Patient → entity relation
            self.graph_store.add_triple(
                patient_uri,
                EX.hasEntity,
                entity_uri
            )

            # Type assertion
            self.graph_store.add_triple(
                entity_uri,
                RDF.type,
                EX[entity.entity_type]
            )

            # Label preservation
            self.graph_store.add_triple(
                entity_uri,
                LABEL,
                Literal(entity.text)
            )

        # ---------------------------------------------------------------------
        # RELATIONS
        # ---------------------------------------------------------------------

        for relation in report.relations:

            if relation.source not in entity_uri_map:
                continue

            if relation.target not in entity_uri_map:
                continue

            source_uri = entity_uri_map[relation.source]

            target_uri = entity_uri_map[relation.target]

            predicate = RELATION_MAPPING.get(
                relation.relation_type,
                EX.hasRelation
            )

            self.graph_store.add_triple(
                source_uri,
                predicate,
                target_uri
            )

    # -------------------------------------------------------------------------
    # ONTOLOGY RESOLUTION LOGIC
    # -------------------------------------------------------------------------

    def _resolve_ontology(self, entity):
        """
        Resolve ontology based on entity type.

        Parameters
        ----------
        entity : Entity

        Returns
        -------
        str
            Ontology name (SNOMED, HPO, ...)
        """

        return self.entity_ontology_map.get(
            entity.entity_type,
            "SNOMED"  # safe fallback
        )