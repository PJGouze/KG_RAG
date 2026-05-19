from rdflib import Namespace
from rdflib.namespace import RDFS


# =============================================================================
# BASE NAMESPACE (local KG)
# =============================================================================

EX = Namespace("http://example.org/medical/")


# =============================================================================
# ONTOLOGY REGISTRY (modular & extensible)
# =============================================================================

ONTOLOGY_REGISTRY = {
    "SNOMED": {
        "prefix": "snomed",
        "namespace": Namespace("http://snomed.info/id/")
    },
    "DOID": {
        "prefix": "doid",
        "namespace": Namespace("http://purl.obolibrary.org/obo/DOID_")
    },
    "HPO": {
        "prefix": "hp",
        "namespace": Namespace("http://purl.obolibrary.org/obo/HP_")
    }
}


# =============================================================================
# ACTIVE ONTOLOGY HANDLER
# =============================================================================

def get_ontology(name: str):
    """
    Retrieve ontology configuration from registry.

    Parameters
    ----------
    name : str
        Name of the ontology (e.g., 'SNOMED', 'DOID', 'HPO').

    Returns
    -------
    dict
        Ontology configuration containing prefix and namespace.
    """

    if name not in ONTOLOGY_REGISTRY:
        raise ValueError(
            f"Ontology '{name}' not found. "
            f"Available: {list(ONTOLOGY_REGISTRY.keys())}"
        )

    return ONTOLOGY_REGISTRY[name]


def get_namespace(name: str):
    """
    Get RDF namespace for a given ontology.

    Parameters
    ----------
    name : str
        Ontology name.

    Returns
    -------
    Namespace
        RDFLib Namespace object.
    """

    return get_ontology(name)["namespace"]


def get_prefix(name: str):
    """
    Get prefix for a given ontology.

    Parameters
    ----------
    name : str
        Ontology name.

    Returns
    -------
    str
        Prefix string.
    """

    return get_ontology(name)["prefix"]


# =============================================================================
# RDF PREDICATES (KEPT DOMAIN-LEVEL, NOT ONTOLOGY-DEPENDENT)
# =============================================================================

HAS_ENTITY = EX.hasEntity
HAS_DISEASE = EX.hasDisease
HAS_SYMPTOM = EX.hasSymptom
HAS_PROCEDURE = EX.hasProcedure
HAS_MEDICATION = EX.hasMedication
HAS_RELATION = EX.hasRelation
HAS_REPORT = EX.hasReport
HAS_PATIENT = EX.hasPatient

LABEL = RDFS.label


RELATION_MAPPING = {
    "has_symptom": HAS_SYMPTOM,
    "has_disease": HAS_DISEASE,
    "treated_by": EX.treatedBy,
    "caused_by": EX.causedBy,
    "associated_with": EX.associatedWith
}


# =============================================================================
# SEMANTIC TYPES (LOCAL KG SCHEMA)
# =============================================================================

DISEASE = EX.Disease
SYMPTOM = EX.Symptom
PROCEDURE = EX.Procedure
MEDICATION = EX.Medication
PATIENT = EX.Patient


# =============================================================================
# URI BUILDERS (NOW GENERIC)
# =============================================================================

def create_patient_uri(patient_id: str):
    """
    Create a patient URI in the local knowledge graph.

    Parameters
    ----------
    patient_id : str

    Returns
    -------
    URIRef
    """
    return EX[f"patient/{patient_id}"]


def create_report_uri(report_id: str):
    """
    Create a report URI in the local knowledge graph.
    """
    return EX[f"report/{report_id}"]


def create_concept_uri(ontology_name: str, concept_id: str):
    """
    Create a URI for a concept in a given ontology.

    Parameters
    ----------
    ontology_name : str
        Ontology name (SNOMED, DOID, HPO, ...)

    concept_id : str
        Concept identifier.

    Returns
    -------
    URIRef
    """

    namespace = get_namespace(ontology_name)

    return namespace[concept_id]