from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    """
    Represents a named entity extracted from a clinical report.
    """

    entity_id: str
    text: str
    entity_type: str
    ontology_id: str | None = None


@dataclass
class Relation:
    """
    Represents a semantic relation between two entities.
    """

    source: str
    target: str
    relation_type: str


@dataclass
class ClinicalReport:
    """
    Represents a clinical report and its semantic annotations.
    """

    report_id: str
    patient_id: str
    entities: List[Entity]
    relations: List[Relation]