from typing import Dict

from entities import (
    ClinicalReport,
    Entity,
    Relation
)


def parse_report(data: Dict) -> ClinicalReport:
    """
    Parse raw JSON annotation data into a ClinicalReport object.
    """

    entities = [
        Entity(
            entity_id=e["id"],
            text=e["text"],
            entity_type=e["type"]
        )
        for e in data["entities"]
    ]

    relations = [
        Relation(
            source=r["source"],
            target=r["target"],
            relation_type=r["type"]
        )
        for r in data["relations"]
    ]

    return ClinicalReport(
        report_id=data["report_id"],
        patient_id=data["patient_id"],
        entities=entities,
        relations=relations
    )