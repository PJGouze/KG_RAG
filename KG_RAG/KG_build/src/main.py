from pathlib import Path

from parser import parse_report
from loader import load_json

from pipeline import normalize_entities

from rdf_builder import RDFBuilder
from graph_store import GraphStore

from snomed_linker import SnomedLinker


# =============================================================================
# PROJECT PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent


ANNOTATIONS_DIR = (
    BASE_DIR
    / "data"
    / "annotations"
)

SNOMED_RF2_DESCRIPTION = (
    BASE_DIR
    / "data"
    / "ontologies"
    / "SNOMEDCT"
    / "Snapshot"
    / "Terminology"
    / "sct2_Description_Snapshot-en_INT_20240101.txt"
)

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "graphs"
    / "medical_kg.ttl"
)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Initialize linker
    # -------------------------------------------------------------------------

    linker = SnomedLinker(
        SNOMED_RF2_DESCRIPTION
    )

    # -------------------------------------------------------------------------
    # Initialize graph
    # -------------------------------------------------------------------------

    graph_store = GraphStore()

    rdf_builder = RDFBuilder(
        graph_store
    )

    # -------------------------------------------------------------------------
    # Process reports
    # -------------------------------------------------------------------------

    for file_path in ANNOTATIONS_DIR.glob("*.json"):

        raw_data = load_json(file_path)

        report = parse_report(raw_data)

        report = normalize_entities(
            report,
            linker
        )

        rdf_builder.add_report(report)

    # -------------------------------------------------------------------------
    # Export RDF graph
    # -------------------------------------------------------------------------

    graph_store.serialize(
        OUTPUT_FILE
    )

    print("Knowledge graph generated.")


if __name__ == "__main__":
    main()