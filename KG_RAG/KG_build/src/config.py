from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

REPORTS_DIR = DATA_DIR / "reports"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

RDF_OUTPUT_PATH = OUTPUT_DIR / "graphs" / "medical_kg.ttl"

EX_NAMESPACE = "http://example.org/medical/"

# =============================================================================
# Ontology Configuration
# =============================================================================

ACTIVE_ONTOLOGY = "SNOMED"
SNOMED_NAMESPACE = "http://snomed.info/id/"