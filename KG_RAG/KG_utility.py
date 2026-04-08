import networkx as nx

def build_kg() -> nx.DiGraph:
    """
    Build a directed biomedical Knowledge Graph.

    The graph contains nodes representing biomedical entities
    (e.g., symptoms, diseases, biological processes) and directed
    edges representing semantic relationships.

    Returns
    -------
    nx.DiGraph
        A directed graph where:
        - nodes are entity names (str) and description
        - edges have a "relation" attribute (str)
    """

    G = nx.DiGraph()

    # =========================
    # 1. Nodes (STRUCTURED)
    # =========================
    nodes = {
        "Sepsis": {
            "description": "Life-threatening condition caused by infection leading to organ dysfunction",
            "type": "disease",
            "synonyms": ["septic condition", "systemic infection"]
        },
        "Infection": {
            "description": "Invasion of the body by pathogenic microorganisms",
            "type": "condition",
            "synonyms": ["pathogen invasion"]
        },
        "Bacteria": {
            "description": "Microscopic organisms that can cause infections",
            "type": "pathogen",
            "synonyms": ["bacterial agent"]
        },
        "Fever": {
            "description": "Elevated body temperature, often due to infection",
            "type": "symptom",
            "synonyms": ["high temperature"]
        },
        "Hypotension": {
            "description": "Low blood pressure, common in sepsis and septic shock",
            "type": "symptom",
            "synonyms": ["low blood pressure"]
        },
        "Tachycardia": {
            "description": "Abnormally fast heart rate, often seen in infection",
            "type": "symptom",
            "synonyms": ["high heart rate"]
        },
        "Organ Failure": {
            "description": "Loss of function of one or more organs",
            "type": "condition",
            "synonyms": ["organ dysfunction"]
        },
        "Septic Shock": {
            "description": "Severe sepsis with persistent hypotension and organ failure",
            "type": "disease",
            "synonyms": ["shock due to sepsis"]
        },
        "Severe Hypotension": {
            "description": "Critically low blood pressure requiring intervention",
            "type": "symptom",
            "synonyms": []
        },
        "Multi-Organ Failure": {
            "description": "Failure of multiple organ systems",
            "type": "condition",
            "synonyms": []
        },
        "Bloodstream": {
            "description": "Circulatory system transporting blood",
            "type": "anatomy",
            "synonyms": []
        },
        "Immune Response": {
            "description": "Body defense mechanism against pathogens",
            "type": "process",
            "synonyms": ["immune reaction"]
        },
        "Inflammation": {
            "description": "Biological response to harmful stimuli",
            "type": "process",
            "synonyms": []
        },
        "Organ Dysfunction": {
            "description": "Impaired organ function",
            "type": "condition",
            "synonyms": []
        },
        "Antibiotics": {
            "description": "Drugs used to treat bacterial infections",
            "type": "treatment",
            "synonyms": []
        },
        "Fluid Resuscitation": {
            "description": "Administration of fluids to restore blood volume",
            "type": "treatment",
            "synonyms": []
        },
        "ICU": {
            "description": "Intensive care unit for critically ill patients",
            "type": "location",
            "synonyms": ["intensive care"]
        },
        "Lactate": {
            "description": "Biomarker indicating severity of sepsis and tissue hypoxia",
            "type": "biomarker",
            "synonyms": []
        },
        "Blood Culture": {
            "description": "Test used to detect bacteria in blood",
            "type": "test",
            "synonyms": []
        },
        "SOFA Score": {
            "description": "Clinical score assessing organ failure in sepsis",
            "type": "score",
            "synonyms": []
        },
        "Sepsis Severity": {
            "description": "Degree of severity of sepsis",
            "type": "concept",
            "synonyms": []
        },
    }

    for node, attributes in nodes.items():
        G.add_node(node, **attributes)

    # =========================
    # 2. Edges
    # =========================
    edges = [
        ("Sepsis", "Infection", "caused_by"),
        ("Sepsis", "Bacteria", "often_caused_by"),
        ("Sepsis", "Fever", "has_symptom"),
        ("Sepsis", "Hypotension", "has_symptom"),
        ("Sepsis", "Tachycardia", "has_symptom"),
        ("Sepsis", "Organ Failure", "can_lead_to"),
        ("Sepsis", "Septic Shock", "can_progress_to"),

        ("Septic Shock", "Sepsis", "is_a"),
        ("Septic Shock", "Severe Hypotension", "characterized_by"),
        ("Septic Shock", "Multi-Organ Failure", "can_lead_to"),

        ("Bacteria", "Bloodstream", "can_infect"),
        ("Infection", "Immune Response", "triggers"),
        ("Immune Response", "Inflammation", "causes"),
        ("Inflammation", "Organ Dysfunction", "can_lead_to"),

        ("Sepsis", "Antibiotics", "treated_with"),
        ("Sepsis", "Fluid Resuscitation", "treated_with"),
        ("Sepsis", "ICU", "managed_in"),

        ("Lactate", "Sepsis", "biomarker_of"),
        ("Blood Culture", "Bacteria", "detects"),
        ("SOFA Score", "Sepsis Severity", "assesses"),
    ]

    for source, target, relation in edges:
        G.add_edge(source, target, relation=relation)

    return G
