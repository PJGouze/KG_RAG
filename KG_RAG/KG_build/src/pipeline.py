def normalize_entities(report, linker):
    """
    Link all report entities to ontology concepts.
    """

    for entity in report.entities:
        ontology_id = linker.link_entity(entity.text)
        entity.ontology_id = ontology_id

    return report