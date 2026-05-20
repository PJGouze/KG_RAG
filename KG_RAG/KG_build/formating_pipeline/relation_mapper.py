class RelationMapper:
    """
    Normalize raw medical relation labels into
    KG-compatible predicates.
    """

    RELATION_MAPPING = {
        "AgentPathogene": "hasPathogen",
        "SitePrimaire": "hasPrimarySite",
        "Origine": "hasOrigin"
    }

    @classmethod
    def normalize(cls, relation_type: str):
        """
        Normalize relation type.

        Parameters
        ----------
        relation_type : str
            Raw relation label.

        Returns
        -------
        str
            Normalized relation predicate.
        """

        relation_type = str(
            relation_type
        ).strip()

        return cls.RELATION_MAPPING.get(
            relation_type,
            relation_type
        )