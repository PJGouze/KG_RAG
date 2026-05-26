class RelationMapper:
    """
    Normalize medical relation labels.
    """

    RELATION_MAPPING = {
        "AgentPathogene": "hasPathogen",
        "SitePrimaire": "hasPrimarySite",
        "Origine": "hasOrigin"
    }

    @classmethod
    def normalize(
        cls,
        relation_type: str
    ) -> str:
        """
        Normalize relation label.
        """

        relation_type = str(
            relation_type
        ).strip()

        return cls.RELATION_MAPPING.get(
            relation_type,
            relation_type
        )