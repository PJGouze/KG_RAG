from pathlib import Path
import json

import pandas as pd

from relation_mapper import RelationMapper


class ConversionPipeline:
    """
    Convert a medical relation dataset into
    KG-ready JSON files.

    The pipeline:
    - groups rows by report
    - extracts unique entities
    - extracts graph relations
    - exports one JSON file per report
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        output_dir: str | Path
    ):
        """
        Initialize conversion pipeline.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe containing medical relations.

        output_dir : str | Path
            Directory where formatted JSON files
            will be saved.
        """

        self.df = dataframe.copy()

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def run(self):
        """
        Execute the full conversion pipeline.
        """

        self._clean_dataframe()

        grouped = self.df.groupby("report")

        for report_id, group in grouped:

            report_json = self._build_report_json(
                report_id=report_id,
                group=group
            )

            self._save_json(
                report_id=report_id,
                data=report_json
            )

    # =========================================================================
    # DATA CLEANING
    # =========================================================================

    def _clean_dataframe(self):
        """
        Normalize dataframe columns and text fields.
        """

        self.df.columns = (
            self.df.columns
            .str.strip()
        )

    # =========================================================================
    # REPORT CONSTRUCTION
    # =========================================================================

    def _build_report_json(
        self,
        report_id: str,
        group: pd.DataFrame
    ) -> dict:
        """
        Build KG-ready JSON for one report.

        Parameters
        ----------
        report_id : str
            Report identifier.

        group : pd.DataFrame
            Report-specific dataframe.

        Returns
        -------
        dict
            Structured KG-ready report.
        """

        entities, entity_map = (
            self._extract_entities(group)
        )

        relations = self._extract_relations(
            group=group,
            entity_map=entity_map
        )

        return {
            "report_id": report_id,
            "entities": entities,
            "relations": relations
        }

    # =========================================================================
    # ENTITY EXTRACTION
    # =========================================================================
    def _extract_entities(
        self,
        group: pd.DataFrame
    ):
        """
        Extract unique entities from a report (no typing).
        """

        entity_map = {}
        entities = []
        entity_counter = 0

        for _, row in group.iterrows():

            for text in [row["source_text"], row["target_text"]]:

                if pd.isna(text):
                    continue

                text = str(text).strip()

                if text == "":
                    continue

                if text not in entity_map:

                    entity_id = f"E{entity_counter}"
                    entity_map[text] = entity_id

                    entities.append({
                        "id": entity_id,
                        "text": text
                    })

                    entity_counter += 1

        return entities, entity_map
    # =========================================================================
    # RELATION EXTRACTION
    # =========================================================================
    def _extract_relations(
        self,
        group: pd.DataFrame,
        entity_map: dict
        ):
        """
        Extract graph relations with robust handling
        of missing or inconsistent annotations.
        """

        relations = []

        for _, row in group.iterrows():

            source_text = str(row["source_text"]).strip()
            target_text = str(row["target_text"]).strip()

            # skip if entities missing from map
            if source_text not in entity_map or target_text not in entity_map:
                continue

            raw_relation = row.get("relation_type", None)
            raw_relation = None if pd.isna(raw_relation) else str(raw_relation).strip()

            # ------------------------------------------------------------
            # Collect attribute candidates (even if noisy)
            # ------------------------------------------------------------

            attr_agent = row.get("attribute_RelationAgentPathogene")
            attr_site = row.get("attribute_RelationSitePrimaire")
            attr_orig = row.get("attribute_RelationOrigine")

            candidates = [
                attr_agent,
                attr_site,
                attr_orig
            ]

            candidates = [
                c.strip() for c in candidates
                if isinstance(c, str) and c.strip() != ""
            ]

            # ------------------------------------------------------------
            # Normalize relation type (if exists)
            # ------------------------------------------------------------

            if raw_relation:
                normalized_relation = RelationMapper.normalize(raw_relation)
            else:
                normalized_relation = "UNKNOWN"

            # ------------------------------------------------------------
            # Build relation JSON
            # ------------------------------------------------------------

            relations.append({
                "source": entity_map[source_text],
                "target": entity_map[target_text],

                "relation": {
                    "type": normalized_relation,
                    "candidates": candidates,
                    "source_of_truth": "relation_type" if raw_relation else "attribute"
                }
            })

        return relations

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def _save_json(
        self,
        report_id: str,
        data: dict
    ):
        """
        Save report JSON.

        Parameters
        ----------
        report_id : str
            Report identifier.

        data : dict
            JSON data.
        """

        output_path = (
            self.output_dir /
            f"{report_id}.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )