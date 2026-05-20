from pathlib import Path
import json
import pandas as pd


class ConversionPipeline:
    """
    Converts raw medical relational dataset into KG-ready JSON files.

    Output is independent from KG construction pipeline.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------

    def run(self):
        """
        Execute full conversion pipeline.
        """

        grouped = self.df.groupby("report")

        for report_id, group in grouped:

            kg_json = self._convert_group(report_id, group)

            self._save_json(report_id, kg_json)

    # ---------------------------------------------------------------------
    # CORE TRANSFORMATION
    # ---------------------------------------------------------------------

    def _convert_group(self, report_id, group):
        """
        Convert a single report group into KG-ready JSON.
        """

        entities = self._extract_entities(group)
        relations = self._extract_relations(group)

        return {
            "report_id": report_id,
            "entities": entities,
            "relations": relations
        }

    # ---------------------------------------------------------------------
    # ENTITIES
    # ---------------------------------------------------------------------

    def _extract_entities(self, group):
        """
        Build unique entity list.
        """

        seen = {}
        entities = []
        idx = 0

        for _, row in group.iterrows():

            for text in [row["source_text"], row["target_text"]]:

                if text not in seen:

                    seen[text] = f"E{idx}"
                    idx += 1

                    entities.append({
                        "id": seen[text],
                        "text": text,
                        "type": "ENTITY"
                    })

        return entities

    # ---------------------------------------------------------------------
    # RELATIONS
    # ---------------------------------------------------------------------

    def _extract_relations(self, group):
        """
        Build relations between entities.
        """

        relations = []

        for _, row in group.iterrows():

            relations.append({
                "source_text": row["source_text"],
                "target_text": row["target_text"],
                "type": row["relation_type"]
            })

        return relations

    # ---------------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------------

    def _save_json(self, report_id, data):
        """
        Save KG-ready JSON.
        """

        path = self.output_dir / f"{report_id}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            