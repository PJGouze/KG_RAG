import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process


class SnomedLinker:
    """
    Simple SNOMED CT entity linker using fuzzy matching.
    """

    def __init__(self, snomed_csv_path: str):
        """
        Initialize the linker.

        Parameters
        ----------
        snomed_csv_path : str
            Path to SNOMED dictionary CSV.
        """

        self.df = pd.read_csv(snomed_csv_path)

        self.terms = self.df["term"].tolist()

    def link_entity(self, text: str, threshold: int = 85):
        """
        Link a textual mention to a SNOMED concept.

        Parameters
        ----------
        text : str
            Entity mention.

        threshold : int
            Minimum matching score.

        Returns
        -------
        str | None
            SNOMED concept identifier.
        """

        result = process.extractOne(
            text,
            self.terms,
            scorer=fuzz.token_sort_ratio
        )

        if result is None:
            return None

        matched_term, score, _ = result

        if score < threshold:
            return None

        row = self.df[self.df["term"] == matched_term].iloc[0]

        return str(row["concept_id"])