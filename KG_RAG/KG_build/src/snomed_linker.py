from pathlib import Path

import pandas as pd

from rapidfuzz import process
from rapidfuzz import fuzz


class SnomedLinker:
    """
    SNOMED CT entity linker based on RF2 descriptions.

    This linker maps clinical entity mentions to SNOMED concepts.
    """

    def __init__(self, rf2_description_path: str):
        """
        Initialize SNOMED linker.

        Parameters
        ----------
        rf2_description_path : str
            Path to SNOMED RF2 Description file.
        """

        self.rf2_description_path = Path(rf2_description_path)

        self.df = None

        self.term_to_concepts = {}

        self.terms = []

        self._load_descriptions()

        self._build_index()

    # -------------------------------------------------------------------------
    # LOADING
    # -------------------------------------------------------------------------

    def _load_descriptions(self):
        """
        Load SNOMED RF2 descriptions file.
        """

        self.df = pd.read_csv(
            self.rf2_description_path,
            sep="\t",
            dtype=str
        )

        # Keep only active concepts
        self.df = self.df[
            self.df["active"] == "1"
        ]

    # -------------------------------------------------------------------------
    # INDEXING
    # -------------------------------------------------------------------------

    def _build_index(self):
        """
        Build term → concept index.
        """

        for _, row in self.df.iterrows():

            term = row["term"].lower()

            concept_id = row["conceptId"]

            if term not in self.term_to_concepts:

                self.term_to_concepts[term] = []

            self.term_to_concepts[term].append(
                concept_id
            )

        self.terms = list(
            self.term_to_concepts.keys()
        )

    # -------------------------------------------------------------------------
    # ENTITY LINKING
    # -------------------------------------------------------------------------

    def link_entity(
        self,
        text: str,
        threshold: int = 85
    ):
        """
        Link entity text to SNOMED concept.

        Parameters
        ----------
        text : str
            Clinical entity mention.

        threshold : int
            Minimum fuzzy matching score.

        Returns
        -------
        str | None
            SNOMED concept ID if found.
        """

        if not text:
            return None

        match = process.extractOne(
            text.lower(),
            self.terms,
            scorer=fuzz.token_sort_ratio
        )

        if match is None:
            return None

        matched_term, score, _ = match

        if score < threshold:
            return None

        concept_ids = self.term_to_concepts[
            matched_term
        ]

        return concept_ids[0]