import json
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> Dict:
    """
    Load a JSON file.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    Dict
        Parsed JSON content.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)