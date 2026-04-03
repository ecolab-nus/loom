"""JSON utilities for loading ETG dumps."""

import json
from pathlib import Path
from typing import Union


def load_variants(input_path: Union[str, Path]) -> list[dict]:
    """Load the staged ETG JSON file and return the list of variants."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "variants" in data:
        return data["variants"]
    
    raise ValueError(f"Malformed ETG JSON at {input_path}")
