from pathlib import Path
from typing import Any

from crag.models.utils import read_jsonl


def load_dataset_file(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return read_jsonl(file_path)
