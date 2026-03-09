import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from crag.webapp.app import main

warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
    category=UserWarning,
    module=r"pydantic\.main",
)

if __name__ == "__main__":
    main()
