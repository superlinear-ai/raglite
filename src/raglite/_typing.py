"""RAGLite typing."""

from typing import Any

import numpy as np

FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]
IntVector = np.ndarray[tuple[int], np.dtype[np.intp]]
