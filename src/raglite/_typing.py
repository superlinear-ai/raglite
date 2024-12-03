"""RAGLite typing."""

import io
import pickle
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.operators import Operators
from sqlalchemy.types import Float, LargeBinary, TypeDecorator, TypeEngine, UserDefinedType

from raglite._config import RAGLiteConfig

ChunkId = str
DocumentId = str
EvalId = str
IndexId = str

FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]
FloatVector = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
IntVector = np.ndarray[tuple[int], np.dtype[np.intp]]


class SearchMethod(Protocol):
    def __call__(
        self, query: str, *, num_results: int = 3, config: RAGLiteConfig | None = None
    ) -> tuple[list[str], list[float]]: ...


class NumpyArray(TypeDecorator[np.ndarray[Any, np.dtype[np.floating[Any]]]]):
    """A NumPy array column type for SQLAlchemy."""

    impl = LargeBinary

    def process_bind_param(
        self, value: np.ndarray[Any, np.dtype[np.floating[Any]]] | None, dialect: Dialect
    ) -> bytes | None:
        """Convert a NumPy array to bytes."""
        if value is None:
            return None
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False, fix_imports=False)
        return buffer.getvalue()

    def process_result_value(
        self, value: bytes | None, dialect: Dialect
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]] | None:
        """Convert bytes to a NumPy array."""
        if value is None:
            return None
        return np.load(io.BytesIO(value), allow_pickle=False, fix_imports=False)  # type: ignore[no-any-return]


class PickledObject(TypeDecorator[object]):
    """A pickled object column type for SQLAlchemy."""

    impl = LargeBinary

    def process_bind_param(self, value: object | None, dialect: Dialect) -> bytes | None:
        """Convert a Python object to bytes."""
        if value is None:
            return None
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)

    def process_result_value(self, value: bytes | None, dialect: Dialect) -> object | None:
        """Convert bytes to a Python object."""
        if value is None:
            return None
        return pickle.loads(value, fix_imports=False)  # type: ignore[no-any-return]  # noqa: S301


class HalfVecComparatorMixin(UserDefinedType.Comparator[FloatVector]):
    """A mixin that provides comparison operators for halfvecs."""

    def cosine_distance(self, other: FloatVector) -> Operators:
        """Compute the cosine distance."""
        return self.op("<=>", return_type=Float)(other)

    def dot_distance(self, other: FloatVector) -> Operators:
        """Compute the dot product distance."""
        return self.op("<#>", return_type=Float)(other)

    def euclidean_distance(self, other: FloatVector) -> Operators:
        """Compute the Euclidean distance."""
        return self.op("<->", return_type=Float)(other)

    def l1_distance(self, other: FloatVector) -> Operators:
        """Compute the L1 distance."""
        return self.op("<+>", return_type=Float)(other)

    def l2_distance(self, other: FloatVector) -> Operators:
        """Compute the L2 distance."""
        return self.op("<->", return_type=Float)(other)


class HalfVec(UserDefinedType[FloatVector]):
    """A PostgreSQL half-precision vector column type for SQLAlchemy."""

    cache_ok = True  # HalfVec is immutable.

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def get_col_spec(self, **kwargs: Any) -> str:
        return f"halfvec({self.dim})"

    def bind_processor(self, dialect: Dialect) -> Callable[[FloatVector | None], str | None]:
        """Process NumPy ndarray to PostgreSQL halfvec format for bound parameters."""

        def process(value: FloatVector | None) -> str | None:
            return f"[{','.join(str(x) for x in np.ravel(value))}]" if value is not None else None

        return process

    def result_processor(
        self, dialect: Dialect, coltype: Any
    ) -> Callable[[str | None], FloatVector | None]:
        """Process PostgreSQL halfvec format to NumPy ndarray."""

        def process(value: str | None) -> FloatVector | None:
            if value is None:
                return None
            return np.fromstring(value.strip("[]"), sep=",", dtype=np.float16)

        return process

    class comparator_factory(HalfVecComparatorMixin):  # noqa: N801
        ...


class Embedding(TypeDecorator[FloatVector]):
    """An embedding column type for SQLAlchemy."""

    cache_ok = True  # Embedding is immutable.

    impl = NumpyArray

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[FloatVector]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(HalfVec(self.dim))
        return dialect.type_descriptor(NumpyArray())

    class comparator_factory(HalfVecComparatorMixin):  # noqa: N801
        ...
