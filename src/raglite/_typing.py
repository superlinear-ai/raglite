"""RAGLite typing."""

import io
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from sqlalchemy.engine import Dialect
from sqlalchemy.sql import func
from sqlalchemy.sql.operators import Operators
from sqlalchemy.types import Float, LargeBinary, TypeDecorator, TypeEngine, UserDefinedType

if TYPE_CHECKING:
    from raglite._config import RAGLiteConfig
    from raglite._database import Chunk, ChunkSpan

ChunkId = str
DocumentId = str
EvalId = str
IndexId = str

FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]
FloatVector = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
IntVector = np.ndarray[tuple[int], np.dtype[np.intp]]


class BasicSearchMethod(Protocol):
    def __call__(
        self, query: str, *, num_results: int, config: "RAGLiteConfig | None" = None
    ) -> tuple[list[ChunkId], list[float]]: ...


class SearchMethod(Protocol):
    def __call__(
        self, query: str, *, num_results: int, config: "RAGLiteConfig | None" = None
    ) -> tuple[list[ChunkId], list[float]] | list["Chunk"] | list["ChunkSpan"]: ...


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


class EmbeddingComparator(UserDefinedType.Comparator[FloatVector]):
    """A comparator that provides distance operations."""

    def _is_postgres(self) -> bool:
        return isinstance(self.type, HalfVec)

    def _is_duckdb(self) -> bool:
        return isinstance(self.type, DuckDBVec)

    def cosine_distance(self, other: FloatVector) -> Operators:
        """Compute the cosine distance."""
        if self._is_postgres():
            return self.op("<=>", return_type=Float)(other)
        if self._is_duckdb():
            return func.array_cosine_distance(self.expr, other)
        return self.op("<=>", return_type=Float)(other)

    def dot_distance(self, other: FloatVector) -> Operators:
        """Compute the dot product distance."""
        if self._is_postgres():
            return self.op("<#>", return_type=Float)(other)
        if self._is_duckdb():
            return func.array_negative_inner_product(self.expr, other)
        return self.op("<#>", return_type=Float)(other)

    def euclidean_distance(self, other: FloatVector) -> Operators:
        """Compute the Euclidean distance."""
        if self._is_postgres():
            return self.op("<->", return_type=Float)(other)
        if self._is_duckdb():
            return func.array_distance(self.expr, other)
        return self.op("<->", return_type=Float)(other)

    def l1_distance(self, other: FloatVector) -> Operators:
        """Compute the L1 distance."""
        if self._is_postgres():
            return self.op("<+>", return_type=Float)(other)
        return func.abs(func.sum(self.expr - other))

    def l2_distance(self, other: FloatVector) -> Operators:
        """Compute the L2 distance."""
        if self._is_postgres():
            return self.op("<->", return_type=Float)(other)
        if self._is_duckdb():
            return func.array_distance(self.expr, other)
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

    class comparator_factory(EmbeddingComparator):  # noqa: N801
        ...


class DuckDBVec(UserDefinedType[FloatVector]):
    """A DuckDB floating point array column type for SQLAlchemy."""

    cache_ok = True

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def get_col_spec(self, **kwargs: Any) -> str:
        return f"FLOAT[{self.dim}]" if self.dim is not None else "FLOAT[]"

    def bind_processor(
        self, dialect: Dialect
    ) -> Callable[[FloatVector | None], list[float] | None]:
        def process(value: FloatVector | None) -> list[float] | None:
            return value.tolist() if value is not None else None

        return process

    def result_processor(
        self, dialect: Dialect, coltype: Any
    ) -> Callable[[list[float] | None], FloatVector | None]:
        def process(value: list[float] | None) -> FloatVector | None:
            return np.asarray(value, dtype=np.float32) if value is not None else None

        return process

    class comparator_factory(EmbeddingComparator):  # noqa: N801
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
        if dialect.name == "duckdb":
            return dialect.type_descriptor(DuckDBVec(self.dim))
        return dialect.type_descriptor(NumpyArray())

    class comparator_factory(EmbeddingComparator):  # noqa: N801
        ...
