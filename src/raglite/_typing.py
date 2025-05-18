"""RAGLite typing."""

import io
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from sqlalchemy import literal
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.sql.operators import Operators
from sqlalchemy.types import Float, LargeBinary, TypeDecorator, TypeEngine, UserDefinedType

if TYPE_CHECKING:
    from raglite._config import RAGLiteConfig
    from raglite._database import Chunk, ChunkSpan

ChunkId = str
DocumentId = str
EvalId = str
IndexId = str

DistanceMetric = Literal["cosine", "dot", "l1", "l2"]

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


class EmbeddingDistance(FunctionElement[float]):
    """SQL expression that renders a distance operator per dialect."""

    inherit_cache = True
    type = Float()  # The result is always a scalar float.

    def __init__(self, left: Any, right: Any, metric: DistanceMetric) -> None:
        self.metric = metric
        super().__init__(left, right)


@compiles(EmbeddingDistance, "postgresql")
def _embedding_distance_postgresql(element: EmbeddingDistance, compiler: Any, **kwargs: Any) -> str:
    op_map: dict[DistanceMetric, str] = {
        "cosine": "<=>",
        "dot": "<#>",
        "l1": "<+>",
        "l2": "<->",
    }
    left, right = list(element.clauses)
    operator = op_map[element.metric]
    return f"({compiler.process(left)} {operator} {compiler.process(right)})"


@compiles(EmbeddingDistance, "duckdb")
def _embedding_distance_duckdb(element: EmbeddingDistance, compiler: Any, **kwargs: Any) -> str:
    func_map: dict[DistanceMetric, str] = {
        "cosine": "array_cosine_distance",
        "dot": "array_negative_inner_product",
        "l2": "array_distance",
    }
    left, right = list(element.clauses)
    dim = left.type.dim  # type: ignore[attr-defined]
    func_name = func_map[element.metric]
    right_cast = f"{compiler.process(right)}::FLOAT[{dim}]"
    return f"{func_name}({compiler.process(left)}, {right_cast})"


class EmbeddingComparator(UserDefinedType.Comparator[FloatVector]):
    """An embedding distance comparator."""

    def distance(self, other: FloatVector, *, metric: DistanceMetric) -> Operators:
        rhs = literal(other, type_=self.expr.type)
        return EmbeddingDistance(self.expr, rhs, metric)


class PostgresHalfVec(UserDefinedType[FloatVector]):
    """A PostgreSQL half-precision vector column type for SQLAlchemy."""

    cache_ok = True

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


class DuckDBSingleVec(UserDefinedType[FloatVector]):
    """A DuckDB single precision vector column type for SQLAlchemy."""

    cache_ok = True

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def get_col_spec(self, **kwargs: Any) -> str:
        return f"FLOAT[{self.dim}]" if self.dim is not None else "FLOAT[]"

    def bind_processor(
        self, dialect: Dialect
    ) -> Callable[[FloatVector | None], list[float] | None]:
        """Process NumPy ndarray to DuckDB single precision vector format for bound parameters."""

        def process(value: FloatVector | None) -> list[float] | None:
            return np.ravel(value).tolist() if value is not None else None

        return process

    def result_processor(
        self, dialect: Dialect, coltype: Any
    ) -> Callable[[list[float] | None], FloatVector | None]:
        """Process DuckDB single precision vector format to NumPy ndarray."""

        def process(value: list[float] | None) -> FloatVector | None:
            return np.asarray(value, dtype=np.float32) if value is not None else None

        return process


class Embedding(TypeDecorator[FloatVector]):
    """An embedding column type for SQLAlchemy."""

    cache_ok = True
    impl = NumpyArray
    comparator_factory: type[EmbeddingComparator] = EmbeddingComparator

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[FloatVector]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgresHalfVec(self.dim))
        if dialect.name == "duckdb":
            return dialect.type_descriptor(DuckDBSingleVec(self.dim))
        return dialect.type_descriptor(NumpyArray())
