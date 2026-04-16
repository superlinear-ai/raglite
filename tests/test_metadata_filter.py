"""Test metadata filter SQL builders."""

import json

from sqlalchemy import column
from sqlalchemy.dialects import postgresql

from raglite._metadata_filter import build_metadata_filter_condition, build_metadata_filter_sql


def test_build_metadata_filter_condition_postgresql_multiple_values() -> None:
    """Build PostgreSQL conditions with OR inside a field and AND across fields."""
    condition = build_metadata_filter_condition(
        column("metadata"),
        {"domain": ["open", "music"], "topic": "physics"},
        dialect="postgresql",
    )
    assert condition is not None

    compiled_condition = condition.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]
    compiled_sql = str(compiled_condition)
    assert " OR " in compiled_sql
    assert " AND " in compiled_sql
    assert "CAST(metadata AS JSONB) @>" in compiled_sql

    compiled_parameters = list(compiled_condition.params.values())
    assert {"domain": ["open"]} in compiled_parameters
    assert {"domain": ["music"]} in compiled_parameters
    assert {"topic": ["physics"]} in compiled_parameters


def test_build_metadata_filter_sql_postgresql_multiple_values() -> None:
    """Build PostgreSQL raw SQL fragments with OR semantics for list values."""
    sql_fragment, parameters = build_metadata_filter_sql(
        {"domain": ["open", "music"], "topic": "physics"},
        dialect="postgresql",
    )

    assert "metadata::jsonb @> :metadata_filter_0" in sql_fragment
    assert "metadata::jsonb @> :metadata_filter_1" in sql_fragment
    assert "metadata::jsonb @> :metadata_filter_2" in sql_fragment
    assert " OR " in sql_fragment
    assert " AND " in sql_fragment

    assert json.loads(parameters["metadata_filter_0"]) == {"domain": ["open"]}
    assert json.loads(parameters["metadata_filter_1"]) == {"domain": ["music"]}
    assert json.loads(parameters["metadata_filter_2"]) == {"topic": ["physics"]}


def test_build_metadata_filter_sql_postgresql_empty_value_is_unsatisfiable() -> None:
    """Treat empty value lists as unsatisfiable for PostgreSQL SQL generation."""
    sql_fragment, parameters = build_metadata_filter_sql({"domain": []}, dialect="postgresql")
    assert sql_fragment == " AND 1=0"
    assert parameters == {}
