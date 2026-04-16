"""Helpers to build metadata filter conditions with consistent semantics."""

import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy import and_, false, or_
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import col, func

from raglite._database import _adapt_metadata
from raglite._typing import MetadataFilter, MetadataValue


def build_metadata_filter_condition(
    metadata_column: Any,
    metadata_filter: MetadataFilter | None,
    *,
    dialect: str,
) -> Any:
    """Build a SQLAlchemy condition for metadata filtering.

    A list of values within the same field uses OR semantics.
    Different fields are combined with AND semantics.
    """
    normalized_metadata_filter = _adapt_metadata(metadata_filter)
    if not normalized_metadata_filter:
        return None

    field_conditions: list[Any] = []
    for metadata_name, metadata_values in normalized_metadata_filter.items():
        if not metadata_values:
            return false()  # empty filters are considered unsatisfiable

        value_conditions: list[Any] = []
        for metadata_value in metadata_values:
            single_value_filter = {metadata_name: [metadata_value]}
            if dialect == "postgresql":
                value_conditions.append(
                    col(metadata_column).cast(JSONB).op("@>")(single_value_filter)  # type: ignore[attr-defined]
                )
            elif dialect == "duckdb":
                value_conditions.append(
                    func.json_contains(
                        col(metadata_column), func.json(json.dumps(single_value_filter))
                    )
                )
            else:
                error_message = f"Unsupported dialect: {dialect}."
                raise ValueError(error_message)
        field_conditions.append(or_(*value_conditions))  # combine values for the same field with OR
    return and_(*field_conditions)  # combine different fields with AND


def build_metadata_filter_sql(
    metadata_filter: Mapping[str, list[MetadataValue] | MetadataValue] | None,
    *,
    dialect: str,
) -> tuple[str, dict[str, str]]:
    """Build SQL fragment and bound parameters for metadata filtering.

    A list of values within the same field uses OR semantics.
    Different fields are combined with AND semantics.

    Returns
    -------
    sql_fragment : str
        A SQL fragment to be included in the WHERE clause, with placeholders for parameters.
    parameters : dict
        A dictionary of parameter names and their corresponding JSON string values to be used in
        the query execution
    """
    normalized_metadata_filter = _adapt_metadata(metadata_filter)
    if not normalized_metadata_filter:
        return "", {}

    field_sql_conditions: list[str] = []
    parameters: dict[str, str] = {}
    parameter_index = 0

    for metadata_name, metadata_values in normalized_metadata_filter.items():
        if not metadata_values:
            return " AND 1=0", {}

        value_sql_conditions: list[str] = []
        for metadata_value in metadata_values:
            parameter_name = f"metadata_filter_{parameter_index}"
            parameter_index += 1
            single_value_filter = json.dumps({metadata_name: [metadata_value]})
            parameters[parameter_name] = single_value_filter

            if dialect == "postgresql":
                value_sql_conditions.append(f"metadata::jsonb @> :{parameter_name}")
            elif dialect == "duckdb":
                value_sql_conditions.append(f"json_contains(metadata, JSON(:{parameter_name}))")
            else:
                error_message = f"Unsupported dialect: {dialect}."
                raise ValueError(error_message)
        field_sql_conditions.append(f"({' OR '.join(value_sql_conditions)})")
    return f" AND {' AND '.join(field_sql_conditions)}", parameters
