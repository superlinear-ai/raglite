"""Test RAGLite's document deletion."""

from typing import Any

import numpy as np
from sqlmodel import Session, SQLModel

from raglite._config import RAGLiteConfig
from raglite._database import (
    Document,
    create_database_engine,
)
from raglite._delete import delete_documents, delete_documents_by_metadata
from raglite._insert import insert_documents


def get_table_states(session: Session) -> dict[str, list[dict[str, Any]]]:
    """Get the current state of all tables in the database."""
    state = {}
    for table_name, table in SQLModel.metadata.tables.items():
        stmt = table.select().order_by(*table.primary_key.columns)
        rows = session.execute(stmt).all()
        row_dicts = [row._asdict() for row in rows]
        for row in row_dicts:
            for k, v in row.items():
                if isinstance(v, np.ndarray):
                    row[k] = v.tolist()
        state[table_name] = row_dicts
    return state


def test_delete(raglite_test_config: RAGLiteConfig) -> None:
    """Test document deletion."""
    with Session(create_database_engine(raglite_test_config)) as session:
        state_before = get_table_states(session)
    content1 = """# ON THE ELECTRODYNAMICS OF MOVING BODIES## By A. EINSTEIN  June 30, 1905It is known that Maxwell..."""
    document1 = Document.from_text(content1, author="Test Author", classification="A")
    doc1_id = document1.id
    insert_documents([document1], config=raglite_test_config)
    with Session(create_database_engine(raglite_test_config)) as session:
        state_after_insert = get_table_states(session)
    assert state_after_insert != state_before, "State should change after insertion"
    deleted_count = delete_documents([doc1_id, "fake_id"], config=raglite_test_config)
    assert deleted_count == 1, f"Expected 1 document to be deleted, but got {deleted_count}"
    with Session(create_database_engine(raglite_test_config)) as session:
        state_after = get_table_states(session)
        for table_name in state_before:
            assert state_after[table_name] == state_before[table_name], (
                f"After deletion, Table '{table_name}' does not match state before insertion.\n"
            )
        assert state_after.keys() == state_before.keys()


def test_delete_by_metadata(raglite_test_config: RAGLiteConfig) -> None:
    """Test document deletion by metadata."""
    with Session(create_database_engine(raglite_test_config)) as session:
        state_before = get_table_states(session)
    content1 = """# ON THE ELECTRODYNAMICS OF MOVING BODIES## By A. EINSTEIN  June 30, 1905It is known that Maxwell..."""
    document1 = Document.from_text(content1, classification="DELETE_ME")
    insert_documents([document1], config=raglite_test_config)
    document2 = Document.from_text(content1 + " diff", classification="DELETE_ME")
    insert_documents([document2], config=raglite_test_config)
    with Session(create_database_engine(raglite_test_config)) as session:
        state_after_insert = get_table_states(session)
    assert state_after_insert != state_before, "State should change after insertion"
    deleted_count = delete_documents_by_metadata(
        {"classification": "DELETE_ME"}, config=raglite_test_config
    )
    assert deleted_count == 2, f"Expected 2 documents to be deleted, but got {deleted_count}"  # noqa: PLR2004
    with Session(create_database_engine(raglite_test_config)) as session:
        state_after = get_table_states(session)
        for table_name in state_before:
            assert state_after[table_name] == state_before[table_name], (
                f"After deletion, Table '{table_name}' does not match state before insertion.\n"
            )
        assert state_after.keys() == state_before.keys()
