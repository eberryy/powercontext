"""Contract tests for captured CSV row evidence."""

from __future__ import annotations

import asyncio
import json

import pytest

from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceMaterialization,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    CSV_ROW_SOURCE_DEFINITION,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def parse(data: bytes, *, document_id: str = "doc", primary_key: str = "id"):
    return parse_csv_document(CSVDocumentInput(
        document_id=document_id, primary_key=primary_key, content=data,
    ))


def test_quoted_csv_bom_and_raw_values() -> None:
    rows = parse(b'\xef\xbb\xbf" id ",name,note\r\n001,"East, Airport","hello\r\nworld ""yes"""\r\n\r\n',
                 primary_key=" id ")
    assert len(rows) == 1
    assert rows[0].columns == (" id ", "name", "note")
    assert rows[0].values == ("001", "East, Airport", 'hello\r\nworld "yes"')


def test_header_only_and_blank_physical_rows() -> None:
    assert parse(b"id,name\n\n\r\n") == ()
    assert [row.values for row in parse(b"id,name\n1,one\n\n2,two\n")] == [
        ("1", "one"), ("2", "two"),
    ]


@pytest.mark.parametrize("data", [
    b"", b"\xef\xbb\xbf", b"id,id\n1,1\n", b",name\n1,one\n",
    b" \t,name\n1,one\n", b"id,name\n1,one\n1,two\n",
    b"id,name\n,one\n", b"id,name\n \t,one\n",
    b"id,name\n1\n", b"id,name\n1,one,extra\n",
    b'id,name\n1,"unclosed\n', b"id,name\n\xff,invalid\n",
])
def test_invalid_inputs_fail_atomically(data: bytes) -> None:
    with pytest.raises(ValueError):
        parse(data)


def test_exact_header_matching() -> None:
    with pytest.raises(ValueError):
        parse(b" id ,name\n1,one\n", primary_key="id")
    assert parse(b" id ,name\n1,one\n", primary_key=" id ")[0].values == ("1", "one")


def test_no_trim_and_stable_identity() -> None:
    source_id = csv_row_source_id("doc:one", " key/one ")
    assert source_id == "csv:doc%3Aone:%20key%2Fone%20"
    assert source_id != csv_row_source_id("doc", "one")
    assert source_id != csv_row_source_id("doc:one", "key/one")


@pytest.mark.parametrize("document_id, key", [
    (" doc ", "x"), ("", "x"), ("\tfoo", "x"), ("doc", ""),
    ("doc", " \t"), ("doc", "x" * 260),
])
def test_identity_validation(document_id: str, key: str) -> None:
    with pytest.raises(ValueError):
        csv_row_source_id(document_id, key)


def test_rows_resolve_read_project_and_transport() -> None:
    rows = parse("id,name,empty\n001,北方机场,\n002,Second,  \n".encode())
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)
    assert manifest.name == "csv-row"
    assert manifest.version == "1"
    assert len(manifest.projections) == 1

    for row in rows:
        source = asyncio.run(registry.resolve(row))
        assert isinstance(source, CSVRowSource)
        assert source.materialization is SourceMaterialization.CAPTURED
        assert source.name == source.source_id == csv_row_source_id(row.document_id, row.primary_key_value)
        assert asyncio.run(registry.read(source)) == row
        mapping = dict(zip(row.columns, row.values, strict=True))
        evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
        assert evidence.content == json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))
        assert evidence.metadata == {
            "document_id": row.document_id,
            "primary_key": row.primary_key,
            "primary_key_value": row.primary_key_value,
            "columns": list(row.columns),
            "row": mapping,
        }

        observation = project_source_for_transport(registry, source)
        assert observation.source_type == "csv-row"
        assert observation.payload["source_id"] == source.source_id
        assert TextEvidence.model_validate(observation.projection(TEXT_EVIDENCE_PROJECTION_KEY)) == evidence


def test_order_independence_and_changed_nonsignificant_fields() -> None:
    first = parse(b"id,name\n01,Alpha\n02,Beta\n")
    second = parse(b"id,name\n02,Beta\n01,Revised\n")
    assert {row.primary_key_value: csv_row_source_id(row.document_id, row.primary_key_value) for row in first} == {
        row.primary_key_value: csv_row_source_id(row.document_id, row.primary_key_value) for row in second
    }


def test_direct_row_must_be_consistent() -> None:
    bad = CSVRowInput(document_id="doc", primary_key="id", primary_key_value="1",
                      columns=("id", "name"), values=("2", "value"))
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    with pytest.raises(ValueError):
        asyncio.run(registry.resolve(bad))
