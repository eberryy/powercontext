# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture one CSV record per durable, primary-key-addressed Source."""

from __future__ import annotations

import csv
import io
import json
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, model_validator

from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence


class CSVDocumentInput(BaseModel):
    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Self-contained captured row: no original CSV or row offset is needed to read it."""

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED
    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> CSVRowSource:
        expected = csv_row_source_id(self.document_id, self.primary_key_value)
        if self.name != expected or self.source_id != expected:
            raise ValueError("CSV row name and source_id must equal its stable identity")
        return self


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be nonempty and have no surrounding whitespace")


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Encode both identity components separately to prevent delimiter collisions."""

    _validate_document_id(document_id)
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must contain a non-whitespace character")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > 256:
        raise ValueError("CSV row source_id exceeds 256 characters")
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Strictly decode and validate an entire CSV document before returning rows."""

    _validate_document_id(value.document_id)
    if not value.primary_key:
        raise ValueError("primary_key must be nonempty")
    try:
        # utf-8-sig removes a BOM only at the start. newline="" preserves quoted CRLF.
        text = value.content.decode("utf-8-sig", errors="strict")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=",", quotechar='"',
                            doublequote=True, strict=True)
        header = next(reader, None)
        if not header or any(not column.strip() for column in header):
            raise ValueError("CSV header must contain nonblank columns")
        if len(header) != len(set(header)):
            raise ValueError("CSV header contains duplicate columns")
        if value.primary_key not in header:
            raise ValueError("primary_key must match a header column exactly")

        columns = tuple(header)
        pk_index = header.index(value.primary_key)
        seen_keys: set[str] = set()
        result: list[CSVRowInput] = []

        for fields in reader:
            # csv.reader returns [] only for an empty physical record.
            if not fields:
                continue
            if len(fields) != len(columns):
                raise ValueError("CSV row width does not match the header")
            primary_key_value = fields[pk_index]
            if not primary_key_value.strip():
                raise ValueError("CSV row primary key is blank")
            if primary_key_value in seen_keys:
                raise ValueError("CSV row primary key is duplicated")
            # Check the bounded identity for every row, not just during resolution.
            csv_row_source_id(value.document_id, primary_key_value)
            seen_keys.add(primary_key_value)
            result.append(CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(fields),
            ))
        return tuple(result)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("Invalid UTF-8 or malformed CSV document") from error


class _CSVRowAdapter:
    name = "csv-row"
    input_class = CSVRowInput
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        if (not value.columns or len(value.columns) != len(value.values)
                or any(not column.strip() for column in value.columns)
                or len(set(value.columns)) != len(value.columns)
                or value.primary_key not in value.columns
                or value.values[value.columns.index(value.primary_key)] != value.primary_key_value):
            raise ValueError("CSV row input does not match its column and key declarations")
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextProjection:
    name = "powercontext.text-evidence"
    version = "1"
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = (
    AdapterSourceDefinition(
        adapter=_CSVRowAdapter(),
        version="1",
        projections=(_CSVRowTextProjection(),),
    )
)


__all__ = [
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSV_ROW_SOURCE_DEFINITION",
    "parse_csv_document",
    "csv_row_source_id",
]
