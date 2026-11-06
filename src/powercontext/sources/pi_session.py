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

"""Capture complete, immutable evidence on one selected Pi session branch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, model_validator

from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def _event_source_id(session_id: str, event_id: str) -> str:
    if not session_id or not event_id:
        raise ValueError("Pi source identity requires a session ID and event ID")
    source_id = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) > 256:
        # No truncation of either identity component: a bounded stable digest
        # distinguishes long IDs without introducing delimiter collisions.
        identity = json.dumps([session_id, event_id], ensure_ascii=False, separators=(",", ":"))
        source_id = "pi:sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return source_id


class PiEventSource(Source):
    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED
    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_identity(self) -> PiEventSource:
        expected = _event_source_id(self.session_id, self.event_id)
        if self.name != expected or self.source_id != expected:
            raise ValueError("Pi event identity does not match its source name")
        return self


@dataclass(frozen=True, slots=True)
class _PiArchive:
    session_id: str
    nodes: dict[str, dict[str, JsonValue]]
    ordered_ids: tuple[str, ...]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant {value!r}")


def _parse_archive(value: PiSessionInput) -> _PiArchive:
    try:
        text = value.content.decode("utf-8-sig", errors="strict")
    except UnicodeError as error:
        raise ValueError("Pi session is not valid UTF-8") from error

    entries: list[dict[str, JsonValue]] = []
    try:
        for line in text.splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(parsed, dict):
                raise ValueError("Every Pi JSONL entry must be a JSON object")
            entries.append(parsed)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Malformed Pi session JSONL") from error

    if not entries:
        raise ValueError("Pi session has no header")
    header = entries[0]
    if (header.get("type") != "session"
            or type(header.get("version")) is not int
            or header["version"] != 3
            or not isinstance(header.get("id"), str)
            or not header["id"]):
        raise ValueError("Pi session requires a v3 session header with a nonempty ID")

    session_id = header["id"]
    nodes: dict[str, dict[str, JsonValue]] = {}
    order: list[str] = []
    for record in entries[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Pi entry ID must be a nonempty string")
        if event_id in nodes:
            raise ValueError(f"Duplicate Pi entry ID: {event_id}")
        if "parentId" not in record:
            raise ValueError(f"Pi entry {event_id} is missing parentId")
        parent_id = record["parentId"]
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in nodes):
            raise ValueError(f"Pi entry {event_id} has a missing, forward, or cyclic parent")
        nodes[event_id] = record
        order.append(event_id)

    return _PiArchive(session_id=session_id, nodes=nodes, ordered_ids=tuple(order))


def _selected_records(archive: _PiArchive, leaf_id: str | None) -> tuple[dict[str, JsonValue], ...]:
    if leaf_id is None:
        if not archive.ordered_ids:
            return ()
        leaf_id = archive.ordered_ids[-1]
    if leaf_id not in archive.nodes:
        raise ValueError(f"Pi leaf {leaf_id!r} is not present in the session")

    path: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise ValueError("Pi history contains a cycle")
        seen.add(current)
        record = archive.nodes[current]
        path.append(record)
        current = record["parentId"]  # Validated as a prior ID or None.
    path.reverse()
    return tuple(path)


def _visible_blocks(value: object) -> str:
    """Render only visible text and tool-call blocks; never include hidden thinking/images."""

    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    chunks: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            chunks.append(block["text"])
        elif kind == "toolCall":
            name = block.get("name", "")
            tool_id = block.get("id", "")
            arguments = block.get("arguments", {})
            args = json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            chunks.append(f"[tool_call {name} {tool_id}] {args}")
    return "\n".join(chunks)


def _event_from_record(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    kind = record.get("type")
    role = ""
    body = ""
    if kind == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        message_role = message.get("role")
        if not isinstance(message_role, str) or not message_role:
            return None
        role = message_role
        if role == "toolResult":
            text = _visible_blocks(message.get("content"))
            body = f"[tool_result {message.get('toolName', '')} {message.get('toolCallId', '')}] {text}"
        elif role == "bashExecution":
            command = message.get("command", "")
            output = message.get("output", "")
            body = f"$ {command}\n{output}"
        else:
            body = _visible_blocks(message.get("content"))
    elif kind in ("compaction", "branch_summary"):
        role = kind
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
    elif kind == "custom_message":
        role = "custom"
        body = _visible_blocks(record.get("content"))
    else:
        # Graph control entries remain on the path but do not emit evidence.
        return None

    if not body.strip():
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=record["parentId"],
        event_type=kind,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _visible_path(archive: _PiArchive, leaf_id: str | None) -> tuple[PiEventInput, ...]:
    return tuple(
        event
        for record in _selected_records(archive, leaf_id)
        if (event := _event_from_record(archive.session_id, record)) is not None
    )


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the entire archive first; return only visible events on the selected root-to-leaf path."""

    return _visible_path(_parse_archive(value), value.leaf_id)


class _PiEventAdapter:
    name = "pi-event"
    input_class = PiEventInput
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = _event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiTextEvidence:
    name = "powercontext.text-evidence"
    version = "1"
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = (
    AdapterSourceDefinition(
        adapter=_PiEventAdapter(),
        version="1",
        projections=(_PiTextEvidence(),),
    )
)


def _record_checksum(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validated_checkpoint(
    checkpoint: JsonValue | None, archive: _PiArchive,
) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
        raise ValueError("Invalid Pi session connector checkpoint")
    if checkpoint.get("session_id") != archive.session_id:
        raise ValueError("Pi session ID changed since the last successful checkpoint")
    imported = checkpoint.get("imported")
    if not isinstance(imported, dict) or not all(
        isinstance(event_id, str) and isinstance(digest, str)
        for event_id, digest in imported.items()
    ):
        raise ValueError("Invalid Pi connector imported-event checkpoint")

    for event_id, checksum in imported.items():
        previous_record = archive.nodes.get(event_id)
        if previous_record is None or _record_checksum(previous_record) != checksum:
            raise ValueError(f"Previously imported Pi event {event_id!r} was removed or rewritten")
    return dict(imported)


class PiSessionConnector:
    """Incrementally capture a selected branch; lifecycle commits the checkpoint."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        # Parse/validate the whole file and all previously imported event records
        # before submitting anything. No module-level progress state is used.
        archive = _parse_archive(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        events = _visible_path(archive, self.leaf_id)
        imported = _validated_checkpoint(session.checkpoint, archive)

        for event in events:
            if event.event_id in imported:
                continue
            item_id = _event_source_id(event.session_id, event.event_id)
            outcome = await session.submit(item_id, "pi-event", event)
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                return ConnectorRunCompletion(
                    status=ConnectorRunStatus.INCOMPLETE,
                    checkpoint=session.checkpoint,
                )
            imported[event.event_id] = _record_checksum(event.record)

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": 1, "session_id": archive.session_id, "imported": imported},
        )


__all__ = [
    "PiSessionInput",
    "PiEventInput",
    "PiEventSource",
    "parse_pi_session",
    "PI_EVENT_SOURCE_DEFINITION",
    "PiSessionConnector",
]
