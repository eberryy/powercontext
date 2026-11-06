"""Regression checks for Pi branch capture, projection and checkpoint safety."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from powercontext.sources import (
    ConnectorBinding,
    SourceDefinitionRegistry,
    SourceMaterialization,
    SourceRef,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _archive(records, session_id="test", leaf_id=None):
    rows = [{"type": "session", "version": 3, "id": session_id}, *records]
    raw = "\n".join(json.dumps(record, ensure_ascii=False) for record in rows).encode("utf-8")
    return parse_pi_session(PiSessionInput(content=raw, leaf_id=leaf_id))


def test_selected_path_control_tool_result_and_compaction():
    records = [
        {"type": "message", "id": "a", "parentId": None,
         "message": {"role": "user", "content": "hi"}},
        {"type": "message", "id": "left", "parentId": "a",
         "message": {"role": "assistant", "content": "not selected"}},
        {"type": "model_change", "id": "ctrl", "parentId": "a"},
        {"type": "compaction", "id": "c", "parentId": "ctrl", "summary": "prior history"},
        {"type": "message", "id": "call", "parentId": "c",
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "not evidence"},
             {"type": "toolCall", "name": "read", "id": "x", "arguments": {"z": 2, "a": "中文"}},
         ]}},
        {"type": "message", "id": "result", "parentId": "call",
         "message": {"role": "toolResult", "toolName": "read", "toolCallId": "x",
                     "content": [{"type": "text", "text": "result\n" * 3000}]}},
    ]
    events = _archive(records, leaf_id="result")
    assert [event.event_id for event in events] == ["a", "c", "call", "result"]
    assert [event.parent_id for event in events] == [None, "ctrl", "c", "call"]
    assert events[2].content == 'assistant: [tool_call read x] {"a":"中文","z":2}'
    assert events[3].content.startswith("toolResult: [tool_result read x] result\n")
    assert len(events[3].record["message"]["content"][0]["text"]) == len("result\n" * 3000)
    assert [event.event_id for event in _archive(records, leaf_id="left")] == ["a", "left"]


def test_projection_and_native_observation():
    event = _archive([{"type": "message", "id": "a:b", "parentId": None,
                       "message": {"role": "user", "content": "Source text"}}])[0]
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))

    async def scenario():
        source = await registry.resolve(event)
        assert source.materialization is SourceMaterialization.CAPTURED
        assert source.name == source.source_id == "pi:test:a%3Ab"
        assert await registry.read(source) == event
        evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
        assert evidence.content == "user: Source text"
        assert evidence.metadata == {
            "session_id": "test", "event_id": "a:b", "parent_id": None,
            "event_type": "message", "role": "user",
        }
        observation = project_source_for_transport(registry, source)
        assert observation.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"] == evidence.content
        assert observation.payload["record"] == event.record
        assert manifest_for_definition(PI_EVENT_SOURCE_DEFINITION).name == "pi-event"

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", [
    b"", b"{}", b"[]", b"\xff",
    b'{"type":"session","version":2,"id":"test"}',
    b'{"type":"session","version":3,"id":"test"}\n{"type":"model_change","id":"a","parentId":"not-here"}',
    b'{"type":"session","version":3,"id":"test"}\n{"type":"model_change","id":"a","parentId":null}\n{"type":"model_change","id":"a","parentId":null}',
])
def test_reject_bad_archives(bad):
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=bad))


class _Sink:
    def __init__(self):
        self.items = []

    async def submit(self, binding, item_id, definition_name, value):
        self.items.append((item_id, value.record))
        return SourceRef(source_type=definition_name, source_id=item_id)


class _Store:
    def __init__(self):
        self.state = None

    async def load(self, binding):
        return self.state

    async def save(self, binding, checkpoint, *, expected):
        assert self.state == expected
        self.state = checkpoint


def test_branch_switch_replay_and_rewrite_rejection(tmp_path: Path):
    path = tmp_path / "session.jsonl"
    rows = [
        {"type": "message", "id": "root", "parentId": None,
         "message": {"role": "user", "content": "Shared"}},
        {"type": "message", "id": "left", "parentId": "root",
         "message": {"role": "assistant", "content": "Left"}},
        {"type": "model_change", "id": "ctrl", "parentId": "root"},
        {"type": "message", "id": "right", "parentId": "ctrl",
         "message": {"role": "assistant", "content": "Right"}},
    ]

    def write():
        path.write_text("\n".join(json.dumps(record) for record in [
            {"type": "session", "version": 3, "id": "test"}, *rows,
        ]) + "\n", encoding="utf-8")

    write()
    sink, store = _Sink(), _Store()
    lifecycle = ConnectorLifecycle(sink=sink, checkpoints=store)
    binding = ConnectorBinding(
        scope_id="test-scope", binding_id="pi", connector_name="pi-session-history",
        connector_version="1",
    )

    async def run(leaf):
        return await lifecycle.run(PiSessionConnector(path, leaf), binding)

    async def scenario():
        first = await run("left")
        assert len(first.items) == 2
        second = await run("right")
        assert len(second.items) == 1
        replay = await run("right")
        assert not replay.items
        assert len(sink.items) == 3

        checkpoint_before = store.state
        path.write_text(path.read_text(encoding="utf-8") + "{broken json}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            await run("right")
        assert store.state == checkpoint_before

        write()
        rows[0]["message"]["content"] = "rewritten"
        write()
        with pytest.raises(ValueError):
            await run("right")
        assert store.state == checkpoint_before
        rows[0]["message"]["content"] = "Shared"
        write()
        assert not (await run("right")).items

    asyncio.run(scenario())
