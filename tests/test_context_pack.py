"""Portable context-pack protocol regression tests."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)


def test_exact_wire_format_and_digest() -> None:
    items = (b"alpha\\n", "海洋".encode("utf-8"))
    pack = ContextPack.build(items)
    unsigned = {
        "schema": "powercontext.context-pack.v1",
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }
    expected_digest = "sha256:" + hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    encoded = pack.to_bytes()
    payload = json.loads(encoded)
    assert pack.digest == expected_digest
    assert pack.byte_count == len(encoded)
    assert payload == {**unsigned, "digest": expected_digest, "byte_count": len(encoded)}
    assert encoded == json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert ContextPack.from_bytes(encoded) == pack


def test_utf8_content_chunk_boundaries_and_empty() -> None:
    content = "alpha 海洋 beta" * 30
    pack = ContextPack.from_content(content, chunk_bytes=7)
    assert b"".join(pack.items).decode("utf-8") == content
    assert all(len(item) <= 7 for item in pack.items)
    assert all(item.decode("utf-8") for item in pack.items)
    assert ContextPack.from_content("").items == ()
    assert ContextPack.from_content(None).items == ()
    with pytest.raises(ContextPackError):
        ContextPack.from_content("海", chunk_bytes=2)


@pytest.mark.parametrize("size", [0, -1, True, 2.5])
def test_bad_chunk_budget(size) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("text", chunk_bytes=size)


@pytest.mark.parametrize("target", [
    (b"a", b"b", b"new", b"tail"),
    (b"new", b"b", b"old"),
    (b"a", b"new", b"old"),
    (),
])
def test_serialized_delta_rebuilds_exact_target(target) -> None:
    base = ContextPack.build((b"a", b"b", b"old"))
    target_pack = ContextPack.build(target)
    delta = ContextPackDelta.between(base, target_pack)
    assert delta.byte_count == len(delta.to_bytes())
    assert apply_context_pack_delta(
        ContextPack.from_bytes(base.to_bytes()), ContextPackDelta.from_bytes(delta.to_bytes())
    ) == target_pack

    # Even a zero-prefix delta is a valid reconstruction.
    payload = json.loads(delta.to_bytes())
    payload["prefix_items"] = 0
    payload["appended"] = [base64.b64encode(item).decode() for item in target]
    reencoded = json.dumps(payload).encode("utf-8")  # Noncanonical whitespace is allowed on input.
    assert apply_context_pack_delta(base, ContextPackDelta.from_bytes(reencoded)) == target_pack


@pytest.mark.parametrize("field", ["schema", "items", "digest", "byte_count"])
def test_pack_missing_fields_rejected(field) -> None:
    wire = json.loads(ContextPack.build((b"x",)).to_bytes())
    wire.pop(field)
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps(wire).encode())


def test_pack_rejects_tampering_and_duplicates() -> None:
    pack = ContextPack.build((b"x",))
    wire = json.loads(pack.to_bytes())
    for override in [
        {"digest": "sha256:" + "0" * 64},
        {"byte_count": pack.byte_count + 1},
        {"unexpected": "extra"},
        {"items": ["!!!"]},
        {"items": ["/w=="]},
    ]:
        changed = {**wire, **override}
        with pytest.raises(ContextPackError):
            ContextPack.from_bytes(json.dumps(changed).encode())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(b'{"schema":"a","schema":"b"}')


@pytest.mark.parametrize("prefix", [True, 1.5, -1])
def test_delta_invalid_prefix_rejected(prefix) -> None:
    base = ContextPack.build((b"a",))
    delta = ContextPackDelta.between(base, ContextPack.build((b"a", b"b")))
    wire = json.loads(delta.to_bytes())
    wire["prefix_items"] = prefix
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps(wire).encode())


def test_delta_wrong_base_and_target_rejected() -> None:
    base = ContextPack.build((b"one",))
    target = ContextPack.build((b"one", b"two"))
    delta = ContextPackDelta.between(base, target)
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(ContextPack.build((b"other",)), delta)
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, "sha256:" + "0" * 64, 1, (b"two",)))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, target.digest, 2, (b"two",)))


def test_bounded_delta_transfer_for_repeated_context() -> None:
    # A stable prefix and changing suffix should benefit from prefix reuse.
    current = "Long shared meeting history. " * 180
    first = ContextPack.from_content(current)
    previous = ContextPack.from_bytes(first.to_bytes())
    incremental = first.byte_count
    raw_full = len(current.encode("utf-8"))
    for n in range(1, 40):
        updated = current + "tail" * n
        target = ContextPack.from_content(updated)
        delta = ContextPackDelta.between(previous, target)
        previous = apply_context_pack_delta(
            ContextPack.from_bytes(previous.to_bytes()),
            ContextPackDelta.from_bytes(delta.to_bytes()),
        )
        assert b"".join(previous.items).decode("utf-8") == updated
        incremental += delta.byte_count
        raw_full += len(updated.encode("utf-8"))
    assert 1 - incremental / raw_full >= .45
