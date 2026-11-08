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

"""Portable, authenticated UTF-8 context packs and prefix-reusing deltas.

The wire format is independent of any live Runtime or sender-side objects.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ContextPackError(ValueError):
    """An invalid context-pack wire value, chunk boundary or delta."""


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _base64_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _check_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, (tuple, list)):
        raise ContextPackError("Context pack items must be a tuple or list of bytes")
    items = tuple(value)
    for item in items:
        if not isinstance(item, bytes):
            raise ContextPackError("Context pack chunks must be bytes")
        try:
            item.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ContextPackError("Each context chunk must be valid UTF-8") from error
    return items


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("Wire items must be a Base64 string array")
    items: list[bytes] = []
    for encoded in value:
        if not isinstance(encoded, str):
            raise ContextPackError("Wire items must contain only Base64 strings")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (UnicodeError, ValueError, binascii.Error) as error:
            raise ContextPackError("Malformed Base64 chunk") from error
        if base64.b64encode(data).decode("ascii") != encoded:
            raise ContextPackError("Noncanonical Base64 encoding")
        items.append(data)
    return _check_items(items)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ContextPackError(f"Duplicate JSON field: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ContextPackError(f"Invalid JSON constant: {value}")


def _read_object(blob: bytes, *, keys: frozenset[str], schema: str) -> dict[str, object]:
    if not isinstance(blob, bytes):
        raise ContextPackError("Wire payload must be bytes")
    try:
        parsed = json.loads(
            blob.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContextPackError("Malformed context-pack JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != keys or parsed.get("schema") != schema:
        raise ContextPackError("Unexpected context-pack schema or fields")
    return parsed


def _check_digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ContextPackError("Invalid SHA-256 digest")
    return value


def _digest_items(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": _PACK_SCHEMA, "items": _base64_items(items)}
    return "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextPack:
    """Complete context as ordered UTF-8 byte chunks and a portable digest."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        verified = _check_items(items)
        digest = _digest_items(verified)
        body = {"schema": _PACK_SCHEMA, "items": _base64_items(verified), "digest": digest}
        # byte_count occurs in its own encoded representation, so its decimal
        # width must reach a fixed point. Convergence takes at most a few steps.
        count = 0
        while True:
            encoded_size = len(_canonical_json({**body, "byte_count": count}))
            if encoded_size == count:
                return cls(items=verified, digest=digest, byte_count=count)
            count = encoded_size

    @classmethod
    def from_content(
        cls, content: str | bytes | None, *, chunk_bytes: int = 512
    ) -> ContextPack:
        """Greedily pack Unicode scalar values without splitting UTF-8 sequences."""

        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if content is None:
            text = ""
        elif isinstance(content, bytes):
            try:
                text = content.decode("utf-8", errors="strict")
            except UnicodeError as error:
                raise ContextPackError("Content is not valid UTF-8") from error
        elif isinstance(content, str):
            text = content
        else:
            raise ContextPackError("Content must be str, bytes, or None")

        chunks: list[bytes] = []
        current = bytearray()
        try:
            for character in text:
                encoded = character.encode("utf-8", errors="strict")
                if len(encoded) > chunk_bytes:
                    raise ContextPackError("A UTF-8 character cannot fit in chunk_bytes")
                if len(current) + len(encoded) > chunk_bytes:
                    chunks.append(bytes(current))
                    current.clear()
                current.extend(encoded)
        except UnicodeError as error:
            raise ContextPackError("Content contains invalid Unicode") from error
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _canonical_json({
            "schema": _PACK_SCHEMA,
            "items": _base64_items(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        payload = _read_object(
            data,
            keys=frozenset({"schema", "items", "digest", "byte_count"}),
            schema=_PACK_SCHEMA,
        )
        digest = _check_digest(payload["digest"])
        count = payload["byte_count"]
        if type(count) is not int or count < 0:
            raise ContextPackError("Pack byte_count must be a nonnegative integer")
        pack = cls.build(_decode_items(payload["items"]))
        if digest != pack.digest or count != pack.byte_count:
            raise ContextPackError("Pack digest or byte_count does not match its body")
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """Copy a prefix of the base pack, then append all target suffix chunks."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        _validate_pack(base)
        _validate_pack(target)
        prefix = 0
        for left, right in zip(base.items, target.items):
            if left != right:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical_json({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _base64_items(self.appended),
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        payload = _read_object(
            data,
            keys=frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"}),
            schema=_DELTA_SCHEMA,
        )
        base_digest = _check_digest(payload["base_digest"])
        target_digest = _check_digest(payload["target_digest"])
        prefix = payload["prefix_items"]
        if type(prefix) is not int or prefix < 0:
            raise ContextPackError("prefix_items must be a nonnegative integer")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=_decode_items(payload["appended"]),
        )


def _validate_pack(pack: ContextPack) -> None:
    if not isinstance(pack, ContextPack):
        raise ContextPackError("Expected a complete ContextPack")
    rebuilt = ContextPack.build(pack.items)
    if pack.digest != rebuilt.digest or pack.byte_count != rebuilt.byte_count:
        raise ContextPackError("Invalid base or target ContextPack")


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Independently reconstruct and authenticate a target from serialized values."""

    _validate_pack(base)
    if not isinstance(delta, ContextPackDelta):
        raise ContextPackError("Expected a ContextPackDelta")
    _check_digest(delta.base_digest)
    _check_digest(delta.target_digest)
    if delta.base_digest != base.digest:
        raise ContextPackError("Delta base digest does not match the provided base")
    if type(delta.prefix_items) is not int or not 0 <= delta.prefix_items <= len(base.items):
        raise ContextPackError("Delta prefix_items is invalid or out of bounds")
    appended = _check_items(delta.appended)
    rebuilt = ContextPack.build((*base.items[:delta.prefix_items], *appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("Delta target digest does not match reconstructed content")
    return rebuilt


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
