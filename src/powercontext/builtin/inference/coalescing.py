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

"""Per-event-loop embedding singleflight, bounded batching and successful LRU reuse."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage

if TYPE_CHECKING:
    from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True, eq=False)
class _Batch:
    """The provider usage belongs to the batch, not individual text positions."""

    usage: InferenceUsage | None = None


@dataclass(slots=True, eq=False)
class _Flight:
    text: str
    future: asyncio.Future[EmbeddingVector]
    batch: _Batch | None = None


def _consume_future_exception(future: asyncio.Future[EmbeddingVector]) -> None:
    # A caller can cancel before a shared provider request fails. Retrieve any
    # unobserved exception without altering the result other callers may await.
    if not future.cancelled():
        future.exception()


def _sum_usage(batches: tuple[_Batch, ...]) -> InferenceUsage:
    usage = tuple(batch.usage for batch in batches if batch.usage is not None)

    def sum_optional(field: str) -> int | None:
        known = [getattr(item, field) for item in usage if getattr(item, field) is not None]
        return sum(known) if known else None

    return InferenceUsage(
        requests=sum(item.requests for item in usage),
        input_tokens=sum_optional("input_tokens"),
        output_tokens=sum_optional("output_tokens"),
    )


class CoalescingEmbeddingModel:
    """Share in-progress text embeddings and cache only completed successes.

    A pending text is registered synchronously before any await, so concurrent
    calls see the same Flight. Even with coalesce_ms=0, an asyncio scheduling
    turn lets other ready callers join the pending batch.

    Each caller waits on shielded per-text futures, not on provider tasks: one
    cancelled caller cannot cancel work another caller needs. The shared model
    is intended for calls made in one event loop.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        if (
            isinstance(coalesce_ms, bool)
            or not isinstance(coalesce_ms, (float, int))
            or not math.isfinite(coalesce_ms)
            or coalesce_ms < 0
        ):
            raise ValueError("coalesce_ms must be finite and nonnegative")

        self.profile = model.profile
        self._model = model
        self._batch_size = max_batch_size
        self._delay = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size

        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _Flight] = {}
        self._pending: OrderedDict[str, _Flight] = OrderedDict()
        self._dispatcher: asyncio.Task[None] | None = None
        self._workers: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        # No awaits until all input positions are registered: duplicate texts
        # across calls are assigned to precisely one pending/running flight.
        values: dict[str, EmbeddingVector] = {}
        waiting: dict[str, _Flight] = {}
        loop = asyncio.get_running_loop()

        for text in texts:
            if text in values or text in waiting:
                continue
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                values[text] = cached
                continue

            flight = self._inflight.get(text)
            if flight is None:
                future: asyncio.Future[EmbeddingVector] = loop.create_future()
                future.add_done_callback(_consume_future_exception)
                flight = _Flight(text=text, future=future)
                self._inflight[text] = flight
                self._pending[text] = flight
            waiting[text] = flight

        if self._pending and self._dispatcher is None:
            self._dispatcher = asyncio.create_task(self._dispatch_pending())

        if waiting:
            # asyncio.shield is applied to *each* shared future. Cancelling the
            # gather only cancels the private wrappers, never provider work.
            resolved = await asyncio.gather(
                *(asyncio.shield(flight.future) for flight in waiting.values())
            )
            values.update(zip(waiting, resolved, strict=True))

        batches: dict[_Batch, None] = {}
        for flight in waiting.values():
            if flight.batch is None:
                raise RuntimeError("Embedding completed without a provider batch")
            batches[flight.batch] = None

        return EmbeddingResult(
            vectors=tuple(values[text] for text in texts),
            usage=_sum_usage(tuple(batches)),
        )

    async def _dispatch_pending(self) -> None:
        # A zero delay still yields one event-loop turn for concurrently ready
        # calls to join; the registered flights stay shared while running.
        await asyncio.sleep(self._delay)
        pending = tuple(self._pending.values())
        self._pending.clear()
        self._dispatcher = None

        for begin in range(0, len(pending), self._batch_size):
            chunk = pending[begin : begin + self._batch_size]
            batch = _Batch()
            for flight in chunk:
                flight.batch = batch
            worker = asyncio.create_task(self._run_batch(chunk, batch))
            self._workers.add(worker)
            worker.add_done_callback(self._workers.discard)

    async def _run_batch(self, flights: tuple[_Flight, ...], batch: _Batch) -> None:
        try:
            result = await self._model.embed(tuple(flight.text for flight in flights))
            if len(result.vectors) != len(flights):
                raise ValueError("Embedding provider returned a wrong number of vectors")
        except BaseException as error:
            for flight in flights:
                if self._inflight.get(flight.text) is flight:
                    self._inflight.pop(flight.text)
                if not flight.future.done():
                    flight.future.set_exception(error)
            return

        batch.usage = result.usage
        for flight, vector in zip(flights, result.vectors, strict=True):
            if self._cache_size:
                self._cache[flight.text] = vector
                self._cache.move_to_end(flight.text)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

            if self._inflight.get(flight.text) is flight:
                self._inflight.pop(flight.text)
            if not flight.future.done():
                flight.future.set_result(vector)


__all__ = ["CoalescingEmbeddingModel"]
