"""Contract tests for cross-call embedding coalescing and cache accounting."""

from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel


class Metered:
    profile = EmbeddingProfile(profile_id="coalescing-fixture", model="offline", dimension=1)

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(tuple(texts))
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) for text in texts),
            usage=InferenceUsage(requests=1, input_tokens=7, output_tokens=2),
        )


def test_zero_window_still_shares_24_texts_across_six_callers() -> None:
    async def run() -> None:
        base = Metered()
        model = CoalescingEmbeddingModel(base, max_batch_size=6, coalesce_ms=0, cache_size=48)
        texts = tuple(f"item-{index}" for index in range(24))
        outcomes = await asyncio.gather(*(model.embed(texts) for _ in range(6)))
        assert len(base.calls) == 4
        assert sorted(text for batch in base.calls for text in batch) == sorted(texts)
        assert all(len(batch) == 6 and len(batch) == len(set(batch)) for batch in base.calls)
        assert all(
            outcome.vectors == tuple((float(len(text)),) for text in texts)
            and outcome.usage == InferenceUsage(requests=4, input_tokens=28, output_tokens=8)
            for outcome in outcomes
        )
        cached = await asyncio.gather(*(model.embed(texts) for _ in range(6)))
        assert len(base.calls) == 4
        assert all(outcome.usage.requests == 0 for outcome in cached)

    asyncio.run(run())


def test_repeated_positions_and_shared_batches_charge_once() -> None:
    async def run() -> None:
        base = Metered()
        model = CoalescingEmbeddingModel(base, max_batch_size=2, coalesce_ms=10)
        a, b = await asyncio.gather(model.embed(("a", "bb", "a")), model.embed(("bb", "ccc")))
        assert a.vectors == ((1.0,), (2.0,), (1.0,))
        assert b.vectors == ((2.0,), (3.0,))
        assert a.usage.requests == sum(bool({"a", "bb"}.intersection(batch)) for batch in base.calls)
        assert a.usage.input_tokens == 7 * a.usage.requests
        assert b.usage.requests == sum(bool({"bb", "ccc"}.intersection(batch)) for batch in base.calls)
        assert len(base.calls) == 2

    asyncio.run(run())


def test_lru_cache_and_empty_batch() -> None:
    async def run() -> None:
        base = Metered()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=2)
        assert (await model.embed(())).usage.requests == 0
        assert base.calls == []
        for value in ("a", "b", "a", "c", "a", "b"):
            await model.embed((value,))
        assert base.calls == [("a",), ("b",), ("c",), ("b",)]

    asyncio.run(run())


def test_late_waiter_cancellation_and_cache_disabled() -> None:
    class Slow(Metered):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts):
            self.calls.append(tuple(texts))
            self.started.set()
            await self.release.wait()
            return EmbeddingResult(
                vectors=tuple((3.0,) for _ in texts),
                usage=InferenceUsage(requests=1, input_tokens=17),
            )

    async def run() -> None:
        base = Slow()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)
        first = asyncio.create_task(model.embed(("x",)))
        await base.started.wait()
        second = asyncio.create_task(model.embed(("x", "x")))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        base.release.set()
        outcome = await second
        assert outcome.vectors == ((3.0,), (3.0,))
        assert outcome.usage.requests == 1
        assert outcome.usage.input_tokens == 17
        assert base.calls == [("x",)]
        await model.embed(("x",))
        assert base.calls == [("x",), ("x",)]

    asyncio.run(run())


def test_failed_batch_does_not_poison_independent_work_or_cache() -> None:
    class Flaky(Metered):
        async def embed(self, texts):
            self.calls.append(tuple(texts))
            if "bad" in texts and self.calls.count(("bad",)) == 1:
                raise RuntimeError("temporary")
            return EmbeddingResult(
                vectors=tuple((4.0,) for _ in texts),
                usage=InferenceUsage(requests=1),
            )

    async def run() -> None:
        base = Flaky()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
        failed, good = await asyncio.gather(
            model.embed(("bad",)), model.embed(("good",)), return_exceptions=True
        )
        assert isinstance(failed, RuntimeError)
        assert good.vectors == ((4.0,),)
        assert (await model.embed(("bad",))).vectors == ((4.0,),)
        assert (await model.embed(("good",))).usage.requests == 0
        assert base.calls.count(("bad",)) == 2
        assert base.calls.count(("good",)) == 1

    asyncio.run(run())


def test_malformed_vector_count_is_a_retryable_value_error() -> None:
    class Wrong(Metered):
        async def embed(self, texts):
            self.calls.append(tuple(texts))
            if len(self.calls) == 1:
                return EmbeddingResult(vectors=())
            return await super().embed(texts)

    async def run() -> None:
        base = Wrong()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        with pytest.raises(ValueError):
            await model.embed(("x",))
        assert (await model.embed(("x",))).vectors == ((1.0,),)

    asyncio.run(run())


@pytest.mark.parametrize(
    "params",
    [
        {"max_batch_size": 0},
        {"max_batch_size": True},
        {"max_batch_size": 1.5},
        {"cache_size": -1},
        {"cache_size": False},
        {"coalesce_ms": -1},
        {"coalesce_ms": float("nan")},
        {"coalesce_ms": float("inf")},
        {"coalesce_ms": True},
    ],
)
def test_invalid_parameters(params) -> None:
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(Metered(), **params)
