"""Regression tests for the HTTP Memory reranker protocol."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit


def _hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="m", revision=1),
        entry_id=f"entry-{rank}",
        entry_version_id=f"version-{rank}",
        text=f"Candidate text {rank}",
        score=1 / rank,
        matched_by=("fts",),
    )


def test_cross_batch_global_ranking_and_ties() -> None:
    requests = []
    scores = {1: 0.1, 2: 0.9, 3: 0.95, 4: 0.9, 5: -1.0}

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        assert set(data) == {"query", "candidates"}
        assert data["query"] == "database timeout"
        assert 1 <= len(data["candidates"]) <= 2
        for candidate in data["candidates"]:
            assert set(candidate) == {"rank", "text"}
            assert candidate["text"] == _hit(candidate["rank"]).text
        requests.append(data)
        # Deliberately return response records out of order.
        return httpx.Response(200, json={"scores": [
            {"rank": value["rank"], "score": scores[value["rank"]]}
            for value in reversed(data["candidates"])
        ]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid/score", client=client, batch_size=2)
            decision = await reranker.rerank("database timeout", tuple(_hit(i) for i in range(1, 6)), 4)
            assert decision.selected_ranks == (3, 2, 4, 1)
            assert decision.usage.requests == 3
            assert reranker.policy_id == "http-cross-encoder-v1"
        assert len(requests) == 3
        assert sorted(candidate["rank"] for request in requests for candidate in request["candidates"]) == [
            1, 2, 3, 4, 5,
        ]

    asyncio.run(run())


@pytest.mark.parametrize("response_body", [
    {"scores": [], "extra": True},
    {"scores": []},
    {"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.5}]},
    {"scores": [{"rank": 2, "score": 0.5}]},
    {"scores": [{"rank": 1, "score": float("inf")}]},
    {"scores": [{"rank": 1, "score": float("nan")}]},
    {"scores": [{"rank": 1, "score": True}]},
    {"scores": [{"rank": True, "score": 0.5}]},
    {"scores": [{"rank": 1.0, "score": 0.5}]},
    {"scores": [{"rank": 1, "score": "0.5"}]},
    {"scores": [{"rank": 1, "score": 0.5, "surprise": 1}]},
    {"scores": {"rank": 1, "score": 0.5}},
    [{"scores": [{"rank": 1, "score": 0.5}]}],
    {"other": [{"rank": 1, "score": 0.5}]},
])
def test_reject_bad_response_schema(response_body) -> None:
    async def run() -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps(response_body).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(run())


def test_rejects_duplicate_object_keys_and_non_json() -> None:
    for body in [
        b'{"scores":[{"rank":1,"rank":1,"score":0.2}]}',
        b'{"scores":[{"rank":1,"score":0.2}],"scores":[]}',
        b"not json",
    ]:
        async def run() -> None:
            transport = httpx.MockTransport(lambda _: httpx.Response(200, content=body))
            async with httpx.AsyncClient(transport=transport) as client:
                with pytest.raises(ValueError):
                    await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank(
                        "q", (_hit(1),), 1
                    )

        asyncio.run(run())


def test_bad_bounds_do_not_send_http_requests() -> None:
    count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.5}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for batch_size in (0, -1, True, 1.5):
                with pytest.raises(ValueError):
                    HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=batch_size)
            reranker = HTTPMemoryReranker("https://reranker.invalid", client=client)
            for candidates, limit in (((), 1), ((_hit(1),), 0), ((_hit(1),), 2)):
                with pytest.raises(ValueError):
                    await reranker.rerank("q", candidates, limit)
        assert count == 0

    asyncio.run(run())


def test_http_errors_pass_through() -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable"))
        ) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank(
                    "q", (_hit(1),), 1
                )
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("network failed", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            with pytest.raises(httpx.ConnectError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank(
                    "q", (_hit(1),), 1
                )

    asyncio.run(run())
