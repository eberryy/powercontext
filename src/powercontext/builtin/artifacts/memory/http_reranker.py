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

"""HTTP score-based reranking across a complete Memory candidate pool.

Global one-based ranks are preserved through batching, responses are validated
against their *individual* request batches, and selection is made only after
all scores have been collected.
"""

from __future__ import annotations

import json
import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous JSON objects containing duplicate member names."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON response key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON numeric constant: {value}")


def _batch_scores(response: httpx.Response, requested_ranks: set[int]) -> dict[int, float]:
    """Validate the entire response shape and exact per-batch rank coverage."""

    try:
        payload = json.loads(
            response.content,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Reranker response is not valid JSON") from error

    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("Reranker response must contain exactly the scores field")
    scores = payload["scores"]
    if not isinstance(scores, list) or len(scores) != len(requested_ranks):
        raise ValueError("Reranker response must score every requested candidate once")

    batch_scores: dict[int, float] = {}
    for item in scores:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("Each score record must contain exactly rank and score")

        rank = item["rank"]
        score = item["score"]
        # bool is a subclass of int, but is neither a valid rank nor a score.
        if type(rank) is not int or rank not in requested_ranks or rank in batch_scores:
            raise ValueError("Reranker returned a duplicate or out-of-batch rank")
        if type(score) not in (int, float):
            raise ValueError("Reranker score must be numeric")
        try:
            score_value = float(score)
        except (ValueError, OverflowError) as error:
            raise ValueError("Reranker score cannot be represented as a finite number") from error
        if not math.isfinite(score_value):
            raise ValueError("Reranker score must be finite")
        batch_scores[rank] = score_value

    if set(batch_scores) != requested_ranks:
        raise ValueError("Reranker response does not cover all requested ranks")
    return batch_scores


class HTTPMemoryReranker:
    """Use a caller-owned async HTTP client to score and globally rerank hits."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.endpoint = endpoint
        self.client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        count = len(candidates)
        if count < 1:
            raise ValueError("rerank requires at least one candidate")
        if type(limit) is not int or not 1 <= limit <= count:
            raise ValueError("rerank limit must be between one and the candidate count")

        # Rank is always the original position in the entire coarse pool.
        # Sending all batches before sorting is essential: a later batch may
        # contain the highest scoring hit.
        scored: dict[int, float] = {}
        requests = 0
        for start in range(0, count, self.batch_size):
            end = min(start + self.batch_size, count)
            request_items = [
                {"rank": index + 1, "text": candidates[index].text}
                for index in range(start, end)
            ]
            requested_ranks = {item["rank"] for item in request_items}
            # Never swallow httpx transport/HTTP errors. The supplied client
            # owns its transport, headers, authentication and lifetime.
            requests += 1
            response = await self.client.post(
                self.endpoint,
                json={"query": query, "candidates": request_items},
            )
            response.raise_for_status()
            scored.update(_batch_scores(response, requested_ranks))

        selected_ranks = tuple(
            sorted(scored, key=lambda rank: (-scored[rank], rank))[:limit]
        )
        return MemoryRerankDecision(
            selected_ranks=selected_ranks,
            usage=InferenceUsage(requests=requests),
        )


__all__ = ["HTTPMemoryReranker"]
