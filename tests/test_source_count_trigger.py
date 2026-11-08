"""Count-trigger configuration and binding regression tests.

Real native Memory/Topic processing and failure recovery are exercised by the
public processing-01-count-trigger acceptance suite.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from powercontext.builtin.runtime.artifact_processing import ArtifactProcessingBinding
from powercontext.builtin.runtime.config import RuntimeConfig


@pytest.mark.parametrize(
    "field",
    ("memory_source_count_threshold", "topic_memory_source_count_threshold"),
)
def test_threshold_defaults_and_strict_positive_values(field: str) -> None:
    assert getattr(RuntimeConfig(), field) is None
    for value in (None, 1, 2, 3, 100):
        assert getattr(RuntimeConfig(**{field: value}), field) == value
    for value in (True, False, 0, -1, 3.0, "3"):
        with pytest.raises(ValidationError):
            RuntimeConfig(**{field: value})


def test_family_thresholds_do_not_change_window_limits() -> None:
    config = RuntimeConfig(
        memory_source_count_threshold=3,
        topic_memory_source_count_threshold=4,
        source_window_limit=1,
        topic_memory_source_window_limit=2,
    )
    assert config.memory_source_count_threshold == 3
    assert config.topic_memory_source_count_threshold == 4
    assert config.source_window_limit == 1
    assert config.topic_memory_source_window_limit == 2


@pytest.mark.parametrize("value", (True, False, 0, -1, 1.0, "3"))
def test_binding_rejects_invalid_thresholds(value: object) -> None:
    with pytest.raises(ValueError):
        ArtifactProcessingBinding(
            binding_name="count-test",
            artifact_family="memory",
            launcher=object(),
            source_count_threshold=value,
        )


def test_binding_retains_its_original_positional_arguments() -> None:
    binding = ArtifactProcessingBinding("count-test", "memory", object())
    assert binding.source_count_threshold is None
    assert ArtifactProcessingBinding(
        "count-test", "memory", object(), source_count_threshold=3
    ).source_count_threshold == 3
