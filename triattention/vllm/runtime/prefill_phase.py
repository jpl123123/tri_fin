"""Helpers for identifying scheduler prefill phase."""

from __future__ import annotations

from typing import Any


def is_request_scheduled_as_prefill(scheduler_output: Any, req_id: str) -> bool:
    """Return whether the scheduler reports this request as a new/prefill item.

    vLLM's chunked-prefill path can keep a request in ``scheduled_new_reqs``
    across multiple chunks.  That is the most reliable runtime signal that the
    request is still in prompt processing; compressed effective KV length is not,
    because it intentionally stays below the full prompt length after compaction.
    """
    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if not isinstance(scheduled_new_reqs, (list, tuple)):
        return False
    for new_req in scheduled_new_reqs:
        candidate = getattr(new_req, "req_id", None)
        if candidate is None:
            candidate = getattr(new_req, "request_id", None)
        if candidate == req_id:
            return True
    return False


def is_prefill_phase_for_limit(
    *,
    scheduler_output: Any,
    req_id: str,
    scheduled_tokens: int,
    prefill_len: int,
    num_computed_tokens: int | None,
) -> bool:
    """Classify prefill for prefill-only policy gates.

    This intentionally uses logical scheduler/request progress rather than
    compressed effective KV length.  Effective length remains small by design
    after TriAttention compaction, so using it here would keep decode steps stuck
    behind prefill-only limits.
    """
    if is_request_scheduled_as_prefill(scheduler_output, req_id):
        return True
    if int(scheduled_tokens) > 1:
        return True
    if prefill_len <= 0 or num_computed_tokens is None:
        return False
    return int(num_computed_tokens) < int(prefill_len)
