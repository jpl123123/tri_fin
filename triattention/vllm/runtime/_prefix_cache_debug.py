"""Print-based debug breakpoints for TriAttention Prefix-Caching investigation.

This module is a **read-only diagnostic layer**.  It does NOT modify any
existing TriAttention or vLLM behavior.  All breakpoints are gated behind a
single master environment switch so production runs are unaffected when the
switch is off.

Master switch
-------------
    export TRIATTN_DEBUG_PREFIX_CACHE_TRACE=1

When the switch is ``0`` / unset, every helper here is a no-op (it returns
immediately without printing), so the call sites added to the runtime are
effectively zero-cost.

Scope of this debug layer
-------------------------
The investigation follows "Direction 1" from the root-cause report:

    Let the compression-reclaimed blocks keep their prefix-cache hash in the
    BlockPool ``cached_block`` reverse-lookup table, i.e. do NOT call
    ``BlockPool._maybe_evict_cached_block`` inside ``_free_reclaimed_blocks``.

The print breakpoints in this module are designed to *observe* the current
(evict-on-reclaim) behavior end-to-end so we can confirm the root cause and
quantify the risk called out in Direction 1:

    Risk: a reused physical block may be overwritten with new data, leaving a
    stale hash in ``cached_block`` that points at "content-changed" block, so a
    later prefix-cache hit would read wrong KV.

To observe that risk we print, at every relevant transition point:

1. ``trace_evict_reclaimed_block``   - per-block evict inside reclaim
2. ``trace_free_reclaimed_blocks``   - batch free (before/after block_hash)
3. ``trace_reclaim_branch``          - which reclaim branch in
   ``_apply_compression_events`` fired, with freed/kept counts
4. ``trace_allocate_slots_patch``    - whether the patched allocate_slots set
   ``delay_cache_blocks=True`` (skipping hash commit) for a compressed request
5. ``trace_worker_self_trigger``     - worker-side self-triggered compression
   timing (Suspect C: compress on first decode step right after prefill)
6. ``trace_block_reuse_after_free``  - risk probe: when a previously-freed
   block id is re-allocated, print whether its ``block_hash`` was already
   cleared by vLLM (validates the Direction-1 mitigation claim that vLLM
   manages hash lifetime itself on ``allocate_slots``)

All prints are flushed immediately so ordering is preserved across the
multi-process (TP=4) Ascend runtime.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}

_master_env_name = "TRIATTN_DEBUG_PREFIX_CACHE_TRACE"

_cached_master: bool | None = None
_print_lock = threading.Lock()


def _master_enabled() -> bool:
    """Return whether the master debug switch is on.

    Cached after first read so we don't hit ``os.environ`` on every breakpoint.
    """
    global _cached_master
    if _cached_master is None:
        raw = os.environ.get(_master_env_name, "0")
        _cached_master = raw.strip().lower() in _TRUE_VALUES
    return _cached_master


def refresh_master_switch() -> None:
    """Force a re-read of the master switch (useful for tests)."""
    global _cached_master
    _cached_master = None


def _emit(message: str) -> None:
    """Print a debug line with a stable prefix and immediate flush.

    Uses ``print`` (not ``logger``) so the output is unconditional and not
    subject to vLLM logger level / handler config — this is exactly the
    "print 形式断点" the user asked for.
    """
    if not _master_enabled():
        return
    line = f"[TRIATTN-PCTRACE] {message}"
    with _print_lock:
        print(line, file=sys.stderr, flush=True)


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _block_id(block: Any) -> Any:
    return _safe_attr(block, "block_id", None)


def _block_hash_repr(block: Any) -> str:
    bh = _safe_attr(block, "block_hash", None)
    if bh is None:
        return "None"
    try:
        s = str(bh)
    except Exception:
        s = "<unrepr>"
    if len(s) > 32:
        s = s[:32] + "..."
    return s


# ---------------------------------------------------------------------------
# Breakpoint 1: per-block evict inside _evict_reclaimed_block_metadata
# ---------------------------------------------------------------------------

def trace_evict_reclaimed_block(
    *,
    block_pool: Any,
    block: Any,
    stage: str,
) -> None:
    """Print before/after the per-block evict call.

    ``stage`` is ``"enter"`` (before ``_maybe_evict_cached_block``) or
    ``"exit"``  (after).  Observing the before/after ``block_hash`` of the
    reclaimed block lets us confirm that TriAttention is the one stripping the
    hash, and lets us measure how many prompt-mid/tail hashes get evicted per
    compression step.
    """
    if not _master_enabled():
        return
    bid = _block_id(block)
    bh = _block_hash_repr(block)
    pool_id = hex(id(block_pool)) if block_pool is not None else "None"
    has_evict = callable(_safe_attr(block_pool, "_maybe_evict_cached_block", None))
    _emit(
        f"evict_reclaimed_block stage={stage} block_id={bid} "
        f"block_hash={bh} pool_id={pool_id} has_evict_fn={has_evict}"
    )


# ---------------------------------------------------------------------------
# Breakpoint 1b: protected-block reuse-clear inside the patched
# _maybe_evict_cached_block (Path C evict-on-rewrite)
# ---------------------------------------------------------------------------

def trace_protected_block_reuse_clear(
    *,
    block_pool: Any,
    block: Any,
    had_hash: bool,
    cleared: bool,
) -> None:
    """Print when a TriAttention hash-protected block is pulled from the free
    pool for reuse and its stale hash is cleared by Path C's evict-on-rewrite.

    This is the *correctness-critical* Path C transition: it is where the
    Direction-1 stale-hash risk is neutralized.  Observing it lets us confirm
    that (a) protected blocks do get reused in real workloads (120 different
    prompts), (b) their stale hash is non-None at reuse (``had_hash=True``),
    and (c) the clear actually fired (``cleared=True``).  If ``had_hash=True``
    but ``cleared=False`` ever appears, the block would crash upstream
    ``cache_full_blocks`` (``assert blk.block_hash is None``) — that is the
    smoking gun for the pre-fix bug.
    """
    if not _master_enabled():
        return
    bid = _block_id(block)
    pool_id = hex(id(block_pool)) if block_pool is not None else "None"
    _emit(
        f"protected_block_reuse_clear block_id={bid} "
        f"had_hash={had_hash} cleared={cleared} pool_id={pool_id}"
    )


# ---------------------------------------------------------------------------
# Breakpoint 2: batch free inside _free_reclaimed_blocks
# ---------------------------------------------------------------------------

def trace_free_reclaimed_blocks(
    *,
    manager: Any,
    removed_blocks: list[Any],
    stage: str,
) -> None:
    """Print the list of blocks about to be freed / just freed.

    ``stage`` is ``"pre_evict"`` (before the per-block evict loop),
    ``"post_evict_pre_free"`` (after evict loop, before ``free_blocks``), or
    ``"post_free"`` (after ``free_blocks``).

    For each removed block we record its ``block_id`` and whether it still
    carries a ``block_hash``.  At ``post_evict_pre_free`` every block_hash
    should be ``None`` under the current (evict-on-reclaim) behavior — that is
    exactly what Direction 1 proposes to *stop* doing.
    """
    if not _master_enabled():
        return
    n = len(removed_blocks) if removed_blocks is not None else 0
    block_ids: list[str] = []
    block_hashes: list[str] = []
    if removed_blocks:
        for b in removed_blocks:
            block_ids.append(str(_block_id(b)))
            block_hashes.append(_block_hash_repr(b))
    ids_s = ",".join(block_ids) if block_ids else "<empty>"
    hashes_s = ",".join(block_hashes) if block_hashes else "<empty>"
    pool_id = "None"
    if manager is not None:
        bp = _safe_attr(manager, "block_pool", None)
        pool_id = hex(id(bp)) if bp is not None else "None"
    _emit(
        f"free_reclaimed_blocks stage={stage} n={n} pool_id={pool_id} "
        f"block_ids=[{ids_s}] block_hashes=[{hashes_s}]"
    )


# ---------------------------------------------------------------------------
# Breakpoint 3: reclaim branch inside _apply_compression_events
# ---------------------------------------------------------------------------

def trace_reclaim_branch(
    *,
    req_id: Any,
    gid: int,
    branch: str,
    freed_count: int,
    kept_count: int,
    new_count: int = 0,
    required_blocks: Any = None,
    extra: str = "",
) -> None:
    """Print which reclaim branch fired, with freed/kept counts.

    ``branch`` is one of:
      - ``"explicit_groups"``      : block_reclaim.groups payload processed
      - ``"synthesize_no_groups"`` : synthesized because groups was missing
      - ``"synthesize_missing_gids"``: synthesized for gids not in groups
      - ``"skip_prefill_no_groups"``: skipped synthesis during chunked prefill
      - ``"skip_prefill_missing_gids"``: skipped synthesis during chunked prefill

    This lets us confirm the report's claim that, on the first decode step
    right after prefill, a reclaim with ~123 (2k budget) or ~108 (4k budget)
    freed blocks fires.
    """
    if not _master_enabled():
        return
    _emit(
        f"reclaim_branch req_id={req_id} gid={gid} branch={branch} "
        f"freed={freed_count} kept={kept_count} new={new_count} "
        f"required_blocks={required_blocks} {extra}".strip()
    )


# ---------------------------------------------------------------------------
# Breakpoint 4: patched allocate_slots (delay_cache_blocks decision)
# ---------------------------------------------------------------------------

def trace_allocate_slots_patch(
    *,
    request: Any,
    num_new_tokens: Any,
    effective_num_computed: Any,
    logical_num_computed: Any,
    will_delay_cache_blocks: bool,
) -> None:
    """Print the allocate_slots patch decision.

    ``will_delay_cache_blocks=True`` means the patched allocate_slots will set
    ``delay_cache_blocks=True`` and temporarily rewrite
    ``request.num_computed_tokens`` to ``effective_num_computed``.  This is
    Suspect B in the report: once a request has been compressed, it never
    commits new prefix-cache hashes again.
    """
    if not _master_enabled():
        return
    req_id = _safe_attr(request, "request_id", None)
    _emit(
        f"allocate_slots_patch req_id={req_id} num_new_tokens={num_new_tokens} "
        f"effective_num_computed={effective_num_computed} "
        f"logical_num_computed={logical_num_computed} "
        f"will_delay_cache_blocks={will_delay_cache_blocks}"
    )


# ---------------------------------------------------------------------------
# Breakpoint 5: worker-side self-triggered compression (Suspect C)
# ---------------------------------------------------------------------------

def trace_worker_self_trigger(
    *,
    req_id: Any,
    scheduled_tokens: Any,
    existing_estimate: Any,
    prefill_len: Any,
    threshold: Any,
    is_prefill_step_for_threshold: bool,
    defer_chunked_prefill: bool,
    will_compress: bool,
    extra: str = "",
) -> None:
    """Print the worker self-trigger decision.

    The report's Suspect C says that ``DEFER_PREFILL_COMPRESSION_ON_ASCEND=1``
    does NOT prevent compression on the first decode step right after prefill,
    because the guard ``defer_chunked_prefill and is_prefill_step_for_threshold``
    stops being true once prefill is done.  This breakpoint confirms that.
    """
    if not _master_enabled():
        return
    _emit(
        f"worker_self_trigger req_id={req_id} scheduled_tokens={scheduled_tokens} "
        f"existing_estimate={existing_estimate} prefill_len={prefill_len} "
        f"threshold={threshold} is_prefill_step_for_threshold={is_prefill_step_for_threshold} "
        f"defer_chunked_prefill={defer_chunked_prefill} will_compress={will_compress} "
        f"{extra}".strip()
    )


# ---------------------------------------------------------------------------
# Breakpoint 6: risk probe — block reuse after free (Direction-1 risk)
# ---------------------------------------------------------------------------

# Track block_ids we have seen freed by TriAttention reclaim, so that when the
# same block_id shows up allocated for a *different* request later we can print
# whether vLLM has already cleared/refreshed its block_hash.  This directly
# validates the Direction-1 mitigation claim: "vLLM BlockPool itself clears
# stale hash and re-registers on allocate_slots, so as long as TriAttention
# does not actively evict, vLLM manages the hash lifecycle."
_freed_block_ids: set[Any] = set()
_freed_block_ids_lock = threading.Lock()


def record_freed_block_ids(block_ids: list[Any]) -> None:
    """Remember block ids that TriAttention reclaim just freed.

    Called from ``_free_reclaimed_blocks`` after ``block_pool.free_blocks``.
    """
    if not _master_enabled():
        return
    with _freed_block_ids_lock:
        for bid in block_ids:
            if bid is not None:
                _freed_block_ids.add(bid)


def trace_block_reuse_on_allocate(
    *,
    request: Any,
    new_block_ids: list[Any],
    blocks: list[Any],
) -> None:
    """Print when a previously-freed block id is re-allocated.

    For each newly allocated block whose id was previously freed by TriAttention
    reclaim, we print its current ``block_hash``.  Three cases are interesting:

    A. ``block_hash`` is the SAME as the old prompt hash  -> the block has not
       been overwritten yet, so a prefix-cache hit on it is still correct.
    B. ``block_hash`` is None                              -> vLLM already
       cleared the stale hash; a future ``_maybe_cache_full_block`` will
       re-register the new content hash. This is the safe case the
       Direction-1 mitigation relies on.
    C. ``block_hash`` is a NEW non-None hash               -> vLLM already
       re-registered the new content hash. Also safe.

    Observing case (A) persisting into a *different* request's allocation is
    the smoking gun for the Direction-1 risk.
    """
    if not _master_enabled():
        return
    if not new_block_ids:
        return
    with _freed_block_ids_lock:
        reused = [bid for bid in new_block_ids if bid in _freed_block_ids]
    if not reused:
        return
    req_id = _safe_attr(request, "request_id", None)
    block_map: dict[Any, Any] = {}
    for b in blocks:
        bid = _block_id(b)
        if bid in reused:
            block_map[bid] = _block_hash_repr(b)
    for bid in reused:
        bh = block_map.get(bid, "<missing>")
        _emit(
            f"block_reuse_on_allocate req_id={req_id} reused_block_id={bid} "
            f"current_block_hash={bh} "
            f"(A=same-old/stale-risk, B=None/cleared-safe, C=new-hash/safe)"
        )


def trace_block_reuse_summary() -> None:
    """Print a one-shot summary of how many freed block ids were later reused.

    Useful at end-of-step to quantify reuse churn.  Safe to call repeatedly.
    """
    if not _master_enabled():
        return
    with _freed_block_ids_lock:
        total = len(_freed_block_ids)
    _emit(f"block_reuse_summary total_freed_block_ids_tracked={total}")


# ---------------------------------------------------------------------------
# Convenience: announce that the debug layer is active
# ---------------------------------------------------------------------------

def announce_active() -> None:
    """Print a single banner once per process when the switch is on."""
    if not _master_enabled():
        return
    _emit(
        "ACTIVE master_switch=TRIATTN_DEBUG_PREFIX_CACHE_TRACE=1 "
        "direction=1(keep-prefix-cache-hash-on-reclaim) "
        "mode=observe-only(no behavior change)"
    )
