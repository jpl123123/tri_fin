"""V1 ModelRunner input patch helpers.

This module provides the compatibility layer used by legacy/default vLLM V1
GPUModelRunner and vLLM-Ascend NPUModelRunner paths.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import torch

from . import input_patch_state as _patch_state
from .phase_profile import (
    phase_elapsed_ms,
    phase_now,
    phase_profile_enabled,
    record_phase,
)


def _debug_drop_pos_delta() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_POS_DELTA", "0") == "1"


def _debug_drop_seq_base() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_SEQ_BASE", "0") == "1"


def _debug_preserve_rope_positions() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_PRESERVE_ROPE_POSITIONS", "0") == "1"


def _validate_expected_v1_batch_mapping(
    *,
    req_indices: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
    input_batch: Any | None = None,
) -> None:
    expected_rows = _patch_state.ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU
    if expected_rows is None:
        return
    expected_rows_np = expected_rows.detach().cpu().numpy().astype(np.int64, copy=False)
    if expected_rows_np.size == 0:
        return
    expected_rows_np = _remap_expected_rows_by_req_ids(
        expected_rows_np,
        input_batch=input_batch,
    )
    row_mask = (expected_rows_np >= 0) & (expected_rows_np < int(num_reqs))
    if not np.all(row_mask):
        raise RuntimeError(
            "TRIATTN_V1_IDX_MAPPING_MISMATCH:"
            f"num_reqs={num_reqs}:expected={expected_rows_np.tolist()}"
        )

    expected_q_lens = _patch_state.ACTIVE_EXPECTED_QUERY_LENS_CPU
    if expected_q_lens is not None:
        expected_q_lens_np = (
            expected_q_lens.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        if expected_q_lens_np.shape != expected_rows_np.shape:
            raise RuntimeError(
                "TRIATTN_V1_QUERY_LENS_COUNT_MISMATCH:"
                f"rows={expected_rows_np.tolist()}:qlens={expected_q_lens_np.tolist()}"
            )
        actual_q_lens = np.asarray(
            num_scheduled_tokens[expected_rows_np],
            dtype=np.int64,
        )
        if not np.array_equal(actual_q_lens, expected_q_lens_np):
            raise RuntimeError(
                "TRIATTN_V1_QUERY_LENS_MISMATCH:"
                f"actual={actual_q_lens.tolist()}:expected={expected_q_lens_np.tolist()}"
            )

    req_indices_i64 = req_indices.astype(np.int64, copy=False)
    present_rows = set(int(v) for v in np.unique(req_indices_i64).tolist())
    missing_rows = [int(row) for row in expected_rows_np.tolist() if int(row) not in present_rows]
    if missing_rows:
        raise RuntimeError(
            "TRIATTN_V1_TOKEN_ROW_MAPPING_MISMATCH:"
            f"missing={missing_rows}:actual={req_indices_i64.tolist()}"
        )


def _current_v1_req_id_to_index(input_batch: Any) -> dict[Any, int] | None:
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if isinstance(req_id_to_index, dict) and req_id_to_index:
        return req_id_to_index
    req_ids_attr = getattr(input_batch, "req_ids", None)
    try:
        req_ids = req_ids_attr() if callable(req_ids_attr) else req_ids_attr
    except Exception:
        req_ids = None
    if not isinstance(req_ids, list):
        req_ids = getattr(input_batch, "_req_ids", None)
    if not isinstance(req_ids, list):
        return None
    out = {
        req_id: idx
        for idx, req_id in enumerate(req_ids)
        if req_id is not None
    }
    return out or None


def _remap_by_expected_req_ids(
    values_by_expected_row: dict[int, int] | None,
    *,
    input_batch: Any,
) -> dict[int, int] | None:
    if not values_by_expected_row:
        return values_by_expected_row
    expected_req_ids = _patch_state.ACTIVE_EXPECTED_REQ_IDS
    expected_rows = _patch_state.ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU
    if expected_req_ids is None or expected_rows is None:
        return values_by_expected_row
    try:
        expected_rows_np = expected_rows.detach().cpu().numpy().astype(np.int64, copy=False)
    except Exception:
        return values_by_expected_row
    if len(expected_req_ids) != int(expected_rows_np.size):
        return values_by_expected_row
    current_map = _current_v1_req_id_to_index(input_batch)
    if not isinstance(current_map, dict):
        return values_by_expected_row

    remapped: dict[int, int] = {}
    changed = False
    for expected_req_id, expected_row in zip(expected_req_ids, expected_rows_np.tolist()):
        expected_row_i = int(expected_row)
        if expected_row_i not in values_by_expected_row:
            continue
        current_row = current_map.get(expected_req_id)
        if not isinstance(current_row, int):
            continue
        remapped[int(current_row)] = int(values_by_expected_row[expected_row_i])
        changed = changed or int(current_row) != expected_row_i
    if not remapped:
        return values_by_expected_row
    return remapped if changed else values_by_expected_row


def _remap_expected_rows_by_req_ids(
    expected_rows_np: np.ndarray,
    *,
    input_batch: Any | None,
) -> np.ndarray:
    expected_req_ids = _patch_state.ACTIVE_EXPECTED_REQ_IDS
    if expected_req_ids is None or input_batch is None:
        return expected_rows_np
    if len(expected_req_ids) != int(expected_rows_np.size):
        return expected_rows_np
    current_map = _current_v1_req_id_to_index(input_batch)
    if not isinstance(current_map, dict):
        return expected_rows_np
    out = expected_rows_np.copy()
    changed = False
    for i, expected_req_id in enumerate(expected_req_ids):
        current_row = current_map.get(expected_req_id)
        if not isinstance(current_row, int):
            continue
        out[i] = int(current_row)
        changed = True
    return out if changed else expected_rows_np


def _block_table_inner_tables(block_table_obj: Any) -> list[Any]:
    inner_tables = getattr(block_table_obj, "block_tables", None)
    if isinstance(inner_tables, (list, tuple)) and inner_tables:
        return list(inner_tables)
    return [block_table_obj]


def _table_block_size(table: Any) -> int | None:
    for attr_name in ("block_size", "logical_block_size", "physical_block_size"):
        try:
            value = int(getattr(table, attr_name))
        except Exception:
            continue
        if value > 0:
            return value
    return None


def _table_row_block_count(table: Any, row_idx: int) -> int | None:
    num_blocks_per_row = getattr(table, "num_blocks_per_row", None)
    if num_blocks_per_row is None:
        return None
    try:
        return int(num_blocks_per_row[row_idx])
    except Exception:
        return None


def _active_override_rows(
    *,
    req_indices: np.ndarray,
    num_reqs: int,
    input_batch: Any | None = None,
) -> list[int]:
    rows: set[int] = set()
    expected_rows = _patch_state.ACTIVE_EXPECTED_REQ_ROW_INDICES_CPU
    if expected_rows is not None:
        try:
            expected_np = expected_rows.detach().cpu().numpy().astype(np.int64, copy=False)
            expected_np = _remap_expected_rows_by_req_ids(
                expected_np,
                input_batch=input_batch,
            )
            rows.update(
                int(row)
                for row in expected_np.tolist()
                if 0 <= int(row) < int(num_reqs)
            )
        except Exception:
            pass

    if _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None and num_reqs == 1:
        rows.add(0)
    if _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0 and num_reqs == 1:
        rows.add(0)

    for mapping in (
        _remap_by_expected_req_ids(
            _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX,
            input_batch=input_batch,
        ),
        _remap_by_expected_req_ids(
            _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX,
            input_batch=input_batch,
        ),
    ):
        if not mapping:
            continue
        for row in mapping.keys():
            try:
                row_i = int(row)
            except Exception:
                continue
            if 0 <= row_i < int(num_reqs):
                rows.add(row_i)

    if not rows and req_indices.size:
        rows.update(
            int(row)
            for row in np.unique(req_indices.astype(np.int64, copy=False)).tolist()
            if 0 <= int(row) < int(num_reqs)
        )
    return sorted(rows)


def _validate_v1_block_table_bounds(
    *,
    block_table: Any,
    seq_lens_np: np.ndarray,
    req_indices: np.ndarray,
    slot_positions_np: np.ndarray | None,
    num_reqs: int,
    validate_seq_lens: bool,
    input_batch: Any | None = None,
) -> None:
    if block_table is None or num_reqs <= 0:
        return
    tables = _block_table_inner_tables(block_table)
    rows = _active_override_rows(
        req_indices=req_indices,
        num_reqs=num_reqs,
        input_batch=input_batch,
    )
    if not tables or not rows:
        return

    if slot_positions_np is not None and np.any(slot_positions_np < 0):
        negatives = slot_positions_np[slot_positions_np < 0]
        sample = negatives[:16].astype(np.int64, copy=False).tolist()
        raise RuntimeError(
            "TRIATTN_ASCEND_V1_SLOT_POSITION_NEGATIVE:"
            f"min={int(negatives.min(initial=0))}:sample={sample}:"
            f"negative_count={int(negatives.size)}:"
            f"total={int(slot_positions_np.size)}"
        )

    req_indices_i64 = req_indices.astype(np.int64, copy=False)
    slot_positions_i64 = (
        slot_positions_np.astype(np.int64, copy=False)
        if slot_positions_np is not None
        else None
    )
    for row in rows:
        row_mask = req_indices_i64 == int(row)
        max_slot_position: int | None = None
        if slot_positions_i64 is not None and bool(row_mask.any()):
            row_positions = slot_positions_i64[row_mask]
            max_slot_position = int(row_positions.max(initial=-1))
        seq_len = int(seq_lens_np[row]) if validate_seq_lens else None

        for gid, table in enumerate(tables):
            block_size = _table_block_size(table)
            current_blocks = _table_row_block_count(table, row)
            if block_size is None or current_blocks is None:
                continue
            capacity = max(0, int(block_size) * int(current_blocks))
            if validate_seq_lens and seq_len is not None and seq_len > capacity:
                raise RuntimeError(
                    "TRIATTN_ASCEND_V1_BLOCK_TABLE_CAPACITY_MISMATCH:"
                    f"row={row}:gid={gid}:seq_len={seq_len}:"
                    f"blocks={current_blocks}:block_size={block_size}:"
                    f"capacity={capacity}"
                )
            if (
                validate_seq_lens
                and seq_len is not None
                and max_slot_position is not None
                and max_slot_position >= seq_len
            ):
                raise RuntimeError(
                    "TRIATTN_ASCEND_V1_SLOT_POSITION_SEQ_LEN_MISMATCH:"
                    f"row={row}:gid={gid}:max_slot_position={max_slot_position}:"
                    f"seq_len={seq_len}:blocks={current_blocks}:"
                    f"block_size={block_size}:capacity={capacity}"
                )
            if max_slot_position is not None and max_slot_position >= capacity:
                raise RuntimeError(
                    "TRIATTN_ASCEND_V1_SLOT_POSITION_OOB:"
                    f"row={row}:gid={gid}:max_slot_position={max_slot_position}:"
                    f"blocks={current_blocks}:block_size={block_size}:"
                    f"capacity={capacity}"
                )


def _effective_block_table_capacity(input_batch: Any | None, row: int) -> int | None:
    """Return current worker block-table capacity in tokens for ``row``."""
    if input_batch is None:
        return None
    block_table_obj = getattr(input_batch, "block_table", None)
    if block_table_obj is None:
        return None
    capacities: list[int] = []
    for table in _block_table_inner_tables(block_table_obj):
        block_size = _table_block_size(table)
        if block_size is None:
            cache_config = getattr(input_batch, "cache_config", None)
            try:
                block_size = int(getattr(cache_config, "block_size"))
            except Exception:
                block_size = None
        row_blocks = _table_row_block_count(table, int(row))
        if block_size is None or row_blocks is None:
            continue
        capacities.append(max(0, int(block_size) * int(row_blocks)))
    if capacities:
        return min(capacities)
    return None


def _clamp_effective_base_to_capacity(
    *,
    input_batch: Any | None,
    row: int,
    effective_base: int,
    num_scheduled: int,
) -> int:
    """Clamp effective base so rebuilt slot positions fit local capacity."""
    if num_scheduled <= 0:
        return int(effective_base)
    capacity = _effective_block_table_capacity(input_batch, int(row))
    if capacity is None or capacity <= 0:
        return int(effective_base)
    max_effective_base = max(0, int(capacity) - int(num_scheduled))
    return min(int(effective_base), max_effective_base)


def _build_effective_slot_positions(
    *,
    positions_np: np.ndarray,
    req_indices: np.ndarray,
    input_batch: Any | None = None,
) -> np.ndarray | None:
    if _debug_drop_pos_delta():
        return None
    if (
        int(req_indices.size) == 0
        or int(positions_np.size) == 0
    ):
        return None

    max_row = int(req_indices.max(initial=-1))
    num_rows = max_row + 1

    if num_rows == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
        base = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE)
        if base >= int(positions_np[0]):
            return None
        base = _clamp_effective_base_to_capacity(
            input_batch=input_batch,
            row=0,
            effective_base=base,
            num_scheduled=int(positions_np.size),
        )
        return base + np.arange(int(positions_np.size), dtype=positions_np.dtype)

    sparse_bases = _remap_by_expected_req_ids(
        _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX,
        input_batch=input_batch,
    )
    if sparse_bases:
        out = positions_np.copy()
        changed = False
        for req_idx, effective_base in sparse_bases.items():
            row = int(req_idx)
            if row < 0 or row >= num_rows:
                continue
            token_indices = np.nonzero(req_indices == row)[0]
            if token_indices.size == 0:
                continue
            if int(effective_base) >= int(positions_np[token_indices[0]]):
                continue
            clamped_base = _clamp_effective_base_to_capacity(
                input_batch=input_batch,
                row=row,
                effective_base=int(effective_base),
                num_scheduled=int(token_indices.size),
            )
            out[token_indices] = clamped_base + np.arange(
                int(token_indices.size),
                dtype=positions_np.dtype,
            )
            changed = True
        if changed:
            return out

    # Fallback for legacy pos-delta-only tests or patched environments where
    # an effective base is unavailable.
    out = positions_np.copy()

    if num_rows == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0:
        delta = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA)
        if delta >= 0:
            return None
        shifted = out + delta
        np.copyto(out, shifted, where=shifted >= 0)
        return out

    sparse_pos_deltas = _remap_by_expected_req_ids(
        _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX,
        input_batch=input_batch,
    )
    if not sparse_pos_deltas:
        return None

    row_deltas = np.zeros(int(req_indices.max()) + 1, dtype=positions_np.dtype)
    for req_idx, delta in sparse_pos_deltas.items():
        delta_i = int(delta)
        if delta_i >= 0:
            continue
        if 0 <= int(req_idx) < row_deltas.shape[0]:
            row_deltas[int(req_idx)] = delta_i
    if not bool(np.any(row_deltas)):
        return None
    deltas = row_deltas[req_indices]
    shifted = out + deltas
    np.copyto(out, shifted, where=shifted >= 0)
    return out


def _apply_sparse_seq_len_overrides_in_place(
    *,
    seq_lens_np: np.ndarray,
    num_computed_tokens_cpu: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
    input_batch: Any | None = None,
) -> bool:
    if _debug_drop_seq_base():
        return False
    if num_reqs <= 0:
        return False

    applied = False
    if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
        base = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE)
        if base >= int(num_computed_tokens_cpu[0]):
            return False
        new_seq_len = base + int(num_scheduled_tokens[0])
        capacity = _effective_block_table_capacity(input_batch, 0)
        if capacity is not None and capacity > 0 and new_seq_len > capacity:
            new_seq_len = int(capacity)
        seq_lens_np[0] = new_seq_len
        return True

    sparse_bases = _remap_by_expected_req_ids(
        _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX,
        input_batch=input_batch,
    )
    if not sparse_bases:
        return False

    seq_lens_np[:num_reqs] = num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens[:num_reqs]
    for req_idx, effective_base in sparse_bases.items():
        idx = int(req_idx)
        if 0 <= idx < num_reqs and int(effective_base) < int(num_computed_tokens_cpu[idx]):
            new_seq_len = int(effective_base) + int(num_scheduled_tokens[idx])
            capacity = _effective_block_table_capacity(input_batch, idx)
            if capacity is not None and capacity > 0 and new_seq_len > capacity:
                new_seq_len = int(capacity)
            seq_lens_np[idx] = new_seq_len
            applied = True
    return applied


def _sync_v1_seq_lens_to_runner_buffers(
    *,
    runner: Any,
    seq_lens_np: np.ndarray,
    num_reqs: int,
) -> bool:
    """Mirror effective V1 seq_lens into buffers consumed by Ascend metadata."""
    if num_reqs <= 0:
        return False

    synced = False
    seq_lens = getattr(runner, "seq_lens", None)
    copy_to_gpu = getattr(seq_lens, "copy_to_gpu", None)
    if callable(copy_to_gpu):
        copy_to_gpu()
        synced = True
    elif isinstance(seq_lens, torch.Tensor):
        values = torch.as_tensor(
            seq_lens_np[:num_reqs],
            device=seq_lens.device,
            dtype=seq_lens.dtype,
        )
        seq_lens[:num_reqs].copy_(values)
        if int(seq_lens.numel()) > num_reqs:
            seq_lens[num_reqs:].fill_(0)
        synced = True

    optimistic = getattr(runner, "optimistic_seq_lens_cpu", None)
    if isinstance(optimistic, torch.Tensor):
        values = torch.as_tensor(
            seq_lens_np[:num_reqs],
            device=optimistic.device,
            dtype=optimistic.dtype,
        )
        optimistic[:num_reqs].copy_(values)
        if int(optimistic.numel()) > num_reqs:
            optimistic[num_reqs:].fill_(0)
        synced = True
    elif isinstance(optimistic, np.ndarray):
        optimistic[:num_reqs] = seq_lens_np[:num_reqs]
        if optimistic.size > num_reqs:
            optimistic[num_reqs:].fill(0)
        synced = True
    return synced


def _record_active_effective_max_seq_len(
    *,
    seq_lens_np: np.ndarray,
    num_reqs: int,
) -> int | None:
    if num_reqs <= 0:
        _patch_state.set_active_effective_max_seq_len(None)
        return None
    try:
        active = seq_lens_np[:num_reqs]
        max_seq_len = int(active.max(initial=0))
    except TypeError:
        try:
            max_seq_len = int(seq_lens_np[:num_reqs].max())
        except Exception:
            _patch_state.set_active_effective_max_seq_len(None)
            return None
    except Exception:
        _patch_state.set_active_effective_max_seq_len(None)
        return None
    _patch_state.set_active_effective_max_seq_len(max_seq_len)
    return max_seq_len


def make_patched_v1_prepare_inputs(
    original_prepare_inputs: Callable[..., Any],
) -> Callable[..., Any]:
    def _patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        profile_enabled = phase_profile_enabled()
        t0 = phase_now() if profile_enabled else 0.0
        overrides_enabled = bool(_patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED)
        seq_applied = False
        slot_applied = False
        effective_max_seq_len = None
        total_num_scheduled_tokens = 0
        num_reqs = 0
        slot_positions_np: np.ndarray | None = None
        try:
            out = original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

            if not overrides_enabled:
                return out

            _patch_state.mark_active_effective_overrides_consumed()

            total_num_scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
            num_reqs = int(getattr(self.input_batch, "num_reqs", 0))
            if total_num_scheduled_tokens <= 0 or num_reqs <= 0:
                return out

            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            _validate_expected_v1_batch_mapping(
                req_indices=req_indices,
                num_scheduled_tokens=num_scheduled_tokens,
                num_reqs=num_reqs,
                input_batch=self.input_batch,
            )

            slot_positions_np = _build_effective_slot_positions(
                positions_np=positions_np,
                req_indices=req_indices,
                input_batch=self.input_batch,
            )
            if slot_positions_np is not None:
                _validate_v1_block_table_bounds(
                    block_table=self.input_batch.block_table,
                    seq_lens_np=self.seq_lens.np,
                    req_indices=req_indices,
                    slot_positions_np=slot_positions_np,
                    num_reqs=num_reqs,
                    validate_seq_lens=False,
                    input_batch=self.input_batch,
                )
                self.input_batch.block_table.compute_slot_mapping(req_indices, slot_positions_np)
                self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
                slot_applied = True
            seq_applied = _apply_sparse_seq_len_overrides_in_place(
                seq_lens_np=self.seq_lens.np,
                num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
                num_scheduled_tokens=num_scheduled_tokens,
                num_reqs=num_reqs,
                input_batch=self.input_batch,
            )
            if seq_applied:
                self.seq_lens.np[num_reqs:].fill(0)
                _sync_v1_seq_lens_to_runner_buffers(
                    runner=self,
                    seq_lens_np=self.seq_lens.np,
                    num_reqs=num_reqs,
                )
                effective_max_seq_len = _record_active_effective_max_seq_len(
                    seq_lens_np=self.seq_lens.np,
                    num_reqs=num_reqs,
                )
            else:
                _patch_state.set_active_effective_max_seq_len(None)

            if seq_applied:
                _validate_v1_block_table_bounds(
                    block_table=self.input_batch.block_table,
                    seq_lens_np=self.seq_lens.np,
                    req_indices=req_indices,
                    slot_positions_np=None,
                    num_reqs=num_reqs,
                    validate_seq_lens=seq_applied,
                    input_batch=self.input_batch,
                )

            return out
        finally:
            if profile_enabled:
                try:
                    max_sched = int(num_scheduled_tokens.max(initial=0))
                except TypeError:
                    max_sched = int(num_scheduled_tokens.max())
                except Exception:
                    max_sched = None
                record_phase(
                    "ascend_v1_prepare_inputs",
                    phase_elapsed_ms(t0),
                    {
                        "num_reqs": num_reqs,
                        "total_tokens": total_num_scheduled_tokens,
                        "max_scheduled": max_sched,
                        "overrides": int(overrides_enabled),
                        "seq_override": int(seq_applied),
                        "slot_override": int(slot_applied),
                        "effective_max_seq_len": effective_max_seq_len,
                    },
                )

    return _patched_prepare_inputs
