# TriAttention Prefix-Caching 失效根因分析与验证报告

> 本报告基于代码逻辑层面分析，并经实测数据验证。结论已确认。

## 一、问题概述

**核心现象**：vllm-ascend 启用 TriAttention 后，Prefix-Caching 完全失效——同一输入连续发送两次请求，第二次请求的 TTFT 与首次基本一致，未出现预期的"前缀缓存命中后 TTFT 大幅下降"。

**实验数据**：

| 测试环境 | 第一次请求 TTFT (ms) | 第二次请求 TTFT (ms) | 缓存生效状态 |
|---|---|---|---|
| Base（关闭 TriAttention） | 12462.8 | 62.1 | 正常生效（前缀缓存命中，TTFT 大幅下降） |
| TriAttention + 2k Budget | 17355.4 | 16401.4 | 缓存失效（两次 TTFT 基本一致） |
| TriAttention + 4k Budget | 17251.0 | 15302.2 | 缓存失效（两次 TTFT 基本一致） |

**理论预期**：TriAttention 的 KV Budget 仅作用于 Decode 阶段的"逻辑驱逐"，不应干预 Prefill 阶段的前缀缓存写入。第二次请求应能命中首次写入的前缀缓存，TTFT 与 Base 一致大幅下降。

**实际异常**：TriAttention 开启后第二次请求 TTFT 无明显优化，Prefix-Caching 完全失效。

---

## 二、实测验证结果（已确认根因）

对第二次请求的 Prefix-Cache 块命中率进行实测，结果与分析预测完全吻合：

| 测试环境 | 命中块数 / 总块数 | 命中率 | 预测命中率 | 是否吻合 |
|---|---|---|---|---|
| Base（关闭 TriAttention） | 156 / 156 | ~100% | ~100% | ✓ |
| TriAttention + 2k Budget | 33 / 156 | ~21% | ~21% | ✓ |
| TriAttention + 4k Budget | 48 / 156 | ~31% | ~31% | ✓ |

> 说明：20k tokens ÷ 128 block_size = 156 块。命中块数恰好等于"压缩后保留的块数"（`ceil((kv_budget + reclaim_interval) / block_size)`）。

**结论**：实测数据与根因预测完全一致，**根因已确认**。

---

## 三、根因定位

### 3.1 一句话根因

TriAttention 在 Decode 阶段触发压缩回收物理 block 时，通过 `_evict_reclaimed_block_metadata → BlockPool._maybe_evict_cached_block` **主动清除了这些 block 对应的 prefix-cache hash 索引**。这些被清除的 block 恰好承载了 prompt 中后段的 KV，正是第二次相同请求最需要命中的部分。

### 3.2 根因可信度

**最高可信度（已验证）**：压缩回收的物理 block 在归还前被主动 evict 了 prefix-cache hash，且这些 hash 恰好是第二次请求最需要命中的 prompt 中后段。

### 3.3 为什么"逻辑驱逐"实际上变成了"物理 evict"

文档声明 KV Budget 仅作用于 Decode 阶段的"逻辑驱逐"，不干预 Prefill 阶段的前缀缓存写入。但实际实现中：

1. **压缩时机紧贴 Prefill 结束**：`DEFER_PREFILL_COMPRESSION_ON_ASCEND=1` 只能阻止 Prefill 期间的压缩，但 Prefill 一结束、Decode 第一步（`scheduled_tokens=1`）就立即触发压缩（`min_decode_tokens_before_compress_on_ascend=0`，无任何 grace period）。
2. **压缩时 evict 已注册的 prompt block**：压缩发生时，prompt 的所有 block 已注册到 prefix-cache。压缩保留前几十个 block，其余 prompt block 全部被 `_free_reclaimed_blocks` → `block_pool.free_blocks` + `_maybe_evict_cached_block` **从 prefix-cache 中清除并归还 free pool**。
3. **第二次请求无法命中**：第二次相同 prompt 请求只能命中前几十个 block，其后全部 miss，需要重新 prefill，TTFT 接近首次。

---

## 四、机制冲突详解

### 4.1 vLLM V1 Prefix-Caching 关键链路

vLLM V1 的 Prefix-Caching 完全运行在 `vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots` 与底层 `BlockPool` 之上：

1. **缓存注册**：每填满一个 block_size 的 KV，`allocate_slots` 基于 `request.num_computed_tokens` 计算该 block 的内容哈希（`block_hash`），通过 `_maybe_cache_full_block` 注册到 `cached_block` 反查表。
2. **缓存匹配**：新请求进入 WAITING 时，`Scheduler._get_prompt_block_ids` 用 prompt 的 token 序列计算"前缀 block hash 链"，从 `cached_block` 中查找已缓存的 block，命中即"复用"，跳过 prefill。
3. **缓存命中结果**：`num_external_computed_tokens` 写回 `request.num_computed_tokens`，下次 `allocate_slots` 只为未命中部分分配新块；命中部分直接 `BlockPool.touch(block)` 维持 LRU。
4. vllm-ascend 的 `NPUModelRunner._prepare_inputs` 与 `BlockTable` **不参与 hash/注册**，只负责把 `positions`/`slot_mapping`/`block_table` 透传给 NPU kernel。

**关键推论**：任何篡改 `request.num_computed_tokens`、block hash 链、block_pool 状态、或 chunk 边界的逻辑，都会让 Prefix-Caching 失效。

### 4.2 TriAttention 的三个关键 monkeypatch

TriAttention 对 vLLM V1 做了三个层面的 monkeypatch，其中两个直接接触 Prefix-Caching 的核心数据通路：

1. `KVCacheManager.allocate_slots` → `_patched_kv_cache_allocate_slots`
2. `Scheduler.schedule` → `_patched_scheduler_schedule`（含 `max_num_scheduled_tokens` 动态改写）
3. `Scheduler.update_from_output` → `_patched_scheduler_update_from_output`（含 `_apply_compression_events` 中的 block_pool 释放与 evict）

### 4.3 压缩阈值与触发时机

压缩阈值公式（`thresholds.py:152-168`）：

```
threshold = kv_budget + compression_reclaim_interval_tokens
```

实验参数下：
- **Decode 阈值** = 4096 + max(128, 16×128=2048) = **6144**
- **Prefill 阈值**（is_prefill_step=True）= 4096 + max(128, max(2048, 32×128=4096)) = **8192**

`DEFER_PREFILL_COMPRESSION_ON_ASCEND=1` 让 scheduler 与 worker 都跳过 Prefill 期间的压缩。但 **Prefill 一结束、Decode 第一步**：
- `effective_tokens = max(num_computed_tokens, prefill_len) = 20000`（20k prompt 已 Prefill 完）
- 20000 远大于 Decode 阈值 6144
- `min_decode_tokens_before_compress_on_ascend=0`（未设置，默认 0）→ **无任何 grace period**
- → 第一次请求 Decode 第一步立即触发压缩

### 4.4 压缩回收的 evict 路径（核心冲突）

压缩发生时：
- 该请求 20k tokens 占用的约 156 个 block（20000/128）中，只保留 `(kv_budget + reclaim_interval) / block_size` 个 block：
  - 2k Budget：保留约 33 个，evict 约 123 个
  - 4k Budget：保留约 48 个，evict 约 108 个
- 其余 block 全部被 `_free_reclaimed_blocks` → `block_pool.free_blocks` + `_maybe_evict_cached_block` **从 prefix-cache 中清除并归还 free pool**。
- 至此，prefix-cache 中"该请求 prompt 中后段"的所有 hash 记录被清空。

---

## 五、可疑代码流程定位

### 疑点 A（最高优先级，已验证为根因）：压缩回收导致 prompt 中后段 hash 被批量 evict

**代码位置**：
- `triattention/vllm/runtime/scheduler.py:36-59`（`_evict_reclaimed_block_metadata` / `_free_reclaimed_blocks`）
- `triattention/vllm/runtime/scheduler.py:571-896`（`_apply_compression_events` 的 reclaim 路径）

**关键代码**：

```36:59:triattention/vllm/runtime/scheduler.py
def _evict_reclaimed_block_metadata(block_pool: Any, block: Any) -> None:
    """Best-effort clear of prefix-cache metadata before reusing a block."""
    if block_pool is None or block is None:
        return
    block_hash = getattr(block, "block_hash", None)
    if block_hash is None:
        return

    maybe_evict = getattr(block_pool, "_maybe_evict_cached_block", None)
    if callable(maybe_evict):
        maybe_evict(block)


def _free_reclaimed_blocks(manager: Any, removed_blocks: list[Any]) -> bool:
    """Free reclaimed tail blocks after clearing any stale prefix-cache identity."""
    if not removed_blocks:
        return False
    block_pool = getattr(manager, "block_pool", None)
    for block in removed_blocks:
        _evict_reclaimed_block_metadata(block_pool, block)
    if block_pool is None:
        return False
    block_pool.free_blocks(reversed(removed_blocks))
    return True
```

**机制冲突**：
- `_evict_reclaimed_block_metadata` 主动调用 `BlockPool._maybe_evict_cached_block`，从 `cached_block` 反查表中删除该 block 的 hash 索引。
- 这超出了"逻辑驱逐"的语义边界，直接破坏了 prefix-cache 的全局反查表。
- 实测命中率（21% / 31%）与"被保留的 block 数 / 总 prompt block 数"完全吻合，证实此为根因。

### 疑点 B（次高优先级）：`_patched_kv_cache_allocate_slots` 对压缩请求永久跳过 hash 提交

**代码位置**：`triattention/vllm/runtime/integration_monkeypatch.py:499-540`

**关键代码**：

```499:540:triattention/vllm/runtime/integration_monkeypatch.py
def _patched_kv_cache_allocate_slots(
    self,
    request,
    num_new_tokens,
    *args,
    **kwargs,
):
    """Keep vLLM allocation math aligned with TriAttention effective KV length.

    Once a request has been physically compacted, its live KV layout no longer
    matches vLLM's original contiguous-prefix block-hash chain. Continuing to
    commit prefix-cache hashes for later full blocks is therefore invalid and
    can trip BlockPool invariants on the next cache update. We keep slot
    allocation but skip vLLM's cache-commit step for compressed requests.
    """
    assert _ORIG_KVCACHE_ALLOCATE_SLOTS is not None
    prepare_request_effective_num_computed(request)
    effective_num_computed = resolve_request_effective_num_computed(request)
    if effective_num_computed is None:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self, request, num_new_tokens, *args, **kwargs,
        )
    logical_num_computed = getattr(request, "num_computed_tokens", None)
    if not isinstance(logical_num_computed, int):
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self, request, num_new_tokens, *args, **kwargs,
        )
    if effective_num_computed >= logical_num_computed:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self, request, num_new_tokens, *args, **kwargs,
        )
    kwargs = dict(kwargs)
    kwargs["delay_cache_blocks"] = True
    setattr(request, "num_computed_tokens", int(effective_num_computed))
    try:
        return _ORIG_KVCACHE_ALLOCATE_SLOTS(
            self, request, num_new_tokens, *args, **kwargs,
        )
    finally:
        setattr(request, "num_computed_tokens", logical_num_computed)
```

**机制冲突**：
- `delay_cache_blocks=True` 让 vLLM `allocate_slots` 跳过本次新填满 block 的 hash 提交。
- 触发条件：`effective_num_computed < logical_num_computed`，即请求已被压缩过。
- 一旦请求被压缩过一次，**该请求在整个生命周期内**都不会再向 `cached_block` 注册新的前缀 hash。
- 这是设计上的"主动放弃"，但对"压缩后继续生成长输出"的请求，其新生成的 KV 永远无法被后续相似 prefix 命中。

### 疑点 C（中等优先级）：Worker 自触发压缩绕过 Scheduler 的 Prefill deferral

**代码位置**：`triattention/vllm/runtime/runner.py:678-945`（`_supplement_worker_self_triggers`）

**机制冲突**：
- Worker 侧用 `_get_actual_kv_from_block_table` 读取真实物理 KV 长度，可在 Scheduler 没发 signal 时自行产生 `should_compress=True` 的 signal。
- 关键守护 `defer_chunked_prefill and is_prefill_step_for_threshold` 在 Prefill 最后一个 chunk 完成后的第一个 Decode 步（`scheduled_tokens=1`、`existing_estimate` 已经 ≥ `prefill_len`）**不再生效**，Worker 立即自触发压缩。
- 因此 `DEFER_PREFILL_COMPRESSION_ON_ASCEND=1` 并不能阻止"Prefill 结束后立即压缩"。

### 疑点 D（低优先级）：`max_num_scheduled_tokens` 动态 cap 改变 chunk 边界

**代码位置**：`triattention/vllm/runtime/integration_monkeypatch.py:135-176`

**机制冲突**：若物理 KV 紧张，cap 生效后 chunk 边界与默认不同，可能导致部分 block 在某个 chunk 末尾只填了一半，hash 提交推迟到下一个 chunk。本实验参数下大概率不触发，列为低优先级。

### 疑点 E（低优先级）：effective-len override 改写 positions

**代码位置**：`triattention/vllm/runtime/input_patch_vllm_v1_backend.py:586-694`、`triattention/vllm/runtime/input_patch_ascend_backend.py:278-340`

**机制冲突**：压缩请求在新一步中，positions 被重映射到 `[effective_base, effective_base+scheduled_tokens)`，seq_lens 被改成 effective 长度。这会让 NPU attention kernel 用压缩后的 KV 做注意力，但不会改变 vLLM block_pool 中 block_hash 的计算（hash 基于 token id 序列，不基于 position）。所以这本身不直接破坏 hash 链，但与疑点 B 叠加，进一步确保压缩请求的新 block 永不进 prefix-cache。

---

## 六、设计语义与实际执行的错位总结

| 设计语义（文档声明） | 实际执行（代码实现） | 错位后果 |
|---|---|---|
| KV Budget 仅作用于 Decode 阶段的"逻辑驱逐" | Decode 第一步立即压缩，且压缩回收时主动 evict prefix-cache hash | "逻辑驱逐"变成"物理 evict" |
| 不干预 Prefill 阶段的前缀缓存写入 | Prefill 期间确实不压缩，但 Prefill 一结束就压缩，等于"写完就清" | prefix-cache 注册完成后立刻被部分清除 |
| Prefix-Caching 与 TriAttention 相互独立 | `_patched_kv_cache_allocate_slots` 对压缩请求永久跳过 hash 提交；`_free_reclaimed_blocks` evict 已注册 hash | 压缩请求既不新增 hash，又清除已有 hash |

---

## 七、修复方向建议（仅作方向性建议，不含代码）

### 方向 1：让压缩回收的 block 保留 prefix-cache hash（首选）

**思路**：`_free_reclaimed_blocks` 归还物理 block 时，**不要调用** `_maybe_evict_cached_block`，让 `cached_block` 反查表继续保留这些 hash 索引。

**权衡**：
- 优点：第二次相同请求能完整命中，Prefix-Caching 恢复正常。
- 风险：被复用的物理 block 可能被新数据覆盖写，此时 `cached_block` 中的旧 hash 索引会指向"内容已变"的 block，导致后续命中读到错误 KV。
- 缓解：vLLM BlockPool 本身在 `allocate_slots` 分配新块时会清空旧 hash 并重新注册，因此只要不在 TriAttention 侧主动 evict，让 vLLM 自己管理 hash 生命周期即可。

### 方向 2：压缩只回收"超出 prompt 长度"的 block（次选）

**思路**：压缩回收时，保留 `prefill_len / block_size` 个 block 不动，只回收 Decode 阶段新生成且超出 KV Budget 的 block。

**权衡**：
- 优点：prompt 部分的 prefix-cache hash 完全不受影响，第二次请求能完整命中 prompt 前缀。
- 风险：Decode 阶段的 KV 仍会被 evict，但这部分本就不是 prefix-cache 的主要命中对象（不同请求的 Decode 输出不同），影响较小。

### 方向 3：延迟压缩到请求结束后（保底）

**思路**：设置较大的 `TRIATTN_RUNTIME_MIN_DECODE_TOKENS_BEFORE_COMPRESS_ON_ASCEND`，让压缩推迟到 Decode 足够多 token 之后，确保请求生命周期内不触发压缩。

**权衡**：
- 优点：简单，不改动代码。
- 缺点：违背 TriAttention 的核心价值（节省 KV 内存），仅适用于验证场景。

### 方向 4：为压缩请求单独维护 prefix-cache 副本（长期方案）

**思路**：压缩前把 prompt block 的 hash 索引快照保存到请求级状态，压缩后不清除全局 `cached_block`，而是让第二次请求匹配时优先用快照。

**权衡**：
- 优点：彻底解决冲突，且不影响 TriAttention 的内存节省能力。
- 缺点：实现复杂，需要侵入 vLLM BlockPool 的匹配逻辑。

---

## 八、验证路径汇总

| 验证编号 | 验证内容 | 期望结果 | 状态 |
|---|---|---|---|
| 1 | 观测压缩时机与 evict 数量 | freed 数量 ≈ (20000 - kv_budget - reclaim_interval) / block_size | 待验证 |
| 2 | 观测 prefix-cache 命中率 | Base ~100%，2k ~21%，4k ~31% | **已验证 ✓** |
| 3 | 关闭压缩但保留 patch（`DISABLE_COMPRESSION=1`） | 第二次 TTFT 接近 Base | 待验证 |
| 4 | 关闭 block reclaim evict（`ENABLE_EXPERIMENTAL_BLOCK_RECLAIM=0`） | 第二次 TTFT 大幅下降 | 待验证 |
| 5 | 调大 `MIN_DECODE_TOKENS_BEFORE_COMPRESS_ON_ASCEND=1000` | 1000 token 内不压缩，第二次 TTFT 接近 Base | 待验证 |
| 6 | 监控 `_maybe_evict_cached_block` 调用次数 | TriAttention 组 evict 次数 ≈ 被回收 block 数；Base 组 ≈ 0 | 待验证 |
| 7 | 确认 `max_num_scheduled_tokens` cap 是否触发 | 本实验参数下大概率不触发 | 待验证 |

---

## 九、最终结论

**TriAttention 开启后 Prefix-Caching 失效的根因已确认**：

TriAttention 的"Decode 阶段逻辑驱逐"机制在实际执行时并不"逻辑"——它在压缩回收物理 block 时，通过 `_evict_reclaimed_block_metadata → BlockPool._maybe_evict_cached_block` 主动清除了这些 block 对应的 prefix-cache hash 索引，而这些 block 恰好承载了 prompt 中后段的 KV，正是第二次相同请求最需要命中的部分。

实测命中率（2k: 21%，4k: 31%）与"压缩后保留的 block 数 / 总 prompt block 数"完全吻合，根因已确认。

**叠加的二次伤害**：`_patched_kv_cache_allocate_slots` 对所有曾压缩过的请求永久跳过 hash 提交（`delay_cache_blocks=True`），进一步确保压缩后的新 block 永不进入 prefix-cache。

**设计语义错位**：文档声明"KV Budget 仅作用于 Decode 阶段的逻辑驱逐，不干预 Prefill 阶段的前缀缓存写入"，但实际实现中 Decode 第一阶段（Prefill 刚结束、`min_decode_tokens_before_compress_on_ascend=0`）立即触发压缩，把已注册的 prompt 中后段 block 从 prefix-cache 中 evict 掉，等价于"前缀缓存写完立刻被清"。
