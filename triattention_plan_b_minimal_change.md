# 方案 A+B 统一开关实施细节

> 目标：通过 `export TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=1` 一个开关同时启用两个功能：
> - **方案 B**：对所有请求强制 `delay_cache_blocks=True`，跳过 HBM 的 `cache_full_blocks` 哈希写入
> - **方案 A**：请求结束时清理 HBM `cached_block_hash_to_block` 的残留哈希（兜底）
>
> 不动 vLLM 核心、不动 vllm-ascend、不动 TriAttention 其它逻辑。
>
> 仅修改本仓 3 个文件，共 6 处改动。

---

## 改动文件清单

| 文件 | 改动数 | 改动性质 |
|------|--------|---------|
| `triattention/vllm/runtime/config.py` | 2 处 | 新增 1 个字段 + 1 个 env 解析 |
| `triattention/vllm/runtime/integration_monkeypatch.py` | 3 处 | 新增 1 个缓存辅助函数 + 修改 `_patched_kv_cache_allocate_slots` 判定 + 在请求结束处调清理函数 |
| `triattention/vllm/runtime/scheduler.py` | 1 处 | 新增 1 个清理辅助函数 |

---

## 改动 1：`triattention/vllm/runtime/config.py` 新增字段声明

### 位置：第 102 行后（`max_compressions_per_step_on_ascend` 字段之后，空行之前）

### 当前代码（第 95-103 行）

```95:103:triattention/vllm/runtime/config.py
    enable_zero_copy_recency: bool = True
    zero_copy_recency_only_on_ascend: bool = True
    enable_packed_pos_delta_on_ascend: bool = False
    auto_fast_recency_on_ascend: bool = True
    early_install_proxy_on_ascend: bool = True
    preinstall_input_patch: bool = True
    force_eager_multi_req_on_ascend_effective_overrides: bool = False
    max_compressions_per_step_on_ascend: int = 4

    # Optional TriAttention-style scoring path (used by runtime hook when enabled).
```

### 改动后

在第 102 行 `max_compressions_per_step_on_ascend: int = 4` 之后、第 103 行空行之前，插入一行：

```python
    max_compressions_per_step_on_ascend: int = 4
    disable_hbm_prefix_hash: bool = False

    # Optional TriAttention-style scoring path (used by runtime hook when enabled).
```

**字段语义**：默认 False（不影响现有行为）；设 True 时对所有请求强制 `delay_cache_blocks=True`，跳过 HBM 的 `cache_full_blocks` 哈希写入。

---

## 改动 2：`triattention/vllm/runtime/config.py` 新增 env 解析

### 位置：第 361 行后（`max_compressions_per_step_on_ascend=maybe_int(...)` 之后，`sparse_stats_path=...` 之前）

### 当前代码（第 358-362 行）

```358:362:triattention/vllm/runtime/config.py
            max_compressions_per_step_on_ascend=maybe_int(
                "MAX_COMPRESSIONS_PER_STEP_ON_ASCEND",
                cls.max_compressions_per_step_on_ascend,
            ),
            sparse_stats_path=sparse_stats_path_candidate,
```

### 改动后

在 `max_compressions_per_step_on_ascend=maybe_int(...)` 块之后、`sparse_stats_path=...` 之前，插入一个 `disable_hbm_prefix_hash=maybe_bool(...)` 块：

```python
            max_compressions_per_step_on_ascend=maybe_int(
                "MAX_COMPRESSIONS_PER_STEP_ON_ASCEND",
                cls.max_compressions_per_step_on_ascend,
            ),
            disable_hbm_prefix_hash=maybe_bool(
                "DISABLE_HBM_PREFIX_HASH",
                cls.disable_hbm_prefix_hash,
            ),
            sparse_stats_path=sparse_stats_path_candidate,
```

**说明**：`maybe_bool` 是同文件第 136-138 行已有的辅助函数，与其它 bool 配置项解析模式完全一致。对应环境变量 `TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH`（前缀 `TRIATTN_RUNTIME_` 由 `from_env` 的 `prefix` 参数自动加）。

---

## 改动 3：`triattention/vllm/runtime/integration_monkeypatch.py` 新增缓存辅助函数

### 位置：第 581 行后（`_should_defer_prefill_boundary` 函数之后）

### 当前代码（第 570-582 行）

```570:582:triattention/vllm/runtime/integration_monkeypatch.py
def _should_defer_prefill_boundary() -> bool:
    global _DEFER_PREFILL_BOUNDARY_CACHE
    if _DEFER_PREFILL_BOUNDARY_CACHE is not None:
        return _DEFER_PREFILL_BOUNDARY_CACHE
    cfg = TriAttentionRuntimeConfig.from_env()
    _DEFER_PREFILL_BOUNDARY_CACHE = bool(
        getattr(cfg, "defer_prefill_compression", False)
    ) or (
        bool(getattr(cfg, "defer_prefill_compression_on_ascend", False))
        and is_ascend_environment_available()
    )
    return _DEFER_PREFILL_BOUNDARY_CACHE
```

### 改动后

在 `_should_defer_prefill_boundary` 函数之后（第 582 行后）新增一个函数。同时需要在文件顶部全局变量区（第 50-53 行附近）新增一个缓存变量。

#### 3a. 顶部全局变量区（第 53 行后）

当前（第 50-53 行）：

```50:53:triattention/vllm/runtime/integration_monkeypatch.py
_ORIG_KVCACHE_ALLOCATE_SLOTS: Callable[..., Any] | None = None
_ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE: Callable[..., Any] | None = None
_DEFER_PREFILL_BOUNDARY_CACHE: bool | None = None
_ASYNC_BOUNDARY_ENABLED_CACHE: bool | None = None
```

改为（新增一行）：

```python
_ORIG_KVCACHE_ALLOCATE_SLOTS: Callable[..., Any] | None = None
_ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE: Callable[..., Any] | None = None
_DEFER_PREFILL_BOUNDARY_CACHE: bool | None = None
_ASYNC_BOUNDARY_ENABLED_CACHE: bool | None = None
_DISABLE_HBM_PREFIX_HASH_CACHE: bool | None = None
```

#### 3b. 第 582 行后新增函数

```python
def _disable_hbm_prefix_hash_enabled() -> bool:
    global _DISABLE_HBM_PREFIX_HASH_CACHE
    if _DISABLE_HBM_PREFIX_HASH_CACHE is not None:
        return _DISABLE_HBM_PREFIX_HASH_CACHE
    cfg = TriAttentionRuntimeConfig.from_env()
    _DISABLE_HBM_PREFIX_HASH_CACHE = bool(
        getattr(cfg, "disable_hbm_prefix_hash", False)
    )
    return _DISABLE_HBM_PREFIX_HASH_CACHE
```

**说明**：仿照 `_async_compression_boundary_enabled`（559-567 行）的模式，用模块级缓存避免每次 `allocate_slots` 调用都重新 `from_env`。

---

## 改动 4：`triattention/vllm/runtime/integration_monkeypatch.py` 修改 `_patched_kv_cache_allocate_slots` 判定

### 位置：第 528-533 行

### 当前代码（第 514-540 行）

```514:540:triattention/vllm/runtime/integration_monkeypatch.py
    assert _ORIG_KVCACHE_ALLOCATE_SLOTS is not None
    # Ensure effective marker is refreshed — _sync_effective_kv_offsets only
    # covers RUNNING requests, but preempted WAITING requests also need it.
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

### 改动后

把第 528-533 行的判定逻辑替换。**关键**：原逻辑是"只有压缩过的请求才 delay"；新逻辑是"如果开关开了，所有请求都 delay；否则保持原逻辑"。

需要同时处理一个细节：原逻辑在 `delay_cache_blocks=True` 分支里还做了 `setattr(request, "num_computed_tokens", int(effective_num_computed))` + finally 恢复——这是为压缩请求改写 num_computed_tokens 让 vLLM 按有效长度分配。**开关模式下不应改写 num_computed_tokens**（请求未必压缩过，改写会破坏正常分配）。所以两条分支要分开处理。

替换第 528-540 行为：

```python
    disable_hbm_hash = _disable_hbm_prefix_hash_enabled()
    if disable_hbm_hash:
        # 方案 B：TriAttention + Prefix Caching 场景下，对所有请求跳过
        # HBM 的 cache_full_blocks 哈希写入。SSD 侧的哈希来源是
        # request.block_hashes（Request 自带字段，由 block_hasher 计算），
        # 与 HBM 的 cached_block_hash_to_block 完全独立，不受影响。
        kwargs = dict(kwargs)
        kwargs["delay_cache_blocks"] = True
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

**改动要点**：
1. 在原有判定之前插入 `disable_hbm_hash` 分支。
2. `disable_hbm_hash` 分支只设 `delay_cache_blocks=True`，**不**改写 `num_computed_tokens`（因为该分支对所有请求生效，绝大多数请求未压缩，改写会出错）。
3. 原"压缩请求"分支（`effective_num_computed < logical_num_computed`）保持不变，作为 fallback。

---

## 关于 `enable_prefix_caching` 的判断

之前我提过 `if cfg.disable_hbm_prefix_hash and enable_prefix_caching:`，但经过核实，TriAttention runtime 内**完全没有**引用过 `enable_prefix_caching`（本地搜索 `triattention/vllm/runtime/` 下 `enable_caching|enable_prefix_caching` 无匹配）。

**决定不在判定里加 `enable_prefix_caching` 检查**，原因：
1. `delay_cache_blocks` 在 vLLM 里即使 `enable_caching=False` 也是安全的（`kv_cache_manager.py` 中 `if not self.enable_caching or delay_cache_blocks: return ...`，两者等价短路）。
2. 加 `enable_prefix_caching` 检查需要从 `self`（KVCacheManager 实例）读 `enable_caching`，但 `_patched_kv_cache_allocate_slots` 的 `self` 是 KVCacheManager，是否有 `enable_caching` 属性需 vLLM 核心确认（本地不可见）。
3. 用户通过 `export TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=1` 显式开启即视为同意"我对 TriAttention + Prefix Caching 场景负责"，无需代码再二次判断。

**场景隔离靠用户自觉**：只在 TriAttention + Prefix Caching 开启的启动命令里 export 这个变量。非 TriAttention 或非 Prefix Caching 场景不要 export。

---

# 方案 A 续写：请求结束清理 HBM 残留哈希（与方案 B 共用同一开关）

> 方案 B 跳过 HBM 写入，但不清理已写入的哈希。方案 A 作为兜底：在请求结束时清理 HBM `cached_block_hash_to_block` 的残留条目。
>
> **统一开关设计**：方案 A 与方案 B 共用同一个开关 `TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH`。开这一个开关就同时启用"不写 HBM 哈希"（方案 B）+"请求结束清残留"（方案 A）。用户只需 export 一个变量。
>
> 方案 A 在方案 B 的 4 处改动基础上，再增加 2 处改动（共 6 处）。不新增任何配置字段或 env 变量——清理逻辑直接复用方案 B 的 `_disable_hbm_prefix_hash_enabled()` 判定。

---

## 方案 A 改动文件清单（在方案 B 基础上追加）

| 文件 | 追加改动数 | 改动性质 |
|------|-----------|---------|
| `triattention/vllm/runtime/scheduler.py` | 1 处 | 新增 1 个清理辅助函数 |
| `triattention/vllm/runtime/integration_monkeypatch.py` | 1 处 | 在 `_patched_scheduler_update_from_output` 的 finished_req_ids 循环里调清理函数（含 import） |

**注意**：不新增 config 字段、不新增 env 变量、不新增全局缓存变量、不新增辅助判定函数——全部复用方案 B 已有的 `_disable_hbm_prefix_hash_enabled()`。

---

## 改动 5：`triattention/vllm/runtime/scheduler.py` 新增清理辅助函数

### 位置：第 59 行后（`_free_reclaimed_blocks` 函数之后）

### 当前代码（第 49-59 行）

```49:59:triattention/vllm/runtime/scheduler.py
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

### 改动后

在 `_free_reclaimed_blocks` 函数之后（第 59 行后）新增一个函数：

```python
def _cleanup_hbm_prefix_hash_mappings(scheduler: Any) -> int:
    """Best-effort clear all HBM prefix-cache hash mappings.

    Iterates over every block in the HBM BlockPool and evicts any that still
    carry a block_hash from cached_block_hash_to_block. Only touches the hash
    metadata and the reverse-lookup table; does NOT touch ref_cnt, the free
    queue, or physical block allocation. Returns the number of blocks evicted.
    """
    block_pool = getattr(getattr(scheduler, "kv_cache_manager", None), "block_pool", None)
    if block_pool is None:
        return 0
    blocks = getattr(block_pool, "blocks", None)
    if not blocks:
        return 0
    maybe_evict = getattr(block_pool, "_maybe_evict_cached_block", None)
    if not callable(maybe_evict):
        return 0
    evicted = 0
    for block in blocks:
        if getattr(block, "block_hash", None) is not None:
            if maybe_evict(block):
                evicted += 1
    return evicted
```

**设计要点**：
1. **仿照 `_evict_reclaimed_block_metadata`（36-46 行）和 `_compute_max_chunk_for_compression`（508 行）的防御式 `getattr` 模式**，不假设任何属性一定存在。
2. **只清 hash，不动 ref_cnt / free queue**：`_maybe_evict_cached_block` 内部只做 `cached_block_hash_to_block.pop` + `block.reset_hash()`（见 vLLM BlockPool 源码），不改 `ref_cnt`，不把 block 加回 free queue。这保证正在使用的 block（ref_cnt>0）不会被误释放。
3. **遍历所有 block**：用 `block_pool.blocks`（本仓证据 `tests/ut/kv_connector/utils.py:44` `for block in scheduler.kv_cache_manager.block_pool.blocks`），对每个 `block_hash is not None` 的 block 调 evict。
4. **返回 evicted 计数**：便于日志观察清理了多少条目。

---

## 改动 6：`triattention/vllm/runtime/integration_monkeypatch.py` 在请求结束处调清理函数

### 位置：第 242-247 行（`_patched_scheduler_update_from_output` 的 finished_req_ids 循环）

### 当前代码（第 242-248 行）

```242:248:triattention/vllm/runtime/integration_monkeypatch.py
    for req_id in scheduler_output.finished_req_ids:
        self._prefill_lens.pop(req_id, None)
        self._length_threshold_cache.pop(req_id, None)
        self._last_signal_log_steps.pop(req_id, None)
        self._long_context_guard_logged.discard(req_id)
        self._effective_len_tracker.remove_request(req_id)
    return outputs
```

### 改动后

在 finished_req_ids 循环之后、`return outputs` 之前，插入清理逻辑。只需在文件顶部 import `_cleanup_hbm_prefix_hash_mappings`，**不新增全局变量、不新增辅助函数**——清理判定直接复用方案 B 的 `_disable_hbm_prefix_hash_enabled()`。

#### 6a. 顶部 import（第 29 行附近）

当前（第 29 行）：

```29:29:triattention/vllm/runtime/integration_monkeypatch.py
from .scheduler import TriAttentionScheduler
```

改为：

```python
from .scheduler import TriAttentionScheduler, _cleanup_hbm_prefix_hash_mappings
```

#### 6b. 修改 finished_req_ids 循环（第 242-248 行）

把：

```python
    for req_id in scheduler_output.finished_req_ids:
        self._prefill_lens.pop(req_id, None)
        self._length_threshold_cache.pop(req_id, None)
        self._last_signal_log_steps.pop(req_id, None)
        self._long_context_guard_logged.discard(req_id)
        self._effective_len_tracker.remove_request(req_id)
    return outputs
```

改为：

```python
    for req_id in scheduler_output.finished_req_ids:
        self._prefill_lens.pop(req_id, None)
        self._length_threshold_cache.pop(req_id, None)
        self._last_signal_log_steps.pop(req_id, None)
        self._long_context_guard_logged.discard(req_id)
        self._effective_len_tracker.remove_request(req_id)

    # 方案 A：请求结束后清理 HBM 残留哈希。
    # 复用方案 B 的 _disable_hbm_prefix_hash_enabled() 判定——同一个开关
    # 同时控制"不写 HBM 哈希"（方案 B）和"请求结束清残留"（方案 A）。
    # 当任意请求结束时，清空 HBM cached_block_hash_to_block 的所有残留条目。
    if scheduler_output.finished_req_ids and _disable_hbm_prefix_hash_enabled():
        evicted = _cleanup_hbm_prefix_hash_mappings(self)
        if cfg.logging_enabled and evicted > 0:
            logger.debug(
                "TriAttention HBM prefix-hash cleanup: evicted %d entries "
                "after %d request(s) finished",
                evicted, len(scheduler_output.finished_req_ids),
            )

    return outputs
```

**改动要点**：
1. **清理时机**：在 `finished_req_ids` 循环之后（所有请求状态已清理），`return outputs` 之前。
2. **只在有请求结束时才清理**：`if scheduler_output.finished_req_ids and _disable_hbm_prefix_hash_enabled()`，避免每步都遍历 block_pool。
3. **复用方案 B 的开关**：`_disable_hbm_prefix_hash_enabled()` 是方案 B 改动 3b 新增的函数，这里直接复用，不新增任何判定逻辑。
4. **清理范围**：`_cleanup_hbm_prefix_hash_mappings` 遍历所有 block，清所有有 hash 的 block。这是"全量清理"，不区分是哪个请求的 block——因为 HBM `cached_block_hash_to_block` 是全局反查表，且目标是保证"HBM 表不累积过期哈希"。
5. **用 `cfg` 判断日志**：`cfg` 在第 206 行已获取（`cfg = getattr(self, "triattention_config", None)`），复用。

---

## 统一开关的行为说明

**一个开关 `TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=1` 同时启用两个功能：**

| 开关状态 | 方案 B（不写 HBM 哈希） | 方案 A（请求结束清残留） | 效果 |
|---------|------------------------|------------------------|------|
| `0` 或未 export（默认） | 关 | 关 | 现有行为，HBM 正常写哈希，不清 |
| `1` | 开 | 开 | HBM 不写哈希 + 每个请求结束清残留；最稳妥，覆盖所有边界 |

**启用方式**（只需一个 export）：

```bash
export ENABLE_TRIATTENTION=1
export TRIATTN_RUNTIME_KV_BUDGET=4096
# ... 其它 TriAttention 环境变量 ...
export TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=1   # ← 同时启用方案 A + B

vllm serve Qwen3-32B \
    --enable-prefix-caching \
    ... 其它参数 ...
```

关闭则不 export 该变量（或 `export TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=0`）。

---

## 方案 A 的安全性分析

### 1. 不会误释放正在使用的 block

`_cleanup_hbm_prefix_hash_mappings` 只调 `_maybe_evict_cached_block`，后者内部只做：
- `cached_block_hash_to_block.pop(block_hash, block.block_id)` —— 从反查表移除
- `block.reset_hash()` —— 清 block 的 hash 属性

**不动 `ref_cnt`**，**不动 free queue**。所以正在使用的 block（ref_cnt>0）不会被释放，只是失去 hash 索引（本来方案 B 下也不该有）。

### 2. 不影响 SSD

SSD（ascend_store）的 hash 来源是 `request.block_hashes`（Request 实例字段），不读 HBM `cached_block_hash_to_block`（见 `triattention_ascend_store_deep_investigation.md`）。清 HBM 表不影响 SSD。

### 3. 不影响下一个请求的 HBM 命中

开关开启时 HBM 不写哈希（方案 B），所以清不清都不影响 HBM 命中（本就为 0）。方案 A 的清理是针对"开关中途生效或未生效期间已写入的残留"。

### 4. 性能影响

`_cleanup_hbm_prefix_hash_mappings` 遍历 `block_pool.blocks`（所有 HBM block），复杂度 O(num_blocks)。只在有请求结束时触发，不是每步触发。对典型 num_blocks（几千）可忽略。

---

## 完整改动汇总验证清单（方案 A + B 统一开关，共 6 处改动）

- [ ] **改动 1**：`triattention/vllm/runtime/config.py` 第 102 行后有 `disable_hbm_prefix_hash: bool = False`
- [ ] **改动 2**：`triattention/vllm/runtime/config.py` `from_env` 内有 `disable_hbm_prefix_hash=maybe_bool("DISABLE_HBM_PREFIX_HASH", cls.disable_hbm_prefix_hash)`
- [ ] **改动 3**：`triattention/vllm/runtime/integration_monkeypatch.py` 顶部全局变量区有 `_DISABLE_HBM_PREFIX_HASH_CACHE: bool | None = None`
- [ ] **改动 3**：`triattention/vllm/runtime/integration_monkeypatch.py` 有 `_disable_hbm_prefix_hash_enabled()` 函数
- [ ] **改动 4**：`_patched_kv_cache_allocate_slots` 在原判定之前插入了 `disable_hbm_hash` 分支（不改写 `num_computed_tokens`）
- [ ] **改动 5**：`triattention/vllm/runtime/scheduler.py` 有 `_cleanup_hbm_prefix_hash_mappings(scheduler)` 函数
- [ ] **改动 6**：`triattention/vllm/runtime/integration_monkeypatch.py` 顶部 import 了 `_cleanup_hbm_prefix_hash_mappings`
- [ ] **改动 6**：`_patched_scheduler_update_from_output` 在 finished_req_ids 循环后、return 前调用了 `_cleanup_hbm_prefix_hash_mappings`（复用 `_disable_hbm_prefix_hash_enabled()` 判定）

---

## 风险与限制

1. **方案 B 只跳过 HBM 写入，不清理已写入的哈希**。方案 A（同一开关）在请求结束时清理残留，覆盖"开关中途生效"的边界。若开关在进程启动时就 export，第一次请求 Prefill 期间就不会写 HBM 哈希，方案 A 的清理是额外保险。
2. **场景隔离靠用户自觉**：非 TriAttention 场景不要开此开关，否则会关掉该场景的 HBM prefix caching（SSD 仍工作，但 HBM 命中能力丧失）。
3. **同 batch 内请求间无法 HBM 命中**：开启后所有请求的 HBM 哈希表始终为空，同 batch 内不同请求无法通过 HBM 命中彼此的 KV。TriAttention 的 token 级重排本就使这种命中不可靠，可接受。
4. **未改 `_free_reclaimed_blocks` 的 `_evict_reclaimed_block_metadata`**：保留原样。开启后，被回收的 block 本就没有 hash（因没写过），`_maybe_evict_cached_block` 会因 `block_hash is None` 直接 return，无副作用。
5. **全量清理 vs 按请求清理**：方案 A 是"任意请求结束 → 清空所有 HBM 哈希"。如果同 batch 有多个并发请求，一个请求结束会清掉其它未结束请求的 HBM 哈希。但开关开启时 HBM 本就无哈希，无影响。
6. **不清理 `req_to_block_hashes`**：`cpu_kv_cache_manager.py:75` 的 `req_to_block_hashes` 是 SSD 侧的缓存，与 HBM 无关，方案 A 不动它。
7. **不清理 `num_cached_block`**：`single_type_manager.num_cached_block` 是请求级缓存计数，方案 A 不动它。若需彻底，可额外清，但当前设计最小化改动面。

---

## 待运行环境验证的点

以下细节实现不在本仓，需运行环境（已装 vLLM 0.18.0）确认：

1. `delay_cache_blocks=True` 确实在 `cache_blocks` 之前 return（vLLM `kv_cache_manager.py:allocate_slots` 行为）。
2. 开关开启后，跑一个请求，检查 HBM `block_pool.cached_block_hash_to_block` 是否为空、SSD `CPUKVCacheManager.block_pool.cached_block_hash_to_block` 是否含该请求的块。
3. 第二次相同请求是否走 SSD 路径（connector `get_num_new_matched_tokens` 返回非 0）。
4. `_maybe_evict_cached_block` 的行为确认（vLLM 核心 `block_pool.py`）：只清 hash + 反查表，不动 ref_cnt/free queue。
5. `block_pool.blocks` 是否包含所有 block（本仓测试 `utils.py:44` 已验证可遍历）。
6. 开启方案 A 后，请求结束检查 `len(block_pool.cached_block_hash_to_block) == 0`。
7. 开启方案 A 后，正在使用的 block 的 `ref_cnt` 不受影响。
