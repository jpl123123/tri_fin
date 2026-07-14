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

---

# 方案 C 续写：从源头拦截 HBM 哈希写入（patch `BlockPool.cache_full_blocks`）

> **背景**：方案 A+B 实测有 0.5% 精度提升，但与 base 还差 2%。原因是方案 B 的 `delay_cache_blocks=True` 只跳过了 `allocate_slots` 内部那一次 `cache_blocks` 调用，**没有覆盖其它显式调用 `cache_blocks` 的路径**。这些漏掉的路径仍在向 HBM `cached_block_hash_to_block` 写入哈希，干扰命中。
>
> 方案 C 的思路：**找到所有 HBM 哈希写入路径的最底层汇聚点，在那里直接 no-op**，一次性堵住所有写入。

---

## 一、所有 HBM 哈希写入路径的完整排查

通过本仓代码搜索 `cache_blocks(|cache_full_blocks|coordinator\.cache_blocks|single_type_manager\.cache_blocks`，找到以下写入路径：

### 路径 1：`allocate_slots` 内部（方案 B 已覆盖）

- **调用链**：`KVCacheManager.allocate_slots` → 末尾 `self.coordinator.cache_blocks(request, num_tokens_to_cache)` → `single_type_manager.cache_blocks` → `BlockPool.cache_full_blocks`
- **位置**：vLLM 核心 `vllm/v1/core/kv_cache_manager.py`（不在本仓）
- **方案 B 处理**：`delay_cache_blocks=True` 让 `allocate_slots` 在 `coordinator.cache_blocks` 之前提前 return。✅ 已覆盖

### 路径 2：`recompute_scheduler.py:182,205` 显式调用（方案 B 未覆盖）

- **调用链**：`RecomputeScheduler._update_request_after_kv_recv` → `self.kv_cache_manager.cache_blocks(request, ...)` → `coordinator.cache_blocks` → `BlockPool.cache_full_blocks`
- **位置**：`other_code/vllm-ascend-releases-v0.18.0/vllm_ascend/core/recompute_scheduler.py:182,205`

```182:205:other_code/vllm-ascend-releases-v0.18.0/vllm_ascend/core/recompute_scheduler.py
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
        else:
            # Now that the blocks are ready, actually cache them.
            # Use Ascend-specific block_ids logic to handle multi-group KV
            # cache configurations (e.g. MLA) where len(block_ids) > 1.
            block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
            if len(block_ids) == 1:
                num_computed_tokens = len(block_ids[0]) * self.block_size
                # Handle the case where num request tokens less than one block.
                num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            else:
                num_computed_tokens = request.num_tokens
            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
```

- **触发时机**：KV connector recv 完成（SSD 加载完成）后，把已就绪 block 写入 HBM 反查表。
- **方案 B 未覆盖**：`delay_cache_blocks` 只影响 `allocate_slots`，不影响这里的显式 `cache_blocks` 调用。❌ **这是漏掉的主要源头**
- **注释佐证**：第 190 行 "Now that the blocks are ready, actually cache them."、第 204 行 "This will cache the blocks iff caching is enabled." 明确说明这是写 HBM 反查表。

### 路径 3：`cpu_kv_cache_manager.py:164`（SSD 侧，不影响 HBM）

- **位置**：`other_code/vllm-ascend-releases-v0.18.0/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py:164`
- **说明**：这是 cpu_offload SSD 侧 `CPUKVCacheManager` 自己的 `cache_blocks`，写的是 SSD 侧的 `cached_block_hash_to_block`，**不影响 HBM**。ascend_store 路径不经过这里。可忽略。

### 路径汇总

| 路径 | 调用点 | 写入目标 | 方案 B 是否覆盖 |
|------|--------|---------|---------------|
| 1. `allocate_slots` 内部 | vLLM 核心 `kv_cache_manager.py` | HBM `cached_block_hash_to_block` | ✅ 覆盖 |
| 2. `recompute_scheduler` 显式 | `recompute_scheduler.py:182,205` | HBM `cached_block_hash_to_block` | ❌ **未覆盖** |
| 3. cpu_offload SSD 侧 | `cpu_kv_cache_manager.py:164` | SSD `cached_block_hash_to_block` | 不影响 HBM |

**结论**：路径 2 是方案 B 漏掉的 HBM 哈希写入源头。SSD 加载完成后，`recompute_scheduler` 会显式调 `kv_cache_manager.cache_blocks` 把哈希写入 HBM 表，方案 B 的 `delay_cache_blocks` 管不到这里。

---

## 二、最底层汇聚点：`BlockPool.cache_full_blocks`

所有 HBM 哈希写入路径最终都汇聚到 `BlockPool.cache_full_blocks`（vLLM 核心 `vllm/v1/core/block_pool.py`）：

```
路径 1: allocate_slots → coordinator.cache_blocks → single_type_manager.cache_blocks → BlockPool.cache_full_blocks
路径 2: recompute_scheduler.cache_blocks → coordinator.cache_blocks → single_type_manager.cache_blocks → BlockPool.cache_full_blocks
                                                                                                    ↑
                                                                                          所有路径汇聚到这里
```

**本仓证据**：搜索 `cache_full_blocks` 在本仓零匹配（只有 `test_remote_prefill_lifecycle.py:72` 的 `cached_block_hash_to_block` 断言）——说明 `cache_full_blocks` 只在 vLLM 核心内部被调用，是所有写入的最底层入口。

---

## 三、方案 C：patch `BlockPool.cache_full_blocks` 直接 no-op

### 思路

在 `integration_monkeypatch.py` 的 `install_vllm_integration_monkeypatches` 里，**额外 patch `BlockPool.cache_full_blocks`**：当开关开启时，让 `cache_full_blocks` 直接 return（不写 `cached_block_hash_to_block`、不给 block 挂 hash）。

这样**所有**调 `cache_full_blocks` 的路径（路径 1 + 路径 2 + 任何未来新增的路径）都被一次性堵住，无需逐个 patch 上层调用点。

### 方案 C 改动文件清单（在方案 A+B 基础上追加 1 处）

| 文件 | 追加改动数 | 改动性质 |
|------|-----------|---------|
| `triattention/vllm/runtime/integration_monkeypatch.py` | 1 处 | 新增 `_patched_block_pool_cache_full_blocks` 函数 + 在 `install_vllm_integration_monkeypatches` 里 patch `BlockPool.cache_full_blocks` |

---

## 改动 7：`triattention/vllm/runtime/integration_monkeypatch.py` patch `BlockPool.cache_full_blocks`

### 7a. 顶部新增全局变量（紧跟 `_DISABLE_HBM_PREFIX_HASH_CACHE`）

方案 A+B 改动后全局变量区：

```python
_DISABLE_HBM_PREFIX_HASH_CACHE: bool | None = None
```

再加一行：

```python
_DISABLE_HBM_PREFIX_HASH_CACHE: bool | None = None
_ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS: Callable[..., Any] | None = None
```

### 7b. 新增 patch 函数（位置：紧跟 `_disable_hbm_prefix_hash_enabled` 函数之后，约第 582 行后）

```python
def _patched_block_pool_cache_full_blocks(self, *args, **kwargs):
    """方案 C：从源头拦截 HBM 哈希写入。

    BlockPool.cache_full_blocks 是所有 HBM 哈希写入路径的最底层汇聚点：
      - allocate_slots → coordinator.cache_blocks → ... → cache_full_blocks
      - recompute_scheduler.cache_blocks → coordinator.cache_blocks → ... → cache_full_blocks

    当开关开启时直接 return，不写 cached_block_hash_to_block、不给 block 挂 hash。
    这样一次性堵住所有写入路径，无需逐个 patch 上层调用点。
    """
    if _disable_hbm_prefix_hash_enabled():
        # 开关开启：跳过 HBM 哈希写入，直接 return
        return
    # 开关关闭：走原始逻辑
    assert _ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS is not None
    return _ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS(self, *args, **kwargs)
```

**设计要点**：
1. **复用方案 B 的 `_disable_hbm_prefix_hash_enabled()`**——同一个开关，不新增判定。
2. **`self` 是 `BlockPool` 实例**——`cache_full_blocks` 是 `BlockPool` 的方法，patch 后 `self` 就是 block_pool。
3. **`*args, **kwargs` 透传**——不关心 `cache_full_blocks` 的具体签名（vLLM 版本可能变化），开关关闭时原样透传给原始函数。
4. **开关开启时直接 return**——`cache_full_blocks` 返回 `None`，调用方不依赖返回值（它是写入操作）。

### 7c. 在 `install_vllm_integration_monkeypatches` 里注册 patch

### 位置：第 747-748 行后（紧跟 `_ORIG_KVCACHE_ALLOCATE_SLOTS` 的 patch）

### 当前代码（第 747-750 行）

```747:750:triattention/vllm/runtime/integration_monkeypatch.py
        _ORIG_KVCACHE_ALLOCATE_SLOTS = KVCacheManager.allocate_slots
        KVCacheManager.allocate_slots = _patched_kv_cache_allocate_slots
        _ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE = EngineCore.step_with_batch_queue
        EngineCore.step_with_batch_queue = _patched_engine_core_step_with_batch_queue
```

### 改动后

在 `KVCacheManager.allocate_slots` patch 之后、`EngineCore` patch 之前，插入 `BlockPool.cache_full_blocks` 的 patch：

```python
        _ORIG_KVCACHE_ALLOCATE_SLOTS = KVCacheManager.allocate_slots
        KVCacheManager.allocate_slots = _patched_kv_cache_allocate_slots

        # 方案 C：patch BlockPool.cache_full_blocks 从源头拦截 HBM 哈希写入。
        # 覆盖 allocate_slots 之外的显式 cache_blocks 调用（如 recompute_scheduler.py:182,205）。
        try:
            import vllm.v1.core.block_pool as block_pool_mod
            _ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS = block_pool_mod.BlockPool.cache_full_blocks
            block_pool_mod.BlockPool.cache_full_blocks = _patched_block_pool_cache_full_blocks
        except Exception:
            if runtime_logging_enabled():
                logger.warning(
                    "TriAttention: could not patch BlockPool.cache_full_blocks "
                    "(vLLM version may differ). HBM hash write interception "
                    "for non-allocate_slots paths may be incomplete.",
                    exc_info=True,
                )

        _ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE = EngineCore.step_with_batch_queue
        EngineCore.step_with_batch_queue = _patched_engine_core_step_with_batch_queue
```

**设计要点**：
1. **`try/except` 防御**：`vllm.v1.core.block_pool` 的导入路径可能随 vLLM 版本变化，用 try/except 避免 patch 失败导致整个 TriAttention 初始化失败。
2. **位置**：紧跟 `allocate_slots` patch 之后，保持"KV cache 相关 patch 集中在一起"的组织性。
3. **复用 `runtime_logging_enabled()`**：与文件其它 patch 的日志模式一致。

---

## 四、方案 C 与方案 A+B 的关系

| 方案 | 拦截点 | 覆盖路径 | 作用 |
|------|--------|---------|------|
| B | `allocate_slots` 的 `delay_cache_blocks=True` | 路径 1（`allocate_slots` 内部） | 跳过 `allocate_slots` 末尾的 `cache_blocks` |
| C | `BlockPool.cache_full_blocks` no-op | 路径 1 + 路径 2 + 所有未来路径 | 从最底层源头堵住所有 HBM 哈希写入 |
| A | 请求结束清理 `_cleanup_hbm_prefix_hash_mappings` | 兜底清理已写入的残留 | 覆盖开关中途生效的边界 |

**方案 C 实际上让方案 B 变得多余**：因为方案 C 在 `cache_full_blocks` 最底层拦截，`allocate_slots` 内部调 `cache_full_blocks` 时也会被 no-op。但保留方案 B 无害（`delay_cache_blocks=True` 让 `allocate_slots` 提前 return，少执行一些代码，性能略好）。

**推荐组合**：A + B + C（三管齐下，最彻底）。但若只要最小改动，**单独用方案 C 即可覆盖所有写入路径**（方案 B 和 C 有重叠，方案 A 是兜底）。

---

## 五、为什么方案 C 能补上那 2% 差距

方案 B 只堵了路径 1（`allocate_slots`），但路径 2（`recompute_scheduler.py:182,205`）在 SSD 加载完成后仍会调 `kv_cache_manager.cache_blocks` 把哈希写入 HBM 表。这些哈希：

1. **在 HBM 表里残留**：第二次请求 `get_computed_blocks` 查 HBM 表时会命中这些残留哈希，导致部分命中 HBM 而非走 SSD。
2. **命中错误的 KV**：TriAttention 的 token 级重排使 HBM 里的哈希与实际 KV 布局不符，命中后读到错误 KV，精度下降。

方案 C 在 `cache_full_blocks` 最底层 no-op，堵住路径 2，让 HBM 表真正保持为空，第二次请求完全走 SSD，精度应能补上那 2% 差距。

---

## 六、方案 C 的安全性分析

### 1. `cache_full_blocks` 的语义

`cache_full_blocks` 的作用是"把已满 block 的 hash 注册进 `cached_block_hash_to_block` 并给 block 挂 hash"。它**只写哈希索引**，不动 `ref_cnt`、不动 free queue、不分配/释放 block。no-op 它只影响"HBM 命中查找能力"，不影响 block 的生命周期。

### 2. 不影响 block 分配

block 分配在 `allocate_slots` 的 `get_new_blocks` / `allocate_new_blocks` 阶段完成，早于 `cache_full_blocks`。no-op `cache_full_blocks` 不影响 block 已分配的事实。

### 3. 不影响 SSD

SSD（ascend_store）的 hash 来源是 `request.block_hashes`（Request 实例字段），不读 HBM `cached_block_hash_to_block`（见 `triattention_ascend_store_deep_investigation.md`）。no-op HBM 写入不影响 SSD。

### 4. 不影响 `recompute_scheduler` 的其它逻辑

`recompute_scheduler.py:182,205` 调 `cache_blocks` 后，第 208 行 `request.num_computed_tokens = num_computed_tokens` 等后续逻辑不依赖 `cache_blocks` 的返回值或副作用。no-op `cache_full_blocks` 只让 HBM 表不写入，不影响请求状态更新。

### 5. 与方案 A 的清理不冲突

方案 A 的 `_cleanup_hbm_prefix_hash_mappings` 遍历 `block_pool.blocks` 清理有 hash 的 block。方案 C no-op 后，block 本就没有 hash（`cache_full_blocks` 没执行），方案 A 的清理会因 `block_hash is None` 跳过，无副作用。两者叠加安全。

---

## 七、方案 C 完整改动汇总验证清单（在方案 A+B 的 6 处基础上追加）

- [ ] **改动 7a**：`triattention/vllm/runtime/integration_monkeypatch.py` 顶部全局变量区有 `_ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS: Callable[..., Any] | None = None`
- [ ] **改动 7b**：`triattention/vllm/runtime/integration_monkeypatch.py` 有 `_patched_block_pool_cache_full_blocks(self, *args, **kwargs)` 函数，复用 `_disable_hbm_prefix_hash_enabled()` 判定
- [ ] **改动 7c**：`install_vllm_integration_monkeypatches` 内有 `import vllm.v1.core.block_pool` + `_ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS = block_pool_mod.BlockPool.cache_full_blocks` + `block_pool_mod.BlockPool.cache_full_blocks = _patched_block_pool_cache_full_blocks`，用 try/except 防御

---

## 八、方案 C 待运行环境验证的点

1. `vllm.v1.core.block_pool.BlockPool.cache_full_blocks` 的导入路径在 vLLM 0.18.0 是否正确（若不对，try/except 会打 warning，需调整 import 路径）。
2. `cache_full_blocks` 的签名是否兼容 `*args, **kwargs` 透传（它是写入操作，返回 None，调用方不依赖返回值）。
3. 开启方案 C 后，跑一个请求，检查 HBM `block_pool.cached_block_hash_to_block` 在**整个生命周期**（包括 SSD 加载完成后）都为空。
4. 开启方案 C 后，第二次相同请求的命中率为 0（完全走 SSD），精度对比 base。
5. `recompute_scheduler.py:182,205` 调 `cache_blocks` 后的后续逻辑（`request.num_computed_tokens = ...` 等）不受 no-op 影响。

---

# 方案 D 续写：系统性排查与更彻底的拦截

> **背景**：方案 C patch `BlockPool.cache_full_blocks` 后精度没变化（仍差 base 约 2%）。这说明要么 patch 没真正生效，要么 HBM 哈希表根本不是瓶颈。本文系统排查所有可能，并给出更彻底的拦截方案。

---

## 一、HBM 哈希读写的完整路径地图

基于 vLLM 0.18.0 源码（`agent-tools/` 缓存）+ 本仓代码，完整路径如下：

### 写入路径（让 `cached_block_hash_to_block` 非空）

**唯一写入入口**：`BlockPool.cache_full_blocks`（vLLM 核心 `block_pool.py:215`），第 285 行 `self.cached_block_hash_to_block.insert(...)`。

调用 `cache_full_blocks` 的上层路径：

| 上层调用 | 位置 | 方案 B/C 覆盖 |
|---------|------|--------------|
| `SingleTypeKVCacheManager.cache_blocks` → `block_pool.cache_full_blocks` | vLLM 核心 `single_type_kv_cache_manager.py:184` | 方案 C 应覆盖 |
| `KVCacheManager.cache_blocks` → `coordinator.cache_blocks` → `single_type_manager.cache_blocks` | vLLM 核心 `kv_cache_manager.py:588` | 方案 C 应覆盖 |
| `allocate_slots` → `coordinator.cache_blocks`（内部直接调，不经 `KVCacheManager.cache_blocks`） | vLLM 核心 `kv_cache_manager.py:461` | 方案 B 覆盖（`delay_cache_blocks`）/ 方案 C 应覆盖 |
| `recompute_scheduler` 显式 `kv_cache_manager.cache_blocks` | `recompute_scheduler.py:182,205` | 方案 C 应覆盖 |

**关键结论**：`cache_full_blocks` 是**唯一**写入点。方案 C patch 它后，**所有**写入路径都应被堵住。如果 HBM 表仍非空，说明 patch 没生效。

### 读取路径（让第二次请求命中 HBM）

**唯一读取入口**：`BlockPool.get_cached_block`（vLLM 核心 `block_pool.py:188`），第 207 行 `self.cached_block_hash_to_block.get_one_block(...)`。

调用 `get_cached_block` 的上层路径：

| 上层调用 | 位置 |
|---------|------|
| `FullAttentionManager.find_longest_cache_hit` → `block_pool.get_cached_block` | vLLM 核心 `single_type_kv_cache_manager.py`（`FullAttentionManager`） |
| `coordinator.find_longest_cache_hit` → 各 manager 的 `find_longest_cache_hit` | vLLM 核心 `kv_cache_coordinator.py` |
| `KVCacheManager.get_computed_blocks` → `coordinator.find_longest_cache_hit` | vLLM 核心 `kv_cache_manager.py:233` |

**关键结论**：`get_cached_block` 是**唯一**读取点。patch 它让 HBM 查命中永远返回 None，比堵写入更彻底——即使有残留哈希也查不到。

---

## 二、为什么方案 C 可能没生效（排查清单）

方案 C patch `cache_full_blocks` 后精度没变，最可能的原因：

### 2.1 patch 没真正安装

方案 C 用 `try/except` 包裹 patch 注册。如果 `import vllm.v1.core.block_pool` 失败（路径不对、版本差异），except 会静默打 warning，patch 没装上。

**验证方法**：在运行环境检查日志是否有 `"TriAttention: could not patch BlockPool.cache_full_blocks"` warning。如果有，说明 patch 没装。

**修复**：调整 import 路径。本仓证据 `cpu_kv_cache_manager.py:6` `from vllm.v1.core.block_pool import BlockPool` 确认路径是 `vllm.v1.core.block_pool`，应正确。但 vLLM 0.18.0 可能重构了路径。

### 2.2 patch 装了但 `BlockPool` 类引用不对

Python 的 monkey-patch 是改类的方法。但如果 `BlockPool` 实例化在 patch 之前，或存在多个 `BlockPool` 类引用（不同 import 路径），patch 可能不生效。

**验证方法**：在 `_patched_block_pool_cache_full_blocks` 函数入口加日志，确认是否被调用。

### 2.3 `cache_full_blocks` 不是唯一写入点

虽然源码分析显示 `cache_full_blocks` 是唯一写入点，但 vLLM 0.18.0 可能有其它写入路径（如 kv_events 回调、connector 直接操作 `cached_block_hash_to_block`）。

**验证方法**：在 `cached_block_hash_to_block.insert` 处加断点/日志，确认所有写入调用栈。

### 2.4 问题根本不在 HBM 哈希表

如果方案 C 确实让 HBM 表为空了（第二次请求 `get_computed_blocks` 返回 0），但精度仍差 2%，说明**问题不在 HBM 命中**，而在别处：

- **SSD 侧 KV 数据被 TriAttention 压缩改写**：TriAttention 压缩后，HBM 里的 KV 布局变了，SSD 写入的 KV 可能是压缩后的（与原始 prompt 不符），第二次请求从 SSD 读到的是压缩后的 KV。
- **`request.block_hashes` 在压缩后变化**：如果压缩改写了 `request.block_hashes`，第二次请求的哈希与 SSD 里的不匹配。
- **TriAttention 压缩本身改变了计算结果**：即使 KV 完全正确，压缩后的 attention 计算可能与 base 不同（这是 TriAttention 的固有特性，非哈希问题）。

---

## 三、方案 D：更彻底的拦截——patch `get_cached_block`（读取侧）

### 思路

不再堵写入（方案 C），而是堵读取：patch `BlockPool.get_cached_block`，开关开启时直接返回 `None`（缓存未命中）。这样：

- **无论 HBM 表里有没有残留哈希，第二次请求查 HBM 都返回未命中**
- **第二次请求必然走 SSD 路径**
- 比堵写入更彻底，不怕任何遗漏的写入路径

### 优势对比

| 方案 | 拦截侧 | 彻底性 | 风险 |
|------|--------|--------|------|
| B | 写入侧（`allocate_slots`） | 只堵 1 条写入路径 | 漏掉其它写入路径 |
| C | 写入侧（`cache_full_blocks`） | 堵所有写入路径 | 依赖 patch 真正生效 |
| **D** | **读取侧（`get_cached_block`）** | **堵所有读取路径，不怕残留** | HBM 表可能仍有残留（但不影响，因为读不到） |

方案 D 是**最彻底**的——即使写入路径有遗漏，HBM 表有残留，读取侧也读不到。

---

## 改动 8：`triattention/vllm/runtime/integration_monkeypatch.py` patch `BlockPool.get_cached_block`

### 8a. 顶部新增全局变量（紧跟 `_ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS`）

方案 C 改动后全局变量区：

```python
_ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS: Callable[..., Any] | None = None
```

再加一行：

```python
_ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS: Callable[..., Any] | None = None
_ORIG_BLOCK_POOL_GET_CACHED_BLOCK: Callable[..., Any] | None = None
```

### 8b. 新增 patch 函数（紧跟 `_patched_block_pool_cache_full_blocks` 之后）

```python
def _patched_block_pool_get_cached_block(self, *args, **kwargs):
    """方案 D：从读取侧拦截 HBM 哈希命中。

    BlockPool.get_cached_block 是所有 HBM 哈希命中查找的唯一入口：
      - get_computed_blocks → find_longest_cache_hit → get_cached_block

    当开关开启时直接返回 None（未命中），让第二次请求完全走 SSD。
    比堵写入更彻底——即使 HBM 表有残留哈希也读不到。
    """
    if _disable_hbm_prefix_hash_enabled():
        # 开关开启：HBM 永远未命中，强制走 SSD
        return None
    # 开关关闭：走原始逻辑
    assert _ORIG_BLOCK_POOL_GET_CACHED_BLOCK is not None
    return _ORIG_BLOCK_POOL_GET_CACHED_BLOCK(self, *args, **kwargs)
```

**设计要点**：
1. **复用方案 B 的 `_disable_hbm_prefix_hash_enabled()`**——同一个开关。
2. **返回 `None`**：`get_cached_block` 的语义是"返回缓存的 block 或 None"，返回 None 表示未命中，调用方（`find_longest_cache_hit`）会 break 循环，返回空命中列表。
3. **`*args, **kwargs` 透传**：不关心签名变化。

### 8c. 在 `install_vllm_integration_monkeypatches` 里注册 patch

### 位置：紧跟方案 C 的 `cache_full_blocks` patch 之后

方案 C 改动后已有：

```python
        try:
            import vllm.v1.core.block_pool as block_pool_mod
            _ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS = block_pool_mod.BlockPool.cache_full_blocks
            block_pool_mod.BlockPool.cache_full_blocks = _patched_block_pool_cache_full_blocks
        except Exception:
            ...
```

在它之后（同一个 try 块内或新增一个 try 块）追加：

```python
        try:
            import vllm.v1.core.block_pool as block_pool_mod
            _ORIG_BLOCK_POOL_GET_CACHED_BLOCK = block_pool_mod.BlockPool.get_cached_block
            block_pool_mod.BlockPool.get_cached_block = _patched_block_pool_get_cached_block
        except Exception:
            if runtime_logging_enabled():
                logger.warning(
                    "TriAttention: could not patch BlockPool.get_cached_block "
                    "(vLLM version may differ). HBM hash read interception "
                    "may be incomplete.",
                    exc_info=True,
                )
```

**建议**：把方案 C 和方案 D 的 patch 放在**同一个 try 块**里，减少重复 import：

```python
        try:
            import vllm.v1.core.block_pool as block_pool_mod
            # 方案 C：堵写入
            _ORIG_BLOCK_POOL_CACHE_FULL_BLOCKS = block_pool_mod.BlockPool.cache_full_blocks
            block_pool_mod.BlockPool.cache_full_blocks = _patched_block_pool_cache_full_blocks
            # 方案 D：堵读取
            _ORIG_BLOCK_POOL_GET_CACHED_BLOCK = block_pool_mod.BlockPool.get_cached_block
            block_pool_mod.BlockPool.get_cached_block = _patched_block_pool_get_cached_block
        except Exception:
            if runtime_logging_enabled():
                logger.warning(
                    "TriAttention: could not patch BlockPool.cache_full_blocks "
                    "or get_cached_block (vLLM version may differ).",
                    exc_info=True,
                )
```

---

## 四、方案 D 的安全性分析

### 1. `get_cached_block` 返回 None 的语义

`get_cached_block` 返回 None 表示"该 hash 未在 HBM 缓存中找到"。调用方 `find_longest_cache_hit`（`FullAttentionManager`）遇到 None 会 break 循环，返回空命中列表。这是正常的"全 miss"行为，vLLM 设计上就支持。

### 2. 不影响 block 分配和 SSD

- block 分配走 `get_new_blocks`，不经过 `get_cached_block`。
- SSD（ascend_store）查命中走 `LookupKeyClient.lookup` → `m_store.exists`，不经过 HBM 的 `get_cached_block`。

### 3. 不影响第一次请求

第一次请求时 HBM 本就无命中（表空），`get_cached_block` 返回 None 是正常行为。方案 D 只是把"第一次无命中"扩展到"所有请求都无 HBM 命中"。

### 4. 与方案 A/B/C 叠加安全

- 方案 D 堵读取，方案 C 堵写入，两者正交，叠加安全。
- 方案 A 清理残留，与 D 叠加也安全（D 让残留读不到，A 清残留是额外保险）。

---

## 五、推荐配置：A + B + C + D 四管齐下

```bash
export TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=1
```

一个开关同时启用：
- **方案 B**：`allocate_slots` 设 `delay_cache_blocks=True`（堵 `allocate_slots` 内部写入）
- **方案 C**：`cache_full_blocks` no-op（堵所有写入路径）
- **方案 D**：`get_cached_block` 返回 None（堵所有读取路径，最彻底）
- **方案 A**：请求结束清理残留（兜底）

**方案 D 是最彻底的保险**：即使 A/B/C 有任何遗漏，D 让 HBM 永远查不到命中，第二次请求必然走 SSD。

---

## 六、如果方案 D 后精度仍不变的诊断步骤

如果方案 D（堵读取）后精度仍差 2%，说明**问题完全不在 HBM 哈希**，需要诊断其它原因：

### 6.1 确认 HBM 命中确实为 0

在 `KVCacheManager.get_computed_blocks`（vLLM 核心 `kv_cache_manager.py:207`）返回处加日志，确认 `num_new_computed_tokens == 0`。如果非 0，说明方案 D 没生效。

### 6.2 确认 SSD 命中正常

在 `ascend_store/pool_scheduler.py:84` `self.client.lookup(token_len, request.block_hashes)` 处加日志，确认第二次请求返回非 0（SSD 命中）。如果返回 0，说明 SSD 侧也没命中——问题在 SSD 数据或 `request.block_hashes`。

### 6.3 确认 `request.block_hashes` 未被压缩改写

在 `pool_scheduler.py:84` 打印 `len(request.block_hashes)` 和前几个 hash 值，对比第一次和第二次请求是否一致。如果压缩改写了 `block_hashes`，第二次请求的哈希与 SSD 里的不匹配，SSD 也不会命中。

### 6.4 确认 SSD 写入的 KV 是原始的而非压缩后的

TriAttention 压缩会重排 HBM 里的 KV。如果 SSD 写入发生在压缩之后（worker 侧 `save_kv_layer`），写入的可能是压缩后的 KV。第二次请求读到压缩后的 KV，与原始 prompt 不符，精度下降。

**检查点**：`ascend_store/kv_transfer.py:152-245` 的 `_handle_request` 写入时机——是在压缩前还是压缩后？如果压缩后，SSD 里的 KV 是错的，需要调整写入时机或跳过压缩请求的 SSD 写入。

### 6.5 确认 TriAttention 压缩本身是否改变计算结果

即使 KV 完全正确，TriAttention 的 token 级稀疏注意力可能与 base 的全注意力在数值上有微小差异。这是 TriAttention 的固有特性，非哈希问题。对比"关 prefix caching + 开 TriAttention"与"关 prefix caching + 关 TriAttention"的精度差，确认是否是压缩本身的精度损失。

---

## 七、方案 D 完整改动汇总验证清单（在方案 A+B+C 的基础上追加）

- [ ] **改动 8a**：`triattention/vllm/runtime/integration_monkeypatch.py` 顶部全局变量区有 `_ORIG_BLOCK_POOL_GET_CACHED_BLOCK: Callable[..., Any] | None = None`
- [ ] **改动 8b**：`triattention/vllm/runtime/integration_monkeypatch.py` 有 `_patched_block_pool_get_cached_block(self, *args, **kwargs)` 函数，开关开启时返回 None，复用 `_disable_hbm_prefix_hash_enabled()` 判定
- [ ] **改动 8c**：`install_vllm_integration_monkeypatches` 内注册了 `BlockPool.get_cached_block` 的 patch（建议与方案 C 的 `cache_full_blocks` patch 放同一 try 块）

---

## 八、本仓证据汇总（所有 HBM 读写路径）

### 写入路径（`cached_block_hash_to_block.insert`）

| 步骤 | 位置 | 本仓/核心 |
|------|------|----------|
| 唯一写入入口 | `BlockPool.cache_full_blocks` 第 285 行 `cached_block_hash_to_block.insert(...)` | vLLM 核心（`agent-tools/69eccfce-...txt:285`） |
| 上层调用 1 | `SingleTypeKVCacheManager.cache_blocks` → `block_pool.cache_full_blocks` | vLLM 核心（`agent-tools/` WebFetch `single_type_kv_cache_manager.py:184`） |
| 上层调用 2 | `KVCacheManager.cache_blocks` → `coordinator.cache_blocks` | vLLM 核心（`agent-tools/87420273-...txt:588`） |
| 上层调用 3 | `allocate_slots` → `coordinator.cache_blocks` | vLLM 核心（`agent-tools/87420273-...txt:461`） |
| 上层调用 4 | `recompute_scheduler.py:182,205` → `kv_cache_manager.cache_blocks` | 本仓 `other_code/...` |

### 读取路径（`cached_block_hash_to_block.get_one_block`）

| 步骤 | 位置 | 本仓/核心 |
|------|------|----------|
| 唯一读取入口 | `BlockPool.get_cached_block` 第 207 行 `cached_block_hash_to_block.get_one_block(...)` | vLLM 核心（`agent-tools/69eccfce-...txt:207`） |
| 上层调用 1 | `FullAttentionManager.find_longest_cache_hit` → `block_pool.get_cached_block` | vLLM 核心（WebFetch `single_type_kv_cache_manager.py`） |
| 上层调用 2 | `coordinator.find_longest_cache_hit` → 各 manager | vLLM 核心（`agent-tools/35721264-...txt:257`） |
| 上层调用 3 | `KVCacheManager.get_computed_blocks` → `coordinator.find_longest_cache_hit` | vLLM 核心（`agent-tools/87420273-...txt:233`） |

**结论**：写入唯一入口是 `cache_full_blocks`（方案 C 堵），读取唯一入口是 `get_cached_block`（方案 D 堵）。两者都 patch 后，HBM 哈希读写完全瘫痪，第二次请求必然走 SSD。
