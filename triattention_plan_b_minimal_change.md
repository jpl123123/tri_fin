# 方案 B 最小改动实施细节

> 目标：通过 `export TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=1` 启用"HBM 不写 prefix hash"（即对所有请求强制 `delay_cache_blocks=True`），不动 vLLM 核心、不动 vllm-ascend、不动 TriAttention 其它逻辑。
>
> 仅修改本仓 2 个文件，共 4 处改动。

---

## 改动文件清单

| 文件 | 改动数 | 改动性质 |
|------|--------|---------|
| `triattention/vllm/runtime/config.py` | 2 处 | 新增 1 个字段 + 1 个 env 解析 |
| `triattention/vllm/runtime/integration_monkeypatch.py` | 2 处 | 新增 1 个缓存辅助函数 + 修改 `_patched_kv_cache_allocate_slots` 判定 |

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

## 启用方式

启动 vllm serve 时加一个环境变量：

```bash
export ENABLE_TRIATTENTION=1
export TRIATTN_RUNTIME_KV_BUDGET=4096
# ... 其它 TriAttention 环境变量 ...
export TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=1   # ← 新增这一行

vllm serve Qwen3-32B \
    --enable-prefix-caching \
    ... 其它参数 ...
```

关闭则不 export 该变量（或 `export TRIATTN_RUNTIME_DISABLE_HBM_PREFIX_HASH=0`）。

---

## 改动汇总验证清单

改动完成后，用以下方式自检：

- [ ] `triattention/vllm/runtime/config.py` 第 102 行后有 `disable_hbm_prefix_hash: bool = False`
- [ ] `triattention/vllm/runtime/config.py` `from_env` 内有 `disable_hbm_prefix_hash=maybe_bool("DISABLE_HBM_PREFIX_HASH", cls.disable_hbm_prefix_hash)`
- [ ] `triattention/vllm/runtime/integration_monkeypatch.py` 顶部全局变量区有 `_DISABLE_HBM_PREFIX_HASH_CACHE: bool | None = None`
- [ ] `triattention/vllm/runtime/integration_monkeypatch.py` 有 `_disable_hbm_prefix_hash_enabled()` 函数
- [ ] `_patched_kv_cache_allocate_slots` 在原判定之前插入了 `disable_hbm_hash` 分支
- [ ] `disable_hbm_hash` 分支不改写 `num_computed_tokens`
- [ ] 原"压缩请求"分支保持不变

---

## 风险与限制

1. **方案 B 只跳过 HBM 写入，不清理已写入的哈希**。如果开关在请求 Prefill 中途生效，或第一次请求 Prefill 时哈希已写入（在压缩前），HBM 里仍可能残留。要彻底保证"第二次请求必走 SSD"，需配合方案 A（请求结束清理 HBM 残留哈希）。但若开关在进程启动时就 export，第一次请求 Prefill 期间就不会写 HBM 哈希，无残留问题。
2. **场景隔离靠用户自觉**：非 TriAttention 场景不要开此开关，否则会关掉该场景的 HBM prefix caching（SSD 仍工作，但 HBM 命中能力丧失）。
3. **同 batch 内请求间无法 HBM 命中**：开启后所有请求的 HBM 哈希表始终为空，同 batch 内不同请求无法通过 HBM 命中彼此的 KV。TriAttention 的 token 级重排本就使这种命中不可靠，可接受。
4. **未改 `_free_reclaimed_blocks` 的 `_evict_reclaimed_block_metadata`**：保留原样。开启方案 B 后，被回收的 block 本就没有 hash（因没写过），`_maybe_evict_cached_block` 会因 `block_hash is None` 直接 return，无副作用。

---

## 待运行环境验证的点

以下细节实现不在本仓，需运行环境（已装 vLLM 0.18.0）确认：

1. `delay_cache_blocks=True` 确实在 `cache_blocks` 之前 return（vLLM `kv_cache_manager.py:allocate_slots` 行为）。
2. 开关开启后，跑一个请求，检查 HBM `block_pool.cached_block_hash_to_block` 是否为空、SSD `CPUKVCacheManager.block_pool.cached_block_hash_to_block` 是否含该请求的块。
3. 第二次相同请求是否走 SSD 路径（connector `get_num_new_matched_tokens` 返回非 0）。
