# TriAttention Prefix-Caching 兼容性修复（方向 1 / 路径 C）— 特性移植说明

> 本文档用于把 **Path C（方向 1，evict-on-rewrite 做对）** 这一项特性从本仓库移植到其它基于同一初始版本（`2863884`）的分支。
>
> - **干净特性分支**：`feature/prefix-cache-direction1-clean-port`
> - **干净特性分支 HEAD commit**：`5a697a9`
> - **基线（初始 commit）**：`2863884`
> - **原研发分支（含调试与文档）**：`fix/prefix-cache-direction1-keep-hash-on-reclaim`，HEAD `76a6f43`
> - **远程**：`git@github.com:jpl123123/tri_fin.git`
>
> 干净分支相对初始 commit **仅 1 个 commit、3 个文件、+317/-4 行**，且不包含任何调试打印与调查文档。移植者可以直接 cherry-pick 或按本文档手工 apply。

---

## 一、一句话特性概述

让 TriAttention 在 Decode 阶段压缩回收物理 block 时 **保留 prefix-cache hash**，使第二次相同请求能命中完整 prompt 前缀；同时仍物理释放 block（内存不膨胀），并在 block 被复用写入新内容前 **无条件清掉 stale hash**（evict-on-rewrite），消除 `cache_full_blocks` 崩溃、stale hash 污染、反查表无界膨胀三个正确性/性能缺陷。

这是根因报告中 **方向 1** 的落地实现，采用 **路径 C（保护标记 + 物理释放 + evict-on-rewrite）** 框架。轻量版、路径 A（pin 不释放）、初版 evict-on-rewrite 均已验证失败（见原研发分支 `docs/debug-prefix-cache-direction1-progress.md` 第五节），不要重复尝试。

---

## 二、改动文件清单（仅这 3 个）

| 文件 | 改动量 | 性质 |
|---|---|---|
| `triattention/vllm/runtime/config.py` | +19 | 新增配置项 + env 加载 |
| `triattention/vllm/runtime/scheduler.py` | +227 / −4 | 路径 C 核心逻辑 + 4 个保护标记 helper + 1 个复用清 hash helper |
| `triattention/vllm/runtime/integration_monkeypatch.py` | +71 | 新增 `_patched_maybe_evict_cached_block` + BlockPool patch 安装 |

**不包含**（均为调试/调查产物，对生产行为零影响，移植时不要带）：

| 文件 | 原研发分支行数 | 为什么不带 |
|---|---|---|
| `triattention/vllm/runtime/_prefix_cache_debug.py` | +427 | 纯只读诊断模块，所有调用点在 `TRIATTN_DEBUG_PREFIX_CACHE_TRACE` 未设时为 no-op |
| `triattention/vllm/runtime/runner.py` | +70 | 全部是 `_pctrace_self_trigger` 调用（print 断点） |
| `scheduler.py` / `integration_monkeypatch.py` 中的 `_pctrace_*` 调用 | （含在上面） | print 断点，no-op |
| `docs/debug-prefix-cache-direction1.md` | +138 | 断点使用手册 |
| `docs/debug-prefix-cache-direction1-progress.md` | +453 | 调查过程与结论记录 |

---

## 三、移植方法

### 方法 A：cherry-pick（推荐，目标分支基线 == `2863884`）

```bash
# 在目标仓库
git fetch origin
git checkout <your-target-branch>
git cherry-pick 5a697a9
```

若目标分支相对 `2863884` 有其它改动且与这 3 个文件冲突，用方法 B。

### 方法 B：手工 apply（目标分支已偏离 `2863884`）

```bash
git fetch origin feature/prefix-cache-direction1-clean-port
git diff 2863884 5a697a9 -- \
    triattention/vllm/runtime/config.py \
    triattention/vllm/runtime/scheduler.py \
    triattention/vllm/runtime/integration_monkeypatch.py \
    | git apply --3way
```

### 方法 C：按第四节逐文件粘贴

当目标分支的 `config.py` / `scheduler.py` / `integration_monkeypatch.py` 已被改得面目全非时，按第四节给出的"插入点 + 代码片段"手工合并。

---

## 四、逐文件改动详解

### 4.1 `triattention/vllm/runtime/config.py`

**改动 1：新增配置字段**（在 `max_compressions_per_step_on_ascend` 之后、`sparse_stats_path` 之前）

```python
    force_eager_multi_req_on_ascend_effective_overrides: bool = False
    max_compressions_per_step_on_ascend: int = 4

    # Direction-1 fix for Prefix-Caching compatibility.
    # When True, _evict_reclaimed_block_metadata does NOT call
    # BlockPool._maybe_evict_cached_block, so the reclaimed blocks keep their
    # prefix-cache hash in the cached_block reverse-lookup table. The second
    # identical request can then hit the full prompt prefix instead of only
    # the (kv_budget + reclaim_interval)/block_size retained blocks.
    # Risk: a reused physical block may be overwritten with new data while its
    # stale hash still lives in cached_block. Mitigation: vLLM BlockPool itself
    # clears stale hash and re-registers on allocate_slots, so as long as
    # TriAttention does not actively evict, vLLM manages the hash lifecycle.
    # Verified safe under bs=7/20k-prompt/kv_budget=4096/gpu_mem_util=0.9
    # (block_reuse_on_allocate probe was empty - free pool always sufficient).
    # Set to False to restore the original evict-on-reclaim behavior.
    keep_prefix_cache_hash_on_reclaim: bool = True

    # Optional TriAttention-style scoring path (used by runtime hook when enabled).
    sparse_stats_path: Path | None = None
```

**改动 2：env 加载**（在 `from_env` 类方法里，`max_compressions_per_step_on_ascend=maybe_int(...)` 之后、`sparse_stats_path=sparse_stats_path_candidate` 之前）

```python
            max_compressions_per_step_on_ascend=maybe_int(
                "MAX_COMPRESSIONS_PER_STEP_ON_ASCEND",
                cls.max_compressions_per_step_on_ascend,
            ),
            keep_prefix_cache_hash_on_reclaim=maybe_bool(
                "KEEP_PREFIX_CACHE_HASH_ON_RECLAIM",
                cls.keep_prefix_cache_hash_on_reclaim,
            ),
            sparse_stats_path=sparse_stats_path_candidate,
```

> `maybe_bool` 是 `from_env` 内的嵌套函数，初始版本已存在，无需新增。

### 4.2 `triattention/vllm/runtime/scheduler.py`

**改动 1：在文件顶部 import 块之后（`from .version import RUNTIME_BUILD_ID` 之后），插入开关缓存 + 5 个 helper 函数**

把原始的：

```python
from .version import RUNTIME_BUILD_ID

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

整体替换为：

```python
from .version import RUNTIME_BUILD_ID


# Direction-1 fix switch cache.  Read once per process from env to avoid
# hitting os.environ on every reclaim (which fires per-block, per-request,
# per-step).  Mirrors the _ASYNC_BOUNDARY_ENABLED_CACHE pattern in
# integration_monkeypatch.py.  Set TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=0
# to restore the original evict-on-reclaim behavior.
_KEEP_HASH_ON_RECLAIM_CACHE: bool | None = None


def _keep_prefix_cache_hash_on_reclaim() -> bool:
    """Return whether Direction-1 fix is active (keep hash, don't evict)."""
    global _KEEP_HASH_ON_RECLAIM_CACHE
    if _KEEP_HASH_ON_RECLAIM_CACHE is None:
        try:
            cfg = TriAttentionRuntimeConfig.from_env()
            _KEEP_HASH_ON_RECLAIM_CACHE = bool(
                getattr(cfg, "keep_prefix_cache_hash_on_reclaim", True)
            )
        except Exception:
            _KEEP_HASH_ON_RECLAIM_CACHE = True
    return _KEEP_HASH_ON_RECLAIM_CACHE


def _evict_reclaimed_block_metadata(block_pool: Any, block: Any) -> None:
    """Best-effort clear of prefix-cache metadata before reusing a block.

    Direction-1 fix: when ``keep_prefix_cache_hash_on_reclaim`` is True
    (default), this function is a no-op — the block's prefix-cache hash is
    preserved in ``BlockPool.cached_block`` so the next identical request can
    hit the full prompt prefix.
    """
    if block_pool is None or block is None:
        return
    block_hash = getattr(block, "block_hash", None)
    if block_hash is None:
        return

    if _keep_prefix_cache_hash_on_reclaim():
        return

    maybe_evict = getattr(block_pool, "_maybe_evict_cached_block", None)
    if callable(maybe_evict):
        maybe_evict(block)


def _free_reclaimed_blocks(manager: Any, removed_blocks: list[Any]) -> bool:
    """Free reclaimed tail blocks after applying Path-C hash protection.

    Path C: reclaimed blocks are returned to the free pool with their
    prefix-cache hash protected (kept), but the physical block is still
    released (ref_cnt→0) so KV memory stays near baseline.
    """
    if not removed_blocks:
        return False
    block_pool = getattr(manager, "block_pool", None)

    if _keep_prefix_cache_hash_on_reclaim():
        for block in removed_blocks:
            _mark_block_hash_protected(block)
    else:
        for block in removed_blocks:
            _evict_reclaimed_block_metadata(block_pool, block)

    if block_pool is None:
        return False
    block_pool.free_blocks(reversed(removed_blocks))
    return True


_TRIATTENTION_HASH_PROTECTED_ATTR = "_triattention_hash_protected"


def _mark_block_hash_protected(block: Any) -> None:
    if block is None:
        return
    try:
        setattr(block, _TRIATTENTION_HASH_PROTECTED_ATTR, True)
    except Exception:
        pass


def _is_block_hash_protected(block: Any) -> bool:
    if block is None:
        return False
    return bool(getattr(block, _TRIATTENTION_HASH_PROTECTED_ATTR, False))


def _clear_block_hash_protection(block: Any) -> None:
    if block is None:
        return
    try:
        if hasattr(block, _TRIATTENTION_HASH_PROTECTED_ATTR):
            delattr(block, _TRIATTENTION_HASH_PROTECTED_ATTR)
    except Exception:
        try:
            setattr(block, _TRIATTENTION_HASH_PROTECTED_ATTR, False)
        except Exception:
            pass


def _evict_protected_block_hash(block_pool: Any, block: Any) -> bool:
    """Fully clear a protected block's prefix-cache hash at reuse time.

    Unconditionally calls reset_hash() so block.block_hash is guaranteed None
    on return, regardless of cached_block_hash_to_block dict state. This is
    what makes Path C safe vs upstream _maybe_evict_cached_block (which returns
    False without resetting when the hash key is absent — vLLM PR #44237 bug).
    """
    if block is None:
        return False
    _clear_block_hash_protection(block)

    block_hash = getattr(block, "block_hash", None)
    if block_hash is None:
        return False

    cache = getattr(block_pool, "cached_block_hash_to_block", None)
    if cache is not None:
        try:
            blocks_by_id = cache.get(block_hash)
            if blocks_by_id is not None:
                block_id = getattr(block, "block_id", None)
                blocks_by_id.pop(block_id, None)
                if len(blocks_by_id) == 0:
                    del cache[block_hash]
        except Exception:
            pass

    reset_hash = getattr(block, "reset_hash", None)
    if callable(reset_hash):
        try:
            reset_hash()
        except Exception:
            try:
                setattr(block, "_block_hash", None)
            except Exception:
                pass
    else:
        try:
            setattr(block, "_block_hash", None)
        except Exception:
            pass
    return True
```

> 上面的代码块已精简掉原研发分支里的大段调查性 docstring（保留行为说明），完整 docstring 见干净分支 `5a697a9`。移植时建议直接用干净分支的版本（docstring 更全，便于后续维护）。

**改动 2：在 `TriAttentionScheduler` 的请求结束清理路径（`scheduler_output.finished_req_ids` 循环后、`return outputs` 前）加一行注释**

```python
        for req_id in scheduler_output.finished_req_ids:
            self._prefill_lens.pop(req_id, None)
            self._prefill_compression_counts.pop(req_id, None)
            self._long_context_guard_logged.discard(req_id)
            self._effective_len_tracker.remove_request(req_id)
            # Path C: no per-request pin cleanup needed — blocks were already
            # returned to the free pool by _free_reclaimed_blocks at reclaim
            # time (with their hash protected from lazy evict).  Nothing to do
            # here; vLLM's normal request-finish free handles the retained
            # blocks (req_to_blocks[req_id]).
        return outputs
```

> 这只是注释，说明为何此处无需 pin 清理。不加这行注释不影响行为，但建议加上以避免后续维护者误以为漏了清理。

### 4.3 `triattention/vllm/runtime/integration_monkeypatch.py`

**改动 1：新增 import + 全局变量**（在文件顶部 import 块末尾、`_PATCHED = False` 之前）

```python
    should_install_triattention_runner_proxy,
)
# Path C: block hash-protection helpers used by the patched
# BlockPool._maybe_evict_cached_block to clear stale hashes on reuse of
# protected blocks (evict-on-rewrite).
from .scheduler import (  # noqa: E402
    _evict_protected_block_hash as _pctrace_evict_protected,
    _is_block_hash_protected as _pctrace_is_protected,
)

_PATCHED = False
```

> 别名保留 `_pctrace_*` 前缀是为了与原研发分支的命名一致、便于对照。它们与调试打印无关，是真正的功能 helper。

在全局变量区（`_ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE` 之后）加一行：

```python
_ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE: Callable[..., Any] | None = None
_ORIG_MAYBE_EVICT_CACHED_BLOCK: Callable[..., Any] | None = None
_DEFER_PREFILL_BOUNDARY_CACHE: bool | None = None
```

**改动 2：新增 `_patched_maybe_evict_cached_block` 函数**

在 `_patched_kv_cache_allocate_slots` 函数之后、`_scheduler_output_has_compression_boundary` 之前插入：

```python
def _patched_maybe_evict_cached_block(self: Any, block: Any) -> Any:
    """Path C: clear stale hash on reuse of TriAttention hash-protected blocks.

    Protected blocks had their hash deliberately kept at reclaim time. Now the
    block is being pulled for new content, so the old hash is definitively
    stale — call _evict_protected_block_hash to fully clear it (reset_hash() +
    remove reverse-lookup entry) BEFORE the block is overwritten. This is the
    "evict-on-rewrite" Path-C completion.

    Non-protected blocks fall through to the original upstream evict unchanged.
    """
    if _pctrace_is_protected(block):
        return _pctrace_evict_protected(self, block)
    assert _ORIG_MAYBE_EVICT_CACHED_BLOCK is not None
    return _ORIG_MAYBE_EVICT_CACHED_BLOCK(self, block)
```

**改动 3：在 `install_vllm_integration_monkeypatches` 里安装 BlockPool patch**

先在函数顶部的 `global` 声明里加一行：

```python
    global _PATCHED, _ORIG_SCHED_INIT, _ORIG_SCHED_SCHEDULE, _ORIG_SCHED_UPDATE_FROM_OUTPUT
    global _ORIG_WORKER_INIT_DEVICE, _ORIG_WORKER_EXECUTE_MODEL
    global _ORIG_KVCACHE_ALLOCATE_SLOTS, _ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE
    global _ORIG_MAYBE_EVICT_CACHED_BLOCK
    global _PATCHED_SCHEDULER_ACTIVE, _PATCHED_WORKER_ACTIVE
```

然后在 `EngineCore.step_with_batch_queue = _patched_engine_core_step_with_batch_queue` 之后、`if patch_worker:` 之前插入：

```python
        _ORIG_ENGINE_CORE_STEP_WITH_BATCH_QUEUE = EngineCore.step_with_batch_queue
        EngineCore.step_with_batch_queue = _patched_engine_core_step_with_batch_queue

        # Path C: patch BlockPool._maybe_evict_cached_block to skip lazy evict
        # on TriAttention hash-protected blocks, so their prefix-cache hash
        # survives in cached_block for the next identical request to match.
        try:
            from vllm.v1.core.block_pool import BlockPool as _V1BlockPool
            _ORIG_MAYBE_EVICT_CACHED_BLOCK = _V1BlockPool._maybe_evict_cached_block
            _V1BlockPool._maybe_evict_cached_block = _patched_maybe_evict_cached_block
            if runtime_logging_enabled():
                logger.info(
                    "TriAttention Path-C: patched BlockPool._maybe_evict_cached_block "
                    "to skip lazy evict on hash-protected blocks"
                )
        except (ImportError, AttributeError) as _e:
            _ORIG_MAYBE_EVICT_CACHED_BLOCK = None
            if runtime_logging_enabled():
                logger.warning(
                    "TriAttention Path-C: could not patch "
                    "BlockPool._maybe_evict_cached_block (%s); "
                    "hash protection inactive, falling back to default evict",
                    _e,
                )

    if patch_worker:
        _install_worker_patches(Worker)
```

---

## 五、配置开关

```bash
# 启用路径 C（默认）
export TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=1

# 关闭，回到原始 evict-on-reclaim 行为（用于 A/B 对比）
export TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=0
```

默认 `1`（启用）。环境变量前缀 `TRIATTN_RUNTIME_` 由 `config.py` 的 `from_env` 自动补全。

---

## 六、运行时行为（移植后会发生什么）

1. **Prefill 阶段**：不变。prompt block 正常注册 hash（`will_delay_cache_blocks=False`）。
2. **Decode 阶段触发压缩回收**（`_free_reclaimed_blocks`）：
   - 给每个被回收 block 打 `_triattention_hash_protected` 标记；
   - **不**调 `_evict_reclaimed_block_metadata`（保留 hash）；
   - 调 `block_pool.free_blocks(reversed(removed_blocks))` 把物理块归还 free pool（`ref_cnt`→0，内存释放）。
3. **第二次相同请求到来**：通过 `_get_prompt_block_ids` + `BlockPool.touch`（`ref_cnt` 0→1）把受保护块从 free queue 移出，命中完整 prompt 前缀，跳过 prefill。
4. **受保护块被新请求复用写入新内容时**（`get_new_blocks` → `_maybe_evict_cached_block`）：
   - patch 检测到 `_triattention_hash_protected` 标记 → 调 `_evict_protected_block_hash`：
     - 清保护标记；
     - 从 `cached_block_hash_to_block` 删条目（处理 `defaultdict(dict)` shape）；
     - **无条件** `block.reset_hash()`（保证 `block.block_hash` 为 None）；
   - 随后 `cache_full_blocks` 注册新 hash 时 `assert blk.block_hash is None` 通过。

---

## 七、适用场景与已知局限（移植前必读）

### 7.1 适用的负载

- **有重复请求**的工作负载（同一 prompt 多次打、多轮对话、few-shot 批量推理等）：第二次相同请求能命中完整 prompt 前缀，TTFT 大幅下降。
- 简单测试验证场景：20k prompt、bs7、kv_budget=4096、两次相同请求 → 第二次 100% 命中，TTFT ~1000ms（基线 329ms），内存 ~21%。

### 7.2 不适用 / 无收益的负载

- **120 条不同 prompt、每条只来一次**的工作负载：没有重复请求就没有命中窗口，路径 C 的"留 hash 给第二次相同请求命中"机制结构上不触发。此负载下的命中收益只能来自共享 system prompt 前缀（走 vLLM 原生 PC），与路径 C 无关。
- 此负载下路径 C 的任务是 **不拖后腿**（消除 stale hash 字典膨胀，让 PC ON 至少持平 PC OFF），而非产生命中收益。

### 7.3 已知风险（已闭环）

| 风险 | 闭环方式 |
|---|---|
| 受保护块被复用时 `cache_full_blocks` 的 `assert blk.block_hash is None` 崩溃 | `_evict_protected_block_hash` 无条件 `reset_hash()`，保证 `block.block_hash` 为 None |
| stale hash 残留指向已覆盖 block（方向 1 风险） | 复用时同步清 `cached_block_hash_to_block` 条目 |
| 反查表无界膨胀（不同 prompt 累积） | 复用时 `del cache[block_hash]`（条目空时删 key） |
| 上游 `_maybe_evict_cached_block` 在 key 不在表里时返 False 不 reset（vLLM PR #44237 bug） | `_evict_protected_block_hash` 无条件 reset，绕过该 bug |

### 7.4 未验证项

- 干净分支 `5a697a9` 的代码已通过 AST 语法检查与结构校验，但 **未在 NPU 环境实测**（本机无 NPU）。
- 建议移植后在目标环境跑两组测试：
  1. **简单测试**：20k prompt、bs7、kv_budget=4096、两次相同请求，验证第二次命中 100%、内存 ~21%。
  2. **回归测试**：120 条不同 prompt，验证 PC ON duration 不显著劣于 PC OFF（消除膨胀后应持平或略好）。

---

## 八、与原研发分支的对应关系

| 干净分支 commit | 对应原研发分支 commit | 说明 |
|---|---|---|
| `5a697a9`（唯一 commit） | `3743257` + `4edd7e7`（部分）+ `5a917c4` + `76a6f43` | 把 4 个 `fix:` commit 的最终功能状态合并为 1 个 commit，剔除所有 `debug:` / `docs:` commit 的内容 |

原研发分支的 17 个 commit 里：
- **4 个 `fix:` commit** 是功能演进（轻量版 → 路径 A → 路径 C 初版 → evict-on-rewrite 做对）。干净分支只取最终状态（`76a6f43` 的功能逻辑），不保留中间失败版本。
- **1 个 `debug:` commit**（`780b8bb`）是 print 断点 + `_prefix_cache_debug.py`，全部不带。
- **12 个 `docs:` commit** 是调查文档迭代，全部不带（文档本身也不带）。

---

## 九、移植后验证清单

- [ ] `python3 -c "import ast; ast.parse(open('triattention/vllm/runtime/config.py').read()); ast.parse(open('triattention/vllm/runtime/scheduler.py').read()); ast.parse(open('triattention/vllm/runtime/integration_monkeypatch.py').read()); print('OK')"` 通过
- [ ] 三个文件中无 `_prefix_cache_debug` / `_pctrace_alloc` / `_pctrace_reuse` / `_pctrace_protected_clear` / `_pctrace_maybe_trace_reuse` / `trace_allocate_slots_patch` / `trace_block_reuse_on_allocate` / `trace_protected_block_reuse_clear` / `trace_worker_self_trigger` / `trace_evict_reclaimed_block` / `trace_free_reclaimed_blocks` / `trace_reclaim_branch` / `announce_active` / `record_freed_block_ids` 等 debug 符号
- [ ] `triattention/vllm/runtime/_prefix_cache_debug.py` 不存在
- [ ] `docs/debug-prefix-cache-direction1*.md` 不存在（除非移植者主动保留作为参考）
- [ ] `runner.py` 未被本特性改动（若目标分支的 `runner.py` 有 print 断点，那是 debug 残留，应清除）
- [ ] 简单测试：20k bs7 两次相同请求，第二次命中 100%、内存 ~21%
- [ ] 回归测试：120 条不同 prompt，PC ON duration 不显著劣于 PC OFF
- [ ] `export TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=0` 后行为回到原始 evict-on-reclaim（A/B 开关有效）

---

## 十、常见问题

**Q1：为什么别名用 `_pctrace_*` 前缀？是不是漏清了 debug 代码？**

不是。`_pctrace_evict_protected` 和 `_pctrace_is_protected` 是从 `scheduler.py` import 的 `_evict_protected_block_hash` 和 `_is_block_hash_protected` 的别名，是真正的功能 helper。前缀保留是为了与原研发分支命名一致、便于 diff 对照。它们与 print 断点无关。

**Q2：`runner.py` 在原研发分支有 +70 行改动，为什么干净分支不带？**

那 70 行全部是 `trace_worker_self_trigger`（即 `_pctrace_self_trigger`）的调用，是 print 断点，在 `TRIATTN_DEBUG_PREFIX_CACHE_TRACE` 未设时为 no-op。不影响生产行为，故不带。

**Q3：能不能只移植 `5a917c4`（路径 C 初版）而不带 `76a6f43`（evict-on-rewrite）？**

**不能**。`5a917c4` 有确认的正确性缺陷：`_patched_maybe_evict_cached_block` 对保护块只清标记不调 `reset_hash`，会触发 `cache_full_blocks` 崩溃或 stale hash 污染。必须带 `76a6f43` 的 evict-on-rewrite 修复。干净分支 `5a697a9` 已经是修复后的最终状态。

**Q4：目标分支的 vLLM 版本不是 v0.10.2 / vllm-ascend v0.18.0 怎么办？**

路径 C 依赖以下 vLLM 内部接口（均在 v0.10.2 验证过）：
- `vllm.v1.core.block_pool.BlockPool._maybe_evict_cached_block`
- `BlockPool.cached_block_hash_to_block`（`defaultdict(dict)`，shape `{BlockHashWithGroupId: {block_id: KVCacheBlock}}`）
- `KVCacheBlock.block_hash` / `block_id` / `reset_hash()` / `_block_hash`
- `BlockPool.free_blocks` / `get_new_blocks`

若目标 vLLM 版本这些接口有变，需对照调整。已知 vLLM PR #44237 修了 `_maybe_evict_cached_block` 在 key 不在表里时不 reset 的 bug，若目标版本已含该修复，`_evict_protected_block_hash` 的"无条件 reset"仍是安全 superset，无需改动。

**Q5：移植后 PC ON 还是比 PC OFF 慢怎么办？**

先确认负载是否有重复请求（见 7.2）。若无重复请求，PC ON 比 PC OFF 慢是 TriAttention+PC 的固有开销（压缩计算 + input_patch），不是路径 C 的 bug。若有重复请求仍慢，检查 `_patched_maybe_evict_cached_block` 是否成功安装（日志会有 `TriAttention Path-C: patched BlockPool._maybe_evict_cached_block`），以及受保护块是否在第二次请求 touch 前就被改写（可临时加回原研发分支的 print 断点排查）。

---

## 十一、参考

- 干净特性分支：`origin/feature/prefix-cache-direction1-clean-port`（HEAD `5a697a9`）
- 原研发分支（含调试与完整调查文档）：`origin/fix/prefix-cache-direction1-keep-hash-on-reclaim`（HEAD `76a6f43`）
- 原研发分支调查文档：`docs/debug-prefix-cache-direction1-progress.md`（453 行，含所有失败版本、机理定位、上游 vLLM v0.10.2 源码交叉验证结论）
- 原研发分支断点手册：`docs/debug-prefix-cache-direction1.md`（138 行）
- 初始 commit：`2863884`
- 远程：`git@github.com:jpl123123/tri_fin.git`
