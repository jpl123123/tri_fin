# TriAttention Prefix-Caching 方向1 调试断点（print 形式）

> 本分支 `debug/prefix-cache-direction1-print-bp` 仅用于**排查** TriAttention
> 开启后 Prefix-Caching 失效的根因，遵循根因报告中**方向 1** 的思路：
>
> > 让压缩回收的 block 保留 prefix-cache hash（不调用
> > `_maybe_evict_cached_block`）。
>
> 当前阶段**不修改任何原有代码逻辑**，只在关键路径上加入 print 形式的
> 断点，并用一个总开关 `TRIATTN_DEBUG_PREFIX_CACHE_TRACE` 统一门控。

## 一、使用方法

### 1.1 总开关

```bash
export TRIATTN_DEBUG_PREFIX_CACHE_TRACE=1   # 打开 print 断点（默认关闭）
```

- 未设置 / 设为 `0` 时，所有断点函数都会在第 1 行就 return，**对生产运行
  零开销、零行为影响**。
- 打开后，所有输出会以 `[TRIATTN-PCTRACE]` 前缀打印到 `stderr`，并立即
  flush，保证多进程（TP=4）下顺序可还原。

### 1.2 与原运行参数组合

```bash
export ENABLE_TRIATTENTION=1
export TRIATTN_RUNTIME_KV_BUDGET=4096
# ... 其余 TRIATTN_RUNTIME_* 参数同问题分析文档 ...
export TRIATTN_DEBUG_PREFIX_CACHE_TRACE=1   # ← 新增这一行即可

vllm serve Qwen3-32B \
    --max-model-len 40960 --block-size 128 \
    --enable-prefix-caching \
    ...
```

然后跑 aisbench 同一请求连续两次，观察 stderr 中 `[TRIATTN-PCTRACE]`
开头的日志。

## 二、断点列表与含义

所有断点实现在 `triattention/vllm/runtime/_prefix_cache_debug.py`，调用点
分布在 `scheduler.py` / `integration_monkeypatch.py` / `runner.py`。

| # | 断点名 | 调用位置 | 观测目标 |
|---|--------|----------|----------|
| 1 | `trace_evict_reclaimed_block` | `scheduler._evict_reclaimed_block_metadata` (enter/exit) | 每个 block 被 evict 前后的 `block_hash`。**方向1核心观测点**：当前行为 exit 时 hash→None；方向1修复后应保持不变。 |
| 2 | `trace_free_reclaimed_blocks` | `scheduler._free_reclaimed_blocks` (pre_evict / post_evict_pre_free / post_free) | 整批被回收 block 的 id 列表与 hash 列表。用于核对"被回收数 ≈ (20000 - kv_budget - reclaim_interval) / block_size"。 |
| 3 | `trace_reclaim_branch` | `scheduler._apply_compression_events` 5 个分支 | `skip_prefill_no_groups` / `synthesize_no_groups` / `explicit_groups` / `synthesize_missing_gids` / `skip_prefill_missing_gids`。用于确认"Prefill 刚结束、Decode 第一步"由哪个分支触发了大批 evict。 |
| 4 | `trace_allocate_slots_patch` | `integration_monkeypatch._patched_kv_cache_allocate_slots` | 是否对压缩过的请求设置 `delay_cache_blocks=True`（疑点 B：永久跳过 hash 提交）。 |
| 5 | `trace_worker_self_trigger` | `runner._supplement_worker_self_triggers` 4 个决策点 | 疑点 C：`defer_chunked_prefill` guard 何时失效，worker 何时自触发压缩。 |
| 6 | `trace_block_reuse_on_allocate` | `integration_monkeypatch._patched_kv_cache_allocate_slots` 末尾 | **方向1风险观测点**：被 TriAttention 回收过的 block id 被 vLLM 重新分配给其他请求时，打印其当前 `block_hash`，判断属于 A/B/C 哪种情况（见下）。 |
| 7 | `trace_protected_block_reuse_clear` | `integration_monkeypatch._patched_maybe_evict_cached_block`（保护块分支） | **路径 C 修复正确性探针**：保护块被复用时清 stale hash 的事件。`had_hash=True cleared=True` = 修复生效；`had_hash=True cleared=False` = 崩溃前兆（`cache_full_blocks` 即将 assert 失败）。120-prompt 负载下应大量出现前者、零后者。 |

### 2.1 断点 1 输出示例

```
[TRIATTN-PCTRACE] evict_reclaimed_block stage=enter block_id=42 block_hash=abc123... pool_id=0x7f... has_evict_fn=True
[TRIATTN-PCTRACE] evict_reclaimed_block stage=exit  block_id=42 block_hash=None  pool_id=0x7f... has_evict_fn=True
```

- `stage=enter` 的 `block_hash` 是 prompt 中后段 block 的内容 hash —— 正是
  第二次相同请求最需要命中的部分。
- `stage=exit` 的 `block_hash=None` 说明当前实现确实把 hash 从
  `cached_block` 反查表里清掉了。**这就是方向1要阻止的行为。**

### 2.2 断点 6（风险探针）输出解读

```
[TRIATTN-PCTRACE] block_reuse_on_allocate req_id=req-2 reused_block_id=42 current_block_hash=None (A=same-old/stale-risk, B=None/cleared-safe, C=new-hash/safe)
```

三种情况的含义（直接对应报告"方向1 风险"的缓解论证）：

- **A：`current_block_hash` 与旧的 prompt hash 相同**
  → 该物理 block 还没被新数据覆盖写，此时 `cached_block` 里的旧 hash 指向
  的内容**仍然正确**。命中是安全的。
- **B：`current_block_hash=None`**
  → vLLM `BlockPool` 在 `allocate_slots` 分配新块时已经清空了旧 hash。
  后续 `_maybe_cache_full_block` 会用新内容重新注册。这是**安全**的，也是
  报告里"缓解"段提到的机制。
- **C：`current_block_hash` 是一个全新的非 None hash**
  → vLLM 已经用新内容重新注册了 hash。也是**安全**的。

**只有在大量出现 A、并且随后该 block 被写入新内容、但 `cached_block` 里的
旧 hash 还没被刷新时**，才会真正触发报告里描述的"读到错误 KV"风险。本探针
的目的就是把这个时序暴露出来，便于后续判断方向1是否需要额外保护。

## 三、本次排查的预期观测顺序

对一个 20k prompt、kv_budget=4096 的请求，第二次相同请求前，stderr 里应
按时间顺序出现：

1. **Prefill 阶段**
   - `allocate_slots_patch will_delay_cache_blocks=False` 多次（正常注册
     prompt block hash）
2. **Prefill 结束、Decode 第一步**
   - `worker_self_trigger branch=worker_self_trigger_fired effective_kv≈20000
     threshold≈6144 ...`（疑点 C 实锤）
   - `reclaim_branch branch=synthesize_no_groups freed≈108 kept≈48 ...`
     （或 `explicit_groups`，取决于 worker 是否带 groups payload）
   - `evict_reclaimed_block stage=enter block_hash=<非None>` × ~108 次
   - `evict_reclaimed_block stage=exit  block_hash=None` × ~108 次
     （方向1要阻止的就是这一步）
   - `free_reclaimed_blocks stage=post_free block_ids=[...]`
3. **第二次相同请求 Prefill**
   - `allocate_slots_patch will_delay_cache_blocks=False`
   - 此时 `cached_block` 里只剩前 ~48 个 prompt block 的 hash，所以只能命中
     ~31%（与报告实测一致）。
4. **风险探针（如果后续有新请求复用了被回收的 block id）**
   - `block_reuse_on_allocate ... current_block_hash=<B 或 C>` 应占多数，
     佐证 vLLM 自己管理了 hash 生命周期。

## 四、代码改动清单

只新增 / 在原代码末尾插入只读 trace 调用，**未改动任何原有控制流**：

| 文件 | 改动 |
|------|------|
| `triattention/vllm/runtime/_prefix_cache_debug.py` | 新增：全部断点实现 + 总开关 |
| `triattention/vllm/runtime/scheduler.py` | import + 在 `_evict_reclaimed_block_metadata` / `_free_reclaimed_blocks` / `_apply_compression_events` 5 个分支插入 trace 调用 |
| `triattention/vllm/runtime/integration_monkeypatch.py` | import + 在 `_patched_kv_cache_allocate_slots` 4 条返回路径前插入 trace + 末尾风险探针 |
| `triattention/vllm/runtime/runner.py` | import + 在 `_supplement_worker_self_triggers` 4 个决策点插入 trace |

所有插入点都用 `# [PCTRACE]` 注释标记，便于 review 与回退。

## 五、下一步（不在本分支执行）

确认观测结果与报告一致后，在**新分支**上做方向1的最小修复：

- 在 `_evict_reclaimed_block_metadata` 中跳过 `_maybe_evict_cached_block`
  调用（保留 `cached_block` 里的 hash）。
- 用本分支的断点 6 风险探针验证 vLLM 在 `allocate_slots` 侧确实会清空 /
  重注册被复用 block 的 hash（即报告"缓解"段成立）。
- 若探针出现大量 case A 且后续被覆盖写，再考虑加一层"复用前校验"
  保护。
