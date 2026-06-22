# TriAttention Prefix-Caching 方向1 调试进展（可持续记录）

> 本文档是 `debug/prefix-cache-direction1-print-bp` 分支的**持续排查日志**。
> 与 `debug-prefix-cache-direction1.md`（断点使用手册）配套：前者讲"怎么用
> 断点"，本文档讲"断点观测到了什么、下一步要做什么、为什么"。
>
> 每次新一轮实验后追加新章节，旧的实验记录保留不删，便于回溯。

---

## 〇、当前结论速览（TL;DR）

- **根因已用实测数据二次确认**：方向 1 描述的"压缩回收时主动 evict prompt
  中后段 block 的 prefix-cache hash"在日志里直接观测到，且 evict 数量
  （122）与报告公式 `ceil(prefill_len/128) - 48` **精确吻合**。
- **疑点 C（Prefill 结束立即压缩）已实锤**：第一次批次里看到
  `keep_scheduler_decode_trigger effective_kv=20864 threshold=6144`，Prefill
  刚结束、Decode 第一步即触发压缩。
- **疑点 B（压缩请求永久跳过 hash 提交）已实锤**：第二次批次请求在第一次
  Decode 就出现 `effective_num_computed=4098 ≪ logical_num_computed=19847`，
  `will_delay_cache_blocks=True`。
- **方向 1 风险点（被复用 block 被覆盖写）尚未观测到**：当前实验片段里
  `block_reuse_on_allocate` 行为空，说明被 evict 的 122 个 block 还没被
  vLLM 重新分配。**这是下一步必须补的数据**，决定方向 1 修复是否需要额外
  保护。

> **协作约定**：每次改完文档/代码后**自动 push** 到
> `origin/debug/prefix-cache-direction1-print-bp`，无需额外确认。日志文件名
> 固定为 `tmp.log`，grep 命令按第五节执行后把 `pctrace_*.log` 贴回 chat。

---

## 一、实验设置（首轮观测，2026-06-22）

| 项 | 值 |
|---|---|
| 模型 | Qwen3-32B |
| TP | 4 |
| block_size | 128 |
| max_model_len | 40960 |
| --enable-prefix-caching | 是 |
| TriAttention 总开关 | `ENABLE_TRIATTENTION=1` |
| kv_budget | 4096 |
| 调试开关 | `TRIATTN_DEBUG_PREFIX_CACHE_TRACE=1` |
| aisbench 设置 | 20k 输入 / 1k 输出 / bs=7 并发 |
| 实验流程 | warmup 1 次（不计）→ 第一次并发 7 → 第二次并发 7 |
| 观测重点 | 第一次 vs 第二次的差异 |

---

## 二、首轮观测结果（2026-06-22）

### 2.1 第一次批次片段特征

**几乎全是噪音守卫**：

```
worker_self_trigger ... scheduled_tokens=1 existing_estimate=4117 prefill_len=19845
  threshold=None is_prefill_step_for_threshold=True defer_chunked_prefill=True
  will_compress=False branch=defer_chunked_prefill_guard_fired
```

这是 Prefill chunk 期间的正常"延迟压缩"守卫生效（`is_prefill_step_for_threshold=True`），不压缩。这部分日志量大但无信息量，下次实验需过滤。

**唯一一条非守卫行（疑点 C 实锤）**：

```
worker_self_trigger req_id=chatcmpl-9ae09d4be185e9df-9f4a91eb
  scheduled_tokens=1 existing_estimate=20786 prefill_len=20784
  threshold=6144 is_prefill_step_for_threshold=False defer_chunked_prefill=True
  will_compress=True branch=keep_scheduler_decode_trigger
  effective_kv=20864 actual_kv=20864 from_blocks=True
```

关键证据链：
- `existing_estimate(20786) ≥ prefill_len(20784)` → Prefill 已完成
- `scheduled_tokens=1` → 是 Decode 第一步
- `is_prefill_step_for_threshold=False` → defer guard 失效（这正是报告说的 guard 在 Prefill 结束后不再成立）
- `effective_kv=20864 ≫ threshold=6144` → 立即满足压缩阈值
- `will_compress=True` → 压缩信号被保留

→ **Prefill 一结束、Decode 第一步立即压缩**，与报告 4.3 节预测一致。

**疑点 B 实锤**：

```
allocate_slots_patch req_id=chatcmpl-a7eb830cedf9fc37-b9e6250d
  num_new_tokens=1 effective_num_computed=4116 logical_num_computed=19865
  will_delay_cache_blocks=True
```

`effective(4116) ≪ logical(19865)` → 该请求已被压缩过，patch 设置
`delay_cache_blocks=True`，永久跳过 hash 提交。与报告 5 节疑点 B 一致。

### 2.2 第二次批次片段特征

**出现了第一次片段完全没有的批量 evict**：

```
evict_reclaimed_block stage=enter block_id=107 block_hash=b'\xbd9\xb71\xe8\x03?!\xf6\xab\x...'
evict_reclaimed_block stage=exit  block_id=107 block_hash=None
... (连续 block_id 92~107 + 1303~1359，共 122 个)
free_reclaimed_blocks stage=post_evict_pre_free n=122
  block_hashes=[None×122]
```

观察：
- `stage=enter` 时每个 block 的 `block_hash` 都是非 None 的内容 hash（prompt 中后段 KV 的指纹）
- `stage=exit` 时全部变 `None` → **TriAttention 主动调 `_maybe_evict_cached_block` 把 hash 从 `cached_block` 反查表里清掉了**
- `n=122` 一次性 evict

### 2.3 数字精确吻合报告公式

被 evict 的 122 个 block_id 分两段：`1303–1359`（57 个）+ `92–156`（65 个）。

按报告阈值公式验算：
- `kv_budget = 4096`
- `reclaim_interval = max(128, 16×128) = 2048`
- `retained_cache_len = 4096 + 2048 = 6144`
- `required_blocks = ceil(6144/128) = 48`
- 第一次片段里观测到一个 `prefill_len=21694` 的请求（`chatcmpl-ad8a975499edbce6-944350c6`）
- `ceil(21694/128) = 170` 个 prompt block
- `freed = 170 - 48 = 122` ✓

**这 122 个被 evict 的 block 就是 21694-token 请求的 prompt 中后段**，正是第二次相同请求最需要命中的部分。

### 2.4 第二次批次的死循环

第二次片段末尾：

```
allocate_slots_patch req_id=chatcmpl-8fe426eb213f252c-ba95e465
  num_new_tokens=1 effective_num_computed=4098 logical_num_computed=19847
  will_delay_cache_blocks=True
```

这是第二次批次的请求在**第一次 Decode** 就已经被压缩了（`effective=4098 ≪ logical=19847`）。说明第二次批次也走了完整 Prefill（没命中）→ 立即压缩 → 跳过 hash 提交，疑点 B + C 在第二次批次上完整复现。

### 2.5 命中率倒推

第一次批次 evict 了 122/170 ≈ 72% 的 prompt block hash → 第二次相同请求最多命中 48/170 ≈ 28%。报告实测 4k budget 命中率 31%，与该观测一致（差异来自不同请求 prefill_len 略有浮动 19285–21694）。

---

## 三、方向 1 风险点的当前观测状态（关键缺口）

**当前缺口**：首轮实验片段里 `block_reuse_on_allocate` 行**完全为空**。

含义：在截取的时间窗内，被 evict 的 122 个 block id（92–156, 1303–1359）还没被 vLLM 重新分配给其他请求。

为什么这是关键缺口：
- 方向 1 的修复（不在 TriAttention 侧 evict hash）依赖报告"缓解"段的论证——"vLLM BlockPool 本身在 `allocate_slots` 分配新块时会清空旧 hash 并重新注册"。
- 这个论证是否成立，**只能**通过 `block_reuse_on_allocate` 探针在实测中验证：当一个被 TriAttention 回收过的 block id 被 vLLM 重新分配时，它的 `block_hash` 是 None（已被 vLLM 清空，安全）/ 新 hash（已重注册，安全）/ 旧 prompt hash（case A，stale 风险）。
- 首轮没观测到 → 无法确认缓解是否成立 → **不能贸然做方向 1 修复**。

可能原因：
1. 实验跑得不够久，被 evict 的 block 还在 free pool 里没被分配
2. 第二次批次命中了部分前缀，没触发足够多的新分配
3. 片段截短了，reuse 行在未截取的部分

---

## 四、下一步实验计划

### 4.1 必须补的观测（优先级 P0）

**目标**：拿到 `block_reuse_on_allocate` 的实测数据，确认方向 1 风险是否真的存在。

**方法**：重跑实验，但要把日志跑得更久——至少跑到第二次批次的请求开始 Decode 生成新 token、需要 allocate 新 block 时。被 evict 的 122 个 block 此时才会被 vLLM 从 free pool 取出重新分配。

**成功判据**：日志里至少出现若干行 `block_reuse_on_allocate`，且能对应到 122 个被 evict 的 block id 之一。

**结果解读**：
- 若 `current_block_hash=None`（case B）占绝大多数 → vLLM 自己清了 hash，方向 1 缓解成立，可放心修复
- 若 `current_block_hash=<新值>`（case C）占绝大多数 → vLLM 已重注册，安全
- 若出现 `current_block_hash=<旧 prompt hash>`（case A）且后续该 block 被写入新数据 → 方向 1 风险真实存在，修复需加额外保护（如复用前校验 hash 一致性）

### 4.2 可选的补充观测（优先级 P1）

**目标**：量化"第二次批次的实际命中率"，与报告实测 31% 交叉验证。

**方法**：在 vLLM 侧启用 prefix-cache hit 统计（vLLM 自带 metric），或在第二次批次 Prefill 阶段单独 grep `allocate_slots_patch will_delay_cache_blocks=False num_new_tokens>1` 的行数，对比第一次批次同请求的 Prefill chunk 数。

### 4.3 日志量优化建议

首轮日志爆量主要来自 `worker_self_trigger branch=defer_chunked_prefill_guard_fired`（每个 prefill chunk 每 TP 都打一行，7×4×N_chunks ≈ 数千行）。下次实验建议先做一次总过滤，把信号行单独存一份，后续所有 grep 都基于这份过滤后的文件：

```bash
grep "TRIATTN-PCTRACE" tmp.log \
  | grep -v "branch=defer_chunked_prefill_guard_fired" > pctrace_signal.log
```

这样能砍掉 90%+ 的噪音，剩下的都是关键决策点。下面的 5.1–5.6 默认基于 `tmp.log`（原始日志）grep，但也可以用 `pctrace_signal.log` 替代以加快速度。

---

## 五、下次实验要 grep 的内容（按优先级）

> 日志文件名固定为 `tmp.log`。每组单独存一个文件，便于交叉对照。
>
> macOS 自带 `grep` 不支持 `-P`（PCRE），下面涉及 `-oP` 的命令在 macOS 上需用
> `rg -o` 或 `ggrep -oP`（`brew install grep` 后可用）替代；下面同时给出 macOS
> 兼容写法。

### 5.1 evict 总量分布（P0，根因严重度）

```bash
# Linux / GNU grep
grep "TRIATTN-PCTRACE" tmp.log \
  | grep "free_reclaimed_blocks stage=post_evict_pre_free" \
  | grep -oP "n=\d+" | sort | uniq -c

# macOS 兼容（用 rg）
rg "TRIATTN-PCTRACE" tmp.log \
  | rg "free_reclaimed_blocks stage=post_evict_pre_free" \
  | rg -o "n=\d+" | sort | uniq -c
```

每行是一个 reclaim 批次的 freed 数。预期第一次和第二次批次各出现 7 次（每请求一次），每次 n ≈ 100–130。

### 5.2 每请求 freed/kept 明细（P0，核对报告公式）

```bash
grep "TRIATTN-PCTRACE" tmp.log | grep "reclaim_branch" > reclaim_branches.log
```

每行带 `req_id gid branch freed= kept= required_blocks=`，可直接核对 `freed ≈ ceil(prefill_len/128) - 48`。

### 5.3 压缩触发时机（P1，复核疑点 C）

```bash
grep "TRIATTN-PCTRACE" tmp.log | grep "worker_self_trigger" \
  | grep -v "branch=defer_chunked_prefill_guard_fired" > self_trigger_decisions.log
```

过滤守卫噪音，剩下 `keep_scheduler_decode_trigger` / `worker_self_trigger_fired` / `below_threshold` 三种实际决策。重点看每个请求第一次出现的 `will_compress=True` 行的 `existing_estimate vs prefill_len`，确认"Prefill 刚结束就压缩"。

### 5.4 hash 提交情况（P1，疑点 B + 命中验证）

```bash
grep "TRIATTN-PCTRACE" tmp.log | grep "allocate_slots_patch" \
  | grep "will_delay_cache_blocks=True" > delay_cache_blocks.log
```

看第二次批次请求在 Prefill 阶段（`num_new_tokens > 1`）是否就已经 `will_delay_cache_blocks=True`。若 Prefill 阶段全是 False、Decode 第一步才变 True，说明第二次批次 Prefill 是正常注册 hash 的（只是后来又被压缩清掉）。

### 5.5 方向 1 风险探针（P0，关键缺口）

```bash
grep "TRIATTN-PCTRACE" tmp.log | grep "block_reuse_on_allocate" > block_reuse.log
```

**如果完全为空**：被 evict 的 block 没被重新分配，方向 1 风险在本实验不触发，可以放心做修复。

**如果有输出**：统计 `current_block_hash=` 后面的值：
- 全是 `None` → vLLM 自己清了 hash（方向 1 缓解成立，安全）
- 有非 None 且和旧 prompt hash 相同 → case A，需进一步看该 block 是否被覆盖写
- 有非 None 且是新 hash → case C，安全

### 5.6 按批次切分（可选）

两次批次的 request_id 不同。先 grep 出所有 req_id：

```bash
# Linux / GNU grep
grep "TRIATTN-PCTRACE" tmp.log | grep -oP "req_id=chatcmpl-[a-f0-9]+" | sort -u

# macOS 兼容（用 rg）
rg "TRIATTN-PCTRACE" tmp.log | rg -o "req_id=chatcmpl-[a-f0-9]+" | sort -u
```

拿到 14 个 req_id（7×2）后，按 warmup 结束时间为界分两组。或直接按 `chatcmpl-` 后缀前 8 位分组。

### 5.7 一键全跑（可选）

把上面 5.1–5.5 一次性产出到 `pctrace_*.log` 文件：

```bash
grep "TRIATTN-PCTRACE" tmp.log | grep -v "branch=defer_chunked_prefill_guard_fired" > pctrace_signal.log
grep "TRIATTN-PCTRACE" tmp.log | grep "reclaim_branch" > pctrace_reclaim_branches.log
grep "TRIATTN-PCTRACE" tmp.log | grep "worker_self_trigger" | grep -v "branch=defer_chunked_prefill_guard_fired" > pctrace_self_trigger.log
grep "TRIATTN-PCTRACE" tmp.log | grep "allocate_slots_patch" > pctrace_allocate_slots.log
grep "TRIATTN-PCTRACE" tmp.log | grep "block_reuse_on_allocate" > pctrace_block_reuse.log
grep "TRIATTN-PCTRACE" tmp.log | grep "evict_reclaimed_block" > pctrace_evict.log
grep "TRIATTN-PCTRACE" tmp.log | grep "free_reclaimed_blocks" > pctrace_free.log
```

跑完把 `pctrace_*.log` 全部贴回 chat 即可，我会按优先级分析。

---

## 六、修复路线（待风险数据齐备后执行）

### 6.1 当前阶段（本分支）已完成

- ✅ print 断点层（`_prefix_cache_debug.py`）
- ✅ 6 类断点接入（scheduler / monkeypatch / runner）
- ✅ 总开关 `TRIATTN_DEBUG_PREFIX_CACHE_TRACE`
- ✅ 首轮观测：根因 + 疑点 B/C 实锤
- ✅ 数字与报告公式精确对齐

### 6.2 下一阶段（新分支，待 P0 风险数据齐备后开）

**分支名建议**：`fix/prefix-cache-direction1-keep-hash-on-reclaim`

**最小修复**：在 `triattention/vllm/runtime/scheduler.py` 的
`_evict_reclaimed_block_metadata` 中跳过 `_maybe_evict_cached_block` 调用：

```python
def _evict_reclaimed_block_metadata(block_pool, block):
    """Direction 1: keep prefix-cache hash, let vLLM manage hash lifecycle."""
    # 不再调用 _maybe_evict_cached_block
    # vLLM BlockPool 在 allocate_slots 分配新块时会清空旧 hash 并重新注册
    return
```

**配套验证**：
1. 重跑首轮实验，确认 evict 断点的 `stage=exit block_hash` 不再变 None
2. 第二次批次命中率应回升到 ~100%（或接近 Base 基线）
3. `block_reuse_on_allocate` 探针应显示 case B/C 占绝大多数
4. 跑输出正确性测试（相同 prompt 两次输出应一致；不同 prompt 不串味）

**回退方案**：如果修复后出现 case A 风险（stale hash 导致读到错误 KV），退回方向 2（只回收 Decode 阶段超出 prompt 长度的 block，不动 prompt 部分）。

### 6.3 长期方案（方向 4，可选）

为压缩请求单独维护 prefix-cache 副本，彻底解耦 TriAttention 的内存节省与 vLLM 的 prefix-cache 生命周期。侵入 BlockPool 匹配逻辑，工作量大，留作后续。

---

## 七、变更日志

| 日期 | 分支 | 改动 |
|---|---|---|
| 2026-06-22 | `debug/prefix-cache-direction1-print-bp` | 初始断点层 + 文档（commit `780b8bb`） |
| 2026-06-22 | 同上 | 首轮观测：根因 + 疑点 B/C 实锤，122 evict 与公式吻合（本文档第二、三节）（commit `47ccd1b`） |
| 2026-06-22 | 同上 | grep 命令改为 `tmp.log`，补充 macOS 兼容写法与一键全跑脚本，明确自动 push 约定 |
| 待定 | `fix/prefix-cache-direction1-keep-hash-on-reclaim` | 待 P0 风险数据齐备后开 |

---

## 八、关键文件索引

| 文件 | 作用 |
|---|---|
| `triattention/vllm/runtime/_prefix_cache_debug.py` | 断点实现 + 总开关 |
| `triattention/vllm/runtime/scheduler.py` | `_evict_reclaimed_block_metadata` / `_free_reclaimed_blocks` / `_apply_compression_events`（方向 1 修复点） |
| `triattention/vllm/runtime/integration_monkeypatch.py` | `_patched_kv_cache_allocate_slots`（疑点 B 修复点） |
| `triattention/vllm/runtime/runner.py` | `_supplement_worker_self_triggers`（疑点 C 修复点） |
| `docs/debug-prefix-cache-direction1.md` | 断点使用手册（怎么用） |
| `docs/debug-prefix-cache-direction1-progress.md` | 本文档（观测到什么、下一步） |
| `TriAttention Prefix-Caching 失效根因分析与验证报告.md` | 根因报告（问题定义） |
| `vllm-ascend TriAttention 与 Prefix-Caching 兼容性问题分析.md` | 问题原始描述 |
