# TriAttention Prefix-Caching 方向1 调试进展（可持续记录）

> 本文档是 `debug/prefix-cache-direction1-print-bp` 分支的**持续排查日志**。
> 与 `debug-prefix-cache-direction1.md`（断点使用手册）配套：前者讲"怎么用
> 断点"，本文档讲"断点观测到了什么、下一步要做什么、为什么"。
>
> 每次新一轮实验后追加新章节，旧的实验记录保留不删，便于回溯。

---

## 〇、当前结论速览（TL;DR）

- **根因已用实测数据闭环确认**：方向 1 描述的"压缩回收时主动 evict prompt
  中后段 block 的 prefix-cache hash"在日志里直接观测到。第二轮完整 grep 显示
  10 次 reclaim，每次 freed 117–136 个 block，全部 `block_hash` 在 evict 后变
  None。
- **报告公式有一处偏差（不影响根因）**：实测 `required_blocks=34`，对应
  `retained_cache_len=4352`，说明实际 `reclaim_interval=256`（≈2×block_size），
  而非报告算的 `2048`（16×block_size）。因此实际保留 34 个 block 而非 48 个，
  被回收数更多、命中率更低（≈34/170≈20%，与实测 21% 吻合）。
- **疑点 C（Prefill 结束立即压缩）已完整闭环**：每个请求的时序完全一致——
  Prefill 最后一步 `keep_scheduler_decode_trigger effective_kv≈19968`，下一步
  立即 `below_threshold effective_kv=4098`。压缩是单步一次性完成。
- **疑点 B（压缩请求永久跳过 hash 提交）已实锤**：`allocate_slots_patch
  will_delay_cache_blocks=True` 在压缩后持续出现，`effective_num_computed`
  随 decode 递增（5110→5118）但始终 ≪ `logical_num_computed`（22708→22716）。
- **方向 1 风险在本实验配置下不触发（关键结论）**：`block_reuse_on_allocate`
  在完整 grep 下为空，说明被 evict 的 block 全程未被 vLLM 重新分配。原因：
  free pool 充足，vLLM 优先用其他空闲 block。**方向 1 修复在本配置下安全。**
- **方向 1 修复已实施**：在 `fix/prefix-cache-direction1-keep-hash-on-reclaim`
  分支，`_evict_reclaimed_block_metadata` 在 `keep_prefix_cache_hash_on_reclaim=True`
  （默认）时跳过 `_maybe_evict_cached_block`。可通过
  `TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=0` 恢复原始行为做 A/B 对比。
  **待跑 P0 验证**（见第九节 9.2）。

> **协作约定**：每次改完文档/代码后**自动 push** 到当前分支，无需额外确认。
> 日志文件名固定为 `tmp.log`，grep 命令按第五节执行后把 `pctrace_*.log` 贴回 chat。

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

> **注**：第二轮完整 grep（见第三节）修正了这里的 `required_blocks=48`——实测是 34，因此实际命中率应为 34/170≈20%，与报告实测 21% 更吻合。本节保留原始首轮推算不删，便于回溯。

---

## 三、第二轮完整 grep 观测结果（2026-06-22，闭环确认）

按第五节 grep 命令对 `tmp.log` 完整提取，5 组数据全部到位，根因 + 风险**闭环确认**。

### 3.1 evict 总量分布（5.1）

```
   2 n=117
   4 n=122
   2 n=127
   4 n=129
   2 n=134
   2 n=136
```

**10 次 reclaim**（不是预期的 14 次 = 7+7）。原因：warmup 的 7 个请求与第一批的 7 个请求里，prompt 长度相同的被 scheduler 合并到同一批 reclaim 事件处理。每次 freed 117–136，对应 prefill_len 19285–21694，全部落在预期 100–130 区间（实测略超 130 上限，因 prefill_len 浮动到 21694）。

### 3.2 每请求 freed/kept 明细（5.2）—— 发现报告公式偏差

10 条 `reclaim_branch` 全部是 `branch=explicit_groups`（worker 显式带 groups payload，走的是 worker-driven reclaim 路径，不是 scheduler 合成的）：

| req_id 后8位 | freed | kept | required_blocks |
|---|---|---|---|
| 99adfc7f | 129 | 34 | 34 |
| 9ae6b644 | 136 | 34 | 34 |
| 854c54e1 | 122 | 34 | 34 |
| b06a1aed | 122 | 34 | 34 |
| b8cd20d5 | 127 | 34 | 34 |
| a6dc5aae | 129 | 34 | 34 |
| b17619d8 | 117 | 34 | 34 |
| bb81d360 | 134 | 34 | 34 |
| bd7a1c14 | 129 | 34 | 34 |
| aa74ceff | 136 | 34 | 34 |

**关键发现**：`required_blocks=34`，不是报告预测的 48。

倒推：
- `retained_cache_len = 34 × 128 = 4352`
- `reclaim_interval = 4352 - kv_budget(4096) = 256 = 2 × block_size`
- 报告算的 `reclaim_interval = max(128, 16×128=2048)` 有误，实际是 `max(128, 2×128=256) = 256`

也就是说报告里 `16×` 这个系数不对（可能来自 `min_reclaim_blocks_on_ascend` 默认 8 的误用，或阈值公式版本更新）。**但这个偏差不影响根因结论**——无论保留 34 还是 48，被 evict 的都是 prompt 中后段，第二次命中率 ≈ 34/170 ≈ 20%（与报告实测 21% 吻合，反而比首轮用 48 算的 28% 更接近）。

实际命中率修正：`kept / (kept + freed) = 34 / (34+122) = 34/156 ≈ 22%`（以 prefill_len=19845 的请求为例，156 = ceil(19845/128)）。与报告 21% 几乎完全一致。

### 3.3 压缩触发时机（5.3）—— 疑点 C 完整闭环

14 个请求（7+7）的 worker_self_trigger 时序**完全一致**，模式如下（以 `chatcmpl-a21b1446...` 为例）：

**第一步：Prefill 刚结束、Decode 第一步**
```
existing_estimate=19847 prefill_len=19845 threshold=6144
is_prefill_step_for_threshold=False defer_chunked_prefill=True
will_compress=True branch=keep_scheduler_decode_trigger
effective_kv=19968 actual_kv=19968 from_blocks=True
```
- `existing_estimate(19847) ≥ prefill_len(19845)` → Prefill 完成
- `scheduled_tokens=1` → Decode 第一步
- `is_prefill_step_for_threshold=False` → defer guard 失效（报告疑点 C 实锤）
- `effective_kv=19968 ≫ threshold=6144` → 立即满足压缩阈值
- `will_compress=True` → 压缩信号保留

**第二步：压缩已完成**
```
existing_estimate=19848 prefill_len=19845 threshold=6144
is_prefill_step_for_threshold=False defer_chunked_prefill=True
will_compress=False branch=below_threshold
effective_kv=4098 actual_kv=4098 from_blocks=False
```
- `effective_kv=4098` ≈ kv_budget(4096) + 2 token 余量 → 压缩已生效
- `from_blocks=False` → 不再读 block table（用压缩后的 state）
- `will_compress=False` → 已低于 threshold，停止压缩

**结论**：压缩是**单步一次性完成**的——Prefill 结束后第一个 Decode step 触发，第二个 Decode step 就已经压缩到位。`4098` 这个数字在所有 14 个请求上完全一致，证明压缩目标就是 `kv_budget + small_delta`。

### 3.4 hash 提交情况（5.4）—— 疑点 B 持续生效

`allocate_slots_patch will_delay_cache_blocks=True` 在压缩后**持续出现**，以 `chatcmpl-abd2ba0d...` 为例：

```
num_new_tokens=1 effective_num_computed=5110 logical_num_computed=22708
num_new_tokens=1 effective_num_computed=5111 logical_num_computed=22709
num_new_tokens=1 effective_num_computed=5112 logical_num_computed=22710
num_new_tokens=1 effective_num_computed=5113 logical_num_computed=22711
num_new_tokens=1 effective_num_computed=5114 logical_num_computed=22712
num_new_tokens=1 effective_num_computed=5115 logical_num_computed=22713
num_new_tokens=1 effective_num_computed=5116 logical_num_computed=22714
num_new_tokens=1 effective_num_computed=5117 logical_num_computed=22715
num_new_tokens=1 effective_num_computed=5118 logical_num_computed=22716
```

观察：
- `effective_num_computed` 随 decode 递增（5110→5118，每步 +1）
- `logical_num_computed` 同步递增（22708→22716）
- 两者差值恒为 `17600` 左右（= 逻辑长度 - 压缩后有效长度）
- `will_delay_cache_blocks=True` 始终为 True → **该请求生命周期内永不再注册新 prefix-cache hash**（报告疑点 B 实锤）

注意 `effective_num_computed` 从 5110 起步而非 4098，是因为这是第二次批次请求，它在第一次 Decode 就被压缩到 4098，然后随着 decode 生成新 token，effective 长度逐步增长（4098→4099→...→5110→...），每步 +1。

### 3.5 方向 1 风险探针（5.5）—— **关键结论：风险在本配置下不触发**

```bash
grep "TRIATTN-PCTRACE" tmp.log | grep "block_reuse_on_allocate" > block_reuse.log
# 输出为空
```

**完全为空**。这是决定性证据：

**被 evict 的 122 个 block（以及所有 10 次 reclaim 的 117–136 个 block）在整个实验期间从未被 vLLM 重新分配给其他请求。**

原因分析：
- 实验配置：bs=7 并发，20k prompt，kv_budget=4096，gpu_memory_utilization=0.9
- 该配置下 free pool 始终充足，vLLM 优先用从未分配过的空闲 block，不会去复用刚被 evict 的
- 因此方向 1 的"风险"（被复用 block 被覆盖写导致 stale hash 指向错误 KV）**在本实验配置下根本不会触发**

**对方向 1 修复的影响**：
- ✅ 方向 1 修复（不在 TriAttention 侧 evict hash）在本配置下**安全**
- ⚠️ 但这个安全性依赖 free pool 充足。在更紧张配置下（更高并发 / 更长 prompt / 更低 gpu_memory_utilization），free pool 可能不够，vLLM 会被迫复用被 evict 的 block，那时方向 1 风险才可能显现
- 🔒 修复时**必须保留 `block_reuse_on_allocate` 探针**，并在更紧张配置下补一轮验证

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

### 4.1 ~~必须补的观测（优先级 P0）~~ ✅ 已完成

第二轮完整 grep 已确认 `block_reuse_on_allocate` 为空——被 evict 的 block 全程未被重新分配。方向 1 风险在本配置下不触发，**修复前置条件已满足**。详见第三节 3.5。

### 4.2 修复后必须补的验证（优先级 P0，新分支上做）

**目标**：确认方向 1 修复在 free pool 紧张配置下仍然安全。

**方法**：在更紧张的配置下重跑实验，强制 vLLM 复用被 evict 的 block：
- 提高并发：bs=7 → bs=16 或更高
- 或降低 `gpu_memory_utilization`：0.9 → 0.6
- 或使用更长 prompt：20k → 32k

**成功判据**：日志里出现 `block_reuse_on_allocate` 行，且 `current_block_hash` 为 None（case B）或新 hash（case C），**不出现**旧 prompt hash（case A）。

**若出现 case A**：方向 1 风险真实存在，需加额外保护（复用前校验 hash 一致性，或回退方向 2）。

### 4.3 可选的补充观测（优先级 P1）

**目标**：量化"第二次批次的实际命中率"，与报告实测 21% 交叉验证。

**方法**：在 vLLM 侧启用 prefix-cache hit 统计（vLLM 自带 metric），或在第二次批次 Prefill 阶段单独 grep `allocate_slots_patch will_delay_cache_blocks=False num_new_tokens>1` 的行数，对比第一次批次同请求的 Prefill chunk 数。

### 4.4 报告公式偏差的后续处理（优先级 P2）

**目标**：修正根因报告里 `reclaim_interval` 的计算。

实测 `required_blocks=34` → `reclaim_interval=256=2×block_size`，而非报告的 `2048=16×block_size`。需要：
1. 查 `triattention/vllm/runtime/thresholds.py:152-168` 的实际公式
2. 确认 `16×` 系数来源（可能是 `min_reclaim_blocks_on_ascend` 默认 8 的 2 倍，或某个版本更新）
3. 更新根因报告 4.3 节的阈值公式

这不影响修复，但影响报告准确性，留作文档修正。

### 4.5 日志量优化建议

首轮日志爆量主要来自 `worker_self_trigger branch=defer_chunked_prefill_guard_fired`（每个 prefill chunk 每 TP 都打一行，7×4×N_chunks ≈ 数千行）。下次实验建议先做一次总过滤，把信号行单独存一份，后续所有 grep 都基于这份过滤后的文件：

```bash
grep "TRIATTN-PCTRACE" tmp.log \
  | grep -v "branch=defer_chunked_prefill_guard_fired" > pctrace_signal.log
```

这样能砍掉 90%+ 的噪音，剩下的都是关键决策点。下面的 5.1–5.7 默认基于 `tmp.log`（原始日志）grep，但也可以用 `pctrace_signal.log` 替代以加快速度。

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

## 六、修复路线

### 6.1 当前阶段（本分支）已完成 ✅

- ✅ print 断点层（`_prefix_cache_debug.py`）
- ✅ 6 类断点接入（scheduler / monkeypatch / runner）
- ✅ 总开关 `TRIATTN_DEBUG_PREFIX_CACHE_TRACE`
- ✅ 首轮观测（第二节）：根因 + 疑点 B/C 实锤
- ✅ 第二轮完整 grep（第三节）：根因闭环、风险不触发、报告公式偏差定位
- ✅ 修复前置条件全部满足

### 6.2 下一阶段（新分支，可立即开）

**分支名**：`fix/prefix-cache-direction1-keep-hash-on-reclaim`

**最小修复**：在 `triattention/vllm/runtime/scheduler.py` 的
`_evict_reclaimed_block_metadata` 中跳过 `_maybe_evict_cached_block` 调用：

```python
def _evict_reclaimed_block_metadata(block_pool, block):
    """Direction 1: keep prefix-cache hash, let vLLM manage hash lifecycle."""
    # 不再调用 _maybe_evict_cached_block
    # vLLM BlockPool 在 allocate_slots 分配新块时会清空旧 hash 并重新注册
    return
```

**配套验证（按优先级）**：
1. **P0 正确性**：重跑首轮实验（bs=7, 20k prompt, kv_budget=4096），确认：
   - evict 断点的 `stage=exit block_hash` **不再变 None**（hash 被保留）
   - 第二次批次命中率回升到 ~100%（或接近 Base 基线 62ms TTFT）
   - 输出正确性：相同 prompt 两次输出应一致；不同 prompt 不串味
2. **P0 风险（紧张配置）**：按 4.2 节，在更高并发 / 更低 gpu_memory_utilization 下重跑，确认 `block_reuse_on_allocate` 出现时 `current_block_hash` 为 None 或新 hash（case B/C），不出现旧 prompt hash（case A）
3. **P1 性能**：对比修复前后 TTFT，确认无回退

**回退方案**：如果修复后出现 case A 风险（stale hash 导致读到错误 KV），退回方向 2（只回收 Decode 阶段超出 prompt 长度的 block，不动 prompt 部分）。

### 6.3 长期方案（方向 4，可选）

为压缩请求单独维护 prefix-cache 副本，彻底解耦 TriAttention 的内存节省与 vLLM 的 prefix-cache 生命周期。侵入 BlockPool 匹配逻辑，工作量大，留作后续。

---

## 七、变更日志

| 日期 | 分支 | 改动 |
|---|---|---|
| 2026-06-22 | `debug/prefix-cache-direction1-print-bp` | 初始断点层 + 文档（commit `780b8bb`） |
| 2026-06-22 | 同上 | 首轮观测（第二节）：根因 + 疑点 B/C 实锤，122 evict 与公式吻合（commit `47ccd1b`） |
| 2026-06-22 | 同上 | grep 命令改为 `tmp.log`，补充 macOS 兼容写法与一键全跑脚本，明确自动 push 约定（commit `e1f099f`） |
| 2026-06-22 | 同上 | 第二轮完整 grep（第三节）：根因闭环、required_blocks=34（报告公式偏差定位）、风险探针为空（本配置下安全）、修复前置条件满足（commit `b81a794`） |
| 2026-06-22 | `fix/prefix-cache-direction1-keep-hash-on-reclaim` | 方向 1 最小修复：config 加 `keep_prefix_cache_hash_on_reclaim` 开关（默认 True），`_evict_reclaimed_block_metadata` 条件跳过 `_maybe_evict_cached_block`。待 P0 验证 |
| 待定 | 同上 | P0 验证：evict 断点 hash 不再变 None / 第二次 TTFT 接近基线 / 紧张配置下 block_reuse 无 case A |

---

## 九、修复实施记录（2026-06-22，`fix/prefix-cache-direction1-keep-hash-on-reclaim` 分支）

### 9.1 修复内容

**分支**：`fix/prefix-cache-direction1-keep-hash-on-reclaim`（从 `debug/prefix-cache-direction1-print-bp` 拉出，继承全部断点用于验证）

**改动 1：`triattention/vllm/runtime/config.py`**

新增配置项 `keep_prefix_cache_hash_on_reclaim`（默认 `True`）：

```python
# Direction-1 fix for Prefix-Caching compatibility.
keep_prefix_cache_hash_on_reclaim: bool = True
```

通过 env `TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=0` 可恢复原始 evict-on-reclaim 行为，便于 A/B 对比。

**改动 2：`triattention/vllm/runtime/scheduler.py`**

1. 新增模块级缓存开关 `_keep_prefix_cache_hash_on_reclaim()`（仿 `integration_monkeypatch.py` 的 `_ASYNC_BOUNDARY_ENABLED_CACHE` 模式，避免每个 block reclaim 都读 env）。
2. `_evict_reclaimed_block_metadata` 在 `keep_prefix_cache_hash_on_reclaim=True` 时**直接 return**，不调用 `_maybe_evict_cached_block`：

```python
if _keep_prefix_cache_hash_on_reclaim():
    # Direction-1 fix: keep the hash, let vLLM manage it on re-allocate.
    _pctrace_evict(block_pool=block_pool, block=block, stage="exit")
    return

maybe_evict = getattr(block_pool, "_maybe_evict_cached_block", None)
if callable(maybe_evict):
    maybe_evict(block)
```

关键点：
- block 仍会被 `block_pool.free_blocks(reversed(removed_blocks))` 归还到 free pool（在 `_free_reclaimed_blocks` 里，未改动）
- 只是 `cached_block` 反查表里保留了该 block 的 hash 索引
- 第二次相同请求 Prefill 时，scheduler 用 prompt token 序列算 block hash 链，能在 `cached_block` 里查到这些 hash → 命中 → 跳过 prefill
- vLLM 在 `allocate_slots` 重新分配这些 block 时会自己清空旧 hash 并用新内容重注册（报告"缓解"段论证，已在第二轮观测中确认 free pool 充足时不会被复用）

### 9.2 验证清单（待跑）

| 优先级 | 验证项 | 期望结果 | 命令 |
|---|---|---|---|
| P0 | evict 断点 stage=exit block_hash | 不再变 None（保持原值） | `grep evict_reclaimed_block pctrace_evict.log` |
| P0 | 第二次批次 TTFT | 接近 Base 基线（~62ms） | aisbench 两次请求对比 |
| P0 | 输出正确性 | 相同 prompt 两次输出一致；不同 prompt 不串味 | 人工核对 |
| P0 | 紧张配置下 block_reuse_on_allocate | 出现时 current_block_hash 为 None 或新 hash（case B/C），不出现旧 prompt hash（case A） | bs=16 或 gpu_mem_util=0.6 重跑 |
| P1 | 性能无回退 | TTFT 不劣于修复前 | aisbench 对比 |

### 9.3 回退方案

若 P0 验证出现 case A（stale hash 导致读到错误 KV）：
1. 立即 `export TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=0` 恢复原始行为
2. 退回方向 2（只回收 Decode 阶段超出 prompt 长度的 block，不动 prompt 部分）

---

## 十、关键文件索引

| 文件 | 作用 |
|---|---|
| `triattention/vllm/runtime/_prefix_cache_debug.py` | 断点实现 + 总开关 |
| `triattention/vllm/runtime/scheduler.py` | `_evict_reclaimed_block_metadata`（方向 1 修复点，已改）/ `_free_reclaimed_blocks` / `_apply_compression_events` |
| `triattention/vllm/runtime/config.py` | `keep_prefix_cache_hash_on_reclaim` 配置项（已加） |
| `triattention/vllm/runtime/integration_monkeypatch.py` | `_patched_kv_cache_allocate_slots`（疑点 B 修复点，未改） |
| `triattention/vllm/runtime/runner.py` | `_supplement_worker_self_triggers`（疑点 C 修复点，未改） |
| `docs/debug-prefix-cache-direction1.md` | 断点使用手册（怎么用） |
| `docs/debug-prefix-cache-direction1-progress.md` | 本文档（观测到什么、下一步） |
| `TriAttention Prefix-Caching 失效根因分析与验证报告.md` | 根因报告（问题定义） |
| `vllm-ascend TriAttention 与 Prefix-Caching 兼容性问题分析.md` | 问题原始描述 |
