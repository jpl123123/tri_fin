# TriAttention + Prefix-Caching 兼容性修复：系统性指导文档

> **速览**
>
> - **当前分支**：`fix/prefix-cache-direction1-keep-hash-on-reclaim`，HEAD = `5a917c4`
> - **当前版本状态**：路径 C 初版（物理释放 + hash 保护），代码本身无 bug，简单测试通过，但真实负载下 PC ON 反而比 PC OFF 慢
> - **大思路**：肯定要按照路径 C 来做（理由见第三节 3.3）
> - **待解决**：路径 C 在真实负载下 PC ON 的 TPOT/TTFT 不及预期的机理定位与改进（范式三要求已基本调和，当前是性能优化问题）
> - **必须遵守**：方向 1（保留 hash）+ 理想运行范式（物理释放 + hash 保留 + 不改 TriAttention 本身）+ 只参照 `other_code/vllm-ascend-releases-v0.18.0`
> - **建议阅读顺序**：第一节（任务约束）→ 第二节（理想范式）→ 第三节（当前状态与大思路）→ 第六节（关键发现）→ 第七节（方向 1 核心难题与路径 C 定位）

---

## 一、任务原始约束（必须遵守）

### 1.1 输入文档

- **问题描述**：`vllm-ascend TriAttention 与 Prefix-Caching 兼容性问题分析.md`（问题现象、实验数据、分析要求）
- **潜在验证报告**：`TriAttention Prefix-Caching 失效根因分析与验证报告.md`（根因定位、机制冲突详解、修复方向建议）

### 1.2 修复方向约束

直接按根因报告的**方向 1**去排查和修复：

> **方向 1：让压缩回收的 block 保留 prefix-cache hash（首选）**
>
> 思路：`_free_reclaimed_blocks` 归还物理 block 时，不要调用 `_maybe_evict_cached_block`，让 `cached_block` 反查表继续保留这些 hash 索引。
>
> 权衡：
> - 优点：第二次相同请求能完整命中，Prefix-Caching 恢复正常。
> - 风险：被复用的物理 block 可能被新数据覆盖写，此时 `cached_block` 中的旧 hash 索引会指向"内容已变"的 block，导致后续命中读到错误 KV。
> - 缓解：vLLM BlockPool 本身在 `allocate_slots` 分配新块时会清空旧 hash 并重新注册，因此只要不在 TriAttention 侧主动 evict，让 vLLM 自己管理 hash 生命周期即可。

**必须重点排查方向 1 的整体方向，以及重点排查风险**："被复用的物理 block 可能被新数据覆盖写，此时 `cached_block` 中的旧 hash 索引会指向'内容已变'的 block，导致后续命中读到错误 KV。"

### 1.3 代码引用约束

**不要引用任何外部的包**。关于 vLLM 的代码只能去看和参照 `other_code/vllm-ascend-releases-v0.18.0` 这个路径。即使是 vLLM 的 import，本质上也是对这个路径的 import。

> 注：`other_code/vllm-ascend-releases-v0.18.0` 是 vllm-ascend 扩展包，vLLM 核心源码（如 `vllm/v1/core/block_pool.py`、`vllm/v1/core/kv_cache_manager.py`）未包含在该路径，需在运行环境查看。本工作中的 explore 子代理曾在运行环境读取过这些源码，结论记录在第七节。

### 1.4 初始阶段约束（已完成）

初始阶段不修改任何原有代码，只加一个 export 总开关然后用 print 形式打断点。这部分已在 `debug/prefix-cache-trace-print-bp` 分支完成，当前分支继承之。

---

## 二、核心理想运行范式（顶层设计约束）

下述范式是修复的顶层指导原则，任何修复方案都应以此为约束重构驱逐与 hash 留存逻辑：

1. **首轮请求执行 Prefill 阶段，生成完整 KV 哈希映射表**。

2. **基于全局 KV Budget 阈值，物理驱逐超出预算上限的 KV 物理块，但完整保留全部哈希映射关联关系**。命中仍然按全序列长度内容进行匹配，用以保留 prefix cache 的命中能力。

3. **设计目标**：若次轮请求输入 Prompt 与首轮完全一致，依靠未销毁的哈希映射，可跳过完整 Prefill 重计算，实现 100% 缓存命中；并且命中后复用原有的压缩逻辑，仅选中的 KV 作为最终的命中结果。

4. **资源底层逻辑**：物理存储仅留存 KV Budget 限额内的 KV 块，不会占用超额显存。

5. **复用触发逻辑**：当重复 Prompt 复现，直接复用首轮 prefill 和压缩逻辑，直接使用未物理销毁的 Top-K KV 块，无需重新计算重要的 KV。

### 2.1 范式三要求的调和状态

范式同时要求：
- 物理块要释放（第 2、4 条：物理驱逐、不占超额显存）
- hash 要保留（第 2 条：完整保留哈希映射关联关系）
- 命中后复用压缩逻辑（第 3、5 条：仅选中 KV 作为命中结果）

这三条在 vLLM 的 lazy evict 机制下存在张力——物理块释放进 free pool 后，vLLM 复用该块时默认会清掉 hash（详见第六节 6.1）。**当前版本（路径 C 初版 `5a917c4`）已基本调和这三条**：物理块释放（内存 ~21%）、hash 保留（简单测试 100% 命中）、命中后复用压缩逻辑也工作。

当前待解的问题不是"三条无法调和"，而是**真实负载（120 条不同 prompt）下 PC ON 的 TPOT 和 TTFT 表现不及预期**——PC ON 反而比 PC OFF 慢（duration +64614ms, TPOT +8ms），TTFT 降幅远不及基线。这是路径 C 框架下的性能优化问题，不是范式调和问题。

---

## 三、当前状态与大思路判断

### 3.1 当前 HEAD（`5a917c4`）：路径 C 初版

当前分支 HEAD 是 `5a917c4`，基于 `debug/prefix-cache-trace-print-bp` 分支（含全部 print 断点）拉出。完整 commit 链：

```
5a917c4 fix: path C - physical release with hash protection (memory-efficient)  ← 当前 HEAD
4edd7e7 fix: path A - pin reclaimed blocks instead of freeing (full Direction-1)
3743257 fix: keep prefix-cache hash on reclaim (Direction 1)  ← 轻量版
b81a794 docs: record round-2 complete grep results, close root-cause loop  ← debug 分支最后
```

### 3.2 路径 C 的核心机制

1. **reclaim 时**（`_free_reclaimed_blocks`）：给每个被回收 block 打 `_triattention_hash_protected` 标记，不调 `_evict_reclaimed_block_metadata`（不清 hash），然后正常 `free_blocks` 归还 free pool（ref_cnt=0，物理块释放）。

2. **复用时**（`_patched_maybe_evict_cached_block`）：vLLM 从 free pool 取块复用时调 `_maybe_evict_cached_block`，patch 检测到块带保护标记则跳过 hash 清理（保留 hash 让第二次请求能命中）。

3. **保护标记清除**：受保护块被复用写入新数据时，patch 清除保护标记（让后续复用正常）。

### 3.3 大思路判断：肯定要按照路径 C 来做

尽管路径 C 初版在真实负载下表现不佳，但**大思路肯定要按照路径 C 来做**。理由：

- 路径 A（pin 不释放）违反范式第 4 条（不占超额显存），内存 86% 不可接受，已排除
- 轻量版（只跳过 evict 仍 free_blocks）在简单测试下命中率仅 22%，vLLM lazy evict 会清掉保留的 hash，已排除
- evict-on-rewrite 在真实负载下 hash 保不住，已排除
- 路径 C 的"物理释放 + hash 保护"是唯一同时满足"物理块释放"和"hash 保留"的方案框架，符合范式第 2、4 条

路径 C 初版的问题不是思路错，而是在真实负载下的具体表现需要进一步排查和改进。应在路径 C 的框架上继续，重点解决真实负载下 PC ON 反而变慢的问题。

### 3.4 当前版本的实验事实

**此版本代码本身没有 bug**（简单测试通过、内存正常），但问题也很明显——在 120 条不同 prompt 的真实工作负载下，PC ON 相比 PC OFF 没有带来收益，反而更慢。具体的机理定位（为何 PC ON 反而变慢）需要进一步排查，此处只记录实验事实（见第五节 5.4）。

### 3.5 配置开关

```bash
export TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=1  # 启用路径 C（默认）
export TRIATTN_RUNTIME_KEEP_PREFIX_CACHE_HASH_ON_RECLAIM=0  # 关闭，回到原始 evict-on-reclaim
export TRIATTN_DEBUG_PREFIX_CACHE_TRACE=1                   # 开启 print 断点
```

### 3.6 改动文件清单

| 文件 | 改动 |
|---|---|
| `triattention/vllm/runtime/_prefix_cache_debug.py` | 新增：print 断点实现 + 总开关（从 debug 分支继承） |
| `triattention/vllm/runtime/config.py` | 新增 `keep_prefix_cache_hash_on_reclaim` 配置项（默认 True）+ env 加载 |
| `triattention/vllm/runtime/scheduler.py` | `_evict_reclaimed_block_metadata` 条件跳过 evict；`_free_reclaimed_blocks` 路径 C 分支打保护标记；新增 `_mark_block_hash_protected` / `_is_block_hash_protected` / `_clear_block_hash_protection` |
| `triattention/vllm/runtime/integration_monkeypatch.py` | 新增 `_patched_maybe_evict_cached_block` 跳过保护块；patch 安装逻辑 |
| `docs/debug-prefix-cache-direction1.md` | 断点使用手册 |
| `docs/debug-prefix-cache-direction1-progress.md` | 本文档 |

---

## 四、问题背景

TriAttention 开启后，vLLM 的 Prefix-Caching（PC）失效。根因报告已确认：TriAttention 在 Decode 阶段压缩回收物理 block 时，通过 `_evict_reclaimed_block_metadata → BlockPool._maybe_evict_cached_block` 主动清除了这些 block 的 prefix-cache hash，导致第二次相同请求无法命中。

本分支按方向 1 修复，但发现该方向在真实工作负载下有严重副作用。

---

## 五、实验配置、实测数据与所有尝试

### 5.1 实验配置

**简单测试（人为构造的最好情况）**：
- 20k 输入，bs7，打两次**完全相同**的请求
- kv_budget=4096，block_size=128，gpu_memory_utilization=0.9
- Qwen3-32B，TP=4
- 预期：第二次 TTFT 应大幅下降（基线 329ms）

**真实工作负载测试**：
- 120 条**不同** prompt，10k 输入，bs16
- 同上模型/硬件配置
- 预期：PC ON 应比 PC OFF 快（基线如此）

### 5.2 基线实测数据（无 TriAttention）

| 配置 | duration | TTFT | TPOT |
|---|---|---|---|
| Base PC OFF | 957299ms | 5437ms | 116ms |
| Base PC ON | 734217ms | 4182ms | 88ms |

基线下 PC ON 比 PC OFF 快（duration -223082ms, TTFT -1255ms, TPOT -28ms），PC 正常工作。

### 5.3 TriAttention（无 PC 修复）实测数据

| 配置 | duration | TTFT | TPOT |
|---|---|---|---|
| TriAttention PC OFF | 651924ms | 5600ms | 76ms |
| TriAttention PC ON（原始，未修复） | 716538ms | 4859ms | 84ms |

**关键观察**：TriAttention PC OFF 完全正常（比 Base PC OFF 还快，因为 TriAttention 的 decode 加速生效）。但 PC ON 反而比 PC OFF 慢（duration +64614ms, TPOT +8ms），与基线相反。

### 5.4 路径 C 初版（当前 HEAD）的实验事实

路径 C 在 120 条不同 prompt bs16 真实负载下，PC ON 相比 PC OFF：duration +64614ms、TPOT +8ms、TTFT -741ms（仅小幅下降，远不及基线的 -1255ms）。具体的机理定位（为何 PC ON 反而变慢、为何 TTFT 降幅不及基线）需要进一步排查。

简单测试（20k bs7 两次相同请求）下路径 C 通过：TTFT ~1000ms, TPOT ~60ms, prefix ratio ~52%, 内存 ~21%。

### 5.5 所有尝试与失败原因（按时间顺序）

本分支共尝试了 6 个版本的修复，全部失败：

#### 5.5.1 轻量版（commit `3743257`）

**改动**：`_evict_reclaimed_block_metadata` 跳过 `_maybe_evict_cached_block`，但 `free_blocks` 仍归还物理块。

**简单测试结果**：第二次命中率仅 22%（=34/156），TTFT 1015ms（基线 329ms）。

**失败原因**：vLLM BlockPool 是 lazy evict 机制。即使 reclaim 时不清 hash，`free_blocks` 把块归还 free pool（ref_cnt=0）后，后续请求（包括第一次请求自己的 decode）从 free pool 取块复用时，vLLM 会调 `_maybe_evict_cached_block` 清掉 hash。所以保留的 hash 在第二次请求来之前就被 vLLM 自己清了。

**根因定位（explore 确认）**：hash 被清的时机是**第一次请求 decode 后续的 `allocate_slots → get_new_blocks`**。当 effective 长度超过 4352（=34×128 保留块容量）后，每生成 128 个新 token 触发一次 get_new_blocks，从 free pool 头部取走带 hash 的块并 lazy evict。

#### 5.5.2 路径 A（commit `4edd7e7`）

**改动**：被回收的 block 不进 free pool（ref_cnt 保持 >0），登记为 pinned 孤儿块，请求结束时释放。

**简单测试结果**：第二次命中率 ~100% ✓，TTFT ~1000ms ✓，TPOT ~60ms ✓。**简单测试通过**。

**真实负载结果**：KV 内存峰值 86%（基线 21%），破坏了 TriAttention 原生 KV 驱逐逻辑。

**失败原因**：pin 住 122 个物理块不释放，7 并发 × 122 block × 32MB ≈ 27GB 额外占用，free pool 不增长，TriAttention 内置驱逐策略完全失效。**违反范式第 4 条**（物理存储仅留存 KV Budget 限额内的 KV 块，不占超额显存）。

#### 5.5.3 路径 C 初版（commit `5a917c4`）— 当前 HEAD

见第三节 3.1-3.4。

#### 5.5.4 路径 C 修复版（commit `b5ad232`，已回退）

**改动**：路径 C 初版的 patch 在跳过 evict 时只清了保护标记，没清 `block.block_hash` 和反查表条目，导致受保护块被改写后旧 hash 残留（stale hash 污染）。修复版改为 evict-on-rewrite：受保护块即将被改写时先执行完整 evict。

**失败原因**：代码审查发现修复版两个分支（protected / not protected）做的是完全一样的事（都调原始 evict），等价于完全没保护，即 `KEEP=0` 行为。更根本的是：evict-on-rewrite 在 120 条不同 prompt 下无效——受保护块在被第二次请求 touch 之前就被改写 evict，hash 没机会保留。

#### 5.5.5 方向 1：chunk cap 最小下限（commit `b2dc8d8`，已回退）

**改动**：给 `_compute_max_chunk_for_compression` 的 chunk cap 设最小下限 512 token。

**失败原因**：用户指出这是改 TriAttention 本身，不是改 PC 交互。TriAttention PC OFF 用得好好的，说明 chunk 压小不是 bug。方向错了。**违反任务约束 1.2**（方向 1 是改 hash 留存，不是改 chunk 大小）。

#### 5.5.6 关闭路径 C 默认开关（commit `9ec6135`，已回退）

**改动**：`keep_prefix_cache_hash_on_reclaim` 默认值从 True 改为 False。

**失败原因**：这等于完全放弃修复，回到原始未修复状态。用户要求回退到 `5a917c4` 重新评估。

---

## 六、关键发现（explore 确认的源码事实）

### 6.1 vLLM BlockPool 的 hash 生命周期

- `BlockPool._maybe_evict_cached_block` 是唯一清 `block.block_hash` 和删 `cached_block_hash_to_block` 条目的入口
- 它在两个地方被调用：(a) TriAttention 的 `_evict_reclaimed_block_metadata`（reclaim 时），(b) vLLM 的 `get_new_blocks`（从 free pool 取块复用时，即 lazy evict）
- `KVCacheBlock.block_hash` setter 有 `assert self.block_hash is None` 硬断言
- `cache_full_blocks` 注册新 hash 时有 `assert blk.block_hash is None` 硬断言
- `BlockPool.touch` 在 ref_cnt 0→1 时把块从 `free_block_queue` 移除

### 6.2 方向 1 风险的真实性（已验证）

根因报告方向 1 的风险："被复用的物理 block 可能被新数据覆盖写，此时 `cached_block` 中的旧 hash 索引会指向'内容已变'的 block，导致后续命中读到错误 KV。"

**explore 确认这个风险是真实的**。vLLM 的 `_maybe_evict_cached_block` 同时做两件事：清 `block.block_hash` + 从 `cached_block_hash_to_block` 删除条目。如果跳过它（方向 1 的思路），旧 hash 会残留在反查表里指向已改写的 block，后续命中读到错误 KV。

**根因报告的"缓解"论证（"vLLM BlockPool 本身在 allocate_slots 分配新块时会清空旧 hash 并重新注册"）是不完整的**。vLLM 清旧 hash 的唯一入口就是 `_maybe_evict_cached_block`，如果方向 1 跳过它，就没有其他入口清旧 hash。`cache_full_blocks` 只做"注册新 hash"，不做"清旧 hash"。所以"缓解"论证只在"块被复用且走完整的 evict→写新→register 流程"时成立，而方向 1 恰恰打断了第一步。

### 6.3 hash 被清的真正时机

不是第二次请求自己清的，而是**第一次请求 decode 后续的 `allocate_slots → get_new_blocks`**。当 effective 长度超过 4352（=34×128 保留块容量）后，每生成 128 个新 token 触发一次 get_new_blocks，从 free pool 头部取走带 hash 的块并 lazy evict。

### 6.4 简单测试 vs 真实负载的差异

- **简单测试（20k bs7 两次相同请求）**：第二次请求 100% 命中 touch 把受保护块移出 free pool，走"安全分支"，stale 污染不发生
- **真实负载（120 条不同 prompt）**：每个 prompt 只来一次，受保护块等不到 touch，只能被改写。具体后果见 5.4 的实验事实，机理定位需要进一步排查

### 6.5 基线 PC ON 为什么快

基线 PC ON 靠共享 system prompt 前缀命中降 TTFT，靠 prefill 计算量减少让 decode 等待更短降 TPOT。vLLM 原生 LRU evict 控制表规模小，hash 管理开销 < 命中收益。

### 6.6 TriAttention PC OFF 为什么正常

TriAttention 的压缩/decode 加速通过 input_patch 改 positions/seq_lens 实现，与 block_pool 物理状态解耦。chunk 压小和压缩开销在 PC OFF 下都是正常设计，不是 bug。

### 6.7 Prefill 阶段正常注册 hash（实测确认）

实测确认 Prefill 阶段（压缩前）`will_delay_cache_blocks=False`（正常注册 hash）。所以共享 system prompt 前缀命中机制在 TriAttention 下没有被完全破坏。`delay_cache_blocks=True` 只在压缩后对压缩过的请求生效。

---

## 七、方向 1 的核心难题与路径 C 的定位

**方向 1 的原始难题**：要让第二次请求命中，hash 必须在块被复用前保留；要让块被复用时不产生 stale 污染，hash 必须在块被改写前清除。但"复用"和"改写"是同一个动作（`get_new_blocks` 取块后写入新内容），难以在时间上分开。

各路径的表现：
- 路径 A（pin 不释放）：hash 永远保留，但内存 86%（违反范式第 4 条）
- 路径 C（保护标记 + 释放）：**已基本解决上述难题**——物理块释放（内存 ~21%）、hash 保留（简单测试 100% 命中）、stale 污染在简单测试下不发生。但真实负载下 PC ON 反而比 PC OFF 慢（见 5.4 实验事实），这是当前待解的性能问题
- evict-on-rewrite（改写时清）：hash 在改写前清除，但真实负载下第二次请求来不及 touch

**路径 C 是当前最优方案**。它通过"保护标记 + 物理释放"调和了范式三要求（详见 2.1），在简单测试下完全达标。当前的问题是真实负载下 PC ON 的 TPOT/TTFT 不及预期，这是路径 C 框架下的性能优化问题，不是范式调和问题。

**大思路肯定要按照路径 C 来做**（理由见 3.3）。路径 C 初版在真实负载下的具体问题需要进一步排查和改进，而不是放弃路径 C 换方向。

---

## 八、本工作中的错误总结（避免重蹈覆辙）

本工作历时两天，犯了多个方法论错误，导致大量返工。系统记录如下，供后续工作避免重蹈覆辙。

### 8.1 急于改代码，不先想清楚

**表现**：看到 stale hash 污染的分析就立刻改 `_patched_maybe_evict_cached_block`，结果改出两个分支做一样的事（等价于 KEEP=0），完全无效。看到"chunk 压小"分析就立刻加 chunk 下限，结果被用户指出改错了对象。

**教训**：每次改代码前必须先问自己：(1) 这个改动解决的是哪个具体问题？(2) 这个问题是 TriAttention 本身的 bug 还是 PC 交互问题？(3) 改完之后预期数字怎么变？三个问题都想清楚再动手。

### 8.2 把简单测试当全部

**表现**：路径 C 初版在 20k bs7 两次相同请求测试通过后就认为修复成功，没有立刻跑真实负载验证。

**教训**：20k bs7 两次相同请求是人为构造的最好情况（第二次请求 100% 命中 touch 救走受保护块），它只能验证"安全分支"，不能验证"危险分支"（受保护块被改写）。任何修复都必须同时通过简单测试和真实负载测试才算数。

### 8.3 没有区分"TriAttention 本身"和"PC 交互"

**表现**：方向 1（chunk cap 最小下限）改的是 TriAttention 的 `_compute_max_chunk_for_compression`，但用户指出 TriAttention PC OFF 用得好好的，说明 chunk 压小不是 bug。

**教训**：PC OFF 正常 = TriAttention 本身没问题。问题只在 PC ON 与 TriAttention 的交互。任何改动如果影响 PC OFF 行为，就是改错了对象。

### 8.4 错误的根因判断

**表现**：多次错误定位根因：
1. 先认为"TTFT 降不下去是疑点 B（delay_cache_blocks=True）破坏共享前缀命中"——explore 确认 Prefill 阶段正常注册 hash，疑点 B 不是主因
2. 再认为"chunk 压小 + 压缩开销吃掉命中收益"——用户指出这是 TriAttention 本身，不是 PC 交互
3. 最后对路径 C 在真实负载下 PC ON 反而变慢的机理做了定位，但用户要求只记录实验事实、不做事先的机理总结（机理定位需要进一步排查）

**教训**：根因定位必须用 explore 确认源码事实，不能靠推理。每次定位后必须用实测数据交叉验证，不能只看理论分析。

### 8.5 路径 C 修复版的低级代码错误

**表现**：`_patched_maybe_evict_cached_block` 的两个分支（protected / not protected）都调 `_ORIG_MAYBE_EVICT_CACHED_BLOCK`，唯一区别是 protected 分支多清了一个保护标记。这等于完全没保护，等价于 KEEP=0。

**教训**：改完代码必须重新读一遍自己的改动，确认逻辑分支确实有差异。特别是 if/else 两个分支都要检查实际执行路径。

### 8.6 没有及时回退错误方向

**表现**：方向 1（chunk 下限）被用户指出改错后，还继续往下做关闭路径 C 的默认开关，等于放弃修复。应该更早意识到方向错了就回退。

**教训**：一旦确认某个方向错了，立刻回退到上一个已知好的状态，不要在错误方向上继续打补丁。

### 8.7 过度依赖 explore 子代理的结论

**表现**：explore 子代理给出"chunk 压小是主因"的结论后，没有用实测数据交叉验证就直接实施方向 1。explore 的分析基于源码推理，可能忽略实际运行时的其他因素。

**教训**：explore 是工具不是裁判。它的结论必须用实测数据验证后才能作为修复依据。特别是"哪个是主因"这种判断，必须用对比实验确认，不能只靠源码分析。

---

## 九、关键文件索引

| 文件 | 作用 |
|---|---|
| `triattention/vllm/runtime/_prefix_cache_debug.py` | print 断点实现 + 总开关 |
| `triattention/vllm/runtime/scheduler.py` | `_evict_reclaimed_block_metadata` / `_free_reclaimed_blocks` / `_mark_block_hash_protected` / `_apply_compression_events` |
| `triattention/vllm/runtime/config.py` | `keep_prefix_cache_hash_on_reclaim` 配置项（默认 True） |
| `triattention/vllm/runtime/integration_monkeypatch.py` | `_patched_kv_cache_allocate_slots`（疑点 B）/ `_patched_maybe_evict_cached_block`（路径 C） |
| `triattention/vllm/runtime/runner.py` | `_supplement_worker_self_triggers`（疑点 C） |
| `docs/debug-prefix-cache-direction1.md` | 断点使用手册 |
| `docs/debug-prefix-cache-direction1-progress.md` | 本文档（系统性指导与失败总结） |
| `TriAttention Prefix-Caching 失效根因分析与验证报告.md` | 根因报告（问题定义，含方向 1 原文） |
| `vllm-ascend TriAttention 与 Prefix-Caching 兼容性问题分析.md` | 问题原始描述 |
| `other_code/vllm-ascend-releases-v0.18.0/` | vLLM-ascend 源码参考（vllm 核心源码未包含，需在运行环境查看） |

---

## 十、调试工具使用

当前分支继承自 `debug/prefix-cache-trace-print-bp`，包含完整的 print 断点层：

```bash
export TRIATTN_DEBUG_PREFIX_CACHE_TRACE=1  # 开启断点

# 一键全跑
grep "TRIATTN-PCTRACE" tmp.log | grep -v "branch=defer_chunked_prefill_guard_fired" > pctrace_signal.log
grep "TRIATTN-PCTRACE" tmp.log | grep "reclaim_branch" > pctrace_reclaim_branches.log
grep "TRIATTN-PCTRACE" tmp.log | grep "worker_self_trigger" | grep -v "branch=defer_chunked_prefill_guard_fired" > pctrace_self_trigger.log
grep "TRIATTN-PCTRACE" tmp.log | grep "allocate_slots_patch" > pctrace_allocate_slots.log
grep "TRIATTN-PCTRACE" tmp.log | grep "block_reuse_on_allocate" > pctrace_block_reuse.log
grep "TRIATTN-PCTRACE" tmp.log | grep "evict_reclaimed_block" > pctrace_evict.log
grep "TRIATTN-PCTRACE" tmp.log | grep "free_reclaimed_blocks" > pctrace_free.log
```

断点含义见 `docs/debug-prefix-cache-direction1.md`。
