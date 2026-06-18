# vllm\-ascend TriAttention 与 Prefix\-Caching 兼容性问题分析 Prompt

## 一、任务说明

请基于本文档提供的 **环境配置、参数定义、实验现象、业务逻辑预期**，仅从代码逻辑、机制层面系统性分析问题根因。**禁止修改任何代码、禁止直接修复问题**，仅输出原理分析、机制冲突、疑点定位结论。

核心研究目标：排查 vllm\-ascend 中 **TriAttention 开启后，Prefix\-Caching 失效** 的潜在 Bug。

## 二、基础运行环境配置（vllm\-ascend 启动完整参数）

### 2\.1 核心环境变量（TriAttention 调度参数）

```bash
# 全局开关：TriAttention 功能总控
export ENABLE_TRIATTENTION=1

# TriAttention KV 缓存预算：Decode 阶段逻辑保留 Token 数（核心参数）
export TRIATTN_RUNTIME_KV_BUDGET=4096
# TriAttention 稀疏统计模型路径
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=triattention/triattention/triattention/vllm/stats/qwen3_32b_int4_stats.pt
# 分片长度、窗口大小配置
export TRIATTN_RUNTIME_DIVIDE_LENGTH=128
export TRIATTN_RUNTIME_WINDOW_SIZE=128

# 调度与优化策略配置
export TRIATTN_RUNTIME_SCORING_BACKEND=auto
export TRIATTN_RUNTIME_FAST_RECENCY_ACCURACY_GUARD=1
export TRIATTN_RUNTIME_DEFER_PREFILL_COMPRESSION_ON_ASCEND=1
export TRIATTN_RUNTIME_ENABLE_ASYNC_COMPRESSION_BOUNDARY=0
export TRIATTN_RUNTIME_EARLY_INSTALL_PROXY_ON_ASCEND=1
export TRIATTN_RUNTIME_PREINSTALL_INPUT_PATCH=1
export TRIATTN_RUNTIME_ENABLE_PACKED_POS_DELTA_ON_ASCEND=0
export TRIATTN_RUNTIME_TRIM_ASCEND_V1_BLOCK_TABLE=0
export TRIATTN_RUNTIME_FORCE_EAGER_MULTI_REQ_ON_ASCEND_EFFECTIVE_OVERRIDES=0

# 性能与日志配置
export TRIATTN_RUNTIME_MAX_COMPRESSIONS_PER_STEP_ON_ASCEND=4
export TRIATTN_RUNTIME_LOGGING=0
export TRIATTN_RUNTIME_PERF_PROFILE=0
export TRIATTN_RUNTIME_E2E_PROFILE=0
export TRIATTN_RUNTIME_PHASE_PROFILE=0
export HF_DATASETS_OFFLINE=1
export TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM=0
```

### 2\.2 vllm 启动服务参数

```bash
vllm serve Qwen3-32B \
    --max-model-len 40960 \
    --served-model-name Qwen3-32B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --block-size 128 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --port 8000 \
    --enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,32,64]}'
```

## 三、关键参数语义定义（核心规则）

### 3\.1 Base 基线环境定义

`ENABLE_TRIATTENTION=0`：关闭所有 TriAttention 逻辑，完全使用 vllm\-ascend 原生调度、KV 缓存、Prefix\-Caching 机制，作为实验基线。

### 3\.2 TriAttention 核心机制定义

`TRIATTN_RUNTIME_KV_BUDGET` 是本次问题核心参数：

- 作用阶段：**仅 Decode 生成阶段**生效

- 核心逻辑：控制 vllm\-ascend 常驻保留的 Token 数量，超出预算的 Token 为**逻辑驱逐**，非物理删除

- Prefill 阶段规则：TriAttention 开启状态下，Prefill 阶段会完整预填充所有输入 Token，不会主动裁剪、驱逐前缀缓存

- 设计预期：Prefix\-Caching 的前缀缓存机制与 TriAttention 逻辑驱逐机制相互独立，互不冲突

## 四、实验测试方案与环境变量

### 4\.1 统一测试流程

基于 aisbench 服务端压测，**同一输入请求连续发送两次**：输入 Token 长度 20k，输出 Token 长度 1k，统计两次请求的 TTFT（首包响应时间）。

### 4\.2 测试变量分组

- 基线组：Base（ENABLE\_TRIATTENTION=0，原生 Prefix\-Caching）

- 实验组1：TriAttention 开启 \+ KV Budget = 2k

- 实验组2：TriAttention 开启 \+ KV Budget = 4k

## 五、实验数据结果

|测试环境|第一次请求 TTFT（ms）|第二次请求 TTFT（ms）|缓存生效状态|
|---|---|---|---|
|Base（关闭 TriAttention）|12462\.8|62\.1|正常生效（前缀缓存命中，TTFT 大幅下降）|
|TriAttention \+ 2k Budget|17355\.4|16401\.4|缓存失效（两次 TTFT 基本一致）|
|TriAttention \+ 4k Budget|17251\.0|15302\.2|缓存失效（两次 TTFT 基本一致）|

## 六、理论预期 vs 实际现象（核心矛盾）

### 6\.1 理论设计预期

TriAttention 的 KV Budget 仅作用于 **Decode 阶段的逻辑驱逐**，不干预 Prefill 阶段的前缀缓存写入与存储：

1. 第一次请求：Prefill 完整 20k 输入 Token，Prefix\-Caching 正常缓存全部前缀 KV；Decode 阶段仅逻辑驱逐超出 Budget 的 Token

2. 第二次请求：命中前缀缓存，无需重复 Prefill 20k Token，TTFT 应和 Base 环境一致，出现大幅下降

### 6\.2 实际异常现象

开启 TriAttention 后，无论 2k/4k KV Budget，第二次请求 TTFT 无明显优化，**Prefix\-Caching 完全失效**，与原生设计逻辑冲突，可判定存在机制 Bug。

## 七、分析要求（严格约束）

1. **分析维度**：仅基于现有配置、参数语义、vllm\-ascend 执行流程、TriAttention 源码机制做**纯代码/逻辑层面分析**，不落地修改、不调试、不补代码

2. **核心排查方向**

    - TriAttention 逻辑是否覆盖/篡改了 Prefix\-Caching 的缓存注册、匹配、复用逻辑

    - KV Budget 逻辑驱逐是否误触发前缀缓存的物理失效/缓存块清空

    - TriAttention 的 Prefill 阶段拦截、代理逻辑是否跳过了原生前缀缓存写入流程

    - Ascend 平台专属优化参数是否与 Prefix\-Caching 存在兼容性冲突

    - 缓存块调度、多请求执行逻辑是否导致前缀缓存无法命中

3. **输出内容要求**：梳理冲突原理、定位可疑代码流程、总结 Bug 根因猜想、给出针对性验证思路

> （注：部分内容可能由 AI 生成）
