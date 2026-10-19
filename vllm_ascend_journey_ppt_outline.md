# PPT 大纲（12 页）：从一份日志看 vLLM-Ascend 的六大模块层

> **用途**：交付给 PPT 编辑 Agent，据此完成排版与配图。
> **内容来源**：`vllm_ascend_journey.html`（已删减为 6 区块主线版，暗色交互页）。
> **叙事逻辑**：是什么 → 有什么结构 → 逐层从日志看模块 → 总结。
> **全局风格**：暗色底；三色体系贯穿（**蓝 = Ascend 侧**、**灰 = vLLM 侧**、**琥珀 = 两侧协同**）；代码/路径/日志统一等宽字体。

---

## 全局约定（编辑前先读）

- **三色图例**每页角落常驻一个小徽标：`■ Ascend ■ vLLM ■ 协同`。
- **日志锚点**样式：黑底终端块，等宽字体，行内着色 —— 时间戳灰、级别（INFO/WARN）彩色、来源文件蓝、消息白。每页最多 3 行。
- **六层顺序**（P5–P10 一一对应）：平台与前端 → 引擎层 EngineCore → 执行层 → 计算层 → 采样·编译·调优 → 输出层。
- **案例数据**（每页可复用的底数）：vLLM `0.30.0`；模型 `GLM-5.3-Flash-W8A8-MXFP8-0918`（NFS 314 GiB）；`DP=8 / EP=8`、`TP=1 / PP=1`；`max_model_len=133,120`；quantization `ascend`（modelslim，W8A8 + MXFP8）；投机 `MTP`（num_speculative_tokens=5）；一次启动 `16:35:31 → 16:40:24`（293 s）；日志约 `2800` 行。
- **六层角色速记**：前端进 → 引擎调度 → 执行广播 → 计算 → 采样/编译 → 输出返回。

---

## P1 · 标题

**主标题**：当你 `vllm serve` 时，究竟发生了什么？

**副标题**：从一份真实日志，看 vLLM-Ascend 的六大模块层

**角标**：vLLM 0.30.0 ｜ GLM-5.3-Flash-W8A8-MXFP8 ｜ DP=8 / EP=8 ｜ 约 2800 行日志

**视觉**：暗色背景 + 虚化日志流；一条青色光轴贯穿标题；右下角放三色图例徽标。

**备注（口播）**：一次 serve = 293 秒、26 个进程、2800 行日志 —— 这 5 分钟里机器到底干了什么？

---

## P2 · 目录

- **什么是 vLLM-Ascend** —— 一份 MonkeyPatch，不 fork vLLM，而是作为插件存在
- **认识它的结构** —— 六大层与其中的重要模块
- **从一份日志逐层浏览模块**：
  1. 平台与前端
  2. 引擎层 EngineCore
  3. 执行层
  4. 计算层
  5. 采样 · 编译 · 调优
  6. 输出层
- **总结**

**视觉**：竖向流程条；六大层用三色描边（前端/引擎/输出偏灰，平台/计算偏蓝，执行/采样偏琥珀）。

---

## P3 · 什么是 vLLM-Ascend？

**核心一句话**：它是 vLLM 的**一份 MonkeyPatch + 一个插件** —— **不 fork vLLM**，而是"寄生"在 vLLM 上。

**两件事让它存在**：
- **被发现** —— entry points：`setup.py` 登记 `vllm.platform_plugins: ascend = vllm_ascend:register`；vLLM 启动时扫描 site-packages 发现它，调 `register()` 拿回 `NPUPlatform` 类路径并实例化。
- **改行为** —— MonkeyPatch：import `vllm_ascend` 时，`patch/*` 以"import 副作用"把 vLLM 内部属性/方法重绑成 Ascend 实现。

**结构上就是两套代码库**：vLLM 主仓（通用逻辑）＋ vllm-ascend（Ascend 适配层）。

**视觉**：左右两个代码库方块，中间用两条箭头相连 —— 上箭标"插件发现（entry points）"、下箭标"运行时改写（MonkeyPatch）"。

**备注**：强调"没有 fork"是关键设计取舍 —— 跟随上游升级成本低，代价是与 vLLM 内部属性名强耦合，升级需逐条核对 patch。

---

## P4 · 认识 vLLM-Ascend 的结构

**六层 + 重要模块**（这张图是全场地图，P5–P10 依次点亮）：

| 层 | 角色 | 重要模块 |
|---|---|---|
| 1 平台与前端 | 请求入口 + Ascend 接入点 | CLI/Config、插件系统、Platform(NPUPlatform)、patch、Ascend 配置、API Server |
| 2 引擎层 EngineCore | 每 rank 的调度大脑 | EngineCore、Scheduler（+ Ascend 变体）、KV Cache Manager |
| 3 执行层 | 把指令广播到干活的进程 | 进程编排、Executor、NPUWorker |
| 4 计算层 | 真正做计算 + 装资产 | ModelRunner、并行状态/通信器、模型注册/实现、权重加载、量化、投机推理 |
| 5 采样·编译·调优 | 编译成图 + 算显存 + 融合算子 + 就绪 | 算子/内核、编译/图、KV cache、采样、资源调优 |
| 6 输出层 | 把 token 送回去 | 输出处理器、Scheduler 记账、API Server 返回 |

**视觉**：六层骨架图（截 HTML `#skeleton` 骨架区），每层右侧挂出对应模块芯片；三色描边同上。

**备注**：提醒听众记住"从进到出"的六层顺序，后面每页对应一层。

---

## P5 · 从一份日志看「平台与前端」

**这层干什么**：请求从这里进，Ascend 从这里被接入 —— 解析参数、发现 Ascend、接管设备决策、改写 vLLM、对外提供 HTTP 服务。

**关键模块与动作**：
- **CLI/Config**（`vllm/entrypoints/cli/serve.py`）：几十个参数 → 一份贯穿全程的 `VllmConfig`
- **插件系统**（`vllm/plugins/__init__.py`）：entry points 发现 `ascend` → 调 `register()`
- **Platform / NPUPlatform**（`vllm_ascend/platform.py`）：设备、dtype、Attention 后端、量化的总接口
- **patch**（`vllm_ascend/patch/platform/*` + `patch/worker/*`）：约 48 个文件，import 时改写 vLLM（日志 ×24 的来源）
- **Ascend 配置**（`ascend_config.py` + `envs.py`）：`--additional-config` 与 `VLLM_ASCEND_*`
- **API Server**（`vllm/entrypoints/launchers/*`）：8 个进程对外服务

**日志锚点**：
```
INFO  __init__.py:54   Available plugins for group vllm.platform_plugins:  - ascend -> vllm_ascend:register
INFO  __init__.py:272  Platform plugin ascend is activated
INFO  patch_kv_cache_dtype.py:115  Patched CacheConfig.cache_dtype to accept 'int8'
```

**视觉**：五步横向流程条（CLI → 插件 → Platform → patch → 配置），下挂两条高亮日志。

---

## P6 · 从一份日志看「引擎层 EngineCore」

**这层干什么**：每个 DP rank 的"大脑"—— 持有调度器与 KV cache 管理器，跑 `run_busy_loop` 主循环（调度 → 执行 → 输出）。它打印的完整引擎配置行是排障时最该逐字段核对的一行。

**关键模块与动作**：
- **EngineCore**（`vllm/v1/engine/core.py`）：V1 引擎主循环
- **Scheduler**（`vllm/v1/core/sched/scheduler.py`）：准入 + token 预算 + KV block 分配 → `SchedulerOutput`；Ascend 专属变体在 `vllm_ascend/core/`
- **KV Cache Manager**：管理每 rank 的 KV block

**日志锚点**：
```
INFO  core.py:123  Initializing a V1 LLM engine (v0.30.0) with config: model=…, speculative_config=…, data_parallel_size=8, …
INFO  core.py:379  init engine (profile, create kv cache, warmup) took 20.12 s
```

**视觉**：引擎配置行放大展示；下方一个"调度 → 执行 → 输出"的循环示意（三节点闭环）。

---

## P7 · 从一份日志看「执行层」

**这层干什么**：把引擎的指令广播到真正干活的进程。

**关键模块与动作**：
- **进程编排**（`vllm/v1/engine/utils.py`、`vllm/v1/utils.py`、`coordinator.py`）：主进程拉起全部子进程并监控
- **Executor**（`vllm/v1/executor/multiproc_executor.py`）：把 `SchedulerOutput` 广播给各 Worker
- **NPUWorker**（`vllm_ascend/worker/worker.py`）：本进程内的设备与模型；patch 替换了 Worker 进程入口

**记住这个数**：**26 个进程 = 1 主 + 8 API Server + 1 DP 协调器 + 8 EngineCore + 8 NPUWorker**；per-rank 代码 ×8，patch/配置 import 时 ×24。

**日志锚点**：
```
INFO  utils.py:253      Started 8 API server processes
INFO  parallel_state.py:2267  rank 2 … is assigned as DP rank 2, PP rank 0, TP rank 0, EP rank 2
```
（后半行的进程前缀是 `(Worker_DPx_EPx pid=…)`）

**视觉**：进程树（截 HTML `#topology` 拓扑区上半）：主进程 → {API ×8 · DP 协调器 · EngineCore ×8} → NPUWorker ×8；×8 节点画成堆叠卡片。

---

## P8 · 从一份日志看「计算层」

**这层干什么**：真正做计算的地方，同时负责把所有资产装进 NPU。

**关键模块**：
- **ModelRunner**（`vllm_ascend/worker/model_runner_v1.py`）：组输入 → forward → 采样
- **并行状态 / 通信器**（`vllm/distributed/parallel_state.py`、`vllm_ascend/distributed/`）
- **模型注册 / 实现**（`vllm/model_executor/models/registry.py`、`vllm_ascend/models/`）
- **权重加载**（`vllm/model_executor/model_loader/*`、`vllm_ascend/model_loader/*`）
- **量化**（`vllm_ascend/quantization/`）
- **投机推理**（`vllm_ascend/spec_decode/`）

**装资产（启动侧）**：8 个 Worker 各自独立读 NFS checkpoint —— 单 rank **53.31 GB / 50.41 s**；**EP=8 → 本地 36 / 全局 288 专家**；量化 **W8A8 + MXFP8**；MTP draft 与主模型**共享 embedding**。

**运行时**：`_prepare_inputs` 组 attention metadata → forward（Attention 读写 KV、MoE 路由 + 专家并行、ACL 图回放）。

**日志锚点**：
```
INFO  modelslim_config.py:316  Using the vLLM Ascend modelslim Quantization now!
INFO  model_runner_v1.py:4449  Loading model weights took 53.3064 GB
INFO  expert_map_manager.py:245  [EP Rank 2/8] Local/global number of experts: 36/288
```

**视觉**：NFS 上的 checkpoint → 8×Worker 并行装载的示意；配一条单 rank 权重条（53.31 GB）。

---

## P9 · 从一份日志看「采样 · 编译 · 调优」

**这层干什么**：把模型编译成可回放的计算图、把显存算清楚、给算子做融合、最后收尾就绪。

**关键模块**：
- **算子/内核**（`vllm_ascend/ops/`、`ops/triton/`、`csrc/`）：Python 覆盖 + Triton 内核 + CANN 自定义算子
- **编译/图**（`vllm_ascend/compilation/`）：融合 pass + ACL 图捕获/回放
- **KV cache**（`vllm/v1/core/kv_cache_utils.py`、`vllm_ascend/core/kv_cache_interface.py`）
- **采样**（`vllm_ascend/sample/`）：NPU 采样 kernel + 投机校验
- **资源调优**（`vllm_ascend/cpu_binding.py`）：NUMA 绑核

**几个关键数**：KV 容量 **20.03 GiB → 499,200 tokens → 133k 请求并发 3.75×**（4 组 ×4480）；图捕获 **7 s / 0.25 GiB**，之后推理直接 replay。

**日志锚点**：
```
INFO  kv_cache_utils.py:2395  NPU KV cache size: 499,200 tokens, Maximum concurrency for 133,120 tokens per request: 3.75x
INFO  gpu_model_runner.py:6929  Graph capturing finished in 7 secs, took 0.25 GiB
INFO  entry.py:139  Starting vLLM server on http://0.0.0.0:8017
```

**视觉**：KV 容量推导链（20.03 GiB → 499,200 tokens → 3.75×）；下方"图捕获 → replay"两态示意；右侧放上线日志。

---

## P10 · 从一份日志看「输出层」

**这层干什么**：把算出来的 token 送回请求方。

**关键模块与流程**：
- **Scheduler 记账**（`update_from_output`）：更新请求状态、释放/复用 KV block、处理 preemption
- **输出处理器**（`vllm/v1/engine/output_processor.py`）：增量 detokenize（token → 文本）
- **API Server 返回**：构造 `RequestOutput` → 经 ZMQ 回 API server → HTTP 响应（流式时每个增量都走这条路）

**日志锚点**：
```
INFO  request_logger.py:98  Generated response chatcmpl-…: output: '…', finish_reason: stop, usage: {prompt_tokens: 11, completion_tokens: 312, total_tokens: 323}
```

**视觉**：流水线示意：token → 增量 detokenize → `RequestOutput` → HTTP 响应。

---

## P11 · 总结

- **一句话**：vLLM-Ascend = **插件接入 + 平台实现 + 运行时 patch + NPU 算子栈**。
- **六层协作一条线**：平台与前端（进）→ 引擎层（调度）→ 执行层（广播）→ 计算层（算）→ 采样·编译·调优（图/显存/融合）→ 输出层（回）。
- **两个关键机制**：entry points 让它**被发现**；MonkeyPatch 让它**改行为**。
- **排障心法（用这套认知能做什么）**：
  1. 先看 **banner + non-default args** —— 配置错误都在这两行现形
  2. 再看 **引擎配置行（P6）** —— 逐字段核对并行/投机/量化
  3. 显存问题 → **free memory 明细 + Available KV cache** 两行对账
  4. 疑似算子问题 → 找 `(Worker_DPx_EPx)` 前缀，开 DEBUG 看 `cudagraph_mode`

**视觉**：六层骨架光轴收束成一条竖线，线上标 4 个"检查点"。

---

## P12 · 结束

**内容**：谢谢观看 / Q&A。

**附注（小字）**：参考日志来自一次真实 `vllm serve` 运行；**推理阶段 R1–R6 的日志按真实字段重建** —— 原日志在就绪后于 16:43 因 NPU AI Core Error 崩溃，未跑成完整请求。

**视觉**：品牌 logo + 交互页面入口（链接 / 二维码位）。

---

## 附：素材地图（HTML 区块 → 幻灯片）

| 幻灯片 | 可从 HTML 直接截取 | 建议 |
|---|---|---|
| P4 | `#skeleton` 六层骨架 | 需另叠加模块芯片 |
| P5 | `#journey` 选中 L1–L5 的卡片 | 取流程条 + 日志锚点 |
| P6 | `#journey` 选中 L7 | 引擎配置行放大 |
| P7 | `#topology` 进程树上半 | ×8 堆叠卡片 |
| P8 | `#journey` 选中 L9/L10/L12 | 权重条 + 专家数 |
| P9 | `#journey` 选中 L14–L16 | KV 推导链 + 上线日志 |
| P10 | `#journey` 切到推理旅程 R6 | 输出流水线 |
| P11 | Hero 光轴 + 骨架 | 收束成竖向检查点 |

> 页面已删去「显存账本」「投机解码剧场」两节；KV 数字在 P9（L14）内呈现，MTP 只在 P8（L13 加载）与 P9（R5 采样）以文字提及。
