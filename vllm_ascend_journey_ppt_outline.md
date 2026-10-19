# PPT 大纲（18 页）：从一份日志看 vLLM-Ascend 的六大模块层

> **用途**：交付给 PPT 编辑 Agent，据此完成排版与配图。
> **内容来源**：`vllm_ascend_journey.html`（已删减为 6 区块主线版，暗色交互页）。

---

## 全局约定（编辑前先读）

**每页内容预算（硬约束，防止再次塞满）**：
- 标题 1 个；上屏要点 **≤ 4 条**（每条 ≤ 一行，短句）
- 日志块 **≤ 1 个**（≤ 3 行）；配图 **1 张**
- 超出的细节一律写进**讲者备注**，不上屏

**其他约定**：
- **六层顺序**：平台与前端 → 引擎层 EngineCore → 执行层 → 计算层 → 采样·编译·调优 → 输出层
- **案例底数**（各页复用，不重复展开）：vLLM `0.30.0`；模型 `GLM-5.3-Flash-W8A8-MXFP8-0918`（NFS 314 GiB）；`DP=8 / EP=8`、`TP=1 / PP=1`；`max_model_len=133,120`；量化 `ascend`（modelslim，W8A8 + MXFP8）；投机 `MTP`（num_speculative_tokens=5）；启动 `16:35:31 → 16:40:24`（293 s）；日志约 `2800` 行
- **模块的处理原则**：模块名只在**所属层页**出现（点到即止，不逐个介绍）；P5 总览图**不列模块**

---

## P1 · 标题

**主标题**：当你 `vllm serve` 时，究竟发生了什么？

**副标题**：从一份真实日志，看 vLLM-Ascend 的六大模块层

**角标**：vLLM 0.30.0 ｜ GLM-5.3-Flash-W8A8-MXFP8 ｜ DP=8 / EP=8 ｜ 约 2800 行日志

**讲者备注**：一次 serve = 293 秒、26 个进程、2800 行日志。这 5 分钟里机器到底干了什么？

---

## P2 · 目录

- 一、什么是 vLLM-Ascend
- 二、它怎么接入 vLLM
- 三、认识它的结构（六大层）
- 四、逐层从日志看模块
- 五、总结与排障

**讲者备注**：全片走一条主线 —— 是什么 → 怎么接 → 什么结构 → 逐层看 → 收束。

---

## P3 · 什么是 vLLM-Ascend

**一句话（大字号）**：它是 vLLM 的**一份 MonkeyPatch + 一个插件** —— **不 fork vLLM**，而是"寄生"在 vLLM 上。

**三点（每点一行）**：
- **寄生**：作为插件挂在 vLLM 上，vLLM 主仓一行不改
- **改写**：import 时把内部实现换成本地 NPU 实现
- **取舍**：升级跟着上游走，代价是与 vLLM 内部命名强耦合

**视觉**：一个灰色 `vLLM` 方块，外面套一层蓝色 `vllm-ascend` 外壳。

**讲者备注**：对比 fork 的成本 —— fork 要持续 rebase 上游；插件式只维护适配层，但上游改命名就得跟着修 patch。

---

## P4 · 它怎么接入 vLLM（两个机制）

**机制一 · 被发现（entry points）**
- `setup.py` 登记 `vllm.platform_plugins: ascend`
- vLLM 启动扫 site-packages → 调 `register()` → 拿回 `NPUPlatform`

**机制二 · 改行为（MonkeyPatch）**
- `patch/*` 以 import 副作用生效；约 48 个文件
- 把 vLLM 内部属性 / 方法重绑成 Ascend 实现

**日志锚点**：
```
INFO  __init__.py:54   Available plugins for group vllm.platform_plugins: - ascend -> vllm_ascend:register
INFO  __init__.py:272  Platform plugin ascend is activated
```

**视觉**：左右两张并列卡片 —— 左「被发现」、右「改行为」，各引一条箭头指向中间的 vLLM。

**讲者备注**：日志里 `patch_*` 出现 ×24，就是因为这些 patch 在 import 时批量生效。

---

## P5 · 结构总览：六大层

**一句话**：请求从第 1 层进、第 6 层出，中间四层各司其职。

**六层（只列名 + 一个动词）**：
- 1 平台与前端 —— **进**（请求入口 + Ascend 接入）
- 2 引擎层 —— **调度**
- 3 执行层 —— **广播**
- 4 计算层 —— **算**
- 5 采样·编译·调优 —— **编译 / 规划**
- 6 输出层 —— **回**

**视觉**：六层骨架图（截 HTML `#skeleton` 骨架区），每层一个色块 + 层名。**本页不列任何模块。**

**讲者备注**：这是全场的"地图"，后面 P6–P14 每页点亮一层；记住"进→调度→广播→算→编译→回"这条线。

---

## P6 · 层 1 · 平台与前端：它管什么

**职责（一句话）**：请求从这里进，Ascend 从这里被接入。

**五个动作（每个一行）**：
- **CLI/Config**：几十个参数 → 一份 `VllmConfig`
- **插件系统**：发现 `ascend`
- **Platform**：设备 / dtype / 后端 / 量化的总接口
- **patch**：import 时改写 vLLM
- **API Server**：对外提供 HTTP

**视觉**：横向流程条：CLI → 插件 → Platform → patch → 配置；右端接 API Server。

**讲者备注**：本页只讲"有哪些模块、各干什么"；日志实证放下一页。

---

## P7 · 层 1 · 平台与前端：日志实证

**一句话**：三条日志，正好对应「发现 → 激活 → 改写」三步。

**日志锚点**：
```
INFO  __init__.py:54                Available plugins ... - ascend -> vllm_ascend:register
INFO  __init__.py:272               Platform plugin ascend is activated
INFO  patch_kv_cache_dtype.py:115   Patched CacheConfig.cache_dtype to accept 'int8'
```

**视觉**：日志放大居中；右侧三个标注框分别指向三行。

**讲者备注**：banner 与 non-default args 也在这层 —— 排障时最先看这两行，配置错都在这里现形。

---

## P8 · 层 2 · 引擎层 EngineCore

**职责（一句话）**：每个 DP rank 的"大脑"，跑主循环：调度 → 执行 → 输出。

**三个模块（每个一行）**：
- **EngineCore**：V1 引擎主循环
- **Scheduler**：准入 + token 预算 + KV block 分配
- **KV Cache Manager**：管理每 rank 的 KV block

**日志锚点**：
```
INFO  core.py:123  Initializing a V1 LLM engine (v0.30.0) with config: … data_parallel_size=8 …
INFO  core.py:379  init engine (profile, create kv cache, warmup) took 20.12 s
```

**视觉**：`schedule → execute → output` 三节点闭环。

**讲者备注**：引擎配置行是排障最该逐字段核对的一行 —— 并行、投机、量化都在里面。

---

## P9 · 层 3 · 执行层

**职责（一句话）**：把引擎的指令广播到真正干活的进程。

**三个模块（每个一行）**：
- **进程编排**：主进程拉起并监控所有子进程
- **Executor**：把 `SchedulerOutput` 广播给各 Worker
- **NPUWorker**：本进程内的设备与模型

**一个数**：**26 进程 = 1 主 + 8 API Server + 1 DP 协调器 + 8 EngineCore + 8 NPUWorker**

**日志锚点**：
```
INFO  utils.py:253  Started 8 API server processes
INFO  parallel_state.py:2267  rank 2 … is assigned as DP rank 2, PP rank 0, TP rank 0, EP rank 2
```

**视觉**：进程树（截 HTML `#topology` 拓扑区上半）：主 → {API ×8 · DP 协调器 · EngineCore ×8} → NPUWorker ×8。

**讲者备注**：之后所有 Worker 日志都带前缀 `(Worker_DPx_EPx pid=…)`，排障靠它定位是哪个 rank。

---

## P10 · 层 4 · 计算层：模块与运行时

**职责（一句话）**：真正做计算的地方；运行时组输入 → forward → 采样。

**四个模块（每个一行）**：
- **ModelRunner**：组输入、跑 forward、触发采样
- **并行状态 / 通信器**：DP / EP / TP 的通信底座
- **模型注册 / 实现**：GLM 等模型在 Ascend 上的实现
- **投机推理**：MTP 草稿模型管理

**运行时三步**：`_prepare_inputs` → forward（Attention 读写 KV、MoE 路由、ACL 图回放）→ 采样

**视觉**：单 Worker 内部流程：输入 → forward（内部分 Attention / MoE 两支）→ 输出。

**讲者备注**：forward 里主要两件事 —— Attention 读写 KV、MoE 按专家路由（EP=8）。

---

## P11 · 层 4 · 计算层：装资产

**三件事（每个一行）**：
- **权重**：8 个 Worker 各读 NFS checkpoint，单 rank **53.31 GB / 50.41 s**
- **量化**：**W8A8 + MXFP8**（modelslim）
- **专家 + 投机**：EP=8 → **本地 36 / 全局 288** 专家；MTP draft 与主模型**共享 embedding**

**日志锚点**：
```
INFO  model_runner_v1.py:4449  Loading model weights took 53.3064 GB
INFO  expert_map_manager.py:245  [EP Rank 2/8] Local/global number of experts: 36/288
```

**视觉**：NFS 上的 checkpoint → 8×Worker 并行装载的示意（8 条并行箭头）。

**讲者备注**：8 个 rank 各读一遍同一份 checkpoint，NFS 是启动期主要瓶颈；这也是"权重 53.31 GiB"的来源。

---

## P12 · 层 5 · 采样·编译·调优：编译与算子

**职责（一句话）**：把模型编译成可回放的图，并给算子做融合。

**两个模块（每个一行）**：
- **编译 / 图**：融合 pass + ACL 图捕获 / 回放
- **算子 / 内核**：`ops/` Python 覆盖、`ops/triton/` 内核、`csrc/` 自定义算子

**一个数**：图捕获 **7 s / 0.25 GiB**，之后推理直接 replay。

**日志锚点**：
```
INFO  gpu_model_runner.py:6929  Graph capturing finished in 7 secs, took 0.25 GiB
INFO  entry.py:139              ACL graph replay is active
```

**视觉**：两态示意 ——「图捕获（一次性）」→「replay（每次推理）」。

**讲者备注**：编译只做一次；replay 是每次推理都在走的路。

---

## P13 · 层 5 · 采样·编译·调优：KV 显存与就绪

**KV 规划（三个数，一条链）**：**20.03 GiB → 499,200 tokens → 133k 请求并发 3.75×**

**就绪收尾（两行）**：
- NUMA 绑核（资源调优）
- FastAPI 路由挂载 ×8

**日志锚点**：
```
INFO  kv_cache_utils.py:2395  NPU KV cache size: 499,200 tokens, … for 133,120 tokens per request: 3.75x
INFO  entry.py:139            Starting vLLM server on http://0.0.0.0:8017
```

**视觉**：KV 容量推导链（20.03 GiB → 499,200 tokens → 3.75×）；下方一行上线日志。

**讲者备注**：hybrid 模型 4 个 KV group，每组 4480；133,120 = max_model_len，3.75× = 能并发几个满长度请求。

---

## P14 · 层 6 · 输出层

**职责（一句话）**：把算出来的 token 送回请求方。

**三步（每个一行）**：
- **Scheduler 记账**：更新请求状态、释放 / 复用 KV block
- **增量 detokenize**：token → 文本
- **返回**：构造 `RequestOutput` → HTTP 响应（流式时每个增量都走）

**日志锚点**：
```
INFO  request_logger.py:98  Generated response chatcmpl-…: output: '…', finish_reason: stop, usage: {prompt_tokens: 11, completion_tokens: 312, total_tokens: 323}
```

**视觉**：流水线：token → detokenize → `RequestOutput` → HTTP 响应。

**讲者备注**：到这一步，一次请求闭环；KV block 在这里被还回去给下一个请求用。

---

## P15 · 一次请求的完整路径（回顾）

**一句话**：把六层串起来 —— 前端进 → 引擎调度 → 执行广播 → 计算 → 采样 / 编译 → 输出。

**视觉**：横向贯穿六层的请求流线；每层下方只标一个动作词（进 / 调度 / 广播 / 算 / 编译 / 回）。

**讲者备注**：回扣 P5 的地图，确认听众能把六层连成一条线。

---

## P16 · 总结

**一句话（大字号）**：vLLM-Ascend = **插件接入 + 平台实现 + 运行时 patch + NPU 算子栈**。

**两个关键机制**：
- entry points 让它**被发现**
- MonkeyPatch 让它**改行为**

**视觉**：六层光轴收束成一条竖线。

**讲者备注**：不 fork 是理解整个项目的钥匙 —— 结构上就是"通用 vLLM + 一层 Ascend 适配"。

---

## P17 · 排障心法（这套认知怎么用）

**四个检查点（按顺序）**：
1. **banner + non-default args** —— 配置错误都在这两行现形
2. **引擎配置行（P8）** —— 逐字段核对并行 / 投机 / 量化
3. **free memory 明细 + Available KV cache（P13）** —— 显存问题两行对账
4. **`(Worker_DPx_EPx)` 前缀 + 开 DEBUG** —— 定位到具体 rank 的算子问题

**视觉**：竖向四个检查点，前面各配一个图标（参数 / 齿轮 / 内存 / 放大镜）。

**讲者备注**：这页是把前面知识变成"可操作动作"的地方。

---

## P18 · 结束

**内容**：谢谢观看 / Q&A

**附注（小字）**：参考日志来自一次真实 `vllm serve` 运行；**推理阶段 R1–R6 的日志按真实字段重建** —— 原日志在就绪后于 16:43 因 NPU AI Core Error 崩溃，未跑成完整请求。

**视觉**：品牌 logo + 交互页面入口（链接 / 二维码位）。

---

## 附：素材地图（HTML 区块 → 幻灯片）

| 幻灯片 | 可从 HTML 直接截取 | 建议 |
|---|---|---|
| P5 | `#skeleton` 六层骨架 | 纯视觉，不加模块 |
| P6–P7 | `#journey` 选中 L1–L5 的卡片 | P6 取流程条，P7 取日志锚点 |
| P8 | `#journey` 选中 L7 | 引擎配置行放大 |
| P9 | `#topology` 进程树上半 | ×8 画成堆叠卡片 |
| P10 | `#journey` 选中 L9 | 单 Worker 内部流程 |
| P11 | `#journey` 选中 L10 / L12 | 权重条 + 专家数 |
| P12–P13 | `#journey` 选中 L14–L16 | KV 推导链 + 上线日志 |
| P14 | `#journey` 切到推理旅程 R6 | 输出流水线 |
| P15–P16 | Hero 光轴 + 骨架 | 收束成竖向检查点 |

