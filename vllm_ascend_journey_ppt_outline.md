# PPT 大纲：当你 `vllm serve` 时，究竟发生了什么？

> 来源：`vllm_ascend_journey.html`（暗色交互版）
> 规模：16 页 ｜ 建议时长 25–30 分钟 ｜ 建议风格：暗色底 + 三色体系（蓝=Ascend 侧、灰=vLLM 侧、琥珀=两侧协同），等宽字体展示日志，每页一个主视觉
> 叙事主线：开场 → 全景 → 加载旅程 → 两大机制 → 进程拓扑 → 推理旅程 → 模块速查（汇总地图）→ 收尾

---

## 第 0 部分 · 开场（P1–P3）

### P1 封面
- **主标题**：当你 `vllm serve` 时，究竟发生了什么？
- **副标题**：用一次真实启动日志，走读 vLLM-Ascend 的模块与结构
- **角标**：vLLM 0.30.0 · GLM-5.3-Flash-W8A8-MXFP8 · DP=8/EP=8 · 约 2800 行日志
- **画面**：暗色背景 + 虚化日志流 + 一条青色光轴贯穿标题

### P2 为什么要跟着日志走
- vLLM 主仓不认识 NPU：vLLM-Ascend 是「插件 + 运行时改写」的适配层，入口分散在两套代码库
- 文档告诉你"是什么"，日志告诉你"以什么顺序、在哪个进程、实际跑了什么"
- 一次 serve = 293 秒、26 个进程、约 2800 行日志 —— 恰好覆盖全部关键模块
- 本 Talk 两条主线：**加载旅程**（把系统搭起来）+ **推理旅程**（让请求流过去）
- **画面**：左代码目录树、右日志流，中间箭头连通

### P3 案例环境卡
- vLLM **0.30.0**（V1 引擎）
- 模型 **GLM-5.3-Flash-W8A8-MXFP8-0918**（NFS 上 314 GiB）
- 并行 **DP=8 / EP=8 / TP=1 / PP=1**
- 量化 **ascend**（modelslim，W8A8 + MXFP8）
- 投机 **MTP**（num_speculative_tokens=5）· max_model_len=**133,120**
- **画面**：参数卡片墙（可直接截页面 hero chips）

---

## 第 1 部分 · 全景（P4–P5）

### P4 一张骨架，装下两条旅程
- 六层骨架：**接入·平台·前端 → 引擎层 EngineCore → 执行层 → 计算层 → 采样·编译·调优 → 输出层**
- 加载旅程 L1–L16 把骨架搭起来并点火；推理旅程 R1–R6 让请求在其中流动
- 颜色约定贯穿全场：蓝=Ascend、灰=vLLM、琥珀=协同
- **画面**：六层脊柱图（截页面骨架区），两条 journey 走线用不同颜色描出

### P5 4 分 53 秒的启动全景
- 16:35:31 → 16:40:24，共 **293 秒**，16 个阶段按真实时间戳排布
- 块的位置 = 起止时刻，**重叠 = 并行发生**（编排与加载大量并行）
- 三大耗时块：权重加载 ~50s（每 rank）· 引擎初始化 20.12s · 图捕获 7s
- **画面**：甘特图（截页面时间线），圈出三段耗时

---

## 第 2 部分 · 加载旅程（P6–P9）

### P6 发现与接入（L1–L5）
- **L1 CLI/配置**：几十个参数 → 一份贯穿全程的 `VllmConfig`；解析出架构串 `Glm5NextForConditionalGeneration`
- **L2 插件**：entry points 发现 `ascend` → 调 `register()` → 拿回 `NPUPlatform` 类路径
- **L3 平台**：NPUPlatform 接管一切设备决策 —— 重置 GPU 专属参数、选 Attention 后端、注册量化
- **L4 patch**：约 48 个 patch 文件在 import 时改写 vLLM 行为（×24 的来源）
- **L5 配置**：`ascend_config.py`（--additional-config）+ `VLLM_ASCEND_*` 环境变量
- 日志锚点：`Platform plugin ascend is activated` / `Patched cache_dtype to accept 'int8'`
- **画面**：五步横向流程条，下挂两条高亮日志

### P7 进程与引擎（L6–L8）
- **L6 编排**：主进程拉起 8 API server + 1 DP 协调器 + 8 EngineCore；任一子进程异常 → 整体收敛
- **L7 EngineCore**：每 rank 的大脑 —— 持有调度器与 KV 管理，跑 busy loop；打印**完整引擎配置行**（排障最该核对的一行）
- **L8 执行器**：`multiproc_executor` 广播指令；patch 把 Worker 入口换成 **NPUWorker**（日志前缀 `Worker_DPx_EPx`）
- 日志锚点：`Initializing a V1 LLM engine (v0.30.0)…` / `rank 2 … assigned as DP rank 2, EP rank 2`
- **画面**：进程树上三层（对应拓扑区上半）

### P8 模型装载（L9–L13）
- **L9 ModelRunner**：8 个 Worker 各自独立读 NFS checkpoint —— 单 rank **53.31 GB / 50.41s**
- **L10 并行**：EP=8 → 本地 36 / 全局 288 专家；Ascend 追加 `_MC2`/`_KVPP` 等专属通信组
- **L11 注册**：architectures → 模型类；需要自定义算子/缓存布局的用原生实现，其余薄封装
- **L12 量化**：modelslim W8A8+MXFP8；`hf_to_vllm_mapper` 做权重 key 映射
- **L13 MTP**：draft 模型与主模型**共享 embedding**，独立包一层 ACL 图
- **画面**：NFS → 8×Worker 装载示意 + 单 rank 权重条

### P9 容量与就绪（L14–L16）
- **L14 KV 规划**：按剩余显存反推容量 —— 20.03 GiB → **499,200 tokens**；133k 长请求最大并发 **3.75×**（4 组 ×4480）
- **L15 编译**：融合 pass（norm_quant、act_quant）+ ACL 图捕获 7s / 0.25 GiB → 推理时直接 replay
- **L16 就绪**：NUMA 绑核（topo_affinity）→ FastAPI 路由 ×8 → `Starting vLLM server on http://0.0.0.0:8017`
- **画面**：KV 容量推导链（20.03 GiB → 499,200 tokens → 3.75×）+ 上线日志放大

---

## 第 3 部分 · 两个关键机制（P10–P11）

### P10 机制一：entry points · vLLM 怎么"看见"Ascend
- vLLM 不需要被 fork：Python 打包标准的 entry points 就是官方扩展点
- `setup.py` 登记 `vllm.platform_plugins: ascend = vllm_ascend:register`
- 启动时 `load_plugins_by_group` 扫描 site-packages → 发现 ascend → 调 `register()`
- 拿回类路径字符串 → vLLM import 并实例化 **NPUPlatform**：此后所有设备决策走 Ascend 实现
- **画面**：五步扫描流程（截页面机制区动画终态）

### P11 机制二：MonkeyPatch · 没有 hook 的地方怎么改
- 没有官方扩展点的位置：用 **import 副作用**重绑 vLLM 内部属性/方法
- 两条路径：`patch/platform/`（~29 文件，进程早期全局生效）+ `patch/worker/`（~19 文件，Worker init 时）
- 三个真实例子：
  - 放开 KV cache dtype 的 int8 校验（MLA / indexer KV 需要）
  - attention block size 对齐 4480（三种缓存 page 对齐，避免整块浪费）
  - 替换 Worker 进程入口为 NPUWorker
- 代价：与 vLLM 内部属性名强耦合，升级需逐条核对（`patch/__init__.py` 有登记清单）
- **画面**：before/after 代码对照 + 三张例子卡

---

## 第 4 部分 · 进程拓扑（P12）

### P12 26 个进程：为什么日志重复 8 次
- **1 主进程 + 8 ApiServer + 1 DP 协调器 + 8 EngineCore + 8 NPUWorker = 26 个进程**
- per-rank 代码在 8 个进程各跑一遍（×8）；patch / 配置在每个进程 import 时各跑一次（×24）
- 日志量演示：只看一个 rank → 2800 行 ≈ 350 行（约 1/8，95% 是重复）
- **画面**：进程拓扑图 + 过滤开关前后对比

---

## 第 5 部分 · 推理旅程（P13）

### P13 一次请求的六步（R1–R6）
- **R1 接入**：FastAPI → InputProcessor（renderer + tokenizer）→ `EngineCoreRequest`，经 ZMQ 入引擎
- **R2 调度**：`Scheduler.schedule` 决定本步跑谁、各跑多少 token、KV block 怎么分（不逐请求打 INFO）
- **R3 执行**：Executor 广播 → `NPUWorker.execute_model`
- **R4 计算**：`_prepare_inputs` → forward：Attention 读写 KV、MoE 路由 + 专家并行、ACL 图回放
- **R5 采样**：NPU 采样 kernel；MTP 场景 draft 出 5 个候选 + RejectionSampler 校验接受前缀
- **R6 输出**：`update_from_output` 记账 → 增量 detokenize → HTTP 响应
- **画面**：六步泳道图（骨架图 + R1–R6 走线）

---

## 第 6 部分 · 模块速查（P14）

### P14 模块速查：24 个模块一张图
- 走完两条旅程，用一张汇总图收束：24 个模块按六层分组，蓝/灰/琥珀标注归属
- 用途：看到某个文件或某种行为 → 反查它在哪一层、归谁管；也便于现场问答时定位
- **画面**：速查网格（截页面速查区或按层重排）

---

## 第 7 部分 · 收尾（P15–P16）

### P15 排障心法（用这套认知做什么）
1. 先看 **banner + non-default args** —— 配置错误都在这两行现形
2. 再看 **L7 引擎配置行** —— 逐字段核对并行/投机/量化
3. 显存问题 → **free memory 明细 + Available KV cache** 两行对账
4. 疑似算子问题 → 找 `(Worker_DPx_EPx)` 前缀；开 DEBUG 看 `cudagraph_mode`
- 附注：本次日志 16:43 因 NPU AI Core Error 崩溃；R1–R6 按真实字段重建
- **画面**：日志片段上标注 4 个检查点

### P16 结尾
- 一句话：**vLLM-Ascend = 插件接入 + 平台实现 + 运行时 patch + NPU 算子栈**
- 两条旅程、一个拓扑、一张速查图 —— 看懂一次 serve，就看懂了整个系统
- 交互页面入口（链接/二维码位）
- **画面**：脊柱光轴收束成 logo

---

## 附：制作建议
- **视觉素材**：甘特图、骨架图、进程拓扑 SVG 均可直接从 HTML 页面截图（暗色底与 PPT 风格一致）
- **日志排版**：统一用等宽字体 + 时间戳/级别/来源着色，每页锚点日志不超过 3 行
- **节奏**：第 2、5 部分（两条旅程）是主体（约 15 分钟）；若压缩到 15 分钟版，可砍 P3/P14，P6–P9 合并为两页
