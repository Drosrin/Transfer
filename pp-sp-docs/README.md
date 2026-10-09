# PP 与 SP 学习文档（vLLM-Ascend）

基于 vllm-ascend `c40cf7bc1` 与 vllm `7d47ac2ddb`（均为 2026-10-08 main）源码整理的三份文档，建议按以下顺序阅读：

| 顺序 | 文档 | 内容 | 适合谁 |
| --- | --- | --- | --- |
| 1 | [PP_SP_方案与原理](PP_SP_方案与原理.md) | 背景、原理、演进史（FlashComm 兴衰、vLLM V1/V2 架构），代码无关 | 想理解"为什么"的所有读者 |
| 2 | [PP_SP_使用指南](PP_SP_使用指南.md) | 简介、部署、常见场景、最佳实践、FAQ | 部署/运维/选型 |
| 3 | [PP_SP_代码走读](PP_SP_代码走读.md) | 从参数到执行的完整代码链路，含大量代码片段与行号 | 想改代码/深入调试的开发者 |

图示说明：文档内的 `mermaid` 代码块在 GitHub / VS Code（装 Mermaid 插件）/ mkdocs-material 下直接渲染为图；`assets/sp_moe.png` 引自 vllm-ascend 官方文档仓库。

配套上游权威文档：

- [Pipeline Parallelism（官方）](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/pipeline_parallel.html)
- [Sequence Parallelism（官方）](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/sequence_parallelism.html)
- [Dynamic Chunked Pipeline Parallel（官方）](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.html)
