# study/ 文档镜像

本目录是 Obsidian Vault `study/` 笔记的只读镜像，便于在 GitHub 上直接阅读。
**权威版本**仍以 `docs/` 下的对应文件为准（表格格式可能因 Obsidian 插件不同而略有差异）。

| 文档 | 内容 | 仓库权威版 |
|---|---|---|
| [Qwen 3.8 omni 音素识别实验设计.md](Qwen%203.8%20omni%20%E9%9F%B3%E7%B4%A0%E8%AF%86%E5%88%AB%E5%AE%9E%E9%AA%8C%E8%AE%BE%E8%AE%A1.md) | 完整实验设计（v3 设计快照，2026-10-28） | [`docs/experiment-design.md`](../docs/experiment-design.md) |
| [Qwen 3.8 omni 最小化实验设计.md](Qwen%203.8%20omni%20%E6%9C%80%E5%B0%8F%E5%8C%96%E5%AE%9E%E9%AA%8C%E8%AE%BE%E8%AE%A1.md) | ¥70 最小化实验设计（v3，含两档样本与成本估算） | [`docs/experiment-design-minimal.md`](../docs/experiment-design-minimal.md) |
| [Qwen 3.8 omni 最简实验报告.md](Qwen%203.8%20omni%20%E6%9C%80%E7%AE%80%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.md) | 一页结论：主结果、消融、双外部基线、成本（2026-10-30） | [`docs/report-minimal.md`](../docs/report-minimal.md) |
| [Qwen 3.8 omni 讯飞 ISE 基线详情.md](Qwen%203.8%20omni%20%E8%AE%AF%E9%A3%9E%20ISE%20%E5%9F%BA%E7%BA%BF%E8%AF%A6%E6%83%85.md) | 讯飞 ISE 详细基线：鉴权/协议/适配/交互账目/全量结果/性价比 | [`docs/baseline-ise-detail.md`](../docs/baseline-ise-detail.md) |
| [Qwen 3.8 omni 三模型弱项对比.md](Qwen%203.8%20omni%20%E4%B8%89%E6%A8%A1%E5%9E%8B%E5%BC%B1%E9%A1%B9%E5%AF%B9%E6%AF%94.md) | Qwen / SDHY / 讯飞 ISE 三模型弱项与逐对对比 | [`docs/model-weakness-comparison.md`](../docs/model-weakness-comparison.md) |

同步方式：`python scripts/sync_study_docs.py --vault-dir <Obsidian study/ 目录>`。
脚本只做复制与链接改写（Obsidian `[[wiki]]` → Markdown 相对链接），不改正文。
