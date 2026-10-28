# Qwen3.8-Omni 普通话音素识别与一致性检测实验框架

按 `docs/experiment-design.md`（设计文档冻结快照，原始来源：Obsidian Vault
`study/Qwen 3.8 omni 音素识别实验设计.md`）实现的 zero-shot 评测框架：

* **RQ1** Qwen3.8-Omni 能否直接从音频识别声母、韵母、声调？
* **RQ2** 给定目标拼音时，模型能否判断目标文本与音频是否一致？
* **RQ3** 模型对目标文本的依赖有多大？是否会被错误目标“带偏”？
* **RQ4** 音素类别、提示条件、思考模式、音频质量、输出格式如何影响表现？

数据：`HaitongSUN/prosodic-abx` 的 `mandarin_tone_syn` config（commit 固定，
6160 条 16 kHz 单声道 WAV；4 个 TTS voice × 1540；声调 1–4 各 1540）。
模型：阿里云百炼 **Qwen3.8-Omni-Flash**（`qwen3.8-omni-flash`）。

> 当前状态：框架已完成并通过 226 个离线测试；`all-mock` 可在无 API key 的情况下跑通
> `prepare-data → build-tasks → generate(mock) → score → report` 全链路。
> 真实 API 调用尚未执行（本机未配置 `DASHSCOPE_API_KEY` / 业务空间 `DASHSCOPE_BASE_URL`）。

---

## 1. 快速开始

```bash
cd "qwen3.8 pinyin"
uv sync                                   # Python 3.12 + 依赖
uv run pinyin-eval check                  # 环境 / 数据 / 模型配置自检
uv run pinyin-eval prepare-data           # 下载 parquet（~100MB）+ 导出音频 + manifest
uv run pinyin-eval build-tasks --validate # 构造任务 + 检测集 + 消融子集，并校验最小对计数
uv run pinyin-eval all-mock --audios 40   # 离线全链路自测（mock client，无需 key）
```

离线自测产物在 `results/runs/mocksmoke/`：`report.md`、`metrics.csv`、`items.parquet` 等。

## 2. 真实实验（需要阿里云百炼凭证）

```bash
export DASHSCOPE_API_KEY="sk-..."          # 百炼 API Key
# 百炼要求使用“业务空间服务地址”，形如：
export DASHSCOPE_BASE_URL="https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

uv run pinyin-eval smoke                    # 1 次真实调用，验证 key / 模型 id / base_url
uv run pinyin-eval estimate --tasks all --profile main      # 预估主实验调用量（83,321 次）
uv run pinyin-eval generate --tasks all --profile main      # 主实验全量
uv run pinyin-eval generate --tasks all --profile ablation --conditions all   # 消融
uv run pinyin-eval score --run-id <主实验 run_id>
uv run pinyin-eval report --run-id <主实验 run_id>
```

设计文档的调用预算（已用 `estimate` 实测，含请求去重）：

| 阶段 | 数据 | 条件数 | API 调用 |
|---|---|---:|---:|
| 主实验 | 全量 6160 音频 + 全部最小对 | 1（`think1/clean/json`） | 83,321 |
| 消融 | 固定子集（core 768 + pairs 11,264 条任务） | 8 | 88,952 |
| 合计 | — | — | **172,273** |

> 模型参数、prompt 版本、数据集 commit、mapping sha256、条件网格都会写入
> `results/runs/<run_id>/run_manifest.json`；预测按 `request_id` 去重后逐条落盘，
> 中断后同一 `--run-id` 重跑会自动续跑，不重复计费。

## 3. 实验与数据流

```
data/raw/*.parquet ──prepare-data──> data/audio/**/*.wav + data/manifests/audio.parquet
                                              │
                                    build-tasks│
                                               ▼
                data/manifests/tasks/core.jsonl   # 实验一/二/三视图（共享提示条件任务）
                data/manifests/tasks/pairs.jsonl  # 实验四/五/六视图（最小对，4 提示条件）
                data/manifests/tasks/detection_sets.csv
                data/manifests/tasks/ablation_*_items.txt
                                               │
                                    generate   │   8 个生成条件 = 思考开/关 × 干净/SNR20dB × JSON/自由文本
                                               ▼
   results/runs/<run_id>/{generations.jsonl, planned_calls.jsonl, run_manifest.json}
                                               │
                                      score    ▼
   results/runs/<run_id>/{items.parquet, detection_rows.parquet, score_summary.json}
                                               │
                                      report   ▼
   results/runs/<run_id>/{report.md, metrics.csv, detection.csv, ablation.csv,
                          comparisons.csv, baselines.csv, confusion/*.csv, summary.json}
```

### 3.1 实验一 ~ 三（全量 6160 音频）

同一批音频上的 3 个提示条件，三者共享任务、按视图分析：

| 视图 | 提示条件 | 任务 | 主指标 |
|---|---|---|---|
| 实验一 | 给定**正确**无调拼音 | 判断一致 + 报真实声调 | 声调准确率（+检测负类） |
| 实验二 | 给定**错误**拼音 | 5 变体/音频（2 改声母、2 改韵母、1 改声调），判断一致 + 报真实声调 | 声调准确率（+检测正类） |
| 实验三 | 无任何读音信息 | 自由识别声母/韵母/声调 | Initial/Final/Tone/Joint Acc + Macro-F1 |

检测分析（实验一+二）按设计文档配平：每个音频 = 1 个一致样本（负类）+ 1 个不一致样本
（正类，2/2/1 分层随机），并以 audio 聚类 bootstrap；另按 initial/final/tone 分别构建
平衡子集。因 `er`、`weng` 在数据集中没有同韵母的其它声母（无法造“改声母”样本），
initial 子集为 6128 而非 6160——已在报告中注明。

### 3.2 实验四 ~ 六（最小对）

由映射表构造 2440 对最小对立对、11824 个 (pair × voice × tone) 实例（与设计文档计数表
逐项一致，`build-tasks --validate` 会断言）：

| 类别 | 对 / 实例 | 实验 |
|---|---:|---|
| 平翘舌 z/zh, c/ch, s/sh（排除 -i 组） | 36 / 576 | 四（声母） |
| 送气 b/p, d/t, g/k, j/q | 67 / 1072 | 四 |
| n/l | 21 / 336 | 四 |
| 前后鼻 an/ang, en/eng, in/ing | 40 / 640 | 五（韵母） |
| i/ü | 5 / 80 | 五 |
| ian/üan | 3 / 48 | 五 |
| 声调 6 对 | 2268 / 9072 | 六（声调） |

每个实例的每条音频跑 4 个提示条件：自由识别 / 正确目标 / 错误目标（给对侧最小对立项）/
A/B 二选一。主指标分别为 exp4 Initial、exp5 Final、exp6 Tone 的 free 条件准确率，
A/B 准确率与一致性检测单独报告。

### 3.3 消融

* 全因子：`思考 {on,off} × 音频 {clean, noise_20db(SNR 20 dB 加性白噪声)} × 输出 {json,text}`。
* 提示条件也作为因子（实验一~三 3 水平、实验四~六 4 水平）：实验一~三 24 个条件、
  实验四~六 32 个条件。
* 每个提示条件固定 256 条消融子样本（4 voice × 4 tone × 16 条）；最小对按 pair_type
  抽到 256 个实例（i/ü 80、ian/üan 48 全取）。同一子集的 8 个生成条件彼此配对，
  支持 McNemar / 配对 bootstrap。
* 噪声变体缓存于 `data/noisy/{audio_id}__snr20db.wav`（确定性随机种子，16 kHz PCM16）。

## 4. 指标与统计

| 指标 | 定义 | 适用范围 |
|---|---|---|
| Initial / Final / Tone Accuracy & Macro-F1 | 单维度识别 | 实验三、四、五、六（free 条件） |
| Joint Accuracy | 声母+韵母+声调全对 | 实验三（及其它 free 条件） |
| EP / ER / Error-F1 | TP/(TP+FP)、TP/(TP+FN)、2PR/(P+R) | 实验一+二 检测（正类=不一致） |
| FPR / FNR | FP/(FP+TN)、FN/(FN+TP) | 同上 |
| Confusion Matrix | 具体混淆对 | 全部识别维度 |
| A/B Accuracy | 二选一准确率（随机基线 50%） | 实验四~六 |

* 95% CI：bootstrap 2000 次，以 audio 为聚类单位、voice 为分层变量（设计文档要求）。
* 条件比较：同一批样本上的 McNemar 检验 + 配对 bootstrap 差值 CI；多重比较用 Holm 校正。
* 基线：声调 25%、A/B 50%、平衡检测集 50%、initial/final 多数类基线。
* 解析失败（无法解析 / 拒答 / 空回复 / API 错误）单独计入 `n_failures` 并报告比例，
  不进指标分母。设计文档中的 τ=0.5 仅适用于模型输出概率的判定；本项目要求模型输出
  类别判断（`consistent: true/false`），因此 τ 不参与计算，详见 `docs/decisions.md`。

## 5. 仓库结构

```
configs/base.yaml              # 数据/模型/任务/统计全部参数（唯一配置源）
docs/experiment-design.md      # 设计文档冻结快照
docs/contracts.md              # 模块间冻结契约（数据结构、函数签名、计数验收）
docs/decisions.md              # 关键决策与设计文档歧义的处理记录
docs/runbook.md                # 正式实验运行手册（含附录模板）
docs/schemas.md                # 数据产物 schema（manifest / items / detection）
src/pinyin_eval/
  config.py normalize.py records.py prompts.py parsing.py client.py runner.py
  scoring.py cli.py
  data/dataset.py data/minimal_pairs.py
  metrics.py stats.py report.py tasks.py
tests/                         # 226 个离线测试（无网络/无 key）
```

## 6. 模型调用细节（百炼 OpenAI 兼容接口）

* `model="qwen3.8-omni-flash"`，`modalities=["text"]`；
* 音频以 `{"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,...","format":"wav"}}` 传入；
* 结构化输出用 `response_format={"type":"json_object"}`（JSON 条件）；
* 思考模式：Qwen3.8-Omni 默认开启思考；关闭时发送 `reasoning_effort="none"`；
  开启时不额外发送参数（使用模型默认力度），均记录在 `run_manifest.json`；
* 固定解码参数：`temperature=0, top_p=1, max_tokens=256`；
* 失败重试：指数退避 + 抖动，重试次数记录在每条 generation 记录中。

来源：[百炼 Qwen-Omni 文档](https://help.aliyun.com/zh/model-studio/qwen-omni)、
[深度思考文档](https://help.aliyun.com/zh/model-studio/deep-thinking)。

## 7. 可复现性

* 数据集 commit、mapping CSV sha256、config sha256、prompt 版本、模型 id、调用日期
  全部写入 `run_manifest.json`；
* `generations.jsonl` 为 append-only，按 `request_id` 去重；重复运行同一 run-id 自动续跑；
* `data/`、`results/` 均为可再生文件（`.gitignore`），任一环节可用 CLI 重建；
* prompt 全文存于任务清单（`prompt_texts`），提示词变更会改变 `request_id`，
  旧预测不会与新预测混用。

## 8. 测试

```bash
uv run pytest                 # 226 passed（离线）
uv run ruff check src tests   # All checks passed
```

覆盖：规范化/解析、prompt 指纹去重、任务构造（实验一~六、检测集、消融抽样）、
数据准备与噪声注入、指标与统计、mock 端到端（含断点续跑、去重、失败记录）、报告生成
（含报告数字与 metrics/stats 直接调用逐项对拍、bootstrap CI 复算、CSV 列稳定、seed 复现）。

> 报告生成耗时参考：mock 96k items / 8 条件 run（n_boot=2000）约 25 秒；单条件 1.4k items 约 2 秒。
> 真实全量主实验（83k 请求 → ~137k items）报告约 1 分钟量级。

## 9. 已知限制 / TODO

* 尚未进行真实 API 调用：模型 id、`reasoning_effort` 取值、JSON 模式行为需在首次
  `smoke` + 小样本实跑后按 `docs/decisions.md` 记录确认；若百炼实际接口与文档有差异，
  只需调整 `configs/base.yaml` 的 `model.*` 与 `client.py` 的 `build_payload`。
* 预算估计未计入 token 价格：`estimate` 只统计调用次数；成本需按百炼 Omni 定价另算
  （每条 ~0.2k 提示 token + ~0.6s 音频）。
* 最小对实验中 exp4/5/6 的 target-condition 提示沿用“一致性 + 真实声调”输出 schema，
  设计文档“实验×指标矩阵”中 exp4 Initial / exp5 Final / exp6 Tone 的 ✓ 对应 free 条件；
  其余条件作为消融/一致性分析报告（见 `docs/decisions.md`）。
* `--include-wide-nasal` 可把 ian/iang、uan/uang 纳入前后鼻最小对（52 对），
  但会偏离设计文档的 40 对口径，默认关闭。
