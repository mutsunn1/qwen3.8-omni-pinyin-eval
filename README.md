# Qwen3.8-Omni 普通话音素识别与一致性检测实验框架

<!-- badges -->
[![CI](https://github.com/mutsunn1/qwen3.8-omni-pinyin-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/mutsunn1/qwen3.8-omni-pinyin-eval/actions/workflows/ci.yml)
[![Repo](https://img.shields.io/badge/GitHub-qwen3.8--omni--pinyin--eval-181717?logo=github)](https://github.com/mutsunn1/qwen3.8-omni-pinyin-eval)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

按 `docs/experiment-design.md`（设计文档冻结快照，原始来源：Obsidian Vault
`study/Qwen 3.8 omni 音素识别实验设计.md`）实现的 zero-shot 评测框架：

* **RQ1** Qwen3.8-Omni 能否直接从音频识别声母、韵母、声调？
* **RQ2** 给定目标拼音时，模型能否判断目标文本与音频是否一致？
* **RQ3** 模型对目标文本的依赖有多大？是否会被错误目标“带偏”？
* **RQ4** 音素类别、提示条件、思考模式、音频质量、输出格式如何影响表现？

数据：`HaitongSUN/prosodic-abx` 的 `mandarin_tone_syn` config（commit 固定，
6160 条 16 kHz 单声道 WAV；4 个 TTS voice × 1540；声调 1–4 各 1540）。
模型：阿里云百炼 **Qwen3.8-Omni-Flash**（`qwen3.8-omni-flash`）。

> 当前状态：框架已按 **2026-10-28 v2 设计更新**（实验一/二只评声调、新增检测实验、
> 新增 Prompt 模板章节）实现，离线测试全绿；`all-mock` 可在无 API key 的情况下跑通
> `prepare-data → build-tasks → generate(mock) → score → report` 全链路。
> 真实 API 调用尚未执行（本机未配置 `DASHSCOPE_API_KEY` / 业务空间 `DASHSCOPE_BASE_URL`）。
> v2 设计变更对应 Obsidian 设计文档 git commits `f4caf99` / `eb5cbe8`（2026-10-28），
> 快照见 `docs/experiment-design.md`。

---

## 1. 快速开始（统一环境，克隆即用）

所有操作通过 [`uv`](https://docs.astral.sh/uv/) 管理：Python 版本由 `.python-version`
固定为 **3.12**，依赖由 `uv.lock` 锁定，环境变量统一放在 `.env`（模板 `.env.example`）。
不需要手动 `activate` 虚拟环境。

```bash
git clone https://github.com/mutsunn1/qwen3.8-omni-pinyin-eval.git
cd qwen3.8-omni-pinyin-eval

scripts/setup.sh            # 或 make setup：uv sync --locked + 生成 .env + 自检
# 编辑 .env 填入 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL（只跑 mock 时可留空）

make mock                   # 离线全链路自测：prepare-data → build-tasks → generate(mock) → score → report
make test                   # 238 个离线测试
make lint                   # ruff
```

等价的显式命令：

```bash
uv sync --locked                          # 创建 .venv（Python 3.12）并安装锁定依赖
cp .env.example .env                      # 首次运行时生成；.env 已被 gitignore
uv run pinyin-eval check                  # 环境 / 数据 / 模型配置自检
uv run pinyin-eval prepare-data           # 下载 parquet（~100MB）+ 导出音频 + manifest
uv run pinyin-eval build-tasks --validate # 构造任务 + 检测集 + 消融子集，并校验最小对计数
uv run pinyin-eval all-mock --audios 40   # 离线全链路自测（mock client，无需 key）
```

离线自测产物在 `results/runs/<run_id>/`：`report.md`、`metrics.csv`、`items.parquet` 等
（`data/`、`results/` 已 gitignore，可用命令再生）。

### 统一环境说明

| 项 | 统一方式 |
|---|---|
| Python | `.python-version` = 3.12；`pyproject.toml` `requires-python >= 3.12`；uv 自动下载/选择 |
| 依赖 | `uv.lock` 锁定（`uv sync --locked` 复现）；dev 依赖在 `[dependency-groups].dev` |
| 环境变量 | `.env`（模板 `.env.example`，gitignore）；`python-dotenv` 语义的最小加载器，shell 变量优先 |
| 配置入口 | `$PINYIN_EVAL_CONFIG` 或 `--config`；默认 `configs/base.yaml` |
| 常用命令 | `Makefile`（`make setup/check/data/tasks/mock/test/lint/ci`）+ `scripts/setup.sh` |
| CI | `.github/workflows/ci.yml`：`uv sync --locked` → ruff → pytest（离线，不需要 API key） |

## 2. 真实实验（需要阿里云百炼凭证）

```bash
# 推荐：把凭证写进 .env（模板见 .env.example），也支持直接 export
#   DASHSCOPE_API_KEY=sk-...
#   DASHSCOPE_BASE_URL=https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1

uv run pinyin-eval smoke                    # 1 次真实调用，验证 key / 模型 id / base_url
uv run pinyin-eval estimate --tasks all --profile main      # 预估主实验调用量（107,896 次）
uv run pinyin-eval generate --tasks all --profile main      # 主实验全量
uv run pinyin-eval generate --tasks all --profile ablation --conditions all   # 消融
uv run pinyin-eval score --run-id <主实验 run_id>
uv run pinyin-eval report --run-id <主实验 run_id>
```

设计文档的调用预算（已用 `estimate` 实测，含请求去重）：

| 阶段 | 数据 | 条件数 | API 调用 |
|---|---|---:|---:|
| 主实验 | 全量 6160 音频 + 检测实验 + 全部最小对 | 1（`think1/clean/json`） | 107,896 |
| 消融 | 固定子集（core 768 + detection 1,536 + pairs 10,752 条任务） | 8 | 96,960 |
| 合计 | — | — | **204,856** |

> 模型参数、prompt 版本、数据集 commit、mapping sha256、条件网格都会写入
> `results/runs/<run_id>/run_manifest.json`；预测按 `request_id` 去重后逐条落盘，
> 中断后同一 `--run-id` 重跑会自动续跑，不重复计费。

## 3. 实验与数据流

```
data/raw/*.parquet ──prepare-data──> data/audio/**/*.wav + data/manifests/audio.parquet
                                              │
                                    build-tasks│
                                               ▼
                data/manifests/tasks/core.jsonl      # 实验一/二/三视图（tone/tone/free）
                data/manifests/tasks/detection.jsonl # 检测实验（target+target_tone → consistent）
                data/manifests/tasks/pairs.jsonl     # 实验四/五/六视图（最小对，A/B 等）
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

### 3.1 实验一 ~ 三 + 检测实验（全量 6160 音频）

实验一/二共用同一个“给定无调 target、只报声调”的 prompt（实验二用错误无调 target；
prompt 不提示 target 有误），实验三是自由识别；检测实验是独立任务：

| 视图 | 提示条件 | 输出 | 指标 |
|---|---|---|---|
| 实验一 | 给定**正确**无调拼音 | `{"tone": 1-4}` | Tone accuracy / Macro-F1 |
| 实验二 | 给定**错误**无调拼音 | `{"tone": 1-4}` | Tone accuracy / Macro-F1 |
| 实验三 | 无任何读音信息 | `{"initial","final","tone"}` | Initial/Final/Tone/Joint Acc + Macro-F1 |
| **检测实验** | `target`（无调）+ `target_tone` | `{"consistent": true\|false}` | EP/ER/Error-F1/FPR/FNR（正类=不一致） |

* 实验二错误注入：每音频 5 个“只改声母或韵母”的错误无调 target（2 改声母 + 2 改韵母 +
  1 额外段错误；`er/weng` 无同韵母其它声母，只有 3 个变体）。
* 检测实验错误注入：每音频 1 个一致样本 + 5 个不一致变体（2 改声母 / 2 改韵母 / 1 改声调）。
  主检测集按音频 1:1 配平（6160 负 + 6160 正，2/2/1 分层随机）；另建 initial/final/tone
  三个平衡子集。`er/weng` 无改声母变体 → initial 子集 6128 而非 6160，报告注明。
* 所有检测指标以 audio 聚类 bootstrap；检测实验本身的生成条件网格为 8 个
  （thinking/quality/form），提示固定。

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

每条音频跑 4 个提示条件：自由识别（free）/ 正确目标（tone）/ 错误目标（tone，仅 exp4/5）/
A/B 二选一。exp6 的声调对两侧无调 target 相同，v2 prompt 无法表达“错误目标”，因此只跑
free / correct_target / A/B 三个条件（见 `docs/decisions.md`）。主指标分别为 exp4 Initial、
exp5 Final、exp6 Tone 的 free 条件准确率，A/B 准确率单独报告（随机基线 50%）。

### 3.3 消融

* 全因子：`思考 {on,off} × 音频 {clean, noise_20db(SNR 20 dB 加性白噪声)} × 输出 {json,text}`。
* 提示条件也作为因子：实验一~三 3 水平、实验四~六 4 水平（exp6 3 水平，见上）；
  检测实验提示固定，只有 8 个生成条件。
* 每个提示条件/任务固定 256 条消融子样本（4 voice × 4 tone × 16 条）；检测实验保留
  256 个音频的全部 6 个任务（1,536 条）以保证 main/typed 检测子集都可算；最小对按
  pair_type 抽到 256 个实例（i/ü 80、ian/üan 48 全取）。同一子集的 8 个生成条件彼此配对。
* 噪声变体缓存于 `data/noisy/{audio_id}__snr20db.wav`（确定性随机种子，16 kHz PCM16）。

## 4. 指标与统计

| 指标 | 定义 | 适用范围 |
|---|---|---|
| Initial / Final / Tone Accuracy & Macro-F1 | 单维度识别 | 实验三、四、五、六（free 条件） |
| Joint Accuracy | 声母+韵母+声调全对 | 实验三（及其它 free 条件） |
| Tone Accuracy（tone prompt） | 给定无调/错误 target 时的声调识别 | 实验一、实验二 |
| EP / ER / Error-F1 | TP/(TP+FP)、TP/(TP+FN)、2PR/(P+R) | 检测实验（正类=不一致） |
| FPR / FNR | FP/(FP+TN)、FN/(FN+TP) | 检测实验 |
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
tests/                         # 238 个离线测试（无网络/无 key）
```

## 6. 模型调用细节（百炼 OpenAI 兼容接口）

* `model="qwen3.8-omni-flash"`，`modalities=["text"]`，`messages=[system, user]`；
  共用 system prompt 取自设计文档 v2 §五；
* 音频以 `{"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,...","format":"wav"}}` 传入；
* 结构化输出用 `response_format={"type":"json_object"}`（JSON 条件）；
* 输出 schema：tone `{"tone"}`、free `{"initial","final","tone"}`、detect `{"consistent"}`、
  A/B `{"choice"}`；自由文本标记：`声调=`、`声母=…；韵母=…；声调=`、`一致=是/否`、`选择=A/B`；
* 思考模式：Qwen3.8-Omni 默认开启思考；关闭时发送 `reasoning_effort="none"`；
  开启时不额外发送参数（使用模型默认力度），均记录在 `run_manifest.json`；
* 固定解码参数：`temperature=0, top_p=1, max_tokens=256`；
* 失败重试：指数退避 + 抖动，重试次数记录在每条 generation 记录中。

来源：[百炼 Qwen-Omni 文档](https://help.aliyun.com/zh/model-studio/qwen-omni)、
[深度思考文档](https://help.aliyun.com/zh/model-studio/deep-thinking)。

## 7. 可复现性

* 数据集 commit、mapping CSV sha256、config sha256、prompt 版本、模型 id、调用日期
  全部写入 `run_manifest.json`；当前 Python/依赖版本可通过 `uv.lock` 与
  `pinyin-eval check --json` 复核；
* `generations.jsonl` 为 append-only，按 `request_id` 去重；重复运行同一 run-id 自动续跑；
* `data/`、`results/` 均为可再生文件（`.gitignore`），任一环节可用 CLI 重建；
* prompt 全文存于任务清单（`prompt_texts`），提示词变更会改变 `request_id`，
  旧预测不会与新预测混用。

## 8. 测试

```bash
uv run pytest                 # 238 passed（离线）
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
* 最小对实验中 exp4/5/6 的 target-condition 提示沿用实验一/二的“只报声调” schema，
  设计文档“实验×指标矩阵”中 exp4 Initial / exp5 Final / exp6 Tone 的 ✓ 对应 free 条件；
  exp6 无法构造 wrong-target（两侧无调 target 相同），只有 3 个提示条件。
* 实验二的 5 个错误无调变体按“2+2+1（额外改声母或韵母）”实现；若设计意图不同，
  调整 `configs/base.yaml: tasks.exp2` 后重新 `build-tasks` 即可。
* `--include-wide-nasal` 可把 ian/iang、uan/uang 纳入前后鼻最小对（52 对），
  但会偏离设计文档的 40 对口径，默认关闭。

## 10. 引用与许可

* 代码：[MIT License](LICENSE)。
* 数据：[`HaitongSUN/prosodic-abx`](https://huggingface.co/datasets/HaitongSUN/prosodic-abx)
  （`mandarin_tone_syn` config，**CC-BY-4.0**，arXiv:2604.02102）；本仓库内
  `data/mapping/mandarin_tone_syn_target_mapping.csv` 为基于该数据集的 target→initial/final
  映射表，使用/再分发时请保留数据集署名与许可。
* 模型：阿里云百炼 Qwen3.8-Omni-Flash，使用时遵循[阿里云百炼服务条款](https://help.aliyun.com/zh/model-studio/)。
