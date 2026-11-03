# Qwen3.8-Omni 普通话音素识别与一致性检测实验框架

[![CI](https://github.com/mutsunn1/qwen3.8-omni-pinyin-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/mutsunn1/qwen3.8-omni-pinyin-eval/actions/workflows/ci.yml)
[![Repo](https://img.shields.io/badge/GitHub-qwen3.8--omni--pinyin--eval-181717?logo=github)](https://github.com/mutsunn1/qwen3.8-omni-pinyin-eval)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

面向 `HaitongSUN/prosodic-abx`（`mandarin_tone_syn`，6160 条单音节 TTS 音频）的
zero-shot 评测框架，回答四个问题：

* **RQ1** Qwen3.8-Omni 能否直接从音频识别声母、韵母、声调？
* **RQ2** 给定目标拼音时，模型能否判断目标文本与音频是否一致？
* **RQ3** 模型对目标文本的依赖有多大？是否会被错误目标“带偏”？
* **RQ4** 提示条件、思考模式、音频质量、输出格式如何影响表现？

> 设计依据：`docs/experiment-design.md`（v3 冻结快照，2026-10-28；原始笔记
> `study/Qwen 3.8 omni 音素识别实验设计.md`）· 模型：阿里云百炼
> `qwen3.8-omni-flash` · 运行环境：Python 3.12 + uv（`uv.lock` 锁定）。

## 0. 当前状态与成果速览

框架已完整实现 v3 设计并通过真实 API 验证；完成两档最小化实验、两个外部基线
（SDHY P0-v2、讯飞 ISE）以及三模型对比；并完成**整句（utterance）pilot**：
任务 A 整句拼音转写（AISHELL-3）、B 整句发音评测 / C 整句朗读评分（OMPAL），
C1/C2 两个思考条件全量真实运行（run `sentence-v1`）。

| 运行 | 数据 | 条件 | 调用 | 实际成本 | 关键结果 |
|---|---|---:|---:|---:|---|
| `mini3-v1`（96 档） | 1,536 音频 | 4 | 59,968 | ¥12.37 | initial 0.952 / final 0.938 / tone 0.910 / joint 0.816；派生检测 F1 0.912 / FPR 0.184 |
| **`mini128-v1`（128 档，主结果）** | **2,048 音频** | 4 | 79,680 | **¥16.78** | **initial 0.962 / final 0.949 / tone 0.912 / joint 0.838**；派生检测 **F1 0.922 / FPR 0.162** |
| 检测 prompt 探针 | 512 音频 × 6 任务 × 3 变体 | 3 | 9,216 | ¥1.45 | 改 prompt 无法修复过度判不一致（FPR 0.76→0.77） |
| **`sentence-v1`（整句 pilot）** | AISHELL-3 300 句 + OMPAL 1,758 条（+82 native 锚点） | 2（C1/C2） | **7,796** | **¥8.29** | A joint **0.883**（C2 0.917）、句级完全正确 0.323（C2 0.440）；B macro-F1 0.553/0.558/0.601；C Pearson 0.352/0.581/0.393；解析 99.99%，验收通过 |
| 合计（Qwen） | — | — | **156,660** | **¥38.89 / ¥70（55.6%）** | 预算守卫未触发 |

**核心结论**

1. **thinking on 必须避免**：同一批音频上 tone −20pp、joint −19pp、exp6 错误信息 −44pp。
2. **检测不能用“模型直接判一致”**：直接判定 FPR 0.771 / Error-F1 0.720；改为
   “exp3 自由识别 → 程序比较目标”的**派生检测**后 F1 0.922 / FPR 0.162，且零额外调用。
3. **噪声主要伤声母/联合**：−12 ~ −13.5pp；声调鲁棒。自由文本仅对“只输出韵母”的
   exp5 有害（ü 丢失，−7 ~ −25pp），JSON 应作为主格式。
4. 主条件下模型能力：声母 **0.962**、韵母 **0.949**、声调 **0.912**、联合 **0.838**。
5. **整句 pilot**（`sentence-v1`，¥8.29）：整句拼音转写可用（joint 0.883 / C2 0.917，
   按单音节类别加权后与单音节基本持平、声调略好）；**发音错误定位不可用**
   （B macro-F1 0.55–0.60、声调 FNR 0.79，模型偏向“都读对”）；朗读评分只能算中等相关
   （C Pearson 0.35 / 0.58 / 0.39）。**thinking on 在整句上反转**：A/B 显著更好、C 的
   MAE 更差——单音节“thinking 有害”的结论不能外推到整句。

**外部基线（同一批 2,048 音频）**

| 模型 | Initial | Final | Tone | Joint | 检测 F1 / FPR |
|---|---:|---:|---:|---:|---:|
| **Qwen3.8-Omni（C1）** | **0.962** | **0.949** | 0.912 | **0.838** | **0.922 / 0.162** |
| SDHY P0-v2（开源，适配自由识别） | 0.600 | 0.746 | 0.699 | 0.299 | 0.731 / 0.701 |
| 讯飞 ISE（商用，候选打分） | — | — | **0.922** | — | 0.914 / 0.133 |

* SDHY：冻结 XLSR-zh + M1/X2 头，CPU 中位 47.7ms/条；详见
  [docs/baseline-sdhy.md](docs/baseline-sdhy.md)。
* 讯飞 ISE：参考打分 API（静音填充 + 拼音标注卷 + 候选打分）；tone 0.922、
  7 类最小对平均 0.962、检测 AUC 0.909；已用 **5,843 / 10,000** 次交互；
  详见 [docs/baseline-ise-detail.md](docs/baseline-ise-detail.md)。
* 逐维/逐对弱项与互补关系：[三模型弱项对比](docs/model-weakness-comparison.md)；
  完整对照（能力/精度/工程/成本/选型）：[三模型详细对比](docs/three-model-comparison.md)。

**文档入口**

* 一页报告：[docs/report-minimal.md](docs/report-minimal.md) · 详细报告：
  [docs/minimal-experiment-report.md](docs/minimal-experiment-report.md)
* 分档摘要：[mini3-v1](docs/results-minimal-v1.md) · [mini128-v1](docs/results-minimal-128.md)
* 设计：[完整设计 v3](docs/experiment-design.md) ·
  [¥70 最小化设计](docs/experiment-design-minimal.md)
* 整句 pilot：[实验报告](docs/sentence-experiment-report.md) ·
  [设计 v0.1](docs/sentence-experiment-design.md) ·
  自动报告 `results/sentence/sentence-v1/report.md` · 独立审计 `.../verification.md`
* Obsidian 笔记镜像（7 篇）：[study/README.md](study/README.md)

---

## 1. 快速开始（统一环境，克隆即用）

所有操作通过 [`uv`](https://docs.astral.sh/uv/) 管理：Python 固定 3.12，依赖由
`uv.lock` 锁定，环境变量统一放在 `.env`（模板 `.env.example`，已 gitignore）。

```bash
git clone https://github.com/mutsunn1/qwen3.8-omni-pinyin-eval.git
cd qwen3.8-omni-pinyin-eval

scripts/setup.sh            # 或 make setup：uv sync --locked + 生成 .env + 自检
# 编辑 .env 填入 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL（仅跑 mock 可留空）

make mock                   # 离线全链路：prepare-data → build-tasks → generate(mock) → score → report
make test                   # 282 个离线测试
make lint                   # ruff
```

等价显式命令：

```bash
uv run pinyin-eval check                  # 环境 / 数据 / 模型配置自检
uv run pinyin-eval prepare-data           # 下载 parquet（~100MB）+ 导出 WAV + manifest
uv run pinyin-eval build-tasks --validate # 构造任务/检测集/消融子集，校验最小对计数
uv run pinyin-eval all-mock --audios 40   # 无 key 全链路自测
```

| 环境项 | 统一方式 |
|---|---|
| Python | `.python-version` = 3.12；`requires-python >= 3.12` |
| 依赖 | `uv.lock` 锁定（`uv sync --locked` 复现） |
| 环境变量 | `.env`（gitignore）；shell 变量优先；支持 `export` |
| 配置入口 | `$PINYIN_EVAL_CONFIG` 或 `--config`（默认 `configs/base.yaml`） |
| 命令 | `Makefile`（setup/check/data/tasks/mock/test/lint/ci）+ `scripts/setup.sh` |
| CI | `.github/workflows/ci.yml`：`uv sync --locked` → ruff → pytest（离线，无 key） |

## 2. 真实实验

### 2.1 完整设计（v3，未全量运行）

```bash
uv run pinyin-eval smoke                                   # 1 次真实调用
uv run pinyin-eval estimate --tasks all --profile main     # 主实验调用量（126,704）
uv run pinyin-eval generate --tasks all --profile main      # 全量主实验
uv run pinyin-eval generate --tasks all --profile ablation --conditions all  # 消融
uv run pinyin-eval score  --run-id <run_id>
uv run pinyin-eval report --run-id <run_id>
```

| 阶段 | 数据 | 条件数 | API 调用 | 预计成本（expected / worst） |
|---|---|---:|---:|---:|
| 主实验 | 全量 6160 音频 + 检测 + 全部最小对 | 1（`think1/clean/json`） | 126,704 | ¥101.3 / ¥106.7 |
| 消融 | core 768 + pairs 8,448 条任务（检测不参与消融） | 8 | 85,872 | ¥41.5 / ¥70.9 |
| **合计** | — | — | **212,576** | **¥142.8 / ¥177.6** |

> 价格：`qwen3.8-omni-flash` 华北2（输入 ¥0.8/M、输出 ¥2.7/M、音频 7 token/秒），
> 详见 `pinyin_eval.pricing` 与 `pinyin-eval estimate`；`max_tokens=256` 是成本上界。
> **预算 ≤¥70 时请直接使用 §2.2 的最小化路径（已实测完成）。**

### 2.2 ¥70 最小化实验（推荐先跑，已跑通两档）

```bash
# 128 档（主结果）：2,048 音频 / 79,680 次 / 实测 ¥16.78
uv run pinyin-eval --config configs/minimal-128.yaml build-tasks --minimal
uv run pinyin-eval --config configs/minimal-128.yaml estimate \
  --tasks all --profile minimal --conditions minimal
uv run pinyin-eval --config configs/minimal-128.yaml generate \
  --tasks all --profile minimal --conditions minimal \
  --run-id mini128-v1 --tag minimal
uv run pinyin-eval --config configs/minimal-128.yaml score  --run-id mini128-v1
uv run pinyin-eval --config configs/minimal-128.yaml report --run-id mini128-v1
uv run pinyin-eval --config configs/minimal-128.yaml cost   --run-id mini128-v1
uv run pinyin-eval --config configs/minimal-128.yaml derive-detect --run-id mini128-v1

# 96 档：configs/minimal.yaml，59,968 次 / 实测 ¥12.37
```

* 条件：C1 主条件（think off / clean / JSON，唯一运行检测实验的条件）+
  C2 think on / C3 SNR 20 dB 噪声 / C4 自由文本；
* 数据：96 或 128 个按韵母分层抽取的 target × 4 voice × 4 tone；7 类最小对各
  48/64 实例；
* 预算：96 档预计 ¥18.9 / 最坏 ¥50.1；128 档预计 ¥25.0 / 最坏 ¥66.6；
  硬预算 `generation.budget_cny`（两档分别为 70 / 55）兜底；
* 检测实验只在 C1 运行（v3：不参与消融）；`derive-detect` 输出派生检测
  （自由识别 + 程序比较），是推荐的主口径。

### 2.3 外部基线

**SDHY P0-v2（开源声学评测栈）**

```bash
# 一次性：clone（需要 gh 权限）+ 下载 wav2vec2-XLSR-zh 权重（1.28GB，sha256 校验）
# 详见 docs/baseline-sdhy.md §5
.venv/bin/python scripts/eval_sdhy_baseline.py
```

**讯飞 ISE（商用语音评测，WebSocket）**

```bash
# .env 中加入 XFYUN_APPID / XFYUN_APIKEY / XFYUN_APISECRET（不要提交）
set -a && . ./.env && set +a
.venv/bin/python scripts/probe_xfyun_ise.py \
  --audio data/audio/ZH_TTS_A/ZH_TTS_A_ai4.wav --pinyin ai4
.venv/bin/python scripts/eval_xfyun_ise.py --concurrency 6
.venv/bin/python scripts/eval_xfyun_ise.py --reuse-only   # 只重算指标，不再用交互
```

> 详细接口/鉴权/适配/交互账目/性价比：[docs/baseline-ise-detail.md](docs/baseline-ise-detail.md)。

### 2.4 整句 pilot（sentence-v1，已完成）

```bash
# 数据准备（AISHELL-3 300 句 / 30 speakers；OMPAL 1,758 learner + 82 native 锚点）
.venv/bin/python -m pinyin_eval.cli sentence prepare
.venv/bin/python -m pinyin_eval.cli sentence check
.venv/bin/python -m pinyin_eval.cli sentence estimate          # expected ¥5.5 / worst-case ¥19.9
# 真实运行（7,796 次调用；断点续跑 + ¥20 硬预算；实测 ¥8.29 / 112 min）
.venv/bin/python -m pinyin_eval.cli sentence generate --run-id sentence-v1
.venv/bin/python -m pinyin_eval.cli sentence score   --run-id sentence-v1
.venv/bin/python -m pinyin_eval.cli sentence cost    --run-id sentence-v1
.venv/bin/python -m pinyin_eval.cli sentence report  --run-id sentence-v1
```

* 3 个任务 × 2 条件（C1 think off / C2 think on），AISHELL-3 300 句（30 speakers，
  性别/口音/年龄组精确分层）+ OMPAL 1,758 learner（46 learners）；prompt 版本
  `sentence-v1`，输出上限按任务校准（A 1536 / B 768–1280 / C 256–1024 tokens）。
* 验收：解析成功率 A 100% / B 100% / C 99.97%，对齐覆盖率 ≥99.8%，成本 ¥8.29 ≤ ¥20，
  **全部通过**；结果与解释边界见 [整句 pilot 报告](docs/sentence-experiment-report.md)。
* 离线全链路（无 key）：`make sentence-mock`（7,796 次 mock 调用 → 评分 → 报告）。

## 3. 实验与数据流

```
data/raw/*.parquet ──prepare-data──> data/audio/**/*.wav + data/manifests/audio.parquet
                                              │
                                    build-tasks│
                                               ▼
                data/manifests/tasks/core.jsonl       # 实验一/二/三（tone/tone/free）
                data/manifests/tasks/detection.jsonl  # 检测实验（target+tone → consistent）
                data/manifests/tasks/pairs.jsonl      # 实验四/五/六 三子实验
                data/manifests/tasks/detection_sets.csv
                data/manifests/tasks/{ablation,minimal}_*_items.txt
                                               │
                                    generate   │  8 个生成条件 = 思考开关 × 干净/噪声 × JSON/文本
                                               ▼
   results/runs/<run_id>/{generations.jsonl, planned_calls.jsonl, run_manifest.json}
                                               │
                                      score    ▼
   results/runs/<run_id>/{items.parquet, detection_rows.parquet, score_summary.json}
                                               │
                                      report   ▼
   results/runs/<run_id>/{report.md, metrics.csv, detection.csv, ablation.csv,
                          comparisons.csv, baselines.csv, usage.csv, summary.json, ...}
```

### 3.1 实验一 ~ 三 + 检测实验

| 视图 | 提示条件 | 输出 | 指标 |
|---|---|---|---|
| 实验一 | 给定**正确**无调拼音 | `{"tone": 1-4}` | Tone accuracy / Macro-F1 |
| 实验二 | 给定**错误**无调拼音 | `{"tone": 1-4}` | Tone accuracy / Macro-F1 |
| 实验三 | 无任何读音信息 | `{"initial","final","tone"}` | Initial/Final/Tone/Joint Acc + Macro-F1 |
| 检测实验 | `target`（无调）+ `target_tone` | `{"consistent": true\|false}` | EP/ER/F1/FPR/FNR（正类=不一致） |

* 实验二：每音频 5 个错误 target（2 改声母 + 2 改韵母 + 1 额外段错误；
  `er/weng` 只有 3 个变体）。
* 检测：每音频 1 一致 + 5 不一致（2/2/1）；主检测集按音频配平（6160+6160），
  另建 initial/final/tone 平衡子集（initial 6128，因 er/weng 无改声母变体）。
* 检测指标以 audio 聚类 bootstrap；检测实验只在主条件运行。

### 3.2 实验四 ~ 六（最小对三子实验）

由映射表构造 2440 对、11824 个 (pair × voice × tone) 实例
（`build-tasks --validate` 断言）：

| 类别 | 对 / 实例 | 实验 |
|---|---:|---|
| 平翘舌 z/zh, c/ch, s/sh（排除 -i） | 36 / 576 | 四（initial） |
| 送气 b/p, d/t, g/k, j/q | 67 / 1072 | 四 |
| n/l | 21 / 336 | 四 |
| 前后鼻 an/ang, en/eng, in/ing | 40 / 640 | 五（final） |
| i/ü | 5 / 80 | 五 |
| ian/üan | 3 / 48 | 五 |
| 声调 6 对 | 2268 / 9072 | 六（tone） |

每条音频跑 **3 个子实验**：自由检测（无目标信息）/ 正确信息（给 target 三字段）/
错误信息（只改对应维度：exp4 initial、exp5 final、exp6 tone）；每个子实验只输出该
实验维度（`{"initial"}` / `{"final"}` / `{"tone"}`）。**v3 已取消 A/B 任务。**

### 3.3 消融

* `思考 {on,off} × 音频 {clean, SNR 20dB 白噪声} × 输出 {json,text}`；
* 实验一~三 3 个提示水平 → 24 条件；实验四~六每个实验 3 子实验 × 8 → 72 条件；
* **检测实验不参与消融**（只在主条件运行）；
* 固定子集：core 768 条、pairs 8,448 条任务；8 个生成条件彼此配对。

## 4. 指标与统计

| 指标 | 定义 | 适用范围 |
|---|---|---|
| Initial / Final / Tone Accuracy & Macro-F1 | 单维度识别 | 实验三/四/五/六（free） |
| Joint Accuracy | 声母+韵母+声调全对 | 实验三 |
| Tone Accuracy（tone prompt） | 给定 target 的声调识别 | 实验一/二 |
| EP / ER / Error-F1 | TP/(TP+FP)、TP/(TP+FN)、2PR/(P+R) | 检测（正类=不一致） |
| FPR / FNR | FP/(FP+TN)、FN/(FN+TP) | 检测 |
| **派生检测** | exp3 自由识别与声称 target 程序化比较（零额外调用） | 检测主口径 |
| 成本 / 用量 | token、reasoning token、CNY、预算状态 | 全部 run（`usage.csv`、`cost`） |

* 95% CI：bootstrap 2000 次，audio 聚类、voice 分层；
* 条件比较：McNemar + 配对 bootstrap 差值 CI，Holm 校正；
* 基线：声调 25%、平衡检测集 50%、initial/final 多数类；
* 解析失败（无法解析/拒答/空回复/API 错误）单独统计，不进指标分母。

## 5. 仓库结构

```
configs/base.yaml                  # 完整设计配置（数据/模型/任务/统计/计费）
configs/minimal.yaml               # ¥70 最小化设计（96 档）
configs/minimal-128.yaml           # 128 档升级配置（递归 extends 示例）
configs/sentence.yaml              # 整句 pilot（A/B/C × C1/C2，¥20 预算）
docs/experiment-design.md          # 完整设计 v3（冻结快照）
docs/experiment-design-minimal.md  # 最小化设计（¥70）
docs/report-minimal.md             # 一页最简实验报告
docs/minimal-experiment-report.md  # 两档详细汇总报告
docs/results-minimal-v1.md         # mini3-v1 结果
docs/results-minimal-128.md        # mini128-v1 结果
docs/baseline-sdhy.md              # SDHY 基线（下载/适配/结果）
docs/baseline-ise.md               # 讯飞 ISE 基线（简版）
docs/baseline-ise-detail.md        # 讯飞 ISE 详细（接口/协议/账目/性价比）
docs/model-weakness-comparison.md  # 三模型弱项对比
docs/three-model-comparison.md     # 三模型详细对比
docs/sentence-experiment-design.md # 整句 pilot 设计 v0.1（冻结快照）
docs/sentence-experiment-report.md # 整句 pilot 报告（结果/偏差/解释边界）
docs/contracts.md / decisions.md / runbook.md / schemas.md
study/                             # Obsidian study/ 笔记镜像（6 篇，脚本同步）
scripts/
  setup.sh                         # 统一环境初始化
  eval_sdhy_baseline.py            # SDHY 基线评测
  probe_xfyun_ise.py               # 讯飞 ISE 单条探针
  eval_xfyun_ise.py                # 讯飞 ISE 候选打分评测（缓存/续跑）
  analyze_detection_probe.py       # 检测 prompt 偏置分析
  make_detection_probe_subset.py   # 检测探针固定子集
  sync_study_docs.py               # Obsidian → study/ 同步
  prepare_sentence_data.py         # 整句数据准备（AISHELL-3 抽样 + OMPAL 清洗 + 音频）
src/pinyin_eval/
  config.py normalize.py records.py prompts.py parsing.py client.py runner.py
  scoring.py pricing.py derived_detection.py cli.py
  data/dataset.py data/minimal_pairs.py
  metrics.py stats.py report.py tasks.py
  sentence/                        # 整句 pilot：normalize/parsing/scoring/metrics/
                                   # runner/report/cli/mock/prompts/records
tests/                             # 离线测试（无网络 / 无 key；含 sentence 5 个测试文件）
```

## 6. 模型调用与成本控制

* `model="qwen3.8-omni-flash"`，`modalities=["text"]`，system + user messages；
* 音频：`input_audio` data URI（wav，base64）；结构化输出 `response_format=json_object`；
* 输出 schema（v3）：tone / free / initial / final / detect；
  自由文本标记：`声调=`、`声母=`、`韵母=`、`一致=是/否`；
* 固定解码：`temperature=0, top_p=1, max_tokens=256`；thinking 关 = `reasoning_effort="none"`，
  开 = 默认力度（不发送参数）；
* 成本：每次 usage 写入 `generations.jsonl`；`pinyin-eval cost --run-id` 汇总；
  `generation.budget_cny` / `generate --max-cost` 为硬预算，达到后停止新请求；
* 价格快照与三档估算（expected / conservative / worst-case，worst-case 假设打满
  `max_tokens=256`）见 `src/pinyin_eval/pricing.py` 与 `pinyin-eval estimate`；
* 失败重试：指数退避 + 抖动，重试次数记录在 generation 记录中。

来源：[Qwen-Omni 文档](https://help.aliyun.com/zh/model-studio/qwen-omni)、
[深度思考](https://help.aliyun.com/zh/model-studio/deep-thinking)、
[模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)。

## 7. 可复现性

* `run_manifest.json` 记录数据集 commit、mapping/config sha256、prompt 版本、模型 id、
  pricing 快照、预算与调用日期；`pinyin-eval check --json` 可复核环境；
* `generations.jsonl` append-only，按 `request_id` 去重；同 run-id 重跑自动续跑；
* `data/`、`results/`、`third_party/` 均为可再生/本地文件（gitignore），可用 CLI 重建；
* prompt 全文存于任务清单；提示词变更会改变 `request_id`，旧预测不混用；
* 真实运行记录：`mini3-v1` / `mini128-v1`（Qwen）、`baseline-sdhy-min128`、
  `ise-baseline`（见各文档）。

## 8. 测试与 CI

```bash
uv run pytest                 # 282 passed（离线）
uv run ruff check src tests   # All checks passed
```

覆盖：规范化/解析、prompt 指纹去重、任务构造（实验一~六、检测、消融/最小化抽样）、
数据准备与噪声注入、指标与统计、mock 端到端（断点续跑/去重/失败记录）、报告
（数字与 metrics/stats 对拍、bootstrap CI、CSV 列稳定、seed 复现）、配置递归 extends、
pricing/预算守卫、派生检测、检测 prompt 变体探针。

GitHub Actions 每次 push 执行 `uv sync --locked → ruff → pytest`（无需 API key）；
当前 main 分支 **CI 绿**。

## 9. 已知限制与后续

* **完整设计未全量运行**：预算约 ¥142.8（最坏 ¥177.6）；已完成的是两档最小化子集
  （2,048 音频，总体 CI ±1–2pp）。
* **thinking on 不可用**：本任务系统性有害；完整实验应删除/降权该条件。
  另有 2 条 thinking-on 空回复（`max_tokens=256` 被 reasoning 吃满），已记入
  `parse_failures.jsonl`；若保留该条件建议把 max_tokens 提到 512。
* **检测主口径已改为派生检测**：模型直接判一致 FPR 0.77；派生检测 F1 0.922、
  FPR 0.162。模型直接判定仅作辅助指标。
* **Qwen 剩余弱项**：m/n（0.858）、o/uo（0.804）、2/3 声（0.825）、en/eng（0.883，
  低于 SDHY 的 0.922 / ISE 的 1.000）；需数据/prompt 定向补强。
* **ISE 基线**：候选打分口径（非自由识别），已用 5,843/10,000 交互；成本是 Qwen 的
  18.8–150 倍（按任务），仅建议做小样本交叉验证；余额 4,157。
* **SDHY 基线**：CPU fp32 与官方标定 cuda-fp16 不等价；自由识别是适配口径，
  不能直接代表其产品能力。
* **整句 pilot 的解释边界**（`sentence-v1`）：A 抽样来自 AISHELL-3 train 划分
  （可访问镜像无 test 拼音标注），特殊子集仅 6 条儿化；OMPAL 只有 3 位评分者、
  法语 L1 子集，B/C 结论只限定该 learner 集合；A/B/C 语料与 ground truth 不同、
  **不能横向比较**；B 任务当前形态不可做自动纠音（macro-F1 0.55–0.60、FNR 0.79），
  需要任务重设计；C 的相关只有中等（0.35–0.58）且整体偏低约 0.45 分（native 锚点）。
  完整偏差清单见 [整句 pilot 报告 §2](docs/sentence-experiment-report.md)。
* 数据子集抽样导致的解释边界、逐对 CI（±7–14pp）见各结果文档。

## 10. 引用与许可

* 代码：[MIT License](LICENSE)（Copyright 2026 mutsumi）。
* 数据：[`HaitongSUN/prosodic-abx`](https://huggingface.co/datasets/HaitongSUN/prosodic-abx)
  （`mandarin_tone_syn`，**CC-BY-4.0**，arXiv:2604.02102）；本仓库
  `data/mapping/mandarin_tone_syn_target_mapping.csv` 为基于该数据集的
  target→initial/final 映射表，再分发请保留数据集署名与许可。
* 模型：阿里云百炼 Qwen3.8-Omni-Flash，遵循[百炼服务条款](https://help.aliyun.com/zh/model-studio/)。
* 外部基线：SDHY P0-v2（`CHAO-AI-Lab/SDHY_Test_System`，私有）、讯飞语音评测
  （讯飞开放平台，遵循其服务条款；密钥不入库）。
* 整句 pilot 数据：[AISHELL-3](https://www.openslr.org/93/)（Apache-2.0；
  wav 经 `shenyunhang/AISHELL-3` HF 完整镜像获取）、
  [OMPAL](https://github.com/phantomhsieh/OMPAL-corpus)（CC BY 4.0；
  commit `c3aaa9e8…`）；派生清单与音频仅本地使用，再分发请保留署名与许可。
