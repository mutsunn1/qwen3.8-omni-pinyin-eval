# 接口契约（冻结）

本文件是各模块之间的**冻结契约**。实现时必须保持一致；如需变更，先改本文件并在 PR/任务中同步。
设计依据：`docs/experiment-design.md`（来自 Obsidian Vault 的设计文档快照）。

## 0. 目录结构与写入范围

```
src/pinyin_eval/
  config.py           # 配置加载（Lead）
  records.py          # 冻结数据结构（Lead，已提交）
  normalize.py        # 拼音/输出规范化（Lead）
  data/
    dataset.py        # 数据集下载、音频导出、audio manifest（Lead）
    minimal_pairs.py  # 最小对构造（teammate: data-dev）
  tasks.py            # 实验一~六任务构造 + 消融抽样 + 检测集组装（Lead）
  prompts.py          # 提示词模板与版本号（Lead）
  parsing.py          # 模型输出解析（Lead）
  metrics.py          # 指标计算（teammate: metrics-dev）
  stats.py            # bootstrap / 显著性 / 多重比较（teammate: metrics-dev）
  client.py           # DashScope OpenAI 兼容客户端（Lead）
  runner.py           # 生成调度、断点续跑、预算（Lead）
  scoring.py          # 预测结果 -> ItemResult（Lead）
  report.py           # 报告与表格（Lead; 后续可委派）
  cli.py              # 命令行入口（Lead）
configs/              # YAML 配置（Lead）
tests/                # 各模块自测；每个模块负责自己的测试文件
docs/                 # 设计快照、契约、运行协议
```

## 1. 数据事实（已核验，2026-10-28）

数据集：`HaitongSUN/prosodic-abx`，config `mandarin_tone_syn`，commit
`d607806619aba678030dac10ea086fb19347205a`，单 parquet，6160 行。

| 事实 | 值 |
|---|---|
| 列 | `id`, `audio{bytes,path}`, `speaker`, `target`, `label` |
| 音频 | WAV，16 kHz，单声道，PCM_16，0.215–1.334 s |
| speaker | `ZH_TTS_A` / `_B` / `_C` / `_D` 各 1540（voice = A/B/C/D） |
| label | 声调 1–4，各 1540 |
| 原始 target | 385 种拼写；规范 target 378 种 |
| id 规则 | `{speaker}_{raw_target}{label}`，例 `ZH_TTS_A_ai1` |
| 映射表 | `data/mapping/mandarin_tone_syn_target_mapping.csv`（378 行，UTF-8 BOM，`count` 求和 = 6160） |
| 零声母规范化（唯一需要代码实现的规则） | `in→yin, ing→ying, iu→you, u→wu, ui→wei, un→wen, ün→yun` |
| 其余 initial/final | **只查映射表，不硬编码** |
| j/q/x + u | 映射表已给出 `ju→(j,ü)` 等，直接用表的值 |

语义：`target` 是**无调拼音**；`label` 是声调；`initial`/`final` 由映射表提供，
零声母时 `initial=""`。JSON 中 `ü` 保持 Unicode `ü`（不写 `v`）。

## 2. 音频清单 audio manifest

`python -m pinyin_eval.cli prepare-data` 生成：

* `data/audio/{speaker}/{audio_id}.wav` —— 从 parquet `bytes` 原样导出（不解码/重采样）
* `data/manifests/audio.parquet` 与 `data/manifests/audio.csv` —— `AudioRecord` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| audio_id | str | 唯一，= 数据集 `id` |
| row_index | int | parquet 行号 |
| speaker | str | `ZH_TTS_A`… |
| voice | str | `A`/`B`/`C`/`D` |
| raw_target | str | 原始 target（385 种） |
| target | str | 规范 target（378 种） |
| initial | str | 映射表 initial，可为 `""` |
| final | str | 映射表 final |
| tone | int | 1–4 |
| wav_path | str | 相对仓库根，`data/audio/...` |
| duration_s | float | 秒 |
| sample_rate | int | 16000 |
| channels | int | 1 |

合并 target（映射表 `count=32` 的 7 个：yin/ying/you/wu/wei/wen/yun）在数据集中
有两种原始拼写；audio manifest 两条都保留，`target` 相同，`raw_target` 不同。
需要“每个 canonical target 选 1 条音频”时用 `raw_target == target` 优先（全部满足）。

## 3. 任务单元 TaskItem

见 `records.py` 的 `TaskItem`。JSONL 一行一个，UTF-8。

### 3.1 任务目录与去重（重要）

> 2026-10-28 **v3** 设计更新：实验一/二只评声调；新增独立检测实验（target+target_tone →
> `{"consistent"}`）；**取消 A/B 任务**；实验四/五/六各拆为自由检测/正确信息/错误信息
> 三个子实验（输出 initial/final/tone）；**检测实验不参与消融**。
> 详见 `docs/experiment-design.md`（v3 快照）。

三个任务目录：

* `tasks/core.jsonl` —— 实验一/二/三视图，覆盖全部 6160 条音频：
  * `prompt_condition="free"` → 实验三视图（task_type=free）
  * `prompt_condition="correct_target"` → 实验一视图（task_type=tone，给正确无调 target）
  * `prompt_condition="wrong_target"` × 5 变体/音频 → 实验二视图（task_type=tone，
    5 个“只改声母/韵母”的错误无调 target：2 改声母 + 2 改韵母 + 1 额外段错误）
* `tasks/detection.jsonl` —— 检测实验：每音频 1 个一致样本（原始 target+label 声调）
  + 5 个不一致变体（2 改声母 / 2 改韵母 / 1 改声调），task_type=detect。
* `tasks/pairs.jsonl` —— 最小对音频（实验四/五/六视图），每实例侧 3 个子实验：
  * free（无目标信息）、correct_target（给 target_initial/final/tone 全对）、
    wrong_target（只改对应维度：exp4 改 initial、exp5 改 final、exp6 改 tone）；
  * task_type 分别为 exp4=`initial`、exp5=`final`、exp6=`tone`；v3 无 A/B。

**跨实验视图、跨条件的重复请求只发一次**：
`TaskItem.task_type` 决定输出 schema；`TaskItem.prompt_texts={"json":..., "text":...}`，
`TaskItem.system_prompt` 为共用 system prompt。生成请求去重键
`request_id = sha256(f"{media_path}\0{output_form}\0{system_prompt}\0{prompt_texts[output_form]}")[:16]`，
其中 `media_path` 由 audio_quality 决定（clean→`wav_path`，noise_20db→缓存噪声路径）。
runner 对 request_id 去重后调用 API；scoring 用 prompt_texts/system_prompt 重算 request_id
把预测关联回每个 item（同一 request 可对应多个 item，例如实验三 free 与实验四 free 共用一次调用）。

### 3.2 item_id 命名（确定性）

```
exp1|correct_target|{audio_id}
exp2|wrong_target|{audio_id}|{variant_type}{variant_index}   # variant_type∈{initial,final}（+extra）
exp3|free|{audio_id}
detect|detect|{audio_id}                                     # 一致样本（负类）
detect|detect|{audio_id}|{variant_type}{variant_index}       # 不一致变体（正类）
exp4|free|{audio_id}
exp4|correct_target|{audio_id}
exp4|wrong_target|{audio_id}                # 给对侧最小对目标（无调）
exp5|… 同 exp4（experiment 改为 exp5）
exp6|free|{audio_id}|{pair_id}；exp6|correct_target|…；
exp6|wrong_target|{audio_id}|{pair_id}（错误信息只改 target_tone）

注意：最小对任务（实验四~六）的 item_id 一律带 `|{pair_id}` 后缀——同一条音频可同时属于
多个声调最小对（如 ai1 属于 (1,2)(1,3)(1,4) 三对），只靠 audio_id 会冲突。
跨实验/跨 pair 完全相同的提示仍由 request_id 去重为一次 API 调用。
```

合并拼写（7 个 canonical target）在核心任务中产生的预测按 `request_id` 关联；
`audio_id` 仍是唯一键。最小对实例选取音频时优先 `raw_target == target`。

### 3.3 金标与 given_target 规则

* `gold_initial/gold_final/gold_tone` 永远描述**音频**。
* `task_type` 决定输出 schema：`free`（initial/final/tone）、`tone`（只有 tone）、
  `detect`（只有 consistent）、`ab`（只有 choice）。
* `correct_target`（实验一/四/五/六）：`given_target` = 本音频规范无调拼音，`given_tone=None`，
  `consistent_gold=True`。
* `wrong_target` 段错误变体（实验二/四/五）：`given_target` = 改动后的无调拼音，
  `given_tone=None`，`consistent_gold=False`，`variant_type=initial|final`。
* 检测实验：`given_target` + `given_tone` 均给出；一致样本 = 原始 target+label；
  段错误变体 = 改动 target、`given_tone=真实声调`；声调错误变体 = 原始 target、`given_tone≠gold`。
* 实验四~六信息子实验：`given_initial`/`given_final`/`given_tone` 均给出；
  correct_target 全对；wrong_target 只改实验对应维度，其余字段正确。
* `free`：`given_target=None`（子实验 free 的 prompt 不含目标信息）。
* v3 无 A/B；`alternatives/correct_choice` 字段保留但恒为 None。
* `cluster_id` = `audio_id`；检测分析以 audio 聚类。
* `stratum` = `voice|tone`（消融抽样：4 voice × 4 tone × 每层 16 条）。

### 3.4 实验一~三主任务

* exp1：6160 条音频 × `correct_target`，task_type=tone（只评声调）。
* exp2：6160 条音频 × `wrong_target`，每条 5 个“只改声母或韵母”的错误无调 target
  （2 initial + 2 final + 1 extra；`er/weng` 无同韵母其它声母，只有 3 个变体）。
  改动后的目标必须存在于 378 个 canonical target 中；同 final 换 initial 池 /
  同 initial 换 final 池中确定性抽取（配置 seed）。
* exp3：6160 条音频 × `free`，task_type=free。
* 检测实验：6160 条音频，每条 1 一致 + 5 不一致（2/2/1），task_type=detect；
  **v3：检测实验不参与消融**，只在主生成条件运行一次。
* 主实验默认生成条件 = `thinking=True, clean, json`（配置 `main_condition`）。

### 3.5 实验四~六（最小对）

* 由 `minimal_pairs.build_all_pairs(...)` 生成；pair_type→experiment 映射：
  exp4 = {sh_zh, aspiration, nasal_lateral}；exp5 = {front_back_nasal_ptr, i_ü, ian_üan}；
  exp6 = {tone}。
* 每个最小对实例的每条音频生成 3 个子实验：free / correct_target / wrong_target，
  分别输出该实验维度（exp4 initial、exp5 final、exp6 tone）。
* 主实验使用全量最小对实例；消融按最小对池抽样（每个 pair_type ≤256 实例，池子不足全取；
  i/ü 80、ian/üan 48 全取）。

## 4. 条件、运行与生成记录

### 4.1 条件网格

```
prompt_condition ∈ EXPERIMENT_PROMPT_CONDITIONS[exp]
thinking        ∈ {True, False}
audio_quality   ∈ {"clean", "noise_20db"}     # 加性高斯白噪声，SNR=20 dB，逐音频确定性种子
output_form     ∈ {"json", "text"}
```

条件 id = `{prompt_condition}__{think1|think0}__{clean|noise_20db}__{json|text}`。

* 主实验（报告主结果）：`configs/*.yaml` 的 `main_condition`，默认
  `thinking=true, clean, json`，在主任务全量上跑。
* 消融：每个「提示条件/任务 × 生成维度」条件 256 条，按 `voice|tone` 16 层 × 16 条抽样，
  固定 seed；同一子集在 8 个生成条件下复用（配对比较有效）。即：
  * `ablation_core_items.txt` = 3 个提示条件 × 256 条 = 768 条；
  * `ablation_pairs_items.txt` = 按 pair_type 抽取实例（每类 ≤256 实例，i/ü 80、ian/üan 48 全取），
    每个被选中实例的 3 个子实验任务全部保留；
  * v3 **不生成** `ablation_detection_items.txt`（检测实验不参与消融）。
* 噪声音频缓存到 `data/noisy/{audio_id}__snr20db.wav`（16-bit PCM WAV，确定性种子）。

### 4.2 生成记录 GenerationRecord

`results/runs/{run_id}/generations.jsonl`，一行一次调用（字段见 `records.py`）。
断点续跑：按 `(item_id, condition_id)` 去重；已存在成功记录则跳过。
错误记录保留（`error` 非空），重试次数记入 `retries`。

### 4.3 解析与 ItemResult

`scoring.py` 把生成记录 + golden 合并为 `results/runs/{run_id}/items.parquet`：

* `parse_status`：`ok | parse_error | refused | empty | api_error`；
  仅 `ok` 进入指标分子分母，其余计入 `n_parse_failures` 并单独汇报比例。
* `pred_*` 为**规范化后**的值（initial 小写；final 用 `ü`；tone 为 int；v3 无 choice）。
* 检测分析时再派生行：`detection_role ∈ {neg,pos}`，`detection_subset ∈ {main,initial,final,tone}`。

## 5. 最小对构造 minimal_pairs.py

输入：`audio_records: Sequence[AudioRecord]`（只用 `target/initial/final/voice/tone/raw_target`）、
`mapping`（target→(initial,final)）。输出：`list[MinimalPair]`（dataclass 定义见模块内）与
`list[MinimalPairInstance]`（含 audio_id、side）。

规则（必须写死为函数，可复现设计文档计数）：

| pair_type | 规则 | 文档计数（types / instances） |
|---|---|---|
| `sh_zh` | (z,zh),(c,ch),(s,sh) 同 final；排除 final=="i"（-i 组） | 36 / 576 |
| `aspiration` | (b,p),(d,t),(g,k),(j,q) 同 final | 67 / 1072 |
| `nasal_lateral` | (n,l) 同 final | 21 / 336 |
| `front_back_nasal_ptr` | (an,ang),(en,eng),(in,ing) 同 initial；**默认只含这三组**（与文档 40 一致） | 40 / 640 |
| `i_ü` | 同 initial，final i vs ü（含 j/q/x 的 `jü→ju` 等） | 5 / 80 |
| `ian_üan` | 同 initial，final ian vs üan | 3 / 48 |
| `tone` | 同 initial+final，6 对声调 (1,2)(1,3)(1,4)(2,3)(2,4)(3,4)；每 canonical target × voice 一对 | 2268 / 9072 |

实例 = 每个 (pair, voice, tone) 取 1 条音频（合并拼写的 7 个 target 优先 `raw_target==target`）。
计数验收：`pytest tests/test_minimal_pairs.py` 必须复现上表全部数字
（`tests/fixtures` 中可直接读 `data/mapping/*.csv` 构造 target 表，不需音频）。

## 6. metrics.py 契约

输入一律为 `ItemResult.to_row()` 得到的 **dict 行**（或 pandas `Series`，可 `.get` 取值），
输出为纯 dict/list，可 JSON 序列化。所有函数**不得**依赖文件 IO / 全局状态。

```python
def classification_metrics(
    gold: Sequence[Any], pred: Sequence[Any | None], *, labels: Sequence[Any] | None = None
) -> dict
# -> {"n": int, "n_failures": int, "accuracy": float,
#     "macro_f1": float, "labels": [..], "per_class": {label: {"precision","recall","f1","support"}}}
# pred 为 None 表示解析失败：从 n/accuracy/macro_f1 分母中剔除，计入 n_failures。

def recognition_metrics(rows: Sequence[Mapping]) -> dict
# 只使用 parse_status=="ok" 的行；
# gold 取 gold_initial/final/tone，pred 取 pred_initial/final/tone（已被 scoring 规范化）
# -> {"n": int, "n_failures": int,
#     "initial": classification_metrics(...), "final": ..., "tone": ...,
#     "joint_accuracy": float}

def detection_metrics(rows: Sequence[Mapping]) -> dict
# 行来自检测实验（detect）：neg=一致样本，pos=不一致变体；pred_consistent 为 bool|None，
# 语义是“模型判定音频与声称一致”（True=一致）。错误检测的正类是“不一致”：
#   判为不一致 = (pred_consistent is False)
#   TP: pos & 判为不一致; FP: neg & 判为不一致; TN: neg & 判为一致; FN: pos & 判为一致
# -> {"n_neg","n_pos","n_failures_neg","n_failures_pos",
#     "tp","fp","tn","fn","error_precision","error_recall","error_f1",
#     "fpr","fnr","accuracy","balanced_accuracy"}
# 分母为 0 时返回 None。

def ab_metrics(rows: Sequence[Mapping]) -> dict
# -> {"n","n_failures","accuracy","choice_counts": {"A":..,"B":..}}

def confusion_matrix(gold, pred, *, labels=None) -> pandas.DataFrame  # 行=gold 列=pred
def confusion_rows(rows, *, dimension: str)  # "initial"|"final"|"tone" -> list[dict]

def majority_baseline(gold: Sequence[Any]) -> dict   # {"label","accuracy","n"}
def random_baseline(n_classes: int) -> float
def detection_chance_baseline() -> float             # 0.5（平衡检测集）

def recognition_metric_pack(row_groups: Mapping[str, Sequence[Mapping]]) -> dict
# 便捷：{"overall": ..., "by_voice": {...}, "by_condition": {...}}
```

## 7. stats.py 契约

无需 scipy。纯 numpy + 标准库。**bootstrap 以 cluster（audio_id）为重采样单位，
voice 作为分层变量**；同一 cluster 的行一起重采样。

```python
def bootstrap_ci(
    rows: Sequence[Mapping], metric_fn: Callable[[Sequence[Mapping]], Mapping[str, float]],
    *, cluster_key: str = "cluster_id", n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
    stratify_key: str | None = "voice",
) -> dict[str, dict[str, float]]
# -> {metric_name: {"estimate":..,"lo":..,"hi":..}}
# estimate = metric_fn(rows 全量)；分层时在每个 stratum 内独立重采样 cluster 再合并。
# metric_fn 返回 dict；metric_fn 收到的是“被重采样的行列表”（重复 cluster 的行按重复次数出现）。

def paired_bootstrap_difference(
    rows_a, rows_b, metric_fn, *, cluster_key="cluster_id", n_boot=2000, alpha=0.05, seed=0,
) -> dict[str, dict[str, float]]
# 要求 rows_a/rows_b 按同一 item_key 对齐；每个 cluster 的抽样对两侧同步。
# -> {metric: {"estimate":..,"lo":..,"hi":..}}，estimate = metric_a - metric_b。

def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict
# 对齐的两侧正确性；不一致对 b = a对b错，c = a错b对。
# -> {"n_pairs","b","c","statistic","p_value","method"}  # 精确二项双侧（math.comb）
# 样本量大（b+c>1000）用连续校正卡方近似并注明 method。

def holm_correct(pvalues: Sequence[float]) -> list[float]
def bh_correct(pvalues: Sequence[float]) -> list[float]

def paired_outcomes(rows_a, rows_b, *, item_key="item_id", correctness_key="is_correct") -> tuple[list[bool], list[bool]]
# 便捷：按 item_key 对齐，剔除任一侧解析失败的行。
```

## 8. scoring / report 契约（Lead 实现）

`items.parquet` 列 = `ItemResult` 字段（含 `task_type`）。`scoring.score_run(run_dir) -> DataFrame`。
`report.write_report(run_dir, out_dir) -> dict[str, str]` 读取 items.parquet +
`run_manifest.json`（含 config 快照、模型 id、调用日期、数据 commit、prompt 版本、
mapping sha256），输出 `report.md`、`metrics.csv`、`detection.csv`、`ablation.csv`、
`comparisons.csv`、`baselines.csv`、`confusion/*.csv`、`summary.json`；
报告对应 2026-10-28 v3 设计：实验一/二只报 Tone；检测实验报 EP/ER/F1/FPR/FNR；
实验四/五/六报 3 个子实验下的 initial/final/tone 准确率；无 A/B；消融只含 core+pairs；
另含 95% bootstrap CI、基线、混淆矩阵、解析失败率、成本与用量、McNemar + Holm/BH。

## 9. 测试与验收

* 每个模块自带 `tests/test_<module>.py`，`uv run pytest` 全绿（默认不联网）。
* 最小对计数表必须与设计文档完全一致（见第 5 节）。
* 数据流验收：mock 客户端跑通 prepare-data → build-tasks → generate(mock) → score → report
  （`tests/test_cli.py` + `tests/test_runner_scoring.py` + `uv run pinyin-eval all-mock`）。
* 真实 API 仅在有 `DASHSCOPE_API_KEY` 时通过 `pinyin_eval.cli smoke` 验证，
  不在 CI/默认测试中执行。

## 10. 写入范围（Agent Teams）

| 成员 | 允许写 |
|---|---|
| lead | 除下表外的所有文件 |
| metrics-dev | `src/pinyin_eval/metrics.py`, `src/pinyin_eval/stats.py`, `tests/test_metrics.py`, `tests/test_stats.py` |
| data-dev | `src/pinyin_eval/data/minimal_pairs.py`, `tests/test_minimal_pairs.py` |

禁止改动他人文件；发现契约问题先在共享任务/消息中提出，由 Lead 更新本文件。

## 11. 计费、成本与预算（v3 新增）

```python
@dataclass(frozen=True)
class Pricing:  # pinyin_eval.pricing
    input_per_mtok: float = 0.8          # CNY / 1M tokens
    cache_hit_input_per_mtok: float = 0.1
    output_per_mtok: float = 2.7
    audio_tokens_per_second: float = 7   # Qwen3.8-Omni: ceil(duration_s) * 7
    ...
    @classmethod
    def from_config(cls, cfg: Config | None = None) -> Pricing
    def audio_tokens(self, duration_s: float | None) -> int
    def call_cost(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> dict
```

* 实际成本：`(prompt_tokens - cached_tokens)*input_per_mtok/1e6
  + cached_tokens*cache_hit_input_per_mtok/1e6 + completion_tokens*output_per_mtok/1e6`；
  `pinyin_eval cost --run-id` 从 `generations.jsonl` 的 usage 汇总。
* 预估：`runner.estimate_plan_cost(cfg, plans)` 返回 expected / conservative / worst-case
  三档；worst-case 假设每次输出打满 `model.max_output_tokens`。
* 预算 guard：`generation.budget_cny` 或 `generate --max-cost`；达到预算后停止新请求，
  manifest `status="budget_exceeded"`、`counts.n_skipped_budget>0`，可用同一 run-id 续跑。
* manifest 新增字段：`pricing`、`budget_cny`、`counts.{n_calls,cost_cny,input_tokens,
  output_tokens,reasoning_tokens,budget}`；report 输出 `usage.csv` + 成本节。

## 12. 最小化实验设计（¥70 预算，v3 新增）

* 文档：`docs/experiment-design-minimal.md`；配置：`configs/minimal.yaml`（`extends: configs/base.yaml`）。
* `build-tasks --minimal` 产出：
  `minimal_items.txt`（C1 全部任务）、`minimal_no_detection_items.txt`（C2–C4 不含检测）、
  `minimal_{core,detection,pair}_items.txt`、`minimal_summary.json`。
* 生成时 `--profile minimal --conditions minimal`：C1 含检测实验，C2–C4 自动排除检测
  （runner `item_subset_by_generation`）；`budget_cny=70` 生效。
* 默认档位：`minimal.n_targets=96`、`minimal.pair_instances_per_type=48` →
  59,968 requests，expected ¥18.9 / conservative ¥21.3 / worst-case ¥50.1。
* 估算命令：`pinyin-eval --config configs/minimal.yaml estimate --tasks all --profile minimal
  --conditions minimal`。
