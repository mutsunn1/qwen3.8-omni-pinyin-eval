# 数据产物 Schema

所有文件均为 UTF-8；JSONL 每行一个 JSON 对象。路径相对仓库根目录。

## 1. `data/manifests/audio.parquet` / `audio.csv`

`AudioRecord`（`src/pinyin_eval/records.py`）：

| 列 | 类型 | 说明 |
|---|---|---|
| audio_id | str | 数据集 `id`，如 `ZH_TTS_A_zhang1` |
| row_index | int | parquet 行号 |
| speaker | str | `ZH_TTS_A` .. `ZH_TTS_D` |
| voice | str | `A`/`B`/`C`/`D` |
| raw_target | str | 数据集原始 target（385 种拼写） |
| target | str | 规范 target（378 种） |
| initial | str | 映射表声母，零声母为 `""` |
| final | str | 映射表韵母（`ü` 保留 Unicode） |
| tone | int | 1–4 |
| wav_path | str | `data/audio/<speaker>/<audio_id>.wav` |
| duration_s | float | 秒 |
| sample_rate | int | 16000 |
| channels | int | 1 |

## 2. `data/manifests/tasks/*.jsonl`

三个任务目录（2026-10-28 v2）：

| 文件 | 内容 | 实验视图 |
|---|---|---|
| `core.jsonl` | 43,056 条：free + correct_target(tone) + wrong_target(tone) × 5 变体 | 实验一/二/三 |
| `detection.jsonl` | 36,896 条：每音频 1 一致 + 2 改声母 + 2 改韵母 + 1 改声调（task_type=detect） | 检测实验 |
| `pairs.jsonl` | 76,448 条：最小对实例的 free/correct_target/wrong_target(仅 exp4/5)/ab_choice | 实验四/五/六 |

`TaskItem` 字段（省略 `None` 等价空值）：

| 字段 | 说明 |
|---|---|
| item_id | 确定性 ID；核心任务 `exp1\|correct_target\|{audio_id}`，检测实验 `detect\|detect\|{audio_id}[\|变体]`，最小对任务带 `\|{pair_id}` 后缀 |
| experiment | `exp1` / `exp2` / `exp3` / `detect` / `exp4`..`exp6`（分析视图） |
| prompt_condition | `free` / `correct_target` / `wrong_target` / `detect` / `ab_choice` |
| task_type | 输出 schema：`free` / `tone` / `detect` / `ab` |
| audio_id / wav_path / voice | 音频与 voice |
| gold_initial / gold_final / gold_tone | 金标（永远描述音频） |
| cluster_id | bootstrap 聚类单位 = audio_id |
| pair_id / pair_type / pair_side | 最小对任务专用 |
| variant_type / variant_index | 错误变体类型（initial/final/tone；实验二只有 initial/final）与序号 |
| given_target / given_tone | 提示中"声称"的读音（无调拼音 / 带调） |
| consistent_gold | 目标条件下的金标一致性 |
| alternatives / correct_choice | A/B 任务候选（带声调数字）与正确项 |
| stratum | `voice\|tone`，消融分层用 |
| prompt_texts | `{"json": ..., "text": ...}` 两种输出形式的完整 user prompt |
| system_prompt | 共用 system prompt（参与 request 指纹） |
| prompt_version | prompt 版本号（当前 `p2`） |

## 3. `data/manifests/tasks/detection_sets.csv`

| 列 | 说明 |
|---|---|
| subset | `main` / `initial` / `final` / `tone` |
| audio_id / cluster_id / voice / tone | 样本 |
| negative_item_id | 实验一（正确目标）item |
| positive_item_id | 实验二（错误目标）item |
| variant_type | 正类错误类型 |
| negative_given_target / negative_given_tone | 负类（一致样本）的 target / target_tone |
| positive_given_target / positive_given_tone | 正类（不一致变体）的 target / target_tone |

## 4. `data/manifests/tasks/minimal_pair_counts.json`

`{pair_type: {"n_pairs": int, "n_instances": int}}`，用于与设计文档计数表核对。

## 5. `data/manifests/tasks/ablation_*_items.txt`

每行一个 `item_id`：

* `ablation_core_items.txt`：3 个提示条件 × 256 条 = 768 条；
* `ablation_detection_items.txt`：256 个音频 × 全部 6 个检测任务 = 1,536 条；
* `ablation_pairs_items.txt`：按 pair_type 选中的最小对实例的全部任务（10,752 条）。

## 6. `results/runs/<run_id>/run_manifest.json`

关键字段：

| 字段 | 说明 |
|---|---|
| run_id / created_at / finished_at / status | 运行元信息 |
| git_commit / config_sha256 / config | 代码与配置快照 |
| dataset | hf_repo / hf_revision / mapping_csv_sha256 |
| model | name / provider / temperature / top_p / max_tokens / thinking 参数 / mock |
| prompts | version + notes |
| task_sources / item_subset_source | 任务清单与子集文件 |
| conditions | 本次运行的生成条件列表 |
| plan | 请求数、条件分布、唯一 prompt/音频数 |
| counts | n_planned / n_skipped_resume / n_ok / n_error / elapsed_s |
| setup | 是否检测到 API key、base_url |

## 7. `results/runs/<run_id>/generations.jsonl`

`GenerationRecord` 每行一次 API 调用（去重后）：

`run_id, request_id, item_id, experiment, prompt_condition, condition_id, thinking,
audio_quality, output_form, model, prompt_version, response_text, reasoning_text,
finish_reason, usage, latency_s, error, retries, timestamp, media_path`

`condition_id = {prompt_condition}__{think1|think0}__{clean|noise_20db}__{json|text}`；
检测实验的 condition_id 形如 `detect__think1__clean__json`。

## 8. `results/runs/<run_id>/planned_calls.jsonl`

`{"request_id": ..., "condition_id": ...}` 每行一个计划调用；scoring 用它判断哪些
(task × condition) 单元属于本次运行。

## 9. `results/runs/<run_id>/items.parquet`

`ItemResult` 字段 + 派生列：

* `task_type ∈ {free, tone, detect, ab}`（决定哪些正确性列非空）；
* 预测：`pred_consistent / pred_initial / pred_final / pred_tone / pred_choice`（规范化后）；
* 状态：`parse_status ∈ {ok, parse_error, refused, empty, api_error}`；
* 正确性（nullable boolean）：`initial_correct / final_correct / joint_correct`（仅 free）、
  `tone_correct`（free + tone）、`consistent_correct`（detect）、`choice_correct`（ab）；
  `consistency_tone_correct` 保留列但恒为 NA（兼容旧 schema）。

## 10. `results/runs/<run_id>/detection_rows.parquet`

`items.parquet` 的行（来自**检测实验** detect 任务）+ `detection_role ∈ {neg,pos}`、
`detection_subset ∈ {main,initial,final,tone}`、`generation_condition`（`think1__clean__json`
等）、`variant_type`（正类错误类型）。

> 语义：`pred_consistent=True` 表示模型判定**一致**；错误检测的阳性类是**不一致**。
> TP = pos 且判为不一致；FP = neg 且判为不一致；TN = neg 且判为一致；FN = pos 且判为一致。

## 11. `results/runs/<run_id>/score_summary.json`

`n_items / n_tasks / n_generations / n_unique_requests / parse_status 计数 /
per_condition 覆盖率 / per_experiment 计数 / detection_rows`。

## 12. report 产物（`pinyin_eval.report.write_report`）

| 文件 | 内容 / 列 |
|---|---|
| `report.md` | 中文报告（2026-10-28 v2 口径）：摘要、实验×指标矩阵、主结果（总体/逐 voice/逐最小对类型，含 95% CI；实验一/二仅 Tone）、检测实验（main/initial/final/tone、EP/ER/F1/FPR/FNR、基线）、消融三组（core / detection / pairs；8 生成条件）、条件对比较（McNemar + 配对 bootstrap + Holm）、混淆矩阵 top-10、附录（设计更新说明、数据事实、最小对计数、检测集构成、解析失败、模型/参数/版本、工件清单） |
| `metrics.csv` | `scope, experiment, prompt_condition, condition_id, generation_condition, voice, pair_type, subset, dimension, metric, value, ci_lo, ci_hi, n, n_failures` |
| `detection.csv` | `scope, generation_condition, voice, subset, dimension, metric, value, ci_lo, ci_hi, n, n_failures, tp, fp, tn, fn, n_neg, n_pos` |
| `ablation.csv` | `group(core/detection/pairs/parse), compared_with, is_main, condition_id, generation_condition, experiment, prompt_condition, dimension, metric, value, n, n_failures, n_neg, n_pos, is_balanced`（检测组带配平标记） |
| `comparisons.csv` | `family, metric, dimension, axis(thinking/audio_quality/output_form), condition_a, condition_b, n_pairs, acc_a, acc_b, diff, diff_lo, diff_hi, mcnemar_b, mcnemar_c, p, p_holm, method` |
| `baselines.csv` | `scope, experiment, subset, dimension, baseline(random/majority), value, label, n` |
| `confusion/exp3_{initial,final,tone}_overall.csv`、`..._voice_{A-D}.csv` | 混淆矩阵（行=gold，列=pred） |
| `summary.json` | 机器可读摘要：run 元信息、主条件、生成条件列表、解析统计、bootstrap 设置、各实验 headline 指标、工件路径 |

> 报告不重算预测：所有指标都来自 `items.parquet` + `detection_rows.parquet`，并直接调用
> `metrics/*` 函数，保证与手工调用一致。若 run 带 `item_subset_source`（消融子集 run），
> 报告头部会标注“子集口径，非全量”，避免把消融 run 的主结果误当全量。
