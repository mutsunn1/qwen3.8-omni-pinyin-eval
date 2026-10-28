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

`TaskItem` 字段（省略 `None` 等价空值）：

| 字段 | 说明 |
|---|---|
| item_id | 确定性 ID；核心任务 `exp1\|correct_target\|{audio_id}`，最小对任务带 `\|{pair_id}` 后缀 |
| experiment | `exp1`..`exp6`（提示条件对应的分析视图） |
| prompt_condition | `free` / `correct_target` / `wrong_target` / `ab_choice` |
| audio_id / wav_path / voice | 音频与 voice |
| gold_initial / gold_final / gold_tone | 金标（永远描述音频） |
| cluster_id | bootstrap 聚类单位 = audio_id |
| pair_id / pair_type / pair_side | 最小对任务专用 |
| variant_type / variant_index | 实验二错误类型（initial/final/tone）与序号 |
| given_target / given_tone | 提示中"声称"的读音（无调拼音 / 带调） |
| consistent_gold | 目标条件下的金标一致性 |
| alternatives / correct_choice | A/B 任务候选（带声调数字）与正确项 |
| stratum | `voice\|tone`，消融分层用 |
| prompt_texts | `{"json": ..., "text": ...}` 两种输出形式的完整 prompt |
| prompt_version | prompt 版本号 |

## 3. `data/manifests/tasks/detection_sets.csv`

| 列 | 说明 |
|---|---|
| subset | `main` / `initial` / `final` / `tone` |
| audio_id / cluster_id / voice / tone | 样本 |
| negative_item_id | 实验一（正确目标）item |
| positive_item_id | 实验二（错误目标）item |
| variant_type | 正类错误类型 |
| negative_given_target / positive_given_target / positive_given_tone | 提示内容（便于复核） |

## 4. `data/manifests/tasks/minimal_pair_counts.json`

`{pair_type: {"n_pairs": int, "n_instances": int}}`，用于与设计文档计数表核对。

## 5. `data/manifests/tasks/ablation_*_items.txt`

每行一个 `item_id`：

* `ablation_core_items.txt`：3 个提示条件 × 256 条 = 768 条；
* `ablation_pairs_items.txt`：按 pair_type 选中的最小对实例的全部任务。

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

## 8. `results/runs/<run_id>/planned_calls.jsonl`

`{"request_id": ..., "condition_id": ...}` 每行一个计划调用；scoring 用它判断哪些
(task × condition) 单元属于本次运行。

## 9. `results/runs/<run_id>/items.parquet`

`ItemResult` 字段 + 派生列：

* 预测：`pred_consistent / pred_initial / pred_final / pred_tone / pred_choice`（规范化后）；
* 状态：`parse_status ∈ {ok, parse_error, refused, empty, api_error}`；
* 正确性（nullable boolean）：`initial_correct / final_correct / tone_correct /
  joint_correct / consistent_correct / choice_correct / consistency_tone_correct`。

## 10. `results/runs/<run_id>/detection_rows.parquet`

`items.parquet` 的行 + `detection_role ∈ {neg,pos}`、`detection_subset ∈ {main,initial,
final,tone}`、`generation_condition`（`think1__clean__json` 等）、`variant_type`（正类错误类型）。

> 语义：`pred_consistent=True` 表示模型判定**一致**；错误检测的阳性类是**不一致**。
> TP = pos 且判为不一致；FP = neg 且判为不一致；TN = neg 且判为一致；FN = pos 且判为一致。

## 11. `results/runs/<run_id>/score_summary.json`

`n_items / n_tasks / n_generations / n_unique_requests / parse_status 计数 /
per_condition 覆盖率 / per_experiment 计数 / detection_rows`。

## 12. report 产物

`report.md`、`metrics.csv`、`detection.csv`、`ablation.csv`、`comparisons.csv`、
`baselines.csv`、`confusion/*.csv`、`summary.json` —— 由 `pinyin_eval.report.write_report`
生成，列定义见 `report.py` 模块 docstring。
