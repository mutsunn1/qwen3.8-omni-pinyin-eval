# 正式实验运行手册（Runbook）

本手册用于**真实 API 实验**。框架层面的自测（无 key）见 README 第 1 节。

## 0. 前置检查清单

- [ ] 环境统一：`scripts/setup.sh`（或 `make setup`）已执行，Python 3.12 / `uv.lock` 已就位
- [ ] `.env` 已从 `.env.example` 创建并填写凭证（或 shell 已 export 同名变量）
- [ ] `DASHSCOPE_API_KEY` 已配置（百炼 API Key，北京/新加坡地域不同）
- [ ] `DASHSCOPE_BASE_URL` 已配置为**业务空间**兼容地址：
      `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`
- [ ] 账号已开通 Qwen3.8-Omni-Flash（或改 `configs/base.yaml: model.model` 为可用模型 id）
- [ ] 磁盘 ≥ 3 GB（parquet 100MB + 音频 60MB + 任务清单 230MB + 结果）
- [ ] `configs/base.yaml` 已复核：主实验条件、消融抽样、并发/限流、模型参数
- [ ] 已知晓调用预算：完整设计 ≈ 212,576 次（主 126,704 + 消融 85,872，预期 ¥142.8 / 最坏 ¥177.6）；
      或选择 **¥70 最小化设计**：`configs/minimal.yaml` 59,968 次，预期 ¥18.9 / 最坏 ¥50.1）

## 0.1 两条实验路径

| 路径 | 配置 | 请求数 | 成本（expected / worst） | 适用 |
|---|---|---:|---:|---|
| **最小化（推荐先跑）** | `--config configs/minimal.yaml`，见 `docs/experiment-design-minimal.md` | 59,968 | ¥18.9 / ¥50.1（硬预算 70） | pilot / 预算 ≤¥70 |
| 完整设计 | `--config configs/base.yaml` | 212,576 | ¥142.8 / ¥177.6 | 论文级全量结论 |

最小化路径命令：

```bash
uv run pinyin-eval --config configs/minimal.yaml build-tasks --minimal
uv run pinyin-eval --config configs/minimal.yaml generate --tasks all --profile minimal \
  --conditions minimal --run-id mini-v1 --tag minimal
uv run pinyin-eval --config configs/minimal.yaml score  --run-id mini-v1
uv run pinyin-eval --config configs/minimal.yaml report --run-id mini-v1
uv run pinyin-eval --config configs/minimal.yaml cost   --run-id mini-v1
```

## 1. 环境与连通性

```bash
make setup                        # uv sync --locked + 生成 .env（若缺失）
# 编辑 .env：DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL
uv run pinyin-eval check          # 应显示 api_key_present=true
uv run pinyin-eval smoke          # 1 次真实调用；应返回可解析的 text 与 usage
```

`smoke` 失败排查：

| 现象 | 处理 |
|---|---|
| 401 / invalid api key | 检查 `DASHSCOPE_API_KEY` 与地域是否匹配 |
| 404 / model not found | 检查 `configs/base.yaml: model.model`；确认账号已开通该模型 |
| 400 / url error | 百炼要求业务空间 `DASHSCOPE_BASE_URL`，不能用公共地址 |
| 400 / unsupported parameter | 检查 `model.thinking_on_reasoning_effort`；关闭时可设 `null` 试 |
| 429 / rate limit | 降低 `model.concurrency`、设置 `model.qps`（如 1–4） |

## 2. 数据准备

```bash
uv run pinyin-eval prepare-data
uv run pinyin-eval build-tasks --validate
```

验收：

- `prepare-data` 输出 `n_audio=6160, n_voices=4, tones 各 1540, sample_rates=[16000]`；
- `build-tasks --validate` 输出 `minimal-pair counts match the design document ✓`；
- 任务清单行数（v3 设计）：`core.jsonl=43056`、`detection.jsonl=36896`、`pairs.jsonl=70944`；
- 检测集：`main=6160, initial=6128, final=6160, tone=6160`（initial 的 32 条缺口原因见
  `docs/decisions.md` §2.3）；
- 消融子集：`ablation_core_items.txt=768`、`ablation_pairs_items.txt=8448`
  （v3 检测实验不参与消融，无 `ablation_detection_items.txt`）。

## 3. 试跑（Pilot，强烈建议）

正式烧预算前，先用 40 条音频 × **全部 8 个生成条件**验证 prompt / 解析 / thinking 行为：

```bash
# --limit N 表示“每个生成条件最多随机抽 N 条请求”（覆盖 free/correct/wrong/detect/AB 五种
# 提示视图 × 8 个生成条件 = 40 condition_id；25 条/条件 ≈ 1,000 次调用，成本很低）
uv run pinyin-eval generate --tasks all --conditions all --limit 25 \
  --run-id pilot1 --tag pilot
uv run pinyin-eval score --run-id pilot1
uv run pinyin-eval report --run-id pilot1
```

Pilot 验收指标（`results/runs/pilot1/score_summary.json`）：

- 每个条件的 `ok / n` 比例 ≥ 95%（否则先修 prompt 或处理 JSON 模式问题）；
- `parse_failures.jsonl` 中人工抽查 10 条，确认不是系统性解析问题；
- JSON 条件应几乎全部可解析；free-text 条件应能抽取新标记
  `声调=`、`声母=…；韵母=…；声调=`、`一致=是/否`、`选择=A/B`；
- thinking-on 条件应在 `generations.jsonl` 的 `reasoning_text` 中看到思考内容；
- 抽样核对 `generations.jsonl`：`media_path` 指向干净/噪声文件正确、`condition_id` 正确。

若 pilot 发现提示词问题：修改 `src/pinyin_eval/prompts.py` 后必须
**提升 `PROMPT_VERSION`**、重新 `build-tasks`（request_id 会变化），旧 pilot 结果作废。

## 4. 主实验（全量）

```bash
uv run pinyin-eval generate --tasks all --profile main --run-id main-v1 --tag main
```

- 预期 107,896 次调用（含检测实验）；按 `concurrency=4`、单次 3–10 s 估计约 30–50 小时；
- 建议在 `tmux`/后台 job 中运行；中断后原命令 + 原 `--run-id` 重跑即续跑；
- 运行中可随时查看 `results/runs/main-v1/generations.jsonl` 计数与
  `run_manifest.json`（结束前 `status=running`）。

## 5. 消融（8 个生成条件 × 固定子集）

```bash
uv run pinyin-eval generate --tasks all --profile ablation --conditions all \
  --run-id ablation-v1 --tag ablation
```

- 预期 96,960 次调用；
- 该 run 自带主条件（think1/clean/json）在消融子集上的重复调用，用于与其余条件**配对**
  比较；若想省预算可改为 7 个条件（去掉 `--condition` 中的主条件），但需自行保证
  report 能取到主条件基线；
- `--profile ablation` 会自动读取 `data/manifests/tasks/ablation_*_items.txt`。

## 6. 评分与报告

```bash
uv run pinyin-eval score  --run-id main-v1
uv run pinyin-eval report --run-id main-v1

uv run pinyin-eval score  --run-id ablation-v1
uv run pinyin-eval report --run-id ablation-v1
```

产物（`results/runs/<run_id>/`）：

| 文件 | 说明 |
|---|---|
| `items.parquet` | 每条 (task item × condition) 的解析结果与正确性 |
| `detection_rows.parquet` | 检测实验分析的 neg/pos 行（neg=一致样本，pos=不一致变体） |
| `score_summary.json` | 解析状态 / 每条件覆盖率 |
| `report.md` | 主报告（中文） |
| `metrics.csv` | 长表指标（含 95% CI） |
| `detection.csv` | 检测指标（EP/ER/F1/FPR/FNR、分类型子集） |
| `ablation.csv` | 消融网格 |
| `comparisons.csv` | McNemar / 配对 bootstrap / Holm 校正 |
| `baselines.csv` | 随机/多数类基线 |
| `confusion/*.csv` | 混淆矩阵 |
| `summary.json` | 报告关键数字（机器可读） |

## 7. 结果审计清单

- [ ] `score_summary.parse_status`：`parse_error + refused + empty + api_error` 比例 < 5%，
      且没有集中在某个条件（如果 JSON 条件失败率高，优先怀疑 `response_format` 或
      `max_tokens` 截断）
- [ ] `report.md` 的“实验×指标矩阵”覆盖 v3 设计文档第 51 行表格的每一项（✓ / —）：
      实验一/二只报 Tone、检测实验报 EP/ER/F1/FPR/FNR、实验四~六各报三个子实验的
      initial/final/tone（无 A/B）
- [ ] 检测指标（来自检测实验，neg=一致样本 / pos=不一致变体）EP/ER/F1、FPR/FNR 与
      bootstrap CI 齐全；分 voice 与总体都有；解析失败单独统计；检测实验只在主条件运行
- [ ] 成本节：`usage.csv`、每条件 token/成本、预算状态；实际成本与 `estimate` 预期一致
- [ ] 消融表每个条件 n 与预期一致（主实验已覆盖的行标注“复用主实验”）
- [ ] McNemar 的 b/c 与效应量方向符合直觉（如噪声条件应更差）
- [ ] 混淆矩阵与“最小对立组”设计一致（平翘舌、送气、前后鼻、n/l 应是主要混淆来源）
- [ ] 附录信息齐全（见下）

## 8. 附录信息模板（写入论文/报告）

```
数据集: HaitongSUN/prosodic-abx, config mandarin_tone_syn,
        commit d607806619aba678030dac10ea086fb19347205a
        target mapping: mandarin_tone_syn_target_mapping.csv (sha256: <run_manifest>)
模型:   阿里云百炼 qwen3.8-omni-flash（<snapshot id 若使用>）
        服务地址: <DASHSCOPE_BASE_URL 地域/业务空间>
        调用日期: <run_manifest.created_at 起>
解码:   temperature=0, top_p=1, max_tokens=256
思考:   on = 模型默认力度; off = reasoning_effort=none
输出:   JSON 条件 response_format=json_object; 自由文本条件无
        schema: tone {"tone"} / free {"initial","final","tone"} / detect {"consistent"} / A/B {"choice"}
Prompt: 版本 <PROMPT_VERSION>，全文见 tasks/*.jsonl 的 prompt_texts
统计:   bootstrap 2000 次 (cluster=audio, stratify=voice), McNemar + Holm
调用量: 完整设计 ≈212,576（预期 ¥142.8 / 最坏 ¥177.6）;
        或最小化设计 ≈59,968（预期 ¥18.9 / 最坏 ¥50.1，预算 70）
脚本:   git commit <run_manifest.git_commit>; config sha256 <run_manifest.config_sha256>
```

## 9. 常见问题

* **同一 run-id 重跑会不会重复计费？** 不会：已完成且无 error 的
  `(request_id, condition_id)` 会跳过；见 `counts.n_skipped_resume`。
* **如何只重跑失败项？** 删除 `generations.jsonl` 中对应行后重跑同一 run-id；
  或新 run 用 `--item-ids` 指定。
* **想换模型/提示词做对比？** 新 run-id + 新 config；`run_manifest` 会记录差异，
  report 各自生成后再比较。
* **结果目录可以删吗？** `data/` 与 `results/` 都可再生；论文数据请归档 run 目录 +
  config 快照。
