# 关键决策与歧义处理

本文件记录实现时对设计文档（`docs/experiment-design.md`）中未完全明确或与数据事实
不一致之处的处理方式。目标是：**实现可复现、结论可解释**，每一处偏离都可被审计。

> **2026-10-28 v2 设计更新**（Obsidian Vault 设计文档 git commits `f4caf99`
> "实验一/二只评声调，新增检测实验并同步矩阵/检测集/输出评分" 与 `eb5cbe8`
> "新增各实验 Prompt 模板"）。本文件与实现已按 v2 全面更新，旧版内容（实验一/二做
> 一致性判定、exp2 含改声调变体、exp4~六统一 4 提示条件）已废弃。

## 1. 模型与接口

| 项 | 决策 | 依据 |
|---|---|---|
| 模型 id | `qwen3.8-omni-flash` | 设计文档写 “Qwen3.8-Omni”；百炼当前文本输出模型 `qwen3.8-omni-flash` 即 Qwen3.8-Omni-Flash（[文档](https://help.aliyun.com/zh/model-studio/qwen-omni)） |
| 服务地址 | 必须配置 `DASHSCOPE_BASE_URL`（业务空间兼容地址）；未配置时回退 `https://dashscope.aliyuncs.com/compatible-mode/v1` | 百炼文档要求业务空间地址，公共兼容地址是否可用取决于账号/地域 |
| 思考开关 | 开 = 不发送 `reasoning_effort`（Qwen3.8-Omni 默认开启思考）；关 = `reasoning_effort="none"` | 深度思考文档；开态思考力度设计未指定，采用模型默认，config 可改 low/medium/high |
| JSON 输出 | `response_format={"type":"json_object"}` + prompt 内固定 schema | 百炼文档示例支持 json_object |
| 音频输入 | base64 data URI（`data:audio/wav;base64,...`），`format="wav"` | 百炼文档"本地文件（Base64）"要求完整 Data URI；单音节文件远小于 10MB 上限 |
| 请求结构 | `messages=[system, user]`，system 为设计文档 v2 的共用 system prompt | 设计文档 §五 |
| 解码参数 | `temperature=0, top_p=1, max_tokens=256` | 设计文档固定参数 |
| 记录模型版本 | `run_manifest.json` 记录 model id + 本机调用日期；若改用快照 id 只需改 config | 设计文档"固定模型 ID/版本与调用日期，记录在附录" |

> 未做真实 API 验证：以上按 2026-10-28 的百炼文档实现；首次 `smoke` 若返回参数错误，
> 调整 `configs/base.yaml` 的 `model.*` 即可，无需改代码结构。

## 2. 任务与实验结构（v2）

### 2.1 三个任务目录、五个分析视图

| 目录 | 任务 | 分析视图 |
|---|---|---|
| `core.jsonl` | 全量 6160 音频 × {free, correct_target(tone), wrong_target(tone)×5} | 实验一 = correct_target；实验二 = wrong_target；实验三 = free |
| `detection.jsonl` | 全量 6160 音频 × {1 一致 + 2 改声母 + 2 改韵母 + 1 改声调} | **检测实验**（只输出 `{"consistent"}`） |
| `pairs.jsonl` | 2440 对最小对实例 × {free, correct_target(tone), wrong_target(仅 exp4/5), A/B} | 实验四/五/六 |

* 实验一/二共用同一个 prompt 模板（v2 §五），唯一区别是 target 正确/错误；prompt 不提示
  target 是否有错，输出只有 `{"tone"}`。
* 检测实验是独立任务（target + target_tone → `{"consistent"}`），不参与实验一/二/三的
  提示条件全因子；它的生成条件网格是 8 个（thinking × quality × form），提示条件固定。
* 相同 prompt 文本的请求在 runner 层按 `request_id` 只发一次（例如实验三 free 与实验四
  free、实验一 correct_target 与实验四 correct_target 对同一音频共用一次调用）。

### 2.2 实验二的 5 个错误无调变体

v2 设计："每个样本构造 5 个错误无调拼音（只改声母或韵母，不含改声调；声调错误由检测实验
覆盖）"。实现为：

* 2 个改声母变体（同韵母、换声母）；
* 2 个改韵母变体（同声母、换韵母）；
* 1 个额外段错误变体：按音频 hash 确定性地再改一次声母或韵母（仍只改一个段；
  config `tasks.exp2.n_extra_segmental_variants`，设为 0 即退回 4 个变体）。

`er`、`weng` 在 378 个规范音节中没有同韵母的其它声母，只能给 2 个改韵母 + 1 个额外改韵母
= 3 个变体（exp2 因此共少 64 个任务）。所有变体的 target 均为**无调拼音**，不给声调。
这是对"5 个变体"与"只改声母或韵母"两句话的一个自洽解读；如需改成 "2+2+1=同时改声母韵母"
只需扩展 `tasks.py` 的 extra 变体规则。

### 2.3 检测实验与检测集

* 一致样本：原始 `target`（无调拼音）+ `label`（真实声调）；
* 不一致变体：2 改声母（target 改、target_tone=真实声调）、2 改韵母（同）、
  1 改声调（target 不变、target_tone≠真实声调）；
* 检测集：主分析每音频 1 负类（一致样本）+ 1 正类（按 2/2/1 权重分层随机抽 1 个变体），
  得到 6160 负 + 6160 正；另构建 initial/final/tone 三个平衡子集；
* `er`/`weng` 没有改声母变体 → 检测 initial 子集为 6128 而非 6160（main 仍 6160 配平）；
* 所有检测指标按 audio 聚类（同一音频的变体不视为独立样本）。

### 2.4 实验四/五/六

* 4 个提示条件：free、correct_target、wrong_target、A/B。
* **实验六（声调最小对）无法构造 wrong_target**：两侧音频的无调 target 相同，v2 的 target
  prompt 不再包含声调，无法表达"错误目标"。因此 exp6 只生成 free / correct_target /
  A/B 三个提示条件；声调辨别能力由 A/B 与 free 条件覆盖。这是与"实验四~六 4 水平"
  设计表述的一处偏差，已在 `docs/contracts.md` 与报告中注明。
* 指标矩阵中 exp4 Initial / exp5 Final / exp6 Tone 的 ✓ 对应 **free 条件**；target 条件
  只评声调（与实验一/二一致）；A/B 准确率单独报告。

### 2.5 输出 schema 与文本标记（v2）

| 任务 | task_type | JSON | 自由文本 |
|---|---|---|---|
| 实验一/二（及 exp4~六 target 条件） | `tone` | `{"tone": 1-4}` | `声调=1` |
| 实验三（及 exp4~六 free） | `free` | `{"initial","final","tone"}` | `声母=zh；韵母=ang；声调=1`（零声母 `声母=`） |
| 检测实验 | `detect` | `{"consistent": true\|false}` | `一致=是` / `一致=否` |
| 实验四~六 A/B | `ab` | `{"choice": "A"\|"B"}` | `选择=A` |

* 解析器同时兼容旧标记（`一致性=`、`无` 表示零声母等），但新 prompt 使用上表标记；
* `{format_instruction}` 在设计文档给出的通用句后追加了该任务的输出 schema
  （如 `输出格式：{"tone": 1|2|3|4}`），否则模型无法可靠输出可解析格式；
* prompt 版本号 `p2`，任何措辞变化都会改变 `request_id`，旧预测不会混入新 run。

## 3. 抽样与统计

| 项 | 决策 |
|---|---|
| 主实验样本量 | 全量：core 6160 音频×7 任务 + detection 每音频 6 任务 + 全部最小对实例 |
| 主实验生成条件 | 设计未指定主条件；默认 `thinking=on, clean, json`（JSON 是机器评分主格式），config 可改 |
| 消融样本量 | 每个提示条件/任务 256 条（4 voice × 4 tone × 16 条），8 个生成条件共用同一子集 |
| 消融子集 | core 3×256=768；detection 256 音频×6 任务=1536（保证 main/typed 检测都可算）；pairs 按 pair_type ≤256 实例（i/ü 80、ian/üan 48 全取） |
| 检测集 | 主 1:1 配平 6160+6160；typed 子集各 6160+6160（initial 6128 正类）；audio 聚类 |
| bootstrap | percentile bootstrap，2000 次；cluster = audio_id；voice 分层；seed 固定 |
| 显著性 | McNemar（精确二项双侧；b+c>1000 用连续性校正 χ²）；Holm 校正；配对 bootstrap 差值 CI |
| τ=0.5 | **不参与**：v2 明确要求类别输出（`consistent: true/false`），不是概率；若后续改用概率再按 0.5 判定 |
| 解析失败 | `parse_error / refused / empty / api_error` 不计入指标分母，单独报告比例 |
| 基线 | 声调四分类 25%、A/B 50%、平衡检测集 50%、initial/final 多数类基线 |

## 4. 数据事实核对（2026-10-28）

| 设计文档/假设 | 实测 |
|---|---|
| 6160 条、4 TTS voice（各 1540） | ✓ |
| 声调 1–4 各 1540，无轻声 | ✓ |
| 385 原始 target → 378 规范 target | ✓（7 条零声母规范化规则完全覆盖） |
| 音频原始单声道 16 kHz / 16-bit | ✓ WAV PCM_16 16 kHz mono，0.215–1.334 s |
| target 无调、label 为声调 | ✓ |
| `mandarin_tone_syn_target_mapping.csv` | 仓库内 `data/mapping/`（sha256 记录于 run manifest），378 行，count 求和 = 6160 |
| 数据集 commit | `d607806619aba678030dac10ea086fb19347205a` |
| 最小对计数 | 36/67/21/40/5/3/2268 对、576/1072/336/640/80/48/9072 实例，`build-tasks --validate` 断言 ✓ |

## 5. 调用预算（v2，`estimate` 实测）

| 阶段 | 条件数 | API 调用 |
|---|---:|---:|
| 主实验（全量） | 1 | 107,896 |
| 消融（固定子集 × 8 条件） | 8 | 96,960 |
| 合计 | — | **204,856** |

（v1 为 83,321 + 88,952；v2 增加检测实验全量任务与检测消融子集。）

## 6. 未决问题（需要实验负责人确认）

1. **实验二第 5 个变体**：当前实现 = 2 改声母 + 2 改韵母 + 1 额外改声母/韵母；
   若设计意图是 "1 个同时改声母和韵母" 或其他组合，改 `tasks.exp2` 配置后重新
   `build-tasks` 即可（不影响框架）。
2. **实验六 wrong_target**：v2 的 target prompt 无调，声调对两侧无调 target 相同，
   无法构造"错误目标"，已跳过；如希望保留 4 提示条件，需要为 exp6 定义一个带调 target
   的 prompt（例如检测实验模板），请确认。
3. **检测实验的消融单位**：当前消融保留 256 音频的全部 6 个检测任务（1536 条/条件），
   以保证类型子集可算；若希望"每个条件恰好 256 条"，需改为只保留 1 个检测结果。
4. 正式实验用哪个模型快照 id（若百炼提供 `qwen3.8-omni-flash-YYYY-MM-DD`）？
   config 已留位置，建议开跑前固定并记录在附录。
