# 关键决策与歧义处理

本文件记录实现时对设计文档中未完全明确或与数据事实不一致之处的处理方式。目标是
**可复现、可审计**。设计文档以 `docs/experiment-design.md` 快照为准。

> **2026-10-28 v3 设计更新**（Obsidian 设计文档 20:37 版本，sha256 `acf2015a…`）：
> 取消 A/B 任务；实验四/五/六各拆为自由检测/正确信息/错误信息三个子实验；
> 检测实验不参与消融。本文件、实现与文档已按 v3 更新（v2 内容废止）。

## 1. 模型与接口

| 项 | 决策 | 依据 |
|---|---|---|
| 模型 id | `qwen3.8-omni-flash`（可改快照 id） | 百炼当前文本输出模型 |
| 服务地址 | 必须配置 `DASHSCOPE_BASE_URL`（业务空间地址） | 百炼文档要求 |
| 思考开关 | 开 = 不发送 `reasoning_effort`（模型默认）；关 = `reasoning_effort="none"` | 深度思考文档 |
| JSON 输出 | `response_format={"type":"json_object"}` + prompt 内 schema | 百炼支持 json_object |
| 音频输入 | base64 data URI，`format="wav"` | 百炼本地文件（Base64） |
| 请求结构 | `messages=[system, user]`，system 为 v3 共用 prompt | 设计文档 §五 |
| 解码 | `temperature=0, top_p=1, max_tokens=256` | 设计文档固定参数 |
| 记录 | `run_manifest.json` 记录 model id、调用日期、pricing 快照、config sha、数据 commit | 设计文档附录要求 |

## 2. 任务结构（v3）

| 目录 | 内容 | 分析视图 |
|---|---|---|
| `core.jsonl` | 6160 音频 × {free, correct_target(tone), wrong_target(tone)×5} | 实验一=正确信息；实验二=错误信息；实验三=自由检测 |
| `detection.jsonl` | 6160 音频 × {1 一致 + 2 改声母 + 2 改韵母 + 1 改声调} | 检测实验（**仅主条件运行，不参与消融**） |
| `pairs.jsonl` | 2440 对最小对实例，每侧 3 个子实验：free / correct_target / wrong_target | 实验四（initial）/五（final）/六（tone） |

* **实验一/二**共用“给定无调 target、只报声调”的 prompt；实验二 target 有错但 prompt
  不提示。两实验只评 Tone。
* **实验三**自由识别 initial/final/tone。
* **实验四/五/六**：每个子实验只输出对应维度（`{"initial"}` / `{"final"}` / `{"tone"}`）。
  正确信息给 target_initial/final/tone 全对；错误信息只改对应维度，其余正确。
* **检测实验**：target（无调）+ target_tone → `{"consistent": true|false}`。
* 相同 prompt 文本请求按 `request_id` 去重（含 system prompt 与音频质量）。

## 3. 实验二错误注入与不可构造样本

* 实验二每个样本 5 个错误无调 target：2 改声母 + 2 改韵母 + 1 额外“只改声母或韵母”
  （按音频确定性选择；`tasks.exp2.n_extra_segmental_variants=1`）。
* `er`、`weng` 在 378 个规范音节中没有同韵母其它声母 → 只能给 3 个变体（2 改韵母 +
  1 额外改韵母）；exp2 共少 64 个任务。
* 声调错误只存在于**检测实验**（target_tone 错误），实验二不注入声调错误。

## 4. 检测实验与检测集

* 一致样本：原始 target + label；不一致变体：2 改声母 / 2 改韵母 / 1 改声调。
* 主检测集：每音频 1 负类 + 1 正类（2/2/1 分层随机）→ 6160+6160；typed 子集各 6160+6160
  （initial 6128：er/weng 无改声母变体）；按 audio 聚类。
* 检测实验只在主生成条件运行（v3）；不做 detection 消融。

## 5. Prompt 与输出解析

* `PROMPT_VERSION=p3`；system prompt 参与 request 指纹，改 prompt 会改变指纹，旧 run 不混用。
* 设计文档的通用 `{format_instruction}` 后追加该任务输出 schema（如
  `输出格式：{"tone": 1|2|3|4}`），否则模型无法可靠输出可解析格式。这是对文档的
  必要补全，不改变任务语义。
* 自由文本标记：`声调=`、`声母=`、`韵母=`、`声母=…；韵母=…；声调=`、`一致=是/否`；
  解析器兼容旧标记（`一致性=`、`无` 表示零声母等）。
* 解析失败（无法解析/拒答/空回复/API 错误）不计入指标分母，单独报告比例。

## 6. 统计与基线

| 项 | 决策 |
|---|---|
| 主实验样本量 | 全量 6160 音频 + 全部最小对实例 |
| 主条件 | `thinking=on, clean, json`（config 可改） |
| 消融 | 24（实验一~三）+ 72（实验四~六）；检测不参与；每条件固定子集 |
| bootstrap | 2000 次，audio 聚类、voice 分层，seed 固定 |
| 显著性 | McNemar 精确双侧 + 配对 bootstrap 差值 CI；Holm 校正 |
| 基线 | 声调 25%、平衡检测集 50%、initial/final 多数类基线（v3 无 A/B 基线） |
| τ=0.5 | 不适用：模型输出类别判断（consistent），不是概率 |

## 7. 计费与预算（新增）

* 价格快照（2026-10-28，华北2）：输入 ¥0.8/M、缓存命中 ¥0.1/M、输出 ¥2.7/M；
  Qwen3.8-Omni 音频 token = `ceil(时长秒) × 7`（不足 1 秒按 1 秒）。
  来源：[模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)、
  [Qwen-Omni 计费与限流](https://help.aliyun.com/zh/model-studio/qwen-omni)。
* 实际成本一律从 API `usage` 计算：`(prompt-cached)*0.8/1e6 + cached*0.1/1e6
  + completion*2.7/1e6`；`pinyin-eval cost --run-id` 汇总。
* 预算 guard：`generation.budget_cny` / `--max-cost`；达到后停止新请求，
  manifest 记 `budget_exceeded`（在途请求最多 concurrency 条），同 run-id 可续跑。
* 预估算三档：expected / conservative / worst-case；worst-case 假设每次输出打满
  `max_tokens=256`，因此是成本上界。
* 完整设计（v3）预计：主实验 126,704 次（¥101.3 / 最坏 ¥106.7）+ 消融 85,872 次
  （¥41.5 / 最坏 ¥70.9）= **212,576 次，预期 ¥142.8、最坏 ¥177.6**。
* **最小化设计（¥70 预算）**：见 `docs/experiment-design-minimal.md` 与
  `configs/minimal.yaml`；默认 96 target × 16 = 1536 音频 + 7 类×48 实例，
  4 个生成条件，检测仅主条件 → **59,968 次调用，预期 ¥18.9、保守 ¥21.3、
  最坏 ¥50.1**，硬预算 70。可选档位 64/32（¥12.5/¥33.3）与 128/64（¥25.0/¥66.6）。
  预算单位按 **CNY** 理解（百炼计费币种）。

## 8. 数据事实核对（2026-10-28）

| 设计/假设 | 实测 |
|---|---|
| 6160 条、4 TTS voice、声调各 1540 | ✓ |
| 385 原始 target → 378 规范 target | ✓（7 条零声母规范化规则） |
| 音频 WAV 16 kHz / 16-bit 单声道 | ✓ 0.215–1.334 秒 |
| 最小对计数 | 36/67/21/40/5/3/2268 对，576/1072/336/640/80/48/9072 实例，`build-tasks --validate` 断言 ✓ |
| 数据集 commit | `d607806619aba678030dac10ea086fb19347205a` |

## 9. 未决问题（不阻塞框架）

1. 正式实验使用哪个模型快照 id（若百炼提供 `qwen3.8-omni-flash-YYYY-MM-DD`）？
   config 已留位置，建议开跑前固定并记录附录。
2. 真实 API 凭证（`DASHSCOPE_API_KEY` / `DASHSCOPE_BASE_URL`）尚未配置；配置后先
   `smoke` + pilot（最小化设计 156 次调用）再决定跑完整还是最小化。
3. 最小化设计默认档位（96 target / 48 实例/类）与 70 元预算的对应关系已在
   `docs/experiment-design-minimal.md` §3.5 给出；如预算币种不是 CNY，按价格换算调整。
