# 关键决策与歧义处理

本文件记录实现时对设计文档（`docs/experiment-design.md`）中未完全明确或与数据事实
不一致之处的处理方式。目标是：**实现可复现、结论可解释**，每一处偏离都可被审计。

## 1. 模型与接口

| 项 | 决策 | 依据 |
|---|---|---|
| 模型 id | `qwen3.8-omni-flash` | 设计文档写 “Qwen3.8-Omni”；百炼当前文本输出模型 `qwen3.8-omni-flash` 即 Qwen3.8-Omni-Flash（[文档](https://help.aliyun.com/zh/model-studio/qwen-omni)） |
| 服务地址 | 必须配置 `DASHSCOPE_BASE_URL`（业务空间兼容地址）；未配置时回退 `https://dashscope.aliyuncs.com/compatible-mode/v1` | 百炼文档明确要求业务空间地址，公共兼容地址是否可用取决于账号/地域 |
| 思考开关 | 开 = 不发送 `reasoning_effort`（Qwen3.8-Omni 默认开启思考）；关 = `reasoning_effort="none"` | 深度思考文档；开态的思考力度未由设计文档指定，故采用模型默认，并在 config 中可改为 low/medium/high |
| JSON 输出 | `response_format={"type":"json_object"}` + prompt 内固定 schema | 百炼文档示例支持 json_object |
| 音频输入 | base64 data URI（`data:audio/wav;base64,...`），`format="wav"` | 百炼文档"本地文件（Base64）"要求完整 Data URI；短单音节文件远小于 10MB 上限 |
| 解码参数 | `temperature=0, top_p=1, max_tokens=256` | 设计文档固定参数 |
| 记录模型版本 | `run_manifest.json` 记录 model id + 本机调用日期；若改用快照 id（如 `qwen3.8-omni-flash-YYYY-MM-DD`）只需改 config | 设计文档"固定模型 ID/版本与调用日期，记录在附录" |

> 未做真实 API 验证：以上按 2026-10-28 的百炼文档实现；首次 `smoke` 若返回参数错误，
> 调整 `configs/base.yaml` 的 `model.*` 即可，无需改代码结构。

## 2. 任务与实验结构

### 2.1 实验一/二/三是同一批任务的三个视图

设计文档的"实验一~三共 3×2×2×2=24 个条件"中，提示条件本身是因子。但三者的音频集合
完全相同、`free/correct_target/wrong_target` 三种提示对所有音频都要跑（消融要求），
因此实现为**一个共享任务目录**，按提示条件切三个分析视图：

* `core.jsonl`：每条音频 = 1 个 free + 1 个 correct_target + 5 个 wrong_target 变体；
* 实验一视图 = correct_target 行；实验二 = wrong_target 行；实验三 = free 行；
* 相同提示文本只发一次 API（`request_id` 去重），例如实验三 free 与实验四 free
  对同一音频共享一次调用。

### 2.2 实验二的"改声调"如何表示

数据集的 target 是**无调拼音**，因此"改声调"必须让给定目标携带声调信息：

* 改声母 / 改韵母变体：给定 target 仍为无调拼音（只换声母/韵母），一致性判定只看声母/韵母；
* 改声调变体：给定 target = 本音频正确无调拼音 + **错误声调数字**（如 `zhang4`），
  一致性判定须包含声调；
* 金标声调永远取音频真实声调（设计文档明确要求"不采用错误目标拼音中的声调"）。

### 2.3 不可构造的"改声母"样本

`er`、`weng` 在 378 个规范音节中没有同韵母的其它声母（`er` 韵母唯一、`weng` 韵母 `ueng`
唯一），无法构造"改声母"错误目标。处理：跳过这 2 个音节 × 4 voice × 4 tone = 32 条音频
的 initial 变体（共少 64 个任务），exp2 变体总数为 30,736 而非 30,800；
检测集的 initial 子集为 6128 而非 6160（main 子集仍 6160 配平）。报告中如实注明。

### 2.4 最小对立对

* 计数与设计文档表逐项一致（36/67/21/40/5/3/2268 对；576/1072/336/640/80/48/9072 实例），
  `build-tasks --validate` 断言；
* **前后鼻只含 an/ang、en/eng、in/ing 三类韵母**（40 对）。设计文档文字写"所有声母的
  -n/-ng 对（如 …）"，但表中 40 对恰好等于这三类的数量；若把 ian/iang、uan/uang 也算入
  则为 52 对。默认按表格口径（40），并提供 `tasks.minimal_pairs.include_wide_nasal: true`
  / `--include-wide-nasal` 开关；
* 一条音频可属于多个声调最小对（如 `ai1` 属于 1-2、1-3、1-4 三对），因此最小对任务的
  `item_id` 带 `pair_id` 后缀；跨 pair 相同的提示仍按 `request_id` 去重为一次调用；
* 合并拼写的 7 个规范 target（yin/ying/you/wu/wei/wen/yun）在数据集中各有两种原始拼写
  （共 112 条音频）。核心任务使用全部 6160 条音频；最小对实例"每个 target×voice×tone
  取 1 条"时优先取 `raw_target == target` 的拼写。

### 2.5 实验四/五/六的提示条件与指标

设计文档给出 4 个提示条件（free / 正确目标 / 错误目标 / A/B）和指标矩阵
（exp4 Initial、exp5 Final、exp6 Tone）。实现：

* 4 个提示条件全部保留：free 用于测维度识别；正确/错误目标用于测一致性判断与真实声调；
  A/B 用于二选一；
* 指标矩阵中的 ✓ 对应 **free 条件的对应维度准确率**（exp4 声母、exp5 韵母、exp6 声调），
  其余条件作为消融/辅助结果报告，避免把"给定目标泄露答案"的准确率当作识别能力；
* 输出 schema 沿用设计文档固定的三种：一致性 `{"consistent","tone"}`、自由识别
  `{"initial","final","tone"}`、A/B `{"choice"}`。

## 3. 抽样与统计

| 项 | 决策 |
|---|---|
| 主实验样本量 | 全量：核心 6160 音频 × 3 提示条件 + 全部最小对实例 |
| 主实验生成条件 | 设计文档未指定主条件；默认 `thinking=on, clean, json`（JSON 是机器评分主格式），config 可改 |
| 消融样本量 | **每个提示条件 256 条**（4 voice × 4 tone × 16 条），8 个生成条件共用同一子集；`ablation_core_items.txt` = 768 条 |
| 最小对消融 | 每个 pair_type 最多 256 个实例（i/ü 80、ian/üan 48 全取，声调 256 对）；实例的两侧音频与 4 个提示条件全部保留 |
| 检测集 | 每个音频 1 负类（exp1）+ 1 正类（exp2 变体，按 2/2/1 权重分层随机）；另构建 initial/final/tone 三个平衡子集；所有检测指标按 audio 聚类 |
| bootstrap | percentile bootstrap，2000 次；cluster = audio_id；voice 分层；seed 固定 |
| 显著性 | McNemar（精确二项双侧；b+c>1000 用连续性校正 χ²）；Holm 校正；配对 bootstrap 差值 CI 作为效应量 |
| τ=0.5 | **不参与**：任务要求模型输出类别判断而非概率；若后续改用 logprob/概率输出，再按设计文档在 0.5 判定 |
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

## 5. 未决问题（需要实验负责人确认）

1. 正式实验用哪个时间点的模型快照 id（若百炼提供 `qwen3.8-omni-flash-YYYY-MM-DD`）？
   config 已留位置，建议开跑前固定并在附录记录。
2. 消融是否需要在**全量**上跑 8 个生成条件？（本框架默认设计文档口径：主实验全量 +
   消融 256 子集；如果预算允许全量，只需 `generate --tasks all --conditions all`。）
3. 最小对消融的 256 实例/类是否合适？（设计文档说"池子不足 256 的类别使用全部可用对"，
   实现为按实例数截断；如希望按"每个提示条件 256 条音频"口径，改
   `tasks.minimal_pairs.ablation_per_type_cap` 并重新 build-tasks。）
