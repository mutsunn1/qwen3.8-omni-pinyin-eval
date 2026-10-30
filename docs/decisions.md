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

## 8.1 真实 pilot 实测（2026-10-29，华北2 业务空间 ws-9mw2rfxyo1pn3qsv）

* `smoke`：1 次调用成功（`"声调=1"`，45 prompt tokens，7.1 s）。
* pilot（最小化设计 × `--limit 12` = 156 次，13 个 condition）：**156/156 解析成功、0 错误**，
  成本 **¥0.0302**，平均延迟 1.74 s（think off ≈0.8–1.7 s，think on ≈2.9–4.8 s）。
* 实测 prompt tokens ≈117–137/次（含音频 2–14 token），比预估（约 150–200）更低；
  think off 输出 5–7 token、think on reasoning ≈110 token。
* **实测单价推算**：think-off ≈ ¥0.00011/次，think-on ≈ ¥0.00046/次；
  最小化默认档（59,968 次）**实际预计 ≈¥11**，远低于 estimate 的 expected ¥18.9
  与 worst-case ¥50.1；¥70 预算充足。
* **发现的问题与修正（prompt p3.1）**：模型在自由识别时会写 `"v"`（实测
  `{"initial": "q", "final": "v", ...}`）。解析层本就把 v→ü 规范化，但为减少噪声，
  实验三与 exp5 free prompt 明确 “ü 直接写作 ü（不要写成 v）”，并写明声母书写规则。
* 扫描/中断保护：`--concurrency`、`--qps`、`--request-timeout` 全局参数与
  `generate --min-interval`（错峰启动，不降低并行度）已加入，用于账号限流（README 429 处理）。
* 未受影响：HuggingFace/GitHub 在本机被系统代理拦截（Clash 主节点故障），
  但数据与任务清单已在本地，实验不受影响；需要访问 GitHub 时临时关闭代理。

### 8.7 第二个外部基线：讯飞 ISE（2026-10-30）

* 接口：`wss://ise-api.xfyun.cn/v2/open-ise`（HMAC-SHA256 鉴权），凭证只入 `.env`。
* ISE 是参考打分 API，不是自由识别；用**候选打分**适配：静音填充 300ms、
  `一<拼音>一一` 三字标注试卷（标注比例限制）、每音频 2–4 候选取最高分。
* 实际消耗 **5,843 / 10,000** 次交互（含探针）；结果（同 mini128 子集）：
  tone 0.922（Qwen 0.904）；7 类最小对多数持平/更好（aspiration 1.000、
  nasal 0.969、ian/üan 0.948、tone 0.977，但 fbn 0.906 / iü 0.938 低于 Qwen）；
  检测 AUC 0.909（Qwen 派生 0.910）、eval F1 0.914（Qwen 0.923）。
* 局限：候选判别 ≠ 自由识别，绝对分被占位字稀释；仅作相对比较基线。
* 详见 `docs/baseline-ise.md`；复现脚本 `scripts/eval_xfyun_ise.py`。

### 8.6 输出规范化修复 + 基线弱对分析（2026-10-30）

* 实测模型会把零声母 y/w 音节拆成 `("y","e")`、把舌尖元音写成 `("zh","")`；
  旧 normalizer 分别给出 `("","ye")` 和 `("z","h")`，系统性压低 final/initial。
  已修：`YW_SURFACE_RULES`（ya/ye/yan/yu/wu/wo/… → 规范韵母）+ 舌尖元音
  `("zh","") → ("zh","i")`；只重新评分（**无 API 调用**），
  mini128 变为 initial 0.962 / final 0.949 / joint 0.838，派生检测 F1 0.922 / FPR 0.162。
* 与 SDHY 基线逐对比较（同 2,048 音频，自由识别）：
  * 基线最弱声母对 m/n 0.301、n/l 0.375、d/t 0.438、s/sh 0.482、z/zh 0.520 →
    Qwen 0.858 / 0.921 / 0.990 / 1.000 / 0.934；
  * 基线最弱韵母对 üe/ie 0.438、ei/ie 0.469、o/uo 0.473 → Qwen 0.920 / 0.922 / 0.804；
  * 声调 1/3 0.508 → 0.921；2/3 0.664 → 0.825；3/4 0.664 → 0.919；
  * **Qwen 仍弱/更差**：m/n 0.858、o/uo 0.804、2/3 声 0.825、**en/eng 0.883
    （低于基线 0.938）**、ian/üan 0.906 vs 0.917（n=96，不显著）。
* 结论：Qwen 在基线弱对上有 10–50pp 提升，但 m/n、o/uo、2/3 声、en/eng 是明确
  的下一批补强点（可做定向 prompt 或数据层修正）。
* 可复现：`scripts/eval_sdhy_baseline.py`（基线） + `results/baseline-vs-qwen-pairs.json`。

### 8.5 外部声学基线：SDHY P0-v2（2026-10-30）

* 从 `CHAO-AI-Lab/SDHY_Test_System@61b2d05` 拉取 `evaluater_speaking_0.1`；编码器
  `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`（sha256 de031fd4…，与
  `models/p0-baseline.json` 一致），X2/M1 头随仓库发布。
* 本机无 CUDA，用 cpu-fp32 跑；官方标定是 cuda-fp16（不宣称完全等价）。
* 该产品链路是**已知 target 的评测**，本实验需要自由识别 → 适配：initial=M1 CTC
  首音素、final=X2 头逐帧最大、tone=其 F0 特征 + 在不相交 512 条上校准的 4 类规则。
* 结果（2,048 音频，同 mini128-v1 子集）：initial 0.600 / final 0.746 / tone 0.699 /
  joint 0.299；派生检测 F1 0.731 / FPR 0.701；Qwen 对应 0.959/0.939/0.912/0.830、
  0.918/0.170。详见 `docs/baseline-sdhy.md`。
* 结论：该基线可作为“声学编码器 + 小头”的对照，但自由识别是适配口径，不能把差距
  直接解读为产品能力差距。

### 8.4 升级档验证（run mini128-v1，2026-10-30）

* 128 target / 2,048 音频 / 79,680 次调用，实际 **¥16.78**（累计 ¥30.60/¥70）。
* 主结果与 96 target 档一致且 CI 收紧（如 exp3 joint 0.807→0.830，CI ±2.0pp→±1.6pp）。
* 检测：模型直接判定 FPR 0.771 稳定存在；派生检测 F1 **0.918**、FPR **0.170**，
  在 2,048 音频上确认。结论：完整实验检测主口径采用派生检测。
* thinking on 仍系统性有害（−16 ~ −44pp）；噪声损伤声母/联合（−12 ~ −13.5pp）；
  文本输出仅对 exp5 韵母任务有害。

### 8.3 检测偏置专项（2026-10-30）

* 背景：`mini3-v1` 检测 FPR=0.764（一致样本 76% 被判不一致）。
* Prompt 探针：`configs/probe-detect-v{1,2,3}.yaml`（512 音频 ×6 任务，独立 task dir
  `data/manifests/tasks-probe-v*`，共 9,216 次调用 ≈ ¥1.45）。配对比较显示 v1/v2/v3
  均未改善（FPR 0.850/0.982/0.832 vs 基线 0.805）。
* **决策**：完整实验的检测部分改用“自由识别 → 程序比较”派生检测（与声称 target 的
  initial/final/tone 比较），Error-F1 **0.720 → 0.908**、FPR **0.764 → 0.193**，
  零额外调用；模型直接一致性判定仅作为辅助口径报告。
* 复现：`.venv/bin/python scripts/analyze_detection_probe.py`（读取 runs 下
  `mini3-v1` / `probe-detect-v*` 的 detection_rows）。

### 8.2 最小化主实验完成（run mini3-v1，2026-10-29/30）

* 59,968 计划调用全部完成（100%），解析失败 **0**；10 条瞬态 400 经 resume 重试成功。
* **实际花费 ¥12.37 / 预算 ¥70（17.67%）**；对比 estimate expected ¥18.85、worst ¥50.11，
  实测低于预期 34%（prompt tokens 比预估小）。
* 结果（C1 主条件）：声母 0.945、韵母 0.926、声调 0.910、联合 0.807；检测精度 0.565 /
  召回 0.993 / FPR 0.764（过度判“不一致”）。
* 消融：thinking on 显著有害（−17 ~ −42pp）；SNR20 噪声损伤声母/联合（−10 ~ −13pp），
  声调鲁棒；自由文本仅对 exp5 韵母单维度任务有害（ü 丢失，−17 ~ −24pp）。
* 完整结果与解释：`docs/results-minimal-v1.md`。

## 9. 未决问题（不阻塞框架）

1. 正式实验使用哪个模型快照 id（若百炼提供 `qwen3.8-omni-flash-YYYY-MM-DD`）？
   config 已留位置，建议开跑前固定并记录附录。
2. ~~真实 API 凭证尚未配置~~ —— 已于 2026-10-29 配置（业务空间
   `ws-9mw2rfxyo1pn3qsv`，华北2），smoke + pilot 通过，正在运行最小化主实验（`mini3-v1`）。
3. 最小化设计默认档位（96 target / 48 实例/类）与 70 元预算的对应关系已在
   `docs/experiment-design-minimal.md` §3.5 给出；如预算币种不是 CNY，按价格换算调整。
