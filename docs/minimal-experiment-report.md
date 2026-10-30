# Qwen3.8-Omni 普通话音素识别最小化实验报告

> 日期：2026-10-30 · 模型：`qwen3.8-omni-flash`（阿里云百炼，华北2 业务空间）
> 设计：[最小化实验设计（v3, ¥70）](experiment-design-minimal.md) ·
> 配置：`configs/minimal.yaml`（96 target 档）、`configs/minimal-128.yaml`（128 target 档）
> 运行：`mini3-v1`（2026-10-29/30）、`mini128-v1`（2026-10-30）

## 0. 摘要

* 两档分层子集（1,536 / 2,048 条音频）上完成 4 个生成条件 × 全任务结构的真实 API 实验，
  共 **139,648 次调用**，总花费 **¥30.60 / ¥70（43.7%）**，解析成功率 99.997%。
* 主条件（think off / clean / JSON）下：声调识别 **0.914**、声母 **0.959**、韵母 **0.939**、
  声、韵、调联合 **0.838**；结论在 128 档上复现且 CI 收紧 15–20%。
* **Thinking on 系统性有害**（−16 ~ −44 个百分点），主条件须用 thinking off。
* 噪声（SNR 20 dB）主要损伤声母/联合（−12 ~ −13.5pp），声调鲁棒。
* 自由文本仅对“只输出韵母”的任务有害（ü 丢失，−7 ~ −25pp），其余任务与 JSON 无显著差异。
* **检测任务不能直接让模型判一致**：模型 FPR 0.76–0.77（过度判不一致）；改用
  “同一音频的自由识别结果与声称 target 程序化比较”的**派生检测**后，Error-F1 从
  0.72 提升到 **0.91**、FPR 降到 **0.17–0.19**，且零额外调用。

## 1. 方法

### 1.1 数据与子集

* 数据集 `HaitongSUN/prosodic-abx`（commit `d6078066…`），6160 条 16 kHz 单声道 WAV，
  4 个 TTS voice × 1540 条/声调，378 个规范 target。
* 两档最小化子集：**96 target × 4 voice × 4 tone = 1,536 条**；**128 target × 16 = 2,048 条**
  （按韵母分层轮转抽取，覆盖 36 个韵母）；最小对 7 类各 48 / 64 实例，
  每实例两侧 × 自由/正确信息/错误信息 3 个子实验。
* 检测实验使用同一音频的 6 个任务（1 一致 + 2 改声母 + 2 改韵母 + 1 改声调），
  只在主条件运行。

### 1.2 生成条件

| 条件 | thinking | 音频 | 输出 |
|---|---|---|---|
| **C1（主条件）** | off | clean | JSON |
| C2 | on | clean | JSON |
| C3 | off | SNR 20 dB 白噪声 | JSON |
| C4 | off | clean | 自由文本 |

固定参数：`temperature=0, top_p=1, max_tokens=256`；模型 `qwen3.8-omni-flash`；
prompt `p3.1`（零样本，v3 结构）。

### 1.3 统计

* Bootstrap 95% CI：2000 次，audio 聚类、voice 分层；
* 条件对比：同一 item 配对，McNemar 精确检验 + 配对 bootstrap 差值 CI，Holm 校正；
* 主结果为 C1 条件下各实验指标；消融为 C2/C3/C4 vs C1。

### 1.4 派生检测（本次新增结论）

对每条检测样本，用同一音频的 exp3 自由识别预测（initial/final/tone）与“声称的
target + target_tone”比较：

```
derived_consistent = pred_initial == claim_initial
                 and pred_final  == claim_final
                 and pred_tone   == claim_tone
```

该判断不需要额外 API 调用，命令：`pinyin-eval derive-detect --run-id <id>`。

## 2. 结果

### 2.1 主结果（C1 主条件，两档对比）

| 实验/指标 | mini3-v1（1,536 音频） | mini128-v1（2,048 音频） |
|---|---:|---:|
| 实验一：正确信息 Tone | 0.909 [0.894, 0.923] | **0.914 [0.901, 0.925]** |
| 实验二：错误信息 Tone | 0.916 [0.903, 0.928] | **0.919 [0.908, 0.929]** |
| 实验三：Initial | 0.952 [0.941, 0.962] | **0.962 [0.954, 0.970]** |
| 实验三：Final | 0.938 [0.925, 0.950] | **0.949 [0.938, 0.958]** |
| 实验三：Tone | 0.910 [0.896, 0.923] | **0.912 [0.900, 0.923]** |
| 实验三：Joint | 0.816 [0.797, 0.835] | **0.838 [0.822, 0.854]** |
| 检测（模型直接判定）Error-F1 / FPR | 0.720 / 0.764 | 0.720 / 0.771 |
| **检测（派生检测）Error-F1 / FPR** | **0.912 / 0.184** | **0.922 / 0.162** |

> 同子集外部声学基线（SDHY P0-v2，适配自由识别，2,048 音频）：
> initial 0.600 / final 0.746 / tone 0.699 / joint 0.299；Qwen 对应
> 0.962 / 0.949 / 0.912 / 0.838。逐弱对对比见 `docs/baseline-sdhy.md` §5。

* 实验四（声母最小对）子实验：0.969 / 0.964 / 0.977（free / 正确信息 / 错误信息）；
* 实验五（韵母）：0.955 / 0.986 / 0.915；
* 实验六（声调）：0.945 / 0.945 / 0.930（n=128/子实验，CI 较宽）。

### 2.2 消融对比（mini128-v1，2,048 音频）

| 对比 | 代表指标 | 差值 [95% CI] | 结论 |
|---|---|---|---|
| C2 thinking on vs C1 | exp1 tone | **−20.0pp** [−21.8, −18.2] | thinking 有害 |
| | exp3 joint | **−19.1pp** [−21.0, −17.1] | |
| | exp6 错误信息 tone | **−43.8pp** [−52.8, −35.5] | |
| C3 噪声 vs C1 | exp3 initial | **−13.5pp** [−15.1, −11.9] | 声母/联合受损 |
| | exp3 joint | **−12.5pp** [−14.5, −10.5] | |
| | exp1/exp2 tone | 0.0pp（ns） | 声调鲁棒 |
| C4 文本 vs C1 | exp3 initial/tone | ≈0（ns） | 基本无差 |
| | exp5 free final | **−7.4pp** [−10.4, −4.6] | ü 丢失 |
| | exp5 错误信息 final | **−25.0pp** [−29.6, −20.4] | |

96 档（mini3-v1）的方向完全一致（thinking −17 ~ −42pp、噪声 initial −13.1pp、
文本 exp5 −17 ~ −24pp），两档互为复现。

## 3. 成本

| 项 | 调用数 | 实际成本 |
|---|---:|---:|
| mini3-v1（96 target） | 59,968 | ¥12.37 |
| 检测 prompt 偏置探针（3 变体） | 9,216 | ¥1.45 |
| mini128-v1（128 target） | 79,680 | ¥16.78 |
| **合计** | **148,864** | **¥30.60 / ¥70（43.7%）** |

* 两档均受硬预算 guard 约束（本 run 上限 ¥55，总预算 ¥70），实际远低于最坏估算
  （mini128 worst-case estimate ¥66.9，实际 ¥16.8）；
* 实测单价：think-off ≈ ¥0.00011–0.00016/次，think-on ≈ ¥0.0005–0.0007/次（随任务长度）。

## 4. 结论与建议

1. **最小化实验可以支撑趋势级结论**：主指标 CI ±1–2pp；两档结果一致，说明子集偏差可控。
2. **完整实验设计的两处修正建议**（已在 `docs/decisions.md` 记录）：
   * 主条件保持 **think off**，删掉/降权 thinking-on 条件（本任务上系统性有害）；
   * 检测任务改为 **派生检测**（自由识别 + 程序比较），模型直接判一致只作辅助指标。
3. **若做完整实验**：预期 ~21.3 万次调用（预期 ¥143 / 最坏 ¥178），需另立预算；
   建议沿用本报告的 prompt、条件与分析口径，并给 thinking-on 提高 `max_tokens`（512）
   以避免极少数空回复（0.0025%）。
4. 剩余 ¥39.4 预算的合理用途：补跑更大样本的主条件（例如 128 档的 C1 扩展），
   或对文本输出的 ü 丢失做专项 prompt 修正（很小成本）。

## 5. 局限

* 子集为分层抽样，只支持聚合/趋势结论，不支持逐音素、逐最小对类型的精细推断；
* 检测派生规则依赖自由识别质量（本实验 joint 0.83，派生检测 F1 0.91）；
* 实验四/六的最小对样本较少（每子实验 n=384/128），条件间差异需谨慎解读；
* 2 条 thinking-on 空回复（`max_tokens=256` 截断），占 0.0025%；
* 本机无法写入 Obsidian Vault（沙箱限制），文档以 repo 为准。

## 6. 复现

```bash
# 96 档
.venv/bin/pinyin-eval --config configs/minimal.yaml build-tasks --minimal
.venv/bin/pinyin-eval --config configs/minimal.yaml generate --tasks all \
  --profile minimal --conditions minimal --run-id mini3-v1 --tag minimal
.venv/bin/pinyin-eval --config configs/minimal.yaml score  --run-id mini3-v1
.venv/bin/pinyin-eval --config configs/minimal.yaml report --run-id mini3-v1
.venv/bin/pinyin-eval --config configs/minimal.yaml derive-detect --run-id mini3-v1

# 128 档
.venv/bin/pinyin-eval --config configs/minimal-128.yaml build-tasks --minimal
.venv/bin/pinyin-eval --config configs/minimal-128.yaml generate --tasks all \
  --profile minimal --conditions minimal --run-id mini128-v1 --tag minimal
.venv/bin/pinyin-eval --config configs/minimal-128.yaml score  --run-id mini128-v1
.venv/bin/pinyin-eval --config configs/minimal-128.yaml report --run-id mini128-v1
.venv/bin/pinyin-eval --config configs/minimal-128.yaml derive-detect --run-id mini128-v1
```

明细见：[docs/results-minimal-v1.md](results-minimal-v1.md)、
[docs/results-minimal-128.md](results-minimal-128.md)；完整报告与工件在
`results/runs/mini3-v1/`、`results/runs/mini128-v1/`。
