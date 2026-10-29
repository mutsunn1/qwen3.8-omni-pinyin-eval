# Qwen3.8-Omni 音素识别实验：最小化实验设计（¥70 预算，v3）

> 版本：v2（2026-10-29），对齐设计文档 v3（2026-10-28 20:37 快照，
> `docs/experiment-design.md`，sha256 `acf2015a…`）。
> 上游完整设计：[完整实验设计 v3](experiment-design.md)
> 关联实现：`configs/minimal.yaml`、`pinyin_eval.pricing`、`pinyin_eval.runner`（成本预算）、
> `build-tasks --minimal`；`report.py` 输出 `usage.csv`/成本节。
> 预算口径：**CNY 70 元**（阿里云百炼 `qwen3.8-omni-flash`，华北2/北京价格）。

本文件定义一套可在 **¥70 以内**完成的最小可行实验：保留 v3 设计的全部六组实验与
RQ1–RQ4，但把音频、条件、最小对数量压缩到"能给出趋势性结论"的最低限度。结论定位为
**pilot / 可行性级**：验证模型能否完成任务、估计效应方向与量级、为完整实验确定样本量
与 prompt 修正方向；不做细粒度（单音素/单最小对类型）推断。

> v3 关键结构（与 v2 不同）：取消 A/B 任务；实验四/五/六各拆为**自由检测 / 正确信息 /
> 错误信息**三个子实验，分别只输出 initial / final / tone；**检测实验不参与消融**
> （只在主条件运行一次）。

---

## 1. 成本模型（先算钱，再定样本量）

### 1.1 价格快照

来源：[百炼模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)（2026-10-28），
音频 token 规则见 [Qwen-Omni 文档·计费与限流](https://help.aliyun.com/zh/model-studio/qwen-omni)。

| 项目 | 单价 |
|---|---:|
| 输入 token（文本/音频统一） | 0.8 元 / 百万 token |
| 缓存命中输入 token | 0.1 元 / 百万 token |
| 输出 token（含思考/reasoning token） | 2.7 元 / 百万 token |

token 计算：

* 音频输入：`ceil(音频时长秒) × 7`（不足 1 秒按 1 秒）；本数据集音频 0.215–1.334 秒 →
  **每条音频 7–14 token**；
* 文本输入：中文 prompt 约 100–210 字符（system 63 + user 37–144）；
* 输出：JSON/短文本 10–30 token；思考模式输出受 `max_output_tokens=256` 硬上限约束。

单次调用成本：

```
cost = (prompt_tokens - cached_tokens) × 0.8/1e6
     + cached_tokens × 0.1/1e6
     + completion_tokens × 2.7/1e6
```

### 1.2 三重估算

| 口径 | prompt token 假设 | 思考输出假设 | 用途 |
|---|---|---|---|
| expected | 0.7 token/字符 | 300 token（被 256 截断） | 预期花费 |
| conservative | 1.0 token/字符 | 800 token（被 256 截断） | 预算规划 |
| worst case | 1.0 token/字符 | 每次调用都打满 `max_tokens=256` | 预算保证上界 |

`max_output_tokens=256` 是成本安全阀：单次输出成本 ≤ `256 × 2.7/1e6 = ¥0.000691`；
加上输入后单次调用成本 ≤ 约 **¥0.00087**。

### 1.3 预算守卫

runner 支持 `generation.budget_cny`（本设计固定 **70**）与 `--max-cost`：累计实际成本
达到预算后停止发起新请求，已完成请求照常落盘，manifest 状态记 `budget_exceeded`，
可用同一 run-id 续跑。超支最多是在途 `concurrency` 条请求，不会显著越界。

> 目标：**预期 ~¥19、保守 ~¥21、理论最坏 ~¥50**；叠加 pilot 与 5% 重试余量后仍在
> ¥53 以内，距 ¥70 至少留 ¥17 缓冲。

---

## 2. 相对完整设计 v3 的削减

| 维度 | 完整设计 v3 | 最小化设计 | 压缩 |
|---|---|---:|---:|
| 核心/检测音频 | 6,160 条 | **1,536 条**（96 target × 4 voice × 4 tone） | 75% ↓ |
| 生成条件 | 96 个（24 + 72；检测实验不消融） | **4 个**单因素条件 | 96% ↓ |
| 实验四~六 | 全量最小对实例 × 3 子实验 | **7 类 × 48 实例** × 3 子实验 | 98% ↓ |
| API 请求 | 212,576 | **59,968** | 72% ↓ |
| 预计成本（expected） | ¥142.8 | **¥18.9** | 87% ↓ |
| 预计成本（conservative） | — | **¥21.3** | — |
| 最坏成本（max_tokens 打满） | ¥177.6 | **¥50.1** | 72% ↓ |

削减原则：
1. **保留全部 6 组实验**：实验一/二（tone-only）、实验三（自由识别）、检测实验
   （target + target_tone → consistent）、实验四/五/六（各自 3 个子实验）；
2. **保留 4 个关键单因素对比**（相对主条件 C1）：思考开关、干净/白噪声、JSON/自由文本；
   不做全因子，交互作用留给完整实验；
3. **最小对按类型分层抽样**，7 类都有实例；每类实例两侧都跑 3 个子实验；
4. **检测实验只在 C1 运行**（v3：不参与消融），其余条件仅跑 core+pairs 子集。

---

## 3. 实验设计

### 3.1 数据子集

* **核心/检测音频（1,536 条）**：从 378 个规范 target 中按**韵母分层轮转**抽 96 个 target
  （保证 36 个韵母至少各出现 1 次），每个 target 取 4 voice × 4 tone 共 16 条音频
  （合并拼写 target 优先取 `raw_target == target`）；4 voice × 4 tone 每格 96 条。
* **最小对音频**：7 个 pair_type 各抽 48 个实例（`<pair, voice, tone>`，按 voice 轮转
  保证 4 个 voice 覆盖），共 336 实例 / 672 条音频侧；与核心子集独立抽样。
* 抽样由全局 seed 决定，写入 `minimal_items.txt` / `minimal_summary.json`，完全可复现。

### 3.2 任务清单（C1 条件复用；C2–C4 不含检测）

| 来源 | 任务数 | task_type / 说明 |
|---|---:|---|
| 实验三 free | 1,536 | free（initial/final/tone） |
| 实验一 correct_target | 1,536 | tone |
| 实验二 wrong_target ×5 | 7,616 | tone（er/weng 少 2 变体） |
| 检测实验 | 9,152 | detect（1 一致 + 5 不一致） |
| exp4（3 类×48 实例×2 侧×3 子实验） | 864 | initial |
| exp5（3 类×48×2×3） | 864 | final |
| exp6（1 类×48×2×3） | 288 | tone |
| **C1 合计** | **21,856** | 检测实验仅 C1 运行 |
| **C2–C4 合计（core+pairs，不含检测）** | **12,704** | 每条件 |

去重后：C1 21,856 + 3 × 12,704 = **59,968 requests**（task 单元格 59,968 → 唯一请求
59,968；本设计子集下无可去重重复请求）。

### 3.3 生成条件（4 个）

| 编号 | thinking | 音频 | 输出 | 作用 |
|---|---|---|---|---|
| **C1 main** | off | clean | JSON | 主结果 + 唯一检测实验运行条件 |
| C2 thinking | **on** | clean | JSON | RQ4：思考开关效应（不含检测） |
| C3 noise | off | **SNR 20 dB 白噪声** | JSON | RQ4：音频质量效应（不含检测） |
| C4 text | off | clean | **自由文本** | RQ4：输出形式效应（不含检测） |

固定参数：`temperature=0, top_p=1, max_tokens=256`，模型 `qwen3.8-omni-flash`；
thinking 开 = 默认力度，关 = `reasoning_effort="none"`。

### 3.4 请求数与成本表（`estimate` 实测）

| 条件 | 请求数 | expected | conservative | worst case |
|---|---:|---:|---:|---:|
| C1 think off / clean / JSON（含检测） | 21,856 | ¥4.18 | ¥5.12 | ¥18.46 |
| C2 think on / clean / JSON | 12,704 | ¥10.13 | ¥10.66 | ¥10.66 |
| C3 think off / noise / JSON | 12,704 | ¥2.38 | ¥2.91 | ¥10.66 |
| C4 think off / clean / text | 12,704 | ¥2.15 | ¥2.58 | ¥10.33 |
| **合计** | **59,968** | **¥18.85** | **¥21.27** | **¥50.11** |

* 单请求理论最坏成本 ≈ `输入(≤约 230 token) × 0.8/1e6 + 256 × 2.7/1e6 ≈ ¥0.00087`；
  `59,968 × 0.00087 ≈ ¥52`（与表一致）。
* Pilot 156 请求 × 最坏 ¥0.00087 ≈ ¥0.14；5% 重试余量 → 总最坏上界 ≈ **¥52.8**。

### 3.5 可选档位（pilot 后按实测 token 决定）

| 档位 | n_targets / pairs-per-type | 请求数 | expected | conservative | worst case | 预算占用 |
|---|---|---:|---:|---:|---:|---:|
| 精简 | 64 / 32 | 39,872 | ¥12.53 | ¥14.14 | ¥33.32 | 48% |
| **默认（本设计）** | **96 / 48** | **59,968** | **¥18.85** | **¥21.27** | **¥50.11** | **72%** |
| 上限（谨慎） | 128 / 64 | 79,680 | ¥25.03 | ¥28.25 | ¥66.58 | 95% |

规则：pilot 实测单位成本 ≤ expected（约 ¥0.00031/call）且预算允许时可上调档位；
若实测接近 worst（¥0.00084/call），保持默认档或降级。`budget_cny=70` 兜底。

### 3.6 预计耗时（参考）

| 并发 | 预计耗时 |
|---:|---:|
| 4 | 20–40 小时 |
| 8 | 10–20 小时 |

（按单请求 4–8 秒、思考模式更慢估算；实际以账号限流为准。）

---

## 4. 执行流程（3 个阶段）

### Stage 0：环境与连通性（约 ¥0.01）

```bash
uv run pinyin-eval --config configs/minimal.yaml check
uv run pinyin-eval --config configs/minimal.yaml smoke
```

### Stage 1：Pilot 校准（156 次调用，≤¥0.2）

```bash
uv run pinyin-eval --config configs/minimal.yaml build-tasks --minimal
uv run pinyin-eval --config configs/minimal.yaml generate --tasks all --profile minimal \
  --conditions minimal --limit 12 --run-id pilot-mini --tag pilot
uv run pinyin-eval --config configs/minimal.yaml score --run-id pilot-mini
uv run pinyin-eval --config configs/minimal.yaml cost --run-id pilot-mini
```

`--limit 12` = 13 个 condition_id × 12 条（C1 的 free/correct/wrong/detect 4 个 +
C2–C4 各 free/correct/wrong 3 个 = 13；每条件确定性随机抽 12 条）。

验收（任一不满足先修 prompt/解析再进 Stage 2）：
- 各条件解析成功率 ≥95%；`parse_failures.jsonl` 无系统性错误；
- thinking-on 条件 `reasoning_text` 非空；
- `cost --run-id pilot-mini` 给出实际 token/成本：`实际成本 / 156 × 59,968` 外推全量；
- 外推成本 ≤¥40 → 可进默认档；≤¥28 → 可考虑上调档位；>¥45 → 先降档（§7）。

### Stage 2：最小化主实验（59,968 次调用）

```bash
uv run pinyin-eval --config configs/minimal.yaml generate --tasks all --profile minimal \
  --conditions minimal --run-id mini3-v1 --tag minimal
# 中断后原命令 + 原 run-id 续跑；预算 70 由 config 自动生效
uv run pinyin-eval --config configs/minimal.yaml score --run-id mini3-v1
uv run pinyin-eval --config configs/minimal.yaml report --run-id mini3-v1
uv run pinyin-eval --config configs/minimal.yaml cost --run-id mini3-v1
```

验收：
- manifest `status=completed`、`counts.n_calls=59968`（检测实验仅 C1 有请求）；
- `cost` 显示实际成本 ≤¥70（预期 ¥19–21）；
- `report.md` 含主结果/检测/三个子实验/消融（C2–C4）/成本与用量；解析失败率单列；
- 报告头部标注最小化子集口径。

---

## 5. 分析计划

### 5.1 指标与实验映射（v3）

| 实验 | 数据 | 指标 | 样本量 / CI 预期 |
|---|---|---|---|
| 实验一（tone-only，正确信息） | 1,536 音频 | Tone accuracy / Macro-F1 | 每 voice 384；总体 ≈±2.5% |
| 实验二（tone-only，错误信息） | 7,616 任务（1,536 音频） | Tone accuracy / Macro-F1（audio 聚类） | 聚类 1,536 |
| 实验三（自由识别） | 1,536 音频 | Initial/Final/Tone/Joint + Macro-F1 | 总体 ≈±2.5% |
| 检测实验（仅 C1） | main/final/tone 各 1,536 对，initial 1,504 对（er/weng 无改声母变体） | EP/ER/Error-F1/FPR/FNR | 每子集约 1,536 对，CI ≈±2.5% |
| 实验四（声母） | 48 实例/类 ×3 类 = 144 实例（288 侧） | 3 子实验的 Initial accuracy | 每子实验 n=288，CI ≈±6% |
| 实验五（韵母） | 144 实例（288 侧） | 3 子实验的 Final accuracy | 每子实验 n=288，CI ≈±6% |
| 实验六（声调） | 48 实例（96 侧） | 3 子实验的 Tone accuracy | 每子实验 n=96，CI ≈±10% |

检测实验子集规模由最小化检测集决定：main/final/tone 各 1,536 对，initial 1,504 对
（er/weng 无改声母变体，少 32 对）。

### 5.2 统计方案

* 95% bootstrap CI：2000 次，audio 聚类、voice 分层；
* **预注册 3 组单因素比较**（均与 C1 配对、同一 items）：
  C2 vs C1（thinking）、C3 vs C1（噪声）、C4 vs C1（输出形式）；
  McNemar 精确检验 + 配对 bootstrap 差值 CI；每个指标 family 内 Holm 校正；
* 实验四/五/六的 3 个子实验为描述性对比（free vs 正确信息 vs 错误信息），
  不做多重比较宣称；
* 基线：声调四分类 25%、平衡检测集 50%、initial/final 多数类基线（v3 无 A/B 基线）；
* 解析失败不计入指标分母，单独报告比例。

### 5.3 解释边界

* 总体识别/检测准确率 CI ≈ ±2.5 个百分点，可判断"是否显著高于随机/基线"与
  条件间中等以上差异；不能可靠判断 <3–5 个百分点的细微差异；
* 逐 voice（n=384）CI ≈ ±5%，只做描述；
* 实验六每子实验 n=96（≈±10%）、实验四/五 n=288（≈±6%），仅趋势级；
* 96 个 target 覆盖 36 韵母但每韵母样本少，**不做逐音素推断**；
* 结论表述为"在 1,536 条分层子集上"的影响，不外推到全量分布。

---

## 6. 可复现性与产物

| 文件 | 说明 |
|---|---|
| `data/manifests/tasks/minimal_items.txt` | C1 全部 21,856 个 item_id（含检测） |
| `data/manifests/tasks/minimal_no_detection_items.txt` | C2–C4 用的 12,704 个 item_id（不含检测） |
| `data/manifests/tasks/minimal_{core,detection,pair}_items.txt` | 分来源子集 |
| `data/manifests/tasks/minimal_summary.json` | 抽样 target/audio/实例/任务统计 |
| `results/runs/<run_id>/run_manifest.json` | pricing 快照、budget_cny、条件、计数、成本 |
| `results/runs/<run_id>/generations.jsonl` | 每次调用的 usage（成本核算依据） |
| `results/runs/<run_id>/{items,detection_rows}.parquet` | 解析后的分析数据 |
| `results/runs/<run_id>/{report.md,usage.csv,metrics.csv,detection.csv,ablation.csv,...}` | 报告与成本/指标表 |
| `pinyin-eval cost --run-id <id>` | 实际 token/成本按条件汇总 |

复现命令见 §4；所有抽样由 `configs/minimal.yaml` + `project.seed` 决定。

---

## 7. 风险、降级与升级

| 风险 | 表现 | 应对 |
|---|---|---|
| 实际 token 超预期 | pilot 外推成本 >¥45 | ① 去掉 C2（thinking）条件（请求→47,264，成本降约 55%）；② 或 `minimal.n_targets: 64`、`pair_instances_per_type: 32` 降档 |
| JSON 解析失败率高 | pilot parse <95% | 调 prompt/解析规则后重跑 pilot（pilot <¥0.2） |
| 限流/超时 | 大量 api_error | 降低 `model.concurrency`、设 `model.qps`；重试会计入成本，先 pilot 测速 |
| 预算耗尽 | `budget_exceeded` | 同 run-id 续跑前提高 `budget_cny`；报告标注不完整条件 |
| 子集代表性偏差 | 每韵母样本少 | 只做聚合结论；完整实验全量验证 |

**升级路径**：本设计通过后，可用 §3.5 档位扩展（仍 ≤¥70 worst-case），或回到完整
v3 设计（约 21.3 万请求、预期 ¥143 / 最坏 ¥178），并对比 minimal 与 full 的一致性。

---

## 8. 与完整设计的差异汇总（v3）

| 设计要素 | 完整设计 v3 | 最小化设计 |
|---|---|---|
| 实验四~六 | 2440 对全量 × 3 子实验 | 7 类 × 48 实例 × 3 子实验 |
| 检测实验 | 全量、单条件（不消融） | 1,536 音频，仅 C1 |
| 消融 | 96 个条件（24 + 72） | 3 个单因素对比 |
| 统计 | 全量推断 + 多重比较 | 3 组预注册比较 + Holm |
| 成本 | 预期 ¥142.8 / 最坏 ¥177.6 | **预期 ¥18.9 / 最坏 ¥50.1，硬预算 70** |

---

## 9. 一页验收清单

- [ ] `build-tasks --minimal` 产出 6 个 minimal 文件，`minimal_summary.json` 中
      `n_all_items=21856`、`n_no_detection_items=12704`、`n_audio=1536`
- [ ] `estimate --profile minimal --conditions minimal` 输出 `n_requests=59968`、
      `expected_cost_cny≈18.85`、`worst_case_cost_cny≈50.11`、`within_worst_case_budget=true`
- [ ] `smoke` 成功
- [ ] pilot 156 次调用，解析成功率 ≥95%，外推成本 ≤¥40
- [ ] 主实验 `status=completed`、`n_calls=59968`、成本 ≤¥70，检测实验仅 C1 有记录
- [ ] `report.md` 含 3 个子实验表、检测节、消融节（仅 core/pairs）、成本与用量节
- [ ] 归档 run_manifest + generations + report + cost 汇总
