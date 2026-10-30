> 2026-10-30 更新：修复 `y/w` 与舌尖元音（zhi/chi/…）的输出规范化后重新评分
> （**未重新调用 API**）：final 与 joint 提升约 1pp，检测派生指标略升。主指标已按
> 最新值更新；消融表为规范化前提取的版本（方向与显著性不变，精确值以
> `results/runs/mini128-v1/report.md` / `comparisons.csv` 为准）。

# Qwen3.8-Omni 最小化实验 · 升级档结果（run mini128-v1）

> 运行：2026-10-30 14:32 → 16:48（2h16m，无中断）
> 设计：`configs/minimal-128.yaml`（`docs/experiment-design-minimal.md` §3.5 的 128/64 档）
> 模型：`qwen3.8-omni-flash`（华北2）；prompt `p3.1`；temp=0, top_p=1, max_tokens=256
> 数据：**2,048 条**分层音频（128 target × 4 voice × 4 tone）+ 7 类最小对各 64 实例
> 条件：C1 主条件（think off / clean / JSON）+ C2 think on + C3 SNR20 噪声 + C4 自由文本；
> 检测实验仅 C1

## 0. 运行与成本

| 项 | 值 |
|---|---|
| 计划 / 成功调用 | 79,680 / **79,680（100%）** |
| 失败 | 12 条瞬态 400（已 resume 重试成功）+ **2 条思考模式空回复**（0.0025%，计入解析失败） |
| 实际花费 | **¥16.78 / 本 run 上限 ¥55**（30.5%） |
| 累计总花费 | mini3-v1 ¥12.37 + 探针 ¥1.45 + 本 run ¥16.78 = **¥30.60 / ¥70（43.7%）** |
| 平均延迟 | 2.01 s（think off 1.2–1.5s；think on 3.5–5.1s） |
| 与 mini3-v1 对比 | 调用 79,680 vs 59,968；成本 16.78 vs 12.37（实测单价一致 ≈¥0.00021/次） |

## 1. 主结果（C1 主条件）

| 实验 | 指标 | mini3-v1（1,536 音频） | **mini128-v1（2,048 音频）** | CI 变化 |
|---|---|---:|---:|---|
| 实验一 | Tone accuracy | 0.909 [0.894, 0.923] | **0.914 [0.901, 0.925]** | ±1.2pp |
| 实验二 | Tone accuracy | 0.916 [0.903, 0.928] | **0.919 [0.908, 0.929]** | ±1.0pp |
| 实验三 | Initial | 0.952 [0.941, 0.962] | **0.962 [0.954, 0.970]** | ±0.8pp |
| 实验三 | Final | 0.938 [0.925, 0.950] | **0.949 [0.938, 0.958]** | ±1.0pp |
| 实验三 | Tone | 0.910 [0.896, 0.923] | **0.912 [0.900, 0.923]** | ±1.2pp |
| 实验三 | Joint | 0.816 [0.797, 0.835] | **0.838 [0.822, 0.854]** | ±1.6pp |
| 检测（模型判定） | Error-F1 / FPR | 0.720 / 0.764 | **0.720 [0.715, 0.724] / 0.771 [0.753, 0.789]** | 稳定 |
| 检测（派生检测） | Error-F1 / FPR | 0.912 / 0.184 | **0.922 / 0.162** | 稳定/略好 |

> 结论在更大样本上完全复现，CI 收紧约 15–20%（如实验三 Joint ±2.0pp → ±1.6pp）。
> 模型直接判一致的偏置（FPR≈0.77）与派生检测的优势（F1 0.922、FPR 0.162）都在
> 2,048 音频上得到确认。注：`y/w` 拆分与舌尖元音规范化修复后重新评分，final/joint 有 +1pp 提升。

## 2. 消融条件对比（C2/C3/C4 vs C1，2,048 音频）

### 2.1 Thinking on vs off

| family | 指标 | off | on | 差值 [95% CI] |
|---|---|---:|---:|---:|
| exp1 | tone | 0.914 | 0.713 | **−20.0pp** [−21.8, −18.2] |
| exp2 | tone | 0.919 | 0.745 | **−17.3pp** [−18.6, −16.0] |
| exp3 | joint | 0.829 | 0.639 | **−19.1pp** [−21.0, −17.1] |
| exp3 | tone | 0.912 | 0.748 | **−16.3pp** [−18.0, −14.6] |
| exp4.wrong | initial | 0.977 | 0.872 | **−10.4pp** [−13.6, −7.5] |
| exp5.wrong | final | 0.915 | 0.730 | **−18.5pp** [−23.1, −13.9] |
| exp6.wrong | tone | 0.930 | 0.492 | **−43.8pp** [−52.8, −35.5] |

### 2.2 SNR 20 dB 噪声 vs clean

| family | 指标 | clean | noise | 差值 [95% CI] |
|---|---|---:|---:|---:|
| exp3 | initial | 0.959 | 0.824 | **−13.5pp** [−15.1, −11.9] |
| exp3 | joint | 0.830 | 0.705 | **−12.5pp** [−14.5, −10.5] |
| exp4.free | initial | 0.969 | 0.857 | **−11.2pp** [−14.6, −8.0] |
| exp5.free | final | 0.955 | 0.923 | −3.1pp [−5.6, −0.9] |
| exp1/exp2 | tone | 0.914/0.919 | 0.914/0.919 | 0（ns） |

### 2.3 自由文本 vs JSON

| family | 指标 | JSON | text | 差值 [95% CI] |
|---|---|---:|---:|---:|
| exp3 | initial | 0.959 | 0.958 | 0（ns） |
| exp3 | tone | 0.912 | 0.908 | 0（ns） |
| exp3 | joint | 0.830 | 0.790 | −4.0pp [−5.4, −2.5] |
| exp3 | final | 0.939 | 0.904 | **−3.5pp** [−4.8, −2.2] |
| exp5.free | final | 0.955 | 0.881 | **−7.4pp** [−10.4, −4.6] |
| exp5.wrong | final | 0.915 | 0.665 | **−25.0pp** [−29.6, −20.4] |

> 复现 mini3-v1 的方向：text 输出对“只输出韵母”的 exp5 损害最大（ü/拼写丢失），
> 对 JSON 有字段约束的任务几乎无差。

## 3. 结论

1. **结果稳定**：96 → 128 target 档位下所有主结论、消融方向一致，CI 收紧 15–20%。
2. **检测口径**：模型直接判一致 FPR≈0.77 在更大样本上仍是稳定偏置；**派生检测**
   （自由识别 + 程序比较）F1 0.918 / FPR 0.170，已可确定作为完整实验主口径。
3. **thinking on 必须关闭**（−16 ~ −44pp），噪声显著损伤声母/联合，文本输出仅对
   韵母单维度任务有害。
4. **总花费 ¥30.60 / ¥70（43.7%）**，剩余预算可支持：派生检测专项分析（¥0）或
   完整实验的一小部分。

## 4. 工件

- 完整报告：`results/runs/mini128-v1/report.md`（23 个工件：`usage.csv`、`comparisons.csv`、
  `metrics.csv`、`detection.csv`、`ablation.csv`、`parse_failures.jsonl` 等）
- 检测派生分析：`results/detection-probe-summary-mini128-v1.json`
  （脚本 `scripts/analyze_detection_probe.py mini128-v1`）
- 复现命令：

```bash
.venv/bin/pinyin-eval --config configs/minimal-128.yaml build-tasks --minimal
.venv/bin/pinyin-eval --config configs/minimal-128.yaml generate --tasks all \\
  --profile minimal --conditions minimal --run-id mini128-v1 --tag minimal
.venv/bin/pinyin-eval --config configs/minimal-128.yaml score  --run-id mini128-v1
.venv/bin/pinyin-eval --config configs/minimal-128.yaml report --run-id mini128-v1
.venv/bin/pinyin-eval --config configs/minimal-128.yaml cost   --run-id mini128-v1
PYTHONPATH=src .venv/bin/python scripts/analyze_detection_probe.py mini128-v1
```
