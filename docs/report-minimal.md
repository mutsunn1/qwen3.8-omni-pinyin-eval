# Qwen3.8-Omni 普通话音素识别最小化实验报告

> 2026-10-30 · 模型 `qwen3.8-omni-flash` · 数据 `HaitongSUN/prosodic-abx`
> 设计：[最小化实验设计](experiment-design-minimal.md) · 两档运行：`mini3-v1`（96 target / 1,536 音频）、`mini128-v1`（128 target / 2,048 音频）
> 详细版：[minimal-experiment-report.md](minimal-experiment-report.md) · 基线：[SDHY](baseline-sdhy.md)、[讯飞 ISE](baseline-ise.md)

## 一句话结论

在 2,048 条分层单音节 TTS 音频上，Qwen3.8-Omni（think off / clean / JSON）的
声母/韵母/声调准确率为 **0.962 / 0.949 / 0.912**，声韵调联合 **0.838**；
**thinking 模式系统性有害**（−20pp 左右），**检测任务必须用“自由识别→程序比较”的派生口径**
（Error-F1 0.922 / FPR 0.162），模型直接判一致的 FPR 高达 0.77。

## 1. 设置

| 项 | 内容 |
|---|---|
| 数据 | 378 个规范 target，按韵母分层抽 128 个 → 128×4 voice×4 tone = **2,048 条音频**；7 类最小对各 64 实例 |
| 条件 | C1 主条件（think off / clean / JSON）+ C2 think on + C3 SNR 20 dB 噪声 + C4 自由文本；检测实验只在 C1 |
| 调用 | 79,680 次（C1 29,088；C2–C4 各 16,864）；解析成功 99.997% |
| 模型参数 | temperature=0, top_p=1, max_tokens=256；prompt v3.1，零样本 |
| 统计 | Audio 聚类、voice 分层 bootstrap（2000 次）；McNemar + Holm |

## 2. 主结果（C1）

| 实验 / 指标 | 准确率 | 95% CI |
|---|---:|---|
| 实验一（正确信息，Tone） | **0.914** | [0.901, 0.925] |
| 实验二（错误信息，Tone） | **0.919** | [0.908, 0.929] |
| 实验三：Initial | **0.962** | [0.954, 0.970] |
| 实验三：Final | **0.949** | [0.938, 0.958] |
| 实验三：Tone | **0.912** | [0.900, 0.923] |
| 实验三：Joint | **0.838** | [0.822, 0.854] |

**最小对（free 条件，n=96–128/类）**：平翘舌 0.992、送气 0.969、n/l 0.945、
前后鼻 0.961、i/ü 0.984、ian/üan 0.906、声调 0.945。

**检测（8,160 对，C1）**：

| 判定方式 | Error-F1 | FPR |
|---|---:|---:|
| 模型直接判一致（设计原方案） | 0.720 | 0.771 |
| **自由识别 → 程序比较（派生检测）** | **0.922** | **0.162** |

## 3. 消融（C2/C3/C4 vs C1，配对检验）

| 对比 | 代表指标 | 差值 [95% CI] |
|---|---|---:|
| think on vs off | exp1 tone | **−20.0pp** [−21.8, −18.2] |
| | exp3 joint | **−19.1pp** [−21.0, −17.1] |
| | exp6 错误信息 tone | **−43.8pp** [−52.8, −35.5] |
| SNR20 噪声 vs clean | exp3 initial | **−13.5pp** [−15.1, −11.9] |
| | exp3 joint | **−12.5pp** [−14.5, −10.5] |
| | exp1/2 tone | 0（ns） |
| 自由文本 vs JSON | exp3 initial/tone | ≈0（ns） |
| | exp5 free final | **−7.4pp** |
| | exp5 错误信息 final | **−25.0pp**（ü 丢失） |

## 4. 外部基线（同一批 2,048 音频）

| 指标 | Qwen3.8-Omni | SDHY P0-v2（开源声学） | 讯飞 ISE（商用，候选打分） |
|---|---:|---:|---:|
| Initial | **0.962** | 0.600 | — |
| Final | **0.949** | 0.746 | — |
| Tone | 0.912 | 0.699 | **0.922** |
| Joint | **0.838** | 0.299 | — |
| 检测 Error-F1 / FPR | 0.922 / 0.162 | 0.731 / 0.701 | 0.914 / 0.133 |
| 7 类最小对（判别） | 0.906–0.992 | 0.422–0.917 | 0.906–1.000 |

> 口径提醒：ISE 是“2–4 候选强制选择”，比自由识别容易，只作相对比较；
> SDHY 是自由识别适配口径，可直接对比。ISE 在送气、n/l、ian/üan、声调最小对、
> en/eng、o/uo、2/3 声上强于 Qwen；Qwen 在 in/ing、üe/ie、ei/ie、d/t、n/l、前后鼻上更稳。

## 5. 成本

| 项 | 调用/交互 | 实际花费 |
|---|---:|---:|
| mini3-v1（96 档） | 59,968 | ¥12.37 |
| 96 档检测 prompt 探针 | 9,216 | ¥1.45 |
| mini128-v1（128 档） | 79,680 | ¥16.78 |
| 讯飞 ISE 基线 | 5,843 次交互（额度 10,000） | 已有余额 |
| **Qwen API 合计** | **148,864** | **¥30.60 / ¥70（43.7%）** |

## 6. 结论与建议

1. **主条件用 think off**：thinking on 在本任务全面有害，完整实验应删除/降权该条件。
2. **检测改用派生检测**：自由识别结果与声称 target 程序化比较，零额外调用，
   Error-F1 从 0.72 提到 0.92。
3. **数据增强/专项**：噪声主要伤声母与联合识别（−12 ~ −13.5pp），声调鲁棒；
   文本输出只对 exp5（单韵母输出）有害（ü 丢失），JSON 模式应作为主格式。
4. **仍需补强**：Qwen 剩余弱项为 **m/n（0.858）、o/uo（0.804）、2/3 声（0.825）、
   en/eng（0.883，低于 SDHY 基线 0.938）**；ISE 弱项为 in/ing、üe/ie 及 B 声线声调。
5. 剩余 ¥39.4 API 预算与 4,157 次 ISE 交互，可做上述弱项专项或扩档（128→更大样本）。

## 7. 复现

```bash
# Qwen 主实验（128 档）
uv run pinyin-eval --config configs/minimal-128.yaml build-tasks --minimal
uv run pinyin-eval --config configs/minimal-128.yaml generate --tasks all \
  --profile minimal --conditions minimal --run-id mini128-v1 --tag minimal
uv run pinyin-eval --config configs/minimal-128.yaml score  --run-id mini128-v1
uv run pinyin-eval --config configs/minimal-128.yaml report --run-id mini128-v1
uv run pinyin-eval --config configs/minimal-128.yaml derive-detect --run-id mini128-v1

# 基线
.venv/bin/python scripts/eval_sdhy_baseline.py          # SDHY（third_party 环境）
set -a && . ./.env && set +a
.venv/bin/python scripts/eval_xfyun_ise.py --reuse-only # 讯飞 ISE（已缓存 5,843 次）
```

完整工件：`results/runs/mini3-v1/`、`results/runs/mini128-v1/`（report.md 等 23 个文件）；
基线：`results/baseline-sdhy-min128.json`、`results/ise-baseline.json`、`results/ise-vs-qwen.json`。
