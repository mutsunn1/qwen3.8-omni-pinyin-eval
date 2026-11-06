# SDHY P0-v2 原生 target-aware 模式：补跑结果

> 2026-11-06 · 目的：纠正此前用“自由识别适配”评价 SDHY 的口径。
> 同批 4,096 条检测 claim（2,048 音频 × 1 一致 + 1 不一致），与 Qwen/ISE 检测基线同一集合。
> 脚本：[scripts/eval_sdhy_native.py](../scripts/eval_sdhy_native.py) ·
> 原始：`results/baseline-sdhy-native.json` + `.claims.csv`

## 0. 为什么要补跑

此前 `docs/baseline-sdhy.md` 的 0.600 / 0.746 / 0.699 / **0.299** 是**把 SDHY 的
冻结编码器 + M1/X2 头当作自由识别器**（initial 取 CTC 首音素、final 取最佳 non-blank、
tone 用 F0 规则重写）得到的——这是适配口径，**不是** SDHY 产品的原生用法。

SDHY P0-v2 的原生链路是 **已知 target 的评测**：

```
audio + target(initial, final, tone)
  → Wav2Vec2 声学证据（冻结 XLSR-zh）
  → X2 forced alignment（声母/韵母 span）
  → M1.6 质心 margin（声母/韵母）
  → F0 规则（声调）
  → RuleCalibrator：每维 GOOD / NEEDS_WORK / UNCERTAIN
```

这套链路**不能**做自由识别，但可以直接做“目标一致性判定”——正好对应本实验的
检测任务。因此用同一批检测 claim 重新跑，得到可比的 native-mode 数字。

## 1. 运行设置

| 项 | 值 |
|---|---|
| 模型 | `third_party/` 下 SDHY P0-v2（XLSR-zh 权重 sha256 `de031fd4…`，M1/X2 头随仓库） |
| 计算模式 | **cpu-fp32**（本机无 CUDA；官方标定是 cuda-fp16，不等价） |
| 输入 | 2,048 音频 × 2 claim（neg=声称正确、pos=声称错误） = 4,096 claims |
| 判定 | `NEEDS_WORK` 判为“不一致”；全部 `GOOD` 判为“一致”；其余 `UNCERTAIN` |
| 连续分数 | `initial/final = -margin`；`tone = +3 needs_work / -3 good / 0 uncertain`；取 max |
| 阈值 | 在 hash 分半的 calib 半区学 F1 最优阈值，eval 半区报告 |
| 运行时间 | **124.5 秒**（CPU，含 1.28GB 权重校验与加载） |

## 2. 检测结果（同一 4,096 claims）

| 方法 | n | Error-F1 | Precision | Recall | FPR | FNR | Acc | AUC（连续；二值≈平衡准确率） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 直接判一致（given target） | 4,096 | 0.720 | 0.564 | 0.996 | 0.771 | 0.004 | 0.612 | 0.612 |
| **SDHY 原生 verdict 规则**（un­certain→一致） | 4,096 | **0.737** | **0.769** | 0.708 | **0.213** | 0.292 | **0.747** | — |
| SDHY 原生 margin 阈值（eval half） | 2,058 | 0.739 | 0.720 | 0.759 | 0.289 | 0.241 | 0.735 | **0.797** |
| SDHY 自由识别适配（旧口径，参考） | 4,096 | 0.731 | — | — | 0.701 | — | — | — |
| Qwen 派生检测（自由识别+比较） | 4,096 | **0.922** | 0.860 | 0.993 | **0.162** | 0.007 | **0.916** | 0.916 |
| 讯飞 ISE（阈值，512 claim 子集，eval half） | 256 | 0.914 | — | — | 0.133 | 0.047 | 0.910 | 0.909 |

原生 verdict 混淆矩阵（4,096）：

|  | 预测一致 | 预测不一致 |
|---|---:|---:|
| gold 一致（neg） | 1,612 (TN) | 436 (FP) |
| gold 不一致（pos） | 599 (FN) | 1,449 (TP) |

**保守代价**：verdict 规则弃权率 **73.6%**（uncertain），所以 FNR 较高（29.2%）。
只看“有明确结论”的 1,081 条（覆盖 26.4%）：

| n | Error-F1 | Precision | Recall | FPR | FNR | Acc |
|---:|---:|---:|---:|---:|---:|---:|
| 1,081 | **0.833** | 0.734 | 0.964 | 0.380 | 0.036 | 0.799 |

## 3. 原生模式的分维行为

### 3.1 正确目标（2,048 neg claims）——不误伤的能力

| 维度 | GOOD（接受） | NEEDS_WORK（误报） | UNCERTAIN |
|---|---:|---:|---:|
| initial | 61.6% | 6.6% | 31.7% |
| final | 56.7% | 8.8% | 34.4% |
| tone | 43.5% | 8.3% | 48.2% |

### 3.2 错误目标（2,048 pos claims）——定位/拒绝能力

| 错误类型 | n | 目标维度 NEEDS_WORK | 任一维 NEEDS_WORK |
|---|---:|---:|---:|
| initial 错 | 807 | **74.2%** | 78.2% |
| final 错 | 836 | **71.2%** | 75.4% |
| tone 错 | 405 | **36.8%** | 46.4% |

结论：原生模式的 **initial/final 定位/拒绝能力尚可**（~71–74%），**tone 规则弱**
（37%），且整体保守（大量 UNCERTAIN）。这与 SDHY 产品定位一致：发音只出“建议”，
不是高召回的错误检测器；其 tone 是 F0 规则而非学习模型。

## 4. 修正后的结论

1. **不能用 0.299 判断 XLSR/GOP 或 SDHY 模型“不行”**：0.299 是自由识别适配口径，
   而 SDHY 压根不是自由识别系统。
2. **在它原生的 target-aware 模式下**：同 4,096 claims 的给定目标一致性判定
   F1 **0.737**（略高于 Qwen 直接判定的 0.720），FPR **0.213**（远低于 0.771），
   但 FNR 0.292（保守、弃权多）；AUC 0.797。
3. **与最优口径仍有差距**：Qwen 派生检测（自由识别 + 程序比较）F1 0.922 / FPR 0.162，
   ISE 阈值法 0.914 / 0.133；原生 SDHY 的价值在于**可解释的声学证据 + 分维定位 +
   本地部署**，而不是与 Qwen 的全局识别能力硬比。
4. **公平比较方式**：Qwen/SDHY/ISE 在“给定目标一致性判定”上应统一用各自的
   target-known 模式——Qwen 直接判定 0.720、SDHY native 0.737、ISE 0.914；
   而自由识别（Qwen 派生 0.922）是另一条更强的 Qwen 专属路径。

## 5. 复现

```bash
# 需要 third_party/ 基线环境（torch/transformers）
PYTHONPATH=src third_party/.venv-baseline/bin/python scripts/eval_sdhy_native.py
# 只重算指标（复用 claims CSV，不再加载模型）
PYTHONPATH=src third_party/.venv-baseline/bin/python scripts/eval_sdhy_native.py --reuse-only
```

## 6. 局限

* CPU fp32 与官方 cuda-fp16 不是同一份证据；GPU 上数值/判定会变。
* 4,096 claims 仅覆盖 mini128 子集（2,048 音频）；未跑全量 6,160。
* RuleCalibrator 是**保守规则**（未用真人数据校准）；弃权率 73.6% 是产品设计选择，
  不是模型上限。
* 连续分数的 tone 惩罚权重（±3）是本文的人为先验，仅用于阈值口径；verdict 口径
  不依赖该权重。
* ISE 的 0.914 来自 512 claim 子集 + 候选打分，与 4,096 native 集合不完全同集合，
  只作数量级参考。
