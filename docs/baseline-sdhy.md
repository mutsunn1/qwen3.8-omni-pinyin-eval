# 基线：SDHY speaking evaluator（P0-v2）在 prosodic-abx 子集上的结果

> 目的：把 `CHAO-AI-Lab/SDHY_Test_System` 的 `evaluater_speaking_0.1` 发音评测模型
> 拉下来，作为 Qwen3.8-Omni 实验的**声学基线**，在同一批 2,048 条音频上与
> `mini128-v1` 对比。
> 日期：2026-10-30 · 评测脚本：[scripts/eval_sdhy_baseline.py](../scripts/eval_sdhy_baseline.py)

## 1. 模型来源与下载校验

| 项 | 内容 |
|---|---|
| 仓库 | `CHAO-AI-Lab/SDHY_Test_System`（private，已用 `gh` 权限克隆） |
| 路径 | `evaluater_speaking_0.1/`（P0-v2 baseline，`models/p0-baseline.json`） |
| commit | `61b2d0524b6ae1593056c5c1c853e8fb57aa881e` |
| 编码器 | HF `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`（apache-2.0） |
| 权重文件 | `pytorch_model.bin` 1,276,296,151 B |
| 权重 sha256 | `de031fd4b29e0c0667e5346450fadfe1326c89936b888b59c4ede608db763ee4` ✅ 与 `p0-baseline.json` 的 revision 完全一致 |
| 随仓库发布的头 | `models/m1-ctc-head.npz`（M1 音素 CTC，61 类）、`models/x2-align-heads.npz`（X2 initial/final 对齐头）、`models/native-centroids.npz` 等（共 ~0.9 MB） |

本地目录（已 gitignore，不入库）：

```
third_party/SDHY_Test_System/                      # 克隆（depth 1）
third_party/models/wav2vec2-large-xlsr-53-chinese-zh-cn/   # 1.2 GB checkpoint
  config.json / preprocessor_config.json / pytorch_model.bin /
  special_tokens_map.json / vocab.json
third_party/.venv-baseline/                        # torch/transformers 基线环境
```

> 编码器权重下载后已核 sha256；`app/wav2vec2.py::Wav2Vec2Encoder` 在加载前还会再校验一次
> （fail-closed）。X2/M1 头直接使用仓库 `models/*.npz`。

## 2. 运行环境与计算模式

| 项 | 值 |
|---|---|
| Python | 3.12（`third_party/.venv-baseline`） |
| torch | 2.15.0 |
| transformers | 5.22.1 |
| 计算模式 | **cpu-fp32**（本机无 CUDA） |
| 官方标定模式 | `cuda-fp16`（`p0-baseline.json.runtime_mode`） |

> 注意：仓库明确说明 cpu-fp32 与 cuda-fp16 **不是同一份证据**（log 概率最大差 1.42、
> 约 6% 帧 argmax 翻转）。本基线只作为“同一模型在 CPU 上的近似”，不宣称与他们的
> 线上判定完全等价；用于跨模型对比时这个差异是次要的（远小于与 Qwen 的差距）。

## 3. 自由识别适配（关键说明）

他们的产品链路是**已知 target 的评测/对齐**（coverage + centroid margin + 声调规则），
并没有“自由识别 initial/final/tone”的现成输出。为了与本实验（无 target 的识别/检测）
可比，我们做了最小适配：

| 维度 | 适配方式 |
|---|---|
| initial | M1 音素 CTC 头贪心解码序列 → 第一个音素，若属于 21 声母则作为 initial，否则空声母 |
| final | X2 final 对齐头逐帧 logits → 非 blank 标签按帧取最大 → 映射到本实验规范韵母（`ix/iy/iz→i`、`v→ü`、`van→üán` 等） |
| tone | 用他们的 `app/tone.py` F0 提取 + `contour_features`，推导 4 类规则：头尾差 ≥2 st → 2，≤−2 st → 4；否则按“浊音中位音高相对该 voice 基准”和跨度判 1/3。阈值在**与评测集不相交的 512 条**上网格搜索（校准准确率 0.672） |
| 规范映射 | final 标签按 `evaluater_speaking_0.1/evaluation/pinyin.py` 的 umlaut/apical 规则折到本实验 36 韵母 |

> 这不是他们的官方任务指标；把对齐头当自由识别器、用规则补声调，都是显式适配，
> 目的是得到“同一编码器/同一套小头”的可比基线。

## 4. 结果（2,048 音频，与 `mini128-v1` 同一子集）

| 指标 | **SDHY P0-v2 基线（cpu-fp32，适配）** | **Qwen3.8-Omni `mini128-v1`（C1）** | 差距 |
|---|---:|---:|---:|
| Initial accuracy | 0.600 | **0.959** | +35.9pp |
| Final accuracy | 0.746 | **0.939** | +19.3pp |
| Tone accuracy | 0.699 | **0.912** | +21.3pp |
| Joint accuracy | 0.299 | **0.830** | +53.1pp |
| 派生检测 Error-F1 | 0.731 | **0.918** | +18.7pp |
| 派生检测 FPR | 0.701 | **0.170** | −53.1pp |

基线逐 voice / 逐调：

* initial：A 0.615 / B 0.613 / C 0.609 / D 0.561；
* final：A 0.812 / B 0.803 / C 0.773 / D 0.594；
* tone：A 0.574 / B 0.650 / C 0.666 / D **0.906**；
* tone 按调：1 声 0.578、2 声 0.891、3 声 **0.438**、4 声 0.891
  （规则分类器对三声/一声的判别是主要短板）。

原始输出：`results/baseline-sdhy-min128.json` + `results/baseline-sdhy-min128.predictions.csv`
（results/ 已 gitignore，脚本可复现）。

## 5. 复现

```bash
# 1) 克隆（需要 gh 权限）
gh repo clone CHAO-AI-Lab/SDHY_Test_System third_party/SDHY_Test_System -- --depth 1

# 2) 下载编码器（HF，1.2 GB；sha256 见上）
mkdir -p third_party/models/wav2vec2-large-xlsr-53-chinese-zh-cn
cd third_party/models/wav2vec2-large-xlsr-53-chinese-zh-cn
for f in config.json preprocessor_config.json vocab.json special_tokens_map.json; do
  curl -sL -o "$f" "https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/resolve/main/$f"
done
curl -L -o pytorch_model.bin \
  "https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/resolve/main/pytorch_model.bin"
shasum -a 256 pytorch_model.bin   # de031fd4…

# 3) 基线环境
export UV_CACHE_DIR="$PWD/third_party/.uv-cache"
uv venv third_party/.venv-baseline --python 3.12
uv pip install --python third_party/.venv-baseline/bin/python \
  "torch>=2.2" "transformers>=4.40" "soundfile>=0.12" "pandas" "pyarrow" "numpy<2.3"

# 4) 评测（约 10 分钟 CPU）
PYTHONPATH=src third_party/.venv-baseline/bin/python scripts/eval_sdhy_baseline.py
```

## 6. 局限

* 基线是**已知 target 的评测系统**，自由识别是适配层（对齐头 + 规则声调），
  不是其官方能力；不能据此说“SDHY 系统差 36 个点”，只能说“这套声学编码器/头在
  无 target 自由识别上弱于 Qwen”。
* CPU fp32 与官方 cuda-fp16 不是同一计算模式；
* 声调规则分类器只有 0.672 校准准确率，且三声/一声混淆明显；
* 未使用他们的 centroid/OMPAL/VOT 规则（需要已知 target 的强制对齐区间），
  这条链路只适合做“给定 target 的一致性判定”基线，与本文自由识别任务不同口径。
