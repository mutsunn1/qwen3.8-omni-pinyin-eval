# Qwen3.8-Omni 整句拼音评测实验报告（pilot，run `sentence-v1`）

> 2026-11-02 · 模型 阿里百炼 `qwen3.8-omni-flash` · prompt `sentence-v1`
> 设计：[整句评测实验设计](Qwen%203.8%20omni%20%E6%95%B4%E5%8F%A5%E6%8B%BC%E9%9F%B3%E8%AF%84%E6%B5%8B%E5%AE%9E%E9%AA%8C%E8%AE%BE%E8%AE%A1.md)（v0.1, 2026-11-01）
> 自动生成结果：[`results/sentence/sentence-v1/report.md`](../results/sentence/sentence-v1/report.md)
> （指标表、逐 speaker/learner、C1 vs C2 对比、SVG 图、验收检查）
> 数据：AISHELL-3（Apache-2.0）· OMPAL（CC BY 4.0）

## 一句话结论

在整句任务上，Qwen3.8-Omni（C1，think off）的**字级拼音三要素全对率 0.883**
（initial 0.967 / final 0.958 / tone 0.941，句级完全正确率 0.323，tone-sensitive
CER 0.116）；按单音节类别分布加权后，stable 子集与单音节的差距为
声母 **+0.6pp** / 韵母 **−0.1pp** / 声调 **+4.3pp**——**整句上下文没有让拼音识别变差**。
但**发音错误定位（B）不可靠**：三维 macro-F1 仅 0.553 / 0.558 / 0.601，模型强烈偏向
“都读对了”（声调 FNR 0.79、error-F1 0.22）；**朗读评分（C）相关性中等**：
accuracy / fluency / prosody 的 Pearson r = 0.352 / 0.581 / 0.393（三维平均 0.470），
native 满分锚点上模型均分 4.55（人工 5.00）。与单音节实验相反，**thinking on（C2）
在整句转写与错误定位上全面更好**（A joint +3.4pp、句级完全正确 +11.7pp、
B 声调 F1 +10.8pp，均显著），但在 C 的分数校准上更差（MAE +0.29）。
全部 **7,796 次调用解析成功率 99.99%、成本 ¥8.29（预算 ¥20）、验收全部通过**。

## 0. 实验目的与范围

本 pilot 把已有的**单音节 TTS 拼音识别**扩展到**整句**，回答四个问题：

- **RQ-A**：给定整句音频，模型能否逐字输出正确的声母/韵母/声调？
- **RQ-B**：给定音频与中文文本，模型能否定位学习者逐字的 tone/consonant/vowel 错误？
- **RQ-C**：给定音频与中文文本，模型的 accuracy/fluency/prosody 评分与人工分有多相关？
- **RQ-Thinking**：think on（C2）相对 think off（C1）在整句任务上表现如何？

定位是 **pilot + 能力 benchmark**，不是产品替代结论。A/B/C 使用不同语料与不同
ground truth，**不能互相替代，也不能直接横向比较**；唯一可对照的是 A 与已有
单音节结果在共享类别上的聚合比较（设计 §7.4）。

已完成：**设计 → 实现 → 数据准备 → 真实 API 全量运行 → 评分/指标/报告 → 独立审计**。

## 1. 实验设置

| 项 | 内容 |
|---|---|
| 模型 | `qwen3.8-omni-flash`（百炼业务空间兼容地址，见 `.env`；调用日期 2026-11-02） |
| 条件 | **C1** = think off / clean / JSON（主结果）；**C2** = think on / clean / JSON（RQ-Thinking）；同一批 item 配对 |
| 解码 | `temperature=0`、`top_p=1`、`response_format={"type":"json_object"}`；C1 发送 `reasoning_effort="none"`，C2 不发送（模型默认思考） |
| 输出上限 | 设计值 C1=256 / C2=1024；按 pilot 校准为按任务覆盖（见 §2.3） |
| 调用量 | 计划 **7,796** 次 = (A 300 + B 1,758 + C 1,840) × 2 条件；C 的 1,840 = 1,758 learner + 82 native 锚点 |
| 成本 | 预算上限 **¥20**；价格快照 输入 ¥0.8/M、输出 ¥2.7/M、音频 `ceil(秒)×7` token；按 API `usage` 实际记账 |
| 统计 | 聚类 bootstrap 2000 次（A 按 speaker、B/C 按 learner），α=0.05；C1 vs C2 配对 bootstrap；A 句级完全正确率另做 McNemar；任务族内 Holm 校正；解析失败不进指标分母、单独报告 |
| 可复现 | `run_manifest.json`（模型 id、prompt 版本/指纹、config sha256、数据清单 sha256、价格快照、seed、预算、git commit）；`generations.jsonl` 逐调用记录 usage/finish_reason/retry/延迟；run id `sentence-v1` |

### 1.1 任务定义（与设计一致）

- **A 整句拼音转写**：只给音频，逐字输出 `{char, initial, final, tone}`；AISHELL-3 原始
  surface 拼音作 gold；零声母 initial 为空串、`ü` 原样、儿化拆成「基础音节 + `er5`」。
- **B 整句发音评测**：音频 + 目标文本（繁体原样），逐字输出
  `{text, tone_ok, consonant_ok, vowel_ok}`；OMPAL 词级标签（0=错）作 gold；
  正类 = 错误，主指标 macro-F1；零声母 consonant 单列。
- **C 整句朗读评分**：音频 + 目标文本，输出 `{accuracy, fluency, prosody}`（1–5，容许 1 位小数）；
  目标为 OMPAL 3 位评分者均值；native 82 条仅作人类满分锚点，不进入相关/误差计算。

## 2. 数据与实现偏差（相对设计稿）

数据准备与运行中记录了以下偏差；每一条都已在 `data/sentence/manifests/prep_summary.json`
或 `run_manifest.json` 中留痕。它们不改变任务定义与主指标口径。

### 2.1 AISHELL-3（任务 A）

| 项 | 设计稿 | 实际执行 | 原因 |
|---|---|---|---|
| 抽样总体 | 全部 88,035 句 | **train 63,262 句**（174 speakers），过滤后 61,321 句 / 172 speakers | 可访问 HF 镜像无 test 划分的拼音标注；train/test 本就有 speaker 重叠，零样本评估不受影响 |
| 音频镜像 | OpenSLR SLR93 | HF 完整镜像 `shenyunhang/AISHELL-3`（`AISHELL/AISHELL-3` 仅约 9,802/63,262 条 wav，作 fallback） | 官方镜像缺文件；标注/spk-info 与官方一致（sha256 已记录） |
| 不可对齐行 | 设计未给出条数 | **18 条**（逐条记录 id + 原因） | `align_gold` 对儿化/字符数不一致的标注做了严格校验（设计稿提到的「2 条 text 不一致」属 OMPAL，不是 AISHELL） |
| 抽样 | 300 句 / 30 speakers / 每人 10 句 | **完全一致**；性别 18F/12M、口音 22 北/8 南、年龄组 24B/5C/1D **精确命中** | — |
| 覆盖 | 21 声母 / 36 韵母 / 4 声调 | **21/21、36/36、4/4**；补 1 句（韵母 `ueng`，同 speaker 替换并标记） | — |
| 特殊子集 | ≥50 句 | **193 句**（轻声 172、一/不 56、儿化 6，可重叠） | — |
| 音频 | 44.1 kHz/16-bit/mono，不重采样 | 300/300 条按此格式核验；总时长 1,059.2 s（中位 3.21 s） | — |

### 2.2 OMPAL（任务 B/C）

| 项 | 设计稿 | 实际执行 | 原因 |
|---|---|---|---|
| learner 条数 | 1,754 | **1,758** | 设计的「12 corrupted + 2 mismatch」是**全语料**口径（9 learner + 3 native 多字槽；第 2 个 text 不一致属于 native `00100217`）；learner 侧实际只有 **10** 条异常（9 个多字 word-slot + 1 个 text/slot 不一致 `00204117`），删除后 1,758。未为凑 1,754 丢弃有效数据 |
| native | 82 条锚点 | 82 条，全部 5 分；保留其 4 条已知异常（仅作锚点、不评分） | — |
| 标签 | 词级 0/1 | 1,758 条共 **19,598** 个逐字单元（多字槽仅存在于已删除的异常行）；零声母 2,970 个（pypinyin 上下文读音） | — |
| 评分者 | 论文 4 人 / 发布 3 人 | 发布数据 3 人均值；并用 detail 文件复核（1,112 条直接 id 重合 0 不一致） | — |
| 句子 | 84 个固定句 | 84 个不同参考文本；`sentence_id = S00..S83`（发布数据无内部编号） | — |
| 音频 | 保持原始格式 | 1,840 条（1,758 + 82）16 kHz/PCM_16/mono；总时长 10,458.6 s（中位 5.2 s） | — |

### 2.3 输出 token 上限校准（pilot 依据）

设计固定 C1=256 / C2=1024。首轮 pilot（12 items/任务/条件，run `pilot-design-v1`）显示：

| 任务|条件 | 截断数/12 | 解析成功/12 |
|---|---|---|
| A C1 | **3** | 9 |
| B C1 | **2** | 10 |
| 其余（A C2、B C2、C C1/C2） | 0 | 12 |

原因：A 每个汉字一条 JSON（约 30–40 token/字），256 对中位 11 字的句子必然不够。
因此按任务放宽输出上限：**C1：A 1536、B 768、C 256；C2：A 1536、B 1280、C 1024**。
校准后 pilot（run `pilot-cal-v2`，72 次真实调用）**72/72 解析成功、0 截断**。
`max_tokens` 只是输出上界，计费仍按 API 返回的实际 tokens；该偏差已写入
`run_manifest.json` 的 conditions 字段。C2 的 reasoning tokens **尾部可超过** `max_tokens`
（整句中位数在 A 776 / B 574 / C 352，均低于上限；但尾部最大值 3,685 / 2,776 / 2,664
仍以 `finish_reason=stop` 正常结束），因此上限只作安全阀、不是 reasoning 的硬预算。

### 2.4 运行工程

- 主运行命令行并发 **64**（`configs/sentence.yaml` 默认 48；该参数未写入 `run_manifest.json`，
  是已知记录缺口）。服务端限流/超时显著，本次累计 **1,331 次重试**，逐调用记录在
  `generations.jsonl` 的 `retries` 字段。
- **11 条调用记录最终失败**（5 条 400 不可重试 + 6 条超时重试耗尽），全部通过同一
  run-id 的**断点续跑**恢复（成功项不重复计费），最终无失败项；重试记录使
  `generations.jsonl` 共 7,807 条（= 7,796 评分调用 + 11 条失败尝试），见 §4/§5。

## 3. 指标与对齐口径（冻结）

- **A**：N0 规范化（NFKC、小写、声调符号→数字、`v`/`u:`→`ü`、儿化拆分、保留 AISHELL
  surface 声调）；汉字序列 Levenshtein 对齐（OpenCC t2s），替换对仍按拼音比较；
  gold 字无对应 = 删除（算错），模型多出 = 插入（不进分母、单独报告）；
  主指标 = 字级 initial+final+tone 全对率；句级完全正确 = 无删除无插入且所有对齐对三维全对。
  N1（教科书一/不 sandhi）只作诊断；stable 子集排除轻声/一不/儿化，用于与单音节结果对照。
- **B**：逐字按 t2s 后文本对齐；正类 = gold 错误（0）；macro-F1 = error/ok 两类 F1 均值；
  同时报告 error-F1、FPR、FNR、准确率与混淆矩阵；模型缺失单元按「未声称错误」计（fn）。
- **C**：Pearson r（主）、Spearman、MAE、RMSE；native 不进相关/误差；另报 native 锚点
  （金标 5.0 vs 模型均分）。
- **比较**：C1−C2 差值方向固定；配对 bootstrap 重采样同一聚类；A 句级完全正确率用
  McNemar（b/c + 精确二项），任务族内 Holm；效应量报告差值点估计与 CI。
- **验收（设计 §10.2）**：解析成功率 ≥95%（分任务）、A 字对齐覆盖率 ≥95%、
  B/C 文本对齐覆盖率 ≥95%、总成本 ≤¥20；不设准确率硬阈值。

## 4. 成本模型 vs 实测

| 项 | 设计预估 | 实测 |
|---|---|---|
| 调用数 | 7,616（B/C 按 1,754） | **7,796**（B/C 按 1,758 + native C 82）+ 11 次重试记录 |
| 成本 | C1 ¥0.5–1.5 / C2 ¥3–8 / 合计 ¥4–10 | **C1 ¥2.08 + C2 ¥6.20 = ¥8.29** |
| 预算 | ≤¥20 | ¥20 上限，实际占用 41.4%，未触发预算守卫 |
| token | 按 7,616 次估算 | prompt 1,494,046 / completion 2,626,517（其中 reasoning 1,433,123） |
| 解析成功率 | ≥95% | **99.99%**（7,795/7,796 ok；C2 有 1 条模型把评分任务误解为错误标注，属真实行为漂移） |

- **单位成本**（同一口径）：按**调用**计价 A ¥0.00170、B ¥0.00135、C ¥0.00069；
  按 **item（两个条件合计）** A ¥0.00341、B ¥0.00269、C ¥0.00138。整句比单音节
  小实验的单次成本高（音频更长、A/B 输出更长、C2 reasoning 更多），但总量仍在个位数元。
- **工程观测**：并发 64 时有明显服务端限流/超时，累计 1,331 次重试；11 条调用记录最终失败，
  由同 run 断点续跑全部恢复（无最终失败项）；平均单次延迟 54.3 s（C2 的 reasoning 与 A 的
  长 JSON 输出是主要来源），全量运行墙钟 112 min。这是**吞吐**限制，不是结果有效性限制。

## 5. 结果

完整表格、逐 speaker/learner、SVG 与验收见 `results/sentence/sentence-v1/report.md`；
以下为主表摘录（C1 主条件；95% CI 为聚类 bootstrap 2000 次）。

### 5.1 任务 A：整句拼音转写（AISHELL-3，n=300）

| 指标 | C1 | C1 95% CI | C2 | C2 95% CI |
|---|---:|---|---:|---|
| 字级拼音准确率（initial+final+tone 全对） | **0.8832** | [0.8615, 0.9027] | **0.9174** | [0.9018, 0.9321] |
| 声母准确率 | 0.9673 | [0.9544, 0.9785] | 0.9834 | [0.9716, 0.9924] |
| 韵母准确率 | 0.9584 | [0.9449, 0.9693] | 0.9816 | [0.9763, 0.9870] |
| 声调准确率 | 0.9406 | [0.9321, 0.9493] | 0.9443 | [0.9371, 0.9526] |
| 句级完全正确率 | 0.3233 | [0.2633, 0.3867] | 0.4400 | [0.3633, 0.5167] |
| tone-sensitive CER | 0.1162 | [0.0972, 0.1376] | 0.0829 | [0.0682, 0.0985] |
| tone-stripped CER | 0.0643 | [0.0481, 0.0839] | 0.0318 | [0.0211, 0.0445] |
| 插入率 / 删除率 | 0.0003 / 0.0017 | — | 0.0003 / 0.0011 | — |
| 字对齐覆盖率 | 0.9983 | [0.9970, 0.9994] | 0.9989 | [0.9977, 0.9997] |

特殊子集（C1 joint）：stable **0.8887**（n=107）、轻声 0.8813（n=172）、
一/不 0.8785（n=56）、**儿化 0.7576（n=6，样本少、波动大）**。
N1（教科书一/不 sandhi）诊断：joint 0.8930（N1−N0 = +0.98pp），
说明 AISHELL surface 一/不标注的不完整只带来约 1pp 量级的主指标损失。

### 5.2 任务 B：整句发音评测（OMPAL，n=1,758 / 19,598 字）

正类 = gold 错误。三个维度的偏差方向**不一致**：声调的预测错误率 12.03% 低于
人工 13.08%（**漏判**，FNR 0.79）；声母 9.04% 远高于人工 1.93%、韵母 2.61% 高于 1.82%
（**过度标记**，声母 FPR 8.3%）——但两者 absolute error 类 F1 仍低（0.16–0.24），
macro-F1 被“ok 类”抬升。整体上模型既漏掉真实错误、又在声母上误报。

| 维度 | macro-F1 (C1) | 95% CI | C2 | error-F1 | FPR | FNR | 准确率 |
|---|---:|---|---:|---:|---:|---:|---:|
| 声调 | **0.5528** | [0.5403, 0.5656] | **0.6607** | 0.2179 | 0.1069 | **0.7909** | 0.8037 |
| 声母 | **0.5581** | [0.5353, 0.5810] | **0.6438** | 0.1647 | 0.0829 | 0.5330 | 0.9084 |
| 韵母 | **0.6006** | [0.5624, 0.6377] | 0.6116 | 0.2189 | 0.0216 | 0.7339 | 0.9654 |

零声母 consonant 单列（n=2,970 字）：macro-F1 0.6354、error-F1 0.2895、FPR 0.0297、
FNR 0.4884。结论：**B 任务上模型不能可靠定位学习者错误**；C2 有明显改善
（声调 +10.8pp、声母 +8.6pp，均显著；韵母无显著差异），但绝对水平仍不足以做自动纠音。

### 5.3 任务 C：整句朗读评分（OMPAL，n=1,758 非 native）

| 维度 | Pearson | 95% CI | Spearman | MAE | RMSE |
|---|---:|---|---:|---:|---:|
| accuracy | **0.3524** | [0.2561, 0.4250] | 0.3248 | 0.8608 | 1.0190 |
| fluency | **0.5809** | [0.5211, 0.6412] | 0.5829 | 1.0572 | 1.2777 |
| prosody | **0.3932** | [0.3085, 0.4774] | 0.3653 | 0.8973 | 1.0848 |
| 三维平均 | **0.4699** | [0.3886, 0.5478] | 0.4394 | 0.8991 | 1.0595 |

- **C1、非 native（n=1,758）**分数分布：pred 均值 accuracy 3.338 / fluency 3.008 /
  prosody 3.132（标准差 0.594 / 0.974 / 0.694），gold 均值 4.025 / 4.007 / 3.937
  （标准差 0.718 / 0.637 / 0.623）；模型整体偏低 0.7–1.0 分且离散度不低于人工，
  说明不是“全给常数分”而是**校准偏差 + 中等相关**（C1+C2 合并、含 native 的
  均值见 `rows_c.parquet`，不在此单独引用）。
- **native 人类满分锚点（n=82）**：人工三维均分 5.000，模型预测 4.548；
  分维 4.507 / 4.673 / 4.462。模型对 L1 满分语音也平均扣约 0.45 分，
  解释 C 的 MAE 时需扣除这一系统偏移。

### 5.4 C1 vs C2（RQ-Thinking，同 item 配对）

| 任务 | 指标 | Δ(C1−C2) | 95% CI | 结论 |
|---|---|---:|---|---|
| A | 字级 joint | **−0.0341** | [−0.0470, −0.0241] | C2 更高 |
| A | 句级完全正确 | **−0.1167** | [−0.1667, −0.0700] | C2 更高（McNemar b=10/c=45, p<0.001, Holm p<0.001） |
| A | tone-sensitive CER | +0.0333 | [+0.0231, +0.0461] | C1 更差（C2 更低更好） |
| B | 声调 macro-F1 | **−0.1079** | [−0.1277, −0.0876] | C2 更高 |
| B | 声母 macro-F1 | **−0.0857** | [−0.1078, −0.0659] | C2 更高 |
| B | 韵母 macro-F1 | −0.0110 | [−0.0430, +0.0195] | 无显著差异 |
| C | accuracy Pearson | **−0.0959** | [−0.1610, −0.0359] | C2 更高（但见 MAE） |
| C | fluency Pearson | **+0.0502** | [+0.0123, +0.0954] | C1 更高 |
| C | 三维平均 Pearson | −0.0340 | [−0.0837, +0.0131] | 无显著差异 |
| C | 三维平均 MAE | **+0.2931** | [+0.2199, +0.3626] | C1 误差更小（C2 更差） |

**RQ-Thinking 答案**：与单音节实验“thinking 系统性有害（约 −20pp）”的结论**相反**，
整句任务上 thinking on 对 A（转写）和 B（错误定位）**显著有利**，对 C 的
Pearson 混合、MAE 明显更差。推测机制：整句输出需要跨字序列决策与错误定位推理，
reasoning 有帮助；而 C 的 1–5 分校准更依赖直接声学印象，reasoning 会放大模型自身的
“从严”偏移。该结论是 pilot 级（n=300 / 1,758），下一步应扩大样本并固定 thinking 预算。

### 5.5 与单音节结果的聚合对照（设计 §7.4）

不加权聚合（A stable 子集 n=107）：

| 维度 | 单音节 exp3 free（n=2,048） | 句内 A stable（n=107） | 差值 |
|---|---:|---:|---:|
| 联合 | 0.8384 | 0.8887 | +0.0503 |
| 声母 | 0.9624 | 0.9609 | −0.0015 |
| 韵母 | 0.9487 | 0.9522 | +0.0034 |
| 声调 | 0.9116 | 0.9557 | +0.0440 |

按单音节类别分布加权（design §7.4；权重覆盖率 initial 100%、final 99.2%、tone 100%）：

| 维度 | 单音节加权 | 句内 stable 加权 | 差值 | 覆盖类别 |
|---|---:|---:|---:|---:|
| 声母 | 0.9624 | 0.9686 | **+0.0062** | 22/22 |
| 韵母 | 0.9508 | 0.9499 | **−0.0009** | 35/36 |
| 声调 | 0.9116 | 0.9548 | **+0.0432** | 4/4 |

**RQ-Context 答案**：在共享稳定类别上，句内拼音识别与单音节基本持平；
声母/韵母差异在 ±1pp 内，声调在句内反而高约 4pp（句子上下文可能帮助声调判断，
但语料/说话人/风格不同，不能做因果解释）。**不能**据此说“整句比单音节更好/更差”，
只能说明“没有观察到整句上下文导致的系统性退化”。

### 5.6 独立审计与复现口径

- **独立复算**：`scripts/verify_sentence_run.py` 不 import 评分/指标/报告模块、只复用
  `normalize.py` 原语，从 `generations.jsonl` + 数据清单重算全部主指标。与产物对比：
  `metrics.csv` 222/222 行、`report_summary.json` 26/26 字段、`acceptance.json`、
  `cost_summary.json` 与 `report.md` 关键表格 **0 处 >0.1pp 差异**；数据准备 57/57、
  运行记录 6/6、可复现性 17/17、验收线 7/7 全部通过 → **PASS**。
  完整审计见 `results/sentence/sentence-v1/verification.md`。
- **评分口径版本**：生成完成后对解析器做过一次容错修复（模型偶尔把对象包成
  单元素数组 `[{"words":[...]}]`，4 条 B|C2 因此恢复为 ok），并按当前实现重新评分；
  `run_manifest.json` 的 `git_commit` 记录的是实验*开始*时的仓库 HEAD（不含 sentence 实现），
  因此复现以本次实现提交为准（见 §7 末行）。
- **可核验的运行参数缺口**：并发数（主运行 64）未写入 `run_manifest.json`；已在 §2.4 说明。
  其余可复现字段（模型 id、prompt 版本+指纹、config sha256、数据清单 sha256、价格快照、
  seed、预算、重试/usage/finish_reason）均已记录。

## 6. 结论与解释边界

**能否稳定测**：能。7,796 次真实调用解析成功率 99.99%、对齐覆盖率 ≥99.8%、
成本 ¥8.29、验收全部通过；离线 mock（7,796 次）+ 5 个测试文件的回归测试保证可复现。

**效应方向/量级（pilot）**：

1. **A 整句拼音转写可用**：C1 joint 0.883（句级完全正确 32%）；换成 think on 提升到
   0.917（句级完全正确 44%）。分开看维度：声母/韵母 >0.95，声调 0.94 是短板。
2. **B 发音错误定位不可用**：macro-F1 0.55–0.60，模型偏向“都读对”（声调 FNR 0.79），
   即使 C2 提升后 error-F1 也只有 0.22–0.29。若要做纠音产品，需要重设计任务
   （例如给音素对齐或逐字候选、调低“ok”先验、或用派生比较而非直接判）。
3. **C 朗读评分只能算中等相关**：fluency r=0.58 可用作趋势，accuracy r=0.35 /
   prosody r=0.39 偏弱；MAE 0.86–1.06 且整体偏低约 0.45 分（native 锚点佐证）。
   适合“粗筛/趋势”，不适合替代人工评分。
4. **RQ-Thinking 在整句上反转**：thinking on 对 A/B 显著有利，对 C 的相关性混合、
   MAE 更差。单音节“thinking 有害”的结论不能外推到整句。
5. **RQ-Context**：加权后整句与单音节在声母/韵母上基本持平、声调略好（+4.3pp），
   未观察到整句上下文导致的系统性退化；但语料/域不同，不做因果结论。

**解释边界**（同设计 §11）：AISHELL-3 是朗读语料、train/test speaker 重叠（零样本评估
抽样，不声称 unseen speaker）；OMPAL 只有 3 位评分者、法语 L1 子集、无 consent 声明，
B/C 结论只限定该 learner 子集；A/B/C 语料与 ground truth 不同，**不能横向比较**；
A 的抽样来自 train 划分（可访问镜像无 test 拼音标注）且特殊子集仅 6 条儿化；
B/C 的 1,758 与设计 1,754 的差异见 §2.2；C2 的输出上限与 reasoning 记账见 §2.3。

**下一步（样本量/设计）**：A 的 95% CI 约 ±2pp，B 的 macro-F1 CI 约 ±1.5pp，
C 的 Pearson CI 约 ±0.1；若要把 A 句级完全正确率的分辨率压到 ±3pp、C 相关系数压到
±0.05，建议 A 扩到 1,000+ 句、B/C 保持全量或扩到更多 learner；B 需要任务重设计；
C2 建议固定 reasoning token 预算后复测；AISHELL-3 test 划分的拼音标注需另找官方来源。

## 7. 复现步骤

```bash
# 0) 环境（`.env` 中填 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL）
.venv/bin/python -m pinyin_eval.cli sentence prepare     # 数据准备（可续传）
.venv/bin/python -m pinyin_eval.cli sentence check       # 计划调用量自检
.venv/bin/python -m pinyin_eval.cli sentence estimate    # 成本预估（expected / worst-case）
.venv/bin/python -m pinyin_eval.cli sentence generate --run-id sentence-v1   # 真实运行（可续跑 + ¥20 预算守卫）
.venv/bin/python -m pinyin_eval.cli sentence score  --run-id sentence-v1
.venv/bin/python -m pinyin_eval.cli sentence cost   --run-id sentence-v1
.venv/bin/python -m pinyin_eval.cli sentence report --run-id sentence-v1
```

离线全链路（无 API key）：`make sentence-mock`（7,796 次 mock 调用 + 评分 + 报告）。

代码版本：整句实现提交为 `<待回填>`（实验运行时仓库 HEAD 为 `9ad0850`，不含 sentence 代码；
复现请 checkout 实现提交并 `uv sync --locked`）。

## 8. 产物清单

| 产物 | 路径 |
|---|---|
| 数据清单与摘要 | `data/sentence/manifests/{aishell3_items,ompal_b_items,ompal_c_items,ompal_native_items}.jsonl`、`prep_summary.json`、`README.md` |
| 运行记录 | `results/sentence/sentence-v1/{run_manifest.json,generations.jsonl,generate_summary.json}` |
| 评分行 | `results/sentence/sentence-v1/rows_{a,b,c}.parquet`（+ CSV） |
| 指标与报告 | `results/sentence/sentence-v1/{report.md,metrics.csv,comparisons.csv,report_summary.json,acceptance.json}`、`figures/*.svg` |
| 逐 speaker/learner | `per_speaker_a.csv`、`per_learner_b.csv`、`per_learner_c.csv` |
| 单音节对照明细 | `single_syllable_categories.csv` |
| 独立审计 | `results/sentence/sentence-v1/verification.md` |
| 代码 | `src/pinyin_eval/sentence/`（normalize/parsing/scoring/metrics/runner/report/cli/mock）、`scripts/prepare_sentence_data.py`、`scripts/verify_sentence_run.py` |
| 测试 | `tests/test_sentence_{core,scoring,metrics,report,data_prep}.py`、`tests/test_verify_sentence_run.py`（432 项全部通过） |
