# Qwen3.8-Omni 整句拼音评测实验设计（pilot）

> 状态：设计稿 v0.2（2026-11-03）· A/B/C 已完成 pilot 运行（2026-11-02）；**附录 A 任务 D 已于 2026-11-05 完成运行（run `sentence-d-v1`）**
> 实现与结果：[整句 pilot 实验报告](sentence-experiment-report.md)
> 关联：[音素识别实验设计](experiment-design.md) · [最简实验报告](report-minimal.md) ·
> [三模型详细对比](three-model-comparison.md)
> 数据集：AISHELL-3（Apache-2.0）· OMPAL（CC BY 4.0）
> 模型：阿里百炼 `qwen3.8-omni-flash`

## 0. 摘要

本实验把已有的**单音节 TTS 拼音识别**扩展到**整句**，用三个任务回答
“Qwen3.8-Omni 的整句拼音能力如何”：

- **A. 整句拼音转写**：只给音频，逐字输出拼音；AISHELL-3 作 gold。
- **B. 整句发音评测**：音频 + 中文文本，逐字判断 tone / consonant / vowel 是否正确；OMPAL 词级标签作 gold。
- **C. 整句朗读评分**：音频 + 中文文本，输出 accuracy / fluency / prosody 1–5 分；OMPAL 句级人工分作 gold。

定位是 **pilot + 能力 benchmark**，不是产品替代结论。A/B/C 使用不同语料和不同
ground truth，**不能互相替代，也不能直接横向比较**；唯一可对照的是 A 与已有
单音节结果在共享音节/声调上的聚合比较。

## 1. 目标与非目标

### 1.1 研究问题

- **RQ-A**：给定整句音频，Qwen3.8-Omni 能否逐字输出正确的声母/韵母/声调？
- **RQ-B**：给定音频与中文文本，模型能否定位学习者逐字的 tone/consonant/vowel 错误？
- **RQ-C**：给定音频与中文文本，模型的整句 accuracy/fluency/prosody 评分与人工分有多相关？
- **RQ-Context**：整句上下文是否改变模型相对单音节的表现？
- **RQ-Thinking**：thinking on 在整句任务上是否仍然有害？

### 1.2 非目标

- 不做产品替代/选型结论（需要更大样本、外部基线和人工评分）。
- 不做训练或微调；全部 zero-shot。
- 不做段落/多句/自由对话；只做**一条 utterance = 一句朗读文本**。
- 不做逐音素强制对齐；A 用逐字序列对齐。
- 不实现代码；设计定稿后再另开实现任务。

## 2. 领域术语

| 术语 | 含义 |
|---|---|
| **整句 / utterance** | 一条音频对应一句朗读文本（AISHELL-3 一行；OMPAL 一句）。不是段落、不是多句拼接。 |
| **整句拼音转写（A）** | 只给音频，输出逐字拼音；不提供文本。 |
| **整句发音评测（B）** | 给音频 + 中文文本，逐字判断发音错误（tone/consonant/vowel）。 |
| **整句朗读评分（C）** | 给音频 + 中文文本，输出 1–5 分的 accuracy/fluency/prosody。 |
| **surface pinyin** | 实际语流标注：轻声 5、一/不变调、儿化合并等。AISHELL-3 的原始拼音。 |
| **canonical pinyin** | 字典/ underlying 声调，不含语流变调；只用于与单音节结果对照。 |
| **字级拼音准确率** | 逐字比较 initial + final + tone 三者是否全部正确；本实验 A 的主指标。 |
| **句级完全正确率** | 整句所有字级拼音都正确；严格指标。 |
| **tone-sensitive CER** | 拼音音节序列的编辑距离错误率；`ni3` 与 `ni2` 视为不同。 |
| **tone-stripped CER** | 去掉声调后的拼音 CER，用于分离声调错误。 |
| **erhua split** | 把 AISHELL-3 的合并写法 `dianr3` 拆成 `dian3` + `er5`，以便逐字对齐。 |
| **C1 / C2** | C1 = think off / clean / JSON；C2 = think on / clean / JSON。 |
| **派生检测** | 用自由识别结果与目标文本程序化比较得到检测结果；本设计 B 是独立任务，不用派生检测代替。 |

## 3. 数据

### 3.1 AISHELL-3（任务 A）

事实（2026-11-01 核对）：

- 85 h，88,035 条 utterance，218 speakers；Apache-2.0；OpenSLR SLR93。
- 转写为逐字 `汉字 pinyin`，例如 `武 wu3 术 shu4 … 的 de5`；无标点。
- 声调 1–5（轻声 = 5）；`ü` 写作 `v`；儿化合并为 `dianr3` 这种形式。
- 一/不 sandhi 只部分标注：不约 98.7% 符合规则，一只有约 65.9%。
- 音频 44.1 kHz / 16-bit / mono。
- 句长：median 11 字，p95 = 21，max = 39；5–30 字覆盖绝大多数。
- 官方 train/test **不是 speaker-disjoint**（170 speakers 重叠）；zero-shot 下不使用
  “held-out speaker”这个说法，只做评估抽样。

抽样：

- **300 句 / 30 speakers**，每人 10 句。
- speaker 分层：约 18 女 / 12 男；22 北方 / 8 南方；年龄 B/C/D 约 24/5/1。
  若某 speaker 在 5–30 字过滤后不足 10 句，用同层 speaker 替换。
- 句子过滤：5–30 汉字；从全部 88,035 句分层随机抽；固定 seed，生成
  `aishell3_pilot_items.jsonl`（实现阶段产物，本设计只定义）。
- 覆盖检查：21 个声母、36 个韵母、4 个声调的类别至少各出现一次；若抽样后仍缺，
  在同层内补抽并记录。
- 特殊子集：至少 50 句含轻声（5）、一/不、儿化中的至少一类，用于诊断；
  特殊子集允许与主语料重叠。

音频：保持原始 44.1 kHz / 16-bit / mono，不重采样、不做增益；记录每条时长。

### 3.2 OMPAL（任务 B/C）

事实（2026-11-01 核对）：

- 1,850 wavs：1,768 learner + 82 native；46 名法语 L1 学习者；84 个固定句子。
- 标签是 JSON，不是 CSV；逐字 `text: ["字"]` + 词级
  `tone` / `phoneme_consonant` / `phoneme_vowel` 0/1；
  句级 `accuracy` / `fluency` / `prosody` 1–5。
- 实际发布 3 位评分者的数据（论文/README 写 4 位）：句级分是 3 人均值，
  词级标签是 3 人多数票；native 全部为 5 分，不能作为评分目标。
- 有 12 条 utterance 的 word-slot 对齐损坏；2 条 text 与 words 不一致
  （`00100217`、`00204117`）。
- 官方 `train_1..5/test_1..5` 文件不是干净 5-fold（speaker 重叠、覆盖不全），
  本实验不使用。
- 文本为繁体中文，无标点；无 pinyin 字段；CC BY 4.0。

清洗：

- 只保留 **1,768 learner**；删除 12 条 corrupted + 2 条 text/slot 不一致，
  得到 **1,754 条**。
- native 82 条只作“人类满分锚点”，不进入 B/C 的目标分布。
- 46 名 learners 全部保留；不划分 train/test（zero-shot 评估）；按 learner 报告。
- 84 个固定句子用于覆盖检查；不对句子做二次筛选（避免选择偏差）。
- 清洗清单、删除 id、脚本 seed 写入实现阶段 manifest；本设计只定义规则。

### 3.3 数据与许可风险

- AISHELL-3：Apache-2.0；但厂商页面 429，未能核对 vendor EULA；对外发布派生
  benchmark 前需再核。
- AISHELL-3 train/test speaker 重叠：不能声称 unseen speaker。
- OMPAL：CC BY 4.0；无 consent/伦理声明；论文说 4 位评分者、数据只有 3 位；
  法语 L1、1,754 条，结论只能限定在该子集。
- OMPAL 无 pinyin 字段：B/C 的 gold 是标签，不是拼音；不要把它当 A 的 gold。

## 4. 任务定义

### 4.1 A：整句拼音转写

- 输入：整句音频（AISHELL-3）。
- 输出：逐字 JSON 数组：

```json
[
  {"char": "你", "initial": "n", "final": "i", "tone": 3},
  {"char": "好", "initial": "h", "final": "ao", "tone": 3}
]
```

规则：

- 每个汉字一个条目；零声母 `initial` 写空字符串 `""`。
- `tone` 取 1–5，5 = 轻声。
- `ü` 直接写 `ü`（不要写 `v`）。
- 儿化音：基础音节 + `er5` 分开写（例如 `点 dian3` + `儿 er5`）。
- 只输出 JSON 数组，不输出标点、解释或 Markdown。

### 4.2 B：整句发音评测

- 输入：整句音频 + 中文文本（OMPAL 原文本，繁体保持原样）。
- 输出：

```json
{
  "words": [
    {"text": "他", "tone_ok": true, "consonant_ok": true, "vowel_ok": true},
    {"text": "們", "tone_ok": true, "consonant_ok": true, "vowel_ok": true}
  ]
}
```

规则：

- `text` 必须按输入文本逐字原样输出；不得改写、简繁转换或合并。
- 只判断 tone / consonant / vowel 三个维度；不输出拼音。
- 零声母字的 consonant 维度按 OMPAL 标签口径处理；报告中单列，不混入主要结论。
- 只输出 JSON，不输出解释。

### 4.3 C：整句朗读评分

- 输入：整句音频 + 中文文本。
- 输出：

```json
{"accuracy": 4.3, "fluency": 4.7, "prosody": 4.0}
```

规则：

- 三维各 1–5 分，允许 1 位小数。
- 不要输出评价理由；只输出 JSON。
- 目标是 3 位评分者的均值（OMPAL 原始 target）。

### 4.4 任务关系

- A 与 B/C 使用不同语料（native read vs French-L1 learner），**不可互相替代**。
- A 是“纯音频转写”；B/C 是“音频 + 文本的评测”，文本是已知条件。
- A 的结果可与已有单音节结果在“共享音节/声调”上聚合比较；B/C 只与 OMPAL
  人工标签比较，不与 A 或单音节横向比。

## 5. 生成条件与固定参数

| 条件 | Thinking | 音频 | 输出格式 | 用途 |
|---|---|---|---|---|
| **C1** | off | clean | JSON | 主结果 |
| **C2** | on | clean | JSON | RQ-Thinking |

- A/B/C 都跑 C1 和 C2；C1/C2 使用同一批 item，配对比较。
- 模型：`qwen3.8-omni-flash`（百炼）；固定模型 id / 调用日期 / 业务空间，写入 manifest。
- 解码：`temperature=0`、`top_p=1`；C1 `max_tokens=256`，C2 `max_tokens=1024`。
- thinking off：`reasoning_effort="none"`；thinking on：不发送参数（模型默认）。
- 结构化输出：`response_format={"type":"json_object"}`（A 的数组包在 JSON 对象里时，
  由 prompt 明确定义；实现阶段需确认 API 对数组的支持）。
- 音频：AISHELL-3 保持 44.1 kHz；OMPAL 保持原始格式；两者都不重采样，记录采样率/声道。
- 记录 `finish_reason`、`reasoning_text`（C2）、解析失败与重试次数。

## 6. Gold 规范化与对齐

### 6.1 AISHELL-3 拼音规范化

按以下顺序应用（记为 **N0 orthographic normalization**，主分析使用）：

1. Unicode NFKC；小写化。
2. 声调符号 → 数字（若模型输出 `nǐ` 这类形式）。
3. `v` → `ü`；`u:` → `ü`。
4. 去掉标点、空格、换行；一个音节一个 token。
5. 儿化拆分：`dianr3` → `dian3` + `er5`；`nar3` → `na3` + `er5`；
   两个汉字共享一个 `r` 拼音时，第二个汉字（儿）记 `er5`。
6. 保留 AISHELL-3 原始 surface 声调；**不强行改写一/不**（见 6.2）。

主分析 = N0(gold) vs N0(hypothesis)；另做：

- **N1 sandhi-normalized**：对 gold 和 hyp 同时应用教科书规则（不 + T4 → bu2；
  一 + T4 → yi2；其余一 → yi4），作为诊断指标；
- **strict raw**：不做 N0 的儿化拆分/一不处理的原始 surface 对照；
- **stable subset**：排除轻声、一/不、儿化后的子集，用于与单音节对照。

### 6.2 一/不标注不一致

AISHELL-3 的“一”只有约 65.9% 应用变调。主分析保留 raw surface；
N1 是诊断口径，报告“N0 vs N1”的差值，说明 sandhi 规则对指标的影响。
不要在同一个数字里混用 N0 和 N1。

### 6.3 A 输出对齐

- 汉字序列先做 NFKC + 简繁统一（OpenCC t2s；AISHELL 为简体，模型可能输出繁体）。
- 用字符级 Levenshtein 对齐 gold 与模型输出。
- gold 有、模型没有的字：算错误；
- 模型多出的字：不进分母，单独报告插入率；
- 句级完全正确 = 所有 gold 字都匹配且 initial/final/tone 全对，且无插入。
- 解析失败/空回复/拒答单独统计，不进指标分母。

### 6.4 OMPAL 文本对齐（B/C）

- 输入文本为繁体；模型输出 `text` 可能简繁混用；用 OpenCC t2s 统一后对齐。
- 以 OMPAL 的 `words` 位置列表为 gold；模型少字 → 该字三个维度都算错；
  模型多字 → 不进分母，报告插入。
- 12+2 条损坏记录已在数据清洗中删除。
- B 的零声母字 consonant 维度：按 OMPAL 标签保留；报告中单列，不改变标签。

## 7. 指标

### 7.1 A：整句拼音转写

**主指标**：

- **字级拼音准确率（tone-sensitive）**：对齐后，initial + final + tone 三者全对的
  gold 字 / gold 总字数。

**次要指标**：

- initial / final / tone 分维准确率；
- tone-sensitive pinyin CER；
- tone-stripped CER；
- 句级完全正确率；
- 插入/删除率；
- 特殊子集：轻声、一/不、儿化、erhua split 的准确率；
- N1 sandhi 诊断与 stable subset 对照。

**统计**：按 speaker 聚类 bootstrap 95% CI（AISHELL 30 speakers）；2000 次。
逐 speaker 报告描述统计。

### 7.2 B：整句发音评测

- 逐字、逐维度混淆矩阵：正类 = 错误（标签 0）。
- 主指标：tone / consonant / vowel 三个维度的 **macro-F1**。
- 次要：各维 F1、FPR、FNR、准确率。
- 零声母 consonant 单列。
- 按 learner 聚类 bootstrap；2000 次；逐 learner 报告。

### 7.3 C：整句朗读评分

- 目标：OMPAL 3 位评分者均值（accuracy/fluency/prosody）。
- 主指标：每维 **Pearson r**；次要：Spearman ρ、MAE、RMSE。
- 另报告三维平均分与总体相关；按 learner 聚类 bootstrap。
- 不把 native 的常数 5 分纳入相关/MAE 计算；native 只作人类满分锚点。

### 7.4 与单音节结果的对照

- 只在 AISHELL-3 的 **stable subset**（排除轻声、一/不、儿化）上，按共享的
  target/initial/final/tone 类别聚合，和已有单音节 TTS 结果比较字级准确率。
- 用单音节的类别分布加权；不做逐条配对；明确语料（read vs TTS）与 speaker 域不同。
- 报告“同一音节在单音节 vs 句内”的聚合差值，不宣称个体层面因果。

## 8. Prompt 模板

> 实现阶段必须把下面模板与输出 schema 一起固定；prompt 变更需更新
> `prompt_version` 与 request fingerprint。以下只是设计稿。

### 8.1 A：整句拼音转写

```text
你是普通话整句拼音转写助手。请只根据音频，逐字输出每个汉字的拼音。
要求：
1. 按音频中汉字的顺序，每个汉字一个条目。
2. 每个条目包含 char、initial、final、tone；tone 取 1–5（5=轻声）。
3. 零声母 initial 写空字符串；ü 直接写 ü；儿化音把基础音节和“儿”（er5）分开写。
4. 只输出 JSON 数组，不要输出标点、解释或 Markdown。
```

### 8.2 B：整句发音评测

```text
你是普通话发音评测助手。音频是学习者朗读，文本是目标句子。
请逐字判断实际发音，输出 JSON：
{"words":[{"text":"字","tone_ok":true,"consonant_ok":true,"vowel_ok":true}, ...]}
要求：
1. text 按目标文本逐字原样输出（繁体保持繁体），不得改写或合并。
2. 只判断 tone / consonant / vowel 三个维度，不输出拼音。
3. 只输出 JSON，不要解释。
```

### 8.3 C：整句朗读评分

```text
你是普通话朗读评分助手。音频是学习者朗读，文本是目标句子。
请给 accuracy / fluency / prosody 三个维度打分，各 1–5 分，允许 1 位小数。
只输出 JSON：{"accuracy": 1-5, "fluency": 1-5, "prosody": 1-5}
不要输出评分理由。
```

## 9. 抽样、条件与统计

### 9.1 样本量

| 任务 | 语料 | 样本 | C1 | C2 | 总调用 |
|---|---|---:|---:|---:|---:|
| A | AISHELL-3 | 300 句 | 300 | 300 | 600 |
| B | OMPAL | 1,754 条 | 1,754 | 1,754 | 3,508 |
| C | OMPAL | 1,754 条 | 1,754 | 1,754 | 3,508 |
| **合计** | — | — | **3,808** | **3,808** | **7,616** |

### 9.2 统计

- bootstrap 2000 次；A 按 speaker 聚类、B/C 按 learner 聚类；voice 不适用（AISHELL/OMPAL
  为多人语音，不作为分层变量；speaker/learner 是聚类单位）。
- C1 vs C2：同一 item 配对；A/B 用配对 bootstrap / McNemar；C 用配对 bootstrap
  差值 CI（或 Wilcoxon 符号秩，若分布偏）。
- 多指标：每个任务族内 Holm 校正；A/B/C 分开报告，不跨任务合并 p 值。
- 报告效应量：准确率差、F1 差、相关系数差、MAE 差。
- 解析失败不计入指标分母；单独报告失败率。

### 9.3 分析输出

- A：主表（字级准确率/CER/句级完全正确）、分维度、特殊子集、逐 speaker；
- B：三维 F1/FPR/FNR、混淆矩阵、逐 learner；
- C：三维 Pearson/Spearman/MAE/RMSE、散点图、逐 learner；
- C1 vs C2 对比表；
- 与单音节聚合对照表（stable subset）；
- 成本/用量/失败率表。

## 10. 成本与验收

### 10.1 成本

- 7,616 次调用（C1 3,808 + C2 3,808）。
- 价格快照沿用现有项目：输入 ¥0.8/M、输出 ¥2.7/M；音频 `ceil(秒)×7` token。
- 预计：C1 约 ¥0.5–1.5；C2 约 ¥3–8；合计约 **¥4–10**。
- 最坏情况（C2 每次打满 1024 输出 token）：
  `3,808 × (输入约 250×0.8/M + 1024×2.7/M) ≈ ¥12`；加 C1 仍 < ¥20。
- 预算上限 **¥20**；实现前用实际音频时长跑 `estimate` 校准；超预算先降 C2 或缩样本。
- 成本按 API `usage` 记录，不按估算宣称。

### 10.2 验收线

- 解析成功率 ≥ 95%（A/B/C 分任务统计）；
- A 字对齐覆盖率 ≥ 95%；B/C 文本对齐覆盖率 ≥ 95%；
- 成本 ≤ ¥20；时长/失败/重试完整记录；
- 不设准确率硬阈值；pilot 的结论是“能否稳定测、效应方向/量级、下一步样本量”。

## 11. 风险与解释边界

| 风险 | 影响 | 应对 |
|---|---|---|
| AISHELL-3 一/不 surface 不完整 | 主指标可能把正确变调算错 | N0 主分析 + N1 诊断 + 特殊子集单列；不做单一数字定论 |
| 儿化合并/拆分歧义 | 字级对齐错误 | 6.1 的 erhua split 规则；报告 erhua 子集；人工抽查 |
| AISHELL-3 train/test speaker 重叠 | 不能声称 unseen speaker | 明确 zero-shot；只做评估抽样；不写 held-out |
| AISHELL-3 read/TTS 风格 | 不能外推到自发/强口音 | 结论限定 read speech；未来加 WenetSpeech/KeSpeech |
| OMPAL 3 vs 4 raters | 与论文可比性下降 | 明确用发布数据 3 人；不做 4 人 ICC |
| OMPAL 无 consent | 发布/投稿风险 | 标注来源与许可；必要时只内部使用 |
| OMPAL 法语 L1、1,754 条 | 泛化有限 | 结论限定该 learner 子集；不外推所有 L2 |
| B/C 文本简繁/对齐 | 错位 | OpenCC t2s；插入/删除单独报告 |
| B 零声母 consonant | 标签口径不一致 | 单列，不混入主结论 |
| 无时间戳 | 不能做强制对齐 | 逐字/编辑距离对齐；不做音素级时序 |
| 模型输出格式漂移 | 解析失败 | JSON schema + 解析失败率；不静默丢弃 |

## 12. 可复现性

- 抽样：固定 seed；输出 item list、speaker 列表、特殊子集清单。
- 数据版本：AISHELL-3 commit/下载日期；OMPAL repo commit（`phantomhsieh/OMPAL-corpus`）。
- 模型：`qwen3.8-omni-flash` id、业务空间、调用日期、价格快照。
- Prompt：`prompt_version`、全文 sha256；改 prompt 换 version。
- 运行：每个请求的 usage、finish_reason、解析结果、重试；成本汇总。
- 产物（实现阶段）：`sentence_items.jsonl`、`runs/<run_id>/generations.jsonl`、
  `metrics.csv`、`report.md`。本设计阶段只定义字段。

## 13. 未决问题

1. OMPAL 4 vs 3 rater：是否需要联系作者确认？本设计按发布数据 3 人处理。
2. AISHELL-3 vendor EULA：OpenSLR Apache-2.0 是否足够？对外发布前再核。
3. OMPAL consent：发布/投稿是否需要额外说明？
4. 一/不 N1 规则的边界（“一”作序数/单独使用）需要在小样本上人工核对。
5. C2 `max_tokens=1024` 是否足够覆盖句级 reasoning；pilot 看 `finish_reason` 再调。
6. B 的零声母 consonant 标签口径需要 OMPAL 作者/数据抽查确认。

## 14. 附录 A：任务 D——ASR 文本条件下的对齐与评分（v0.2，未运行）

> 本附录是 v0.2 新增；`sentence-v1` 只跑 A/B/C，不包含 D。D 已用独立 run `sentence-d-v1` 完成，
> 结果见 [整句 pilot 报告 §6](sentence-experiment-report.md#6-任务-dasr-文本条件下的对齐与评分v02-附录-a)。
> 目的：真实 ASR 可能自动纠错或误识；给定 ASR 文本后，Qwen 的逐字对齐、错误定位和整句评分会不会退化？

### A.1 RQ-D

- **RQ-D1**：给定音频 + ASR 文本，模型能否逐字判断哪些文本字对应音频、哪些是 ASR 错字/多字？
- **RQ-D2**：ASR 文本错误会不会降低发音错误定位（tone/consonant/vowel）？
- **RQ-D3**：ASR 文本错误会不会降低整句 accuracy/fluency/prosody 评分与人工分的相关性？
- **RQ-D4**：与 gold text 相比，真实 ASR 文本和受控错误文本分别造成多大退化？

D 是“文本条件化”任务，和 A（纯音频自由转写）不同；D 的结论不能替代 A，也不改变 A/B/C 的主结果。

### A.2 数据与文本源

数据：OMPAL 清洗后的 **1,758 条 learner**（同 B/C；见 §3.2）。不使用 native 评分目标。

三种文本源（同一批音频）：

| 条件 | 文本来源 | 作用 |
|---|---|---|
| **D-gold** | OMPAL 原始文本（oracle） | 上界；与 ASR/注入对比 |
| **D-asr** | 固定 ASR 输出 | 真实 ASR 错误、自动纠错 |
| **D-inject-10** | gold 文本 + 10% 字符级错误 | 受控错误；主错误文本条件 |
| D-inject-5 / 20 | gold + 5%/20% 错误 | 可选敏感性 |

ASR 固定为 **FunASR Paraformer-zh**（主）或 **Whisper large-v3**（fallback）；必须是独立于
Qwen3.8-Omni 的 ASR，避免同模型自评。记录 ASR 模型 id、版本/commit、解码参数、运行日期。
ASR 原始输出保留，不做人工纠错；对齐前只做 NFKC、去标点/空格、OpenCC t2s。

注入规则（D-inject）：

- 以 gold 文本的字符为单位，10% 字符被扰动；操作混合约 6:2:2
  （substitution : deletion : insertion）。
- substitution 从同音/近音/常见混淆字表取样；deletion 直接删字；insertion 插入常见字。
- 固定 seed；生成 `d_injected_10.jsonl` 并记录每条扰动位置；5%/20% 只作敏感性，不混入主条件。
- 注入是“受控压力测试”，不声称等价于真实 ASR 错误。

ASR 质量描述（不用于筛选）：

- ASR 文本 vs gold 的字符错误率（CER）、S/D/I 比例；
- **ASR normalization rate**：在 OMPAL 标注为发音错误（0）的 gold 字上，ASR 文本仍等于
  gold 字的比例（描述 ASR 自动纠错的程度，不做因果推断）。

### A.3 任务与输出

输入：音频 + 一条文本（gold / ASR / injected）。文本可能多字、少字、错字。

输出：

```json
{
  "alignment": [
    {"text": "他", "aligned": true, "text_error": false, "error_type": "none", "score": 4.5},
    {"text": "們", "aligned": true, "text_error": false, "error_type": "tone", "score": 3.0}
  ],
  "scores": {"accuracy": 4.3, "fluency": 4.5, "prosody": 4.0}
}
```

字段：

- `text`：按输入文本原样复制（繁体保持繁体）；
- `aligned`：该文本字是否对应音频中的实际发音；
- `text_error`：该文本字本身是否错误（ASR 错字/多字）；`aligned=false` 时通常为 true；
- `error_type`：仅当 `aligned=true` 且 `text_error=false` 时判断，取 `none|tone|consonant|vowel`；
  其他情况填 `null`；
- `score`：该字发音 1–5；无法对齐填 `null`。逐字分数为描述性指标，不做主推断；
- `scores`：整句三维 1–5，允许 1 位小数。

输出数组长度应等于输入文本字数；模型漏字/多字按对齐错误单独统计。

### A.4 Gold 对齐与标签

对每条输入文本 T 与 OMPAL gold 文本 G 做字符级 Levenshtein 对齐：

- match/substitution：T 字对应一个 G 字；`text_correct = (T字 == G字)`；
- insertion：T 字没有 G 对应；
- deletion：G 字没有 T 对应。

由此得到 gold 对齐标签：

- `aligned_gold = (T字有 G 对应)`；
- `text_error_gold = not text_correct`（substitution/insertion）；
- 只有 **text_correct 且 aligned** 的 T 字才进入发音错误定位指标；
- gold 有、T 没有的字（deletion）不进入 T 侧指标，但在 gold-side 覆盖率中报告；
- 发音错误标签用 OMPAL 词级 tone/consonant/vowel 0/1；零声母 consonant 单列。

### A.5 指标

**文本质量（描述）**：ASR CER、S/D/I、ASR normalization rate。

**对齐**：

- `aligned` 的 precision / recall / F1（对 gold aligned）；
- `text_error` 的 precision / recall / F1（对 gold text_error）；
- 输入字覆盖率和 gold-side 覆盖率（漏字/多字）。

**错误定位**（只在 text-correct + aligned 字上）：

- tone / consonant / vowel 的 macro-F1、error-F1、FPR、FNR；
- 与 D-gold 的同口径结果对比；零声母 consonant 单列。

**整句评分**：

- accuracy / fluency / prosody 的 Pearson（主）、Spearman、MAE、RMSE；
- 目标为 OMPAL 3 位评分者均值；native 不进目标；
- D-asr / D-inject 相对 D-gold 的退化差值（配对 bootstrap，按 learner 聚类；
  多指标族内 Holm）。

**逐字分数**：均值、分布、与 OMPAL 0/1 标签的 point-biserial 相关，仅作描述。

### A.6 条件、样本量与成本

- 主条件：**C1（think off / clean / JSON）**；C2 可选，不纳入 D 主表。
- 文本条件：D-gold、D-asr、D-inject-10（主）；D-inject-5/20 可选。
- 调用量：1,758 × 3 = **5,274**（主）；加 5%/20% 为 **8,790**。
- 预算：D 主实验预计 **¥6–10**，上限 **¥15**；加 5%/20% 上限 **¥20**。若沿用
  `sentence-v1` 的 ¥20 总预算（已花 ¥8.29，剩余约 ¥11.7），主条件可跑；不够时缩到
  1,200 条或只跑 gold/asr 两条件；5%/20% 需追加预算。单独记账并在 manifest 标记。
- ASR 推理本地进行，成本/耗时单独记录（不计入 Qwen API 预算）。

### A.7 Prompt（D）

```text
你是普通话发音评测助手。音频是学习者朗读；文本来自 ASR，可能包含错字、多字或少字。
请以音频为准，按输入文本顺序逐字输出 alignment 数组：
- text：原样复制输入字；
- aligned：该字是否对应音频中的实际发音；
- text_error：该字本身是否错误（错字/多字）；
- error_type：若对齐且文本正确，取 none/tone/consonant/vowel，否则填 null；
- score：该字发音 1–5；无法对齐填 null。
并输出整句 scores：accuracy / fluency / prosody，各 1–5，允许 1 位小数。
只输出 JSON：{"alignment":[...],"scores":{...}}，不要解释、不要 Markdown。
```

### A.8 验收与风险

验收：解析成功率 ≥95%；alignment 数组覆盖率 ≥95%；ASR 文本 100% 生成并留档；
D 主实验成本 ≤¥10；不设准确率硬阈值。

风险：

- ASR 自动纠错可能抹掉学习者错误信号；用 D-gold vs D-asr 的退化量描述，不归因单个机制；
- 无时间戳，D 的“对齐”是文本-音频逐字对应，不是 start/end；
- ASR substitution 字的 OMPAL 发音标签不可直接解释，主错误定位只取 text-correct 字；
- 注入错误是受控压力测试，不代表真实 ASR 错误分布；
- ASR 模型版本/许可证、OMPAL 3 位评分者/法语 L1、逐字分数无人工 1–5 gold；
- 文本简繁/标点差异需 OpenCC t2s + 规范化，否则对齐会虚高或虚低。

### A.9 未决问题（D）

> 运行后回答（run `sentence-d-v1`，2026-11-05）：
> 1. ASR 用 FunASR Paraformer-zh（CPU；MPS 在该 FunASR/torch 组合下不可用，已记录 fallback）；
> 2. 主错误率 10% 已跑，5%/20% 未跑（与设计一致，仅敏感性）；
> 3. 逐字 `score` 保留，仅作描述统计（无人工 gold）；
> 4. C2 未纳入 D（主表只做 C1）；
> 5. ASR 质量 pilot：CER(micro) 0.200、S/D/I 2361/55/1509、normalization rate 0.665。


1. 实际可用的 ASR 是 Paraformer-zh 还是 Whisper large-v3？需固定版本与解码。
2. D-inject 主错误率用 10% 是否合适？5/20 是否值得跑。
3. 逐字 `score` 是否保留？OMPAL 只有 0/1 标签，逐字 1–5 无直接 gold。
4. C2 是否纳入 D；若纳入，thinking 预算与成本需重估。
5. ASR 对 learner 语音的 CER 分布未知，需先小样本 pilot 确认文本源质量。

## 15. 参考

- AISHELL-3: <https://www.openslr.org/93/>；论文 arXiv:2010.11567。
- OMPAL: <https://github.com/phantomhsieh/OMPAL-corpus>；Interspeech 2025。
- Qwen3.8-Omni 百炼文档：<https://help.aliyun.com/zh/model-studio/qwen-omni>。
- 现有单音节项目：`qwen3.8 pinyin/docs/experiment-design.md`、
  `docs/minimal-experiment-report.md`、`docs/model-weakness-comparison.md`。
- 评测指标参考：speechocean762（PCC/MAE 口径）；CAPT review（El Kheir et al.）。
