# 讯飞 ISE（语音评测流式版）基线 · 详细信息

> 本文是 [baseline-ise.md](baseline-ise.md) 的完整版：接口契约、鉴权、适配实验、
> 任务清单、交互账目、全部结果与复现方式。
> 日期：2026-10-30 · 脚本：[scripts/eval_xfyun_ise.py](../scripts/eval_xfyun_ise.py)
> · 探针：[scripts/probe_xfyun_ise.py](../scripts/probe_xfyun_ise.py)

---

## 0. 一句话

讯飞 ISE 是**参考打分** API（音频 + 参考试卷 → 分数/错误定位），我们用“静音填充 +
拼音标注三字试卷 + 2–4 候选择优”把它适配成可与 Qwen 比较的基线；在 mini128 子集上
tone 0.922、7 类最小对平均 0.962、检测 AUC 0.909（与 Qwen 派生检测 0.910 基本持平），
共消耗 **5,843 / 10,000** 次交互。

---

## 1. 接口契约

### 1.1 端点与鉴权

| 项 | 值 |
|---|---|
| 协议 | WebSocket（wss） |
| 地址 | `wss://ise-api.xfyun.cn/v2/open-ise` |
| 鉴权方式 | 握手 URL query：`authorization` + `host` + `date` |
| 签名算法 | HMAC-SHA256，headers 固定为 `host date request-line` |
| 音频 | 16 kHz、16-bit、单声道 PCM（`aue=raw`，`auf=audio/L16;rate=16000`） |
| 单次时长上限 | 5 分钟 |
| 单次评测 | 一次连接 = 一个 session；`ssb` 参数帧 → 若干 `auw` 音频帧 → 结果 |

签名原文（`\n` 为真实换行，冒号后有空格）：

```
host: ise-api.xfyun.cn
date: Wed, 30 Oct 2026 11:00:00 GMT
GET /v2/open-ise HTTP/1.1
```

```
signature = base64(hmac_sha256(signature_origin, APISecret))
authorization_origin = api_key="<APIKey>", algorithm="hmac-sha256",
                       headers="host date request-line", signature="<signature>"
authorization = base64(authorization_origin)
URL = wss://ise-api.xfyun.cn/v2/open-ise
      ?authorization=<urlencode(authorization)>&host=ise-api.xfyun.cn&date=<urlencode(date)>
```

对应脚本函数 `auth_url()`（`scripts/eval_xfyun_ise.py`）。`date` 用 RFC1123/GMT。

### 1.2 会话流程

```
客户端                                 服务端
  │ 1) ssb 参数帧                          │
  │  {common:{app_id}, business:{cmd:"ssb", sub:"ise", ent:"cn_vip",
  │    category:"read_syllable", text:"\ufeff一<ai4>一一", tte:"utf-8",
  │    ttp_skip:true, aue:"raw", auf:"audio/L16;rate=16000",
  │    rstcd:"utf8", rst:"entirety", ise_unite:"1",
  │    extra_ability:"syll_phone_err_msg", plev:"0", group:"adult"},
  │   data:{status:0}}                     │
  │ ─────────────────────────────────────▶ │
  │ 2) auw 音频帧（base64 PCM，40ms/帧）    │
  │  首帧: business:{cmd:"auw",aus:1}, data:{status:1,data:...}
  │  中间: aus:2, status:1
  │  末帧: aus:4, status:2                 │
  │ ─────────────────────────────────────▶ │
  │ 3) 结果（可能多帧；status=2 为最终）    │
  │ ◀───────────────────────────────────── │
  │  {code:0, sid, data:{status:2, data:"<base64 XML>"}}
```

`text` 必须是 UTF-8 BOM（`\uFEFF`）开头。`ttp_skip=true` 时文本只在 `ssb` 中传一次；
否则要先走 `cmd=ttp` 文本帧。

### 1.3 返回的 XML（真实样例，截断）

```xml
<?xml version="1.0" encoding="UTF-8"?>
<xml_result>
  <read_syllable lan="cn" type="study" version="7,0,0,1024">
    <rec_paper>
      <read_syllable beg_pos="0" content="爱。" end_pos="50"
          except_info="0" fluency_score="0.000000" integrity_score="0.000000"
          phone_score="50.000000" time_len="50" tone_score="100.000000"
          total_score="75.000000">
        <sentence beg_pos="0" content="爱" end_pos="50" time_len="50">
          <word beg_pos="0" content="爱" end_pos="50" symbol="ai4" time_len="50">
            <syll beg_pos="3" content="爱" dp_message="0" end_pos="35"
                  rec_node_type="paper" symbol="ai4" time_len="32">
              <phone beg_pos="3" content="_a" dp_message="0" end_pos="6"
                     is_yun="0" perr_level_msg="0" perr_msg="1"
                     rec_node_type="paper" time_len="3"/>
              <phone beg_pos="6" content="ai" dp_message="0" end_pos="35"
                     is_yun="1" mono_tone="TONE4" perr_level_msg="0"
                     perr_msg="0" rec_node_type="paper" time_len="29"/>
            </syll>
          </word>
        </sentence>
      </read_syllable>
    </rec_paper>
  </read_syllable>
</xml_result>
```

关键字段：

| 字段 | 含义 |
|---|---|
| `read_syllable.total_score` | 本卷总分（声/韵/调综合） |
| `read_syllable.phone_score` | 声韵分（声母+韵母） |
| `read_syllable.tone_score` | 调型分 |
| `read_syllable.integrity_score` | 完整度（单字卷常为 0，因占位字漏读） |
| `phone.content` | 参考音素（`_a` 表零声母起始、`ai` 表韵母） |
| `phone.is_yun` | 0=声母，1=韵母 |
| `phone.perr_msg` | 0 正确；1 声韵错；2 调型错；3 声韵+调型错 |
| `phone.dp_message` | 0 正常；16 漏读；32 增读；64 回读；128 替换 |
| `phone.mono_tone` | 参考/对齐的单字调（TONE1–4）——注意它跟随参考文本，不是独立识别结果 |
| `syll.symbol` | 规范拼音（带数字调） |

### 1.4 错误码（实测与文档相关）

| code | 含义 | 处理 |
|---|---|---|
| `0` | 成功 | — |
| `48195` | 试卷格式错误/未设置（如单字+拼音标注比例 >1/3、拼音-only 试卷） | 改成合法的三字标注试卷 |
| `10163` | 参数校验失败 | 检查 business 字段 |
| `40007` / `10043` | 音频解码失败 | 确认 raw PCM16 / 16k / mono |
| `10200` | 10 秒未发数据超时 | 会话内及时发送 |
| `11201` | 额度用尽 | 需购买/充值 |

> 本次 5,843 次成功交互中最终无失败；早期调试遇到过 `48195`（格式问题）。

---

## 2. 适配：从“打分 API”到“可比较基线”

### 2.1 难点

1. **单字卷的拼音标注限制**：单字 + 拼音标注会被拒（标注汉字数 ≤ 全卷 1/3）；
   纯拼音试卷也不被解析（实测 `ai4` 被错误映射成别的字）。
2. **短音频分数退化**：数据集音频仅 0.2–0.6 s，直接送评时 `phone_score` 掉到 0。

### 2.2 两个修正（均有对照实测）

**① 静音填充**：对每条音频前后各补 300 ms 静音（`PAD_S=0.3`）：

| 音频 | 参考 | 原始音频 total/phone | 填充后 total/phone |
|---|---|---:|---:|
| `an1` | `an1` | 50 / 0 | **100 / 100** |
| `an1` | `ang1`（错） | 0 / 0 | **75 / 50** |
| `zhang1` | `zhang1` | — | 100 / 100 |
| `zhang1` | `zang1`（错） | — | 75 / 50 |

> 填充后“正确候选 > 错误候选”的排序稳定，且 `phone_score` 可用了。

**② 三字拼音标注试卷**：送

```
\ufeff一<ai4>一一
```

即 3 个汉字、第 1 个用 `<拼音>` 标注（1/3 上限内），ISE 按 `ai4` 评测第一个音节，
后两字记为漏读。绝对分被稀释，但同结构候选之间可比。

**③ 候选打分**：

| 任务 | 候选 | 选择规则 |
|---|---|---|
| 声调分类 | 同一音节 4 个声调（`xxx1`–`xxx4`） | `total_score` 最高 |
| 设计最小对 | 本侧音 vs 对侧音（同声调） | `total_score` 最高 |
| SDHY 弱对 | 同上 | `total_score` 最高 |
| 检测 | 声称的 target+tone | `inconsistency = -total_score`，阈值 -25 |

**④ 拼音 surface 规范化**（否则 ISE 词典对不上）：

* `j/q/x + ü系`：`ju/jue/juan/jun`；
* `l/n + ü系`：`lv/nv`、`lve/nve`；
* 零声母：`yi/ya/ye/yao/you/yan/yin/yang/ying/yong/yu/yue/yuan/yun/wu/wa/wo/wai/wei/wan/wen/wang/weng`。

---

## 3. 评测任务与交互账目

### 3.1 数据

mini128 子集（2,048 条音频，128 target × 4 voice × 4 tone）；最小对 7 类 × 64 实例；
检测 256 一致 + 256 不一致 claim（从 mini128 的 2,048 音频对应的 detection_sets main 中抽，
避免用了不在本 run 的音频——早期版本这里浪费了 ~358 次交互）。

### 3.2 计划调用（去重前）

| 任务 | 明细 | 行数 |
|---|---|---:|
| 声调分类 | 512 音频（每 voice×tone 32）× 4 候选 | 2,048 |
| 设计最小对 | 864 音频侧 × 2 候选 | 1,728 |
| SDHY 弱对 | 15 对 × 64 音频 × 2 候选 | 1,856 |
| 检测 | 512 claim × 1 参考 | 512 |
| **合计（去重前）** | | **6,144** |

**去重后最终计划 unique calls = 5,354**（同一 audio+ref 在不同任务复用，如 tone 与
tone 弱对、design pair 与 weak pair）。

### 3.3 实际交互

| 项 | 数值 |
|---|---:|
| 成功交互（cache 成功行） | **5,843** |
| 其中属于最终计划 | 5,354 |
| 早期调试/被替换的样本（smoke + 旧 detection 采样） | 489 |
| 最终失败 | 0 |
| 余额 | **10,000**（剩余 ≈4,157） |
| 并发 | 6（单次约 0.5–1.5 s；5,000+ 次约 25 分钟） |

缓存文件 `data/ise/cache.jsonl`（每行 audio+ref+分数，成功才入库），支持断点续跑；
`--reuse-only` 只重算指标不调用 API。

---

## 4. 完整结果

### 4.1 声调分类（512 音频 × 4 候选）

| 指标 | 值 |
|---|---:|
| Accuracy | **0.9219** |
| 逐 gold 声调 | 1 声 0.992 / 2 声 0.961 / **3 声 0.750** / 4 声 0.984 |
| 逐 voice | A 0.969 / **B 0.805** / C 0.938 / D 0.977 |

混淆矩阵（行=gold，列=pred）：

| gold\pred | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 1 | 127 | 1 | 0 | 0 |
| 2 | 2 | 123 | 3 | 0 |
| 3 | 7 | **25** | 96 | 0 |
| 4 | 2 | 0 | 0 | 126 |

> 三声是明显弱项：128 个三声中 25 个被判成二声；B 声线（0.805）也显著低于其他 voice。
> 对照 Qwen（同 512 音频自由识别）：**0.904**，三声同样是最弱调但混淆更分散。

### 4.2 设计里的 7 类最小对（同 864 音频侧，free 条件）

| 类型 | n | ISE | Qwen | 胜者 |
|---|---:|---:|---:|---|
| sh/zh 平翘舌 | 128 | **0.992** | **0.992** | 平 |
| 送气 aspiration | 128 | **1.000** | 0.969 | ISE |
| n/l nasal_lateral | 128 | **0.969** | 0.945 | ISE |
| 前后鼻 FBN | 128 | 0.906 | **0.961** | Qwen |
| i/ü | 128 | 0.938 | **0.984** | Qwen |
| ian/üan | 96 | **0.948** | 0.906 | ISE |
| 声调最小对 | 128 | **0.977** | 0.945 | ISE |
| **总平均** | 864 | **0.962** | — | — |

### 4.3 SDHY 基线的弱对（每对 64 音频，平衡两侧）

| 类型 | 对 | n | ISE | Qwen | 胜者 |
|---|---|---:|---:|---:|---|
| initial | m/n | 64 | 0.812 | **0.859** | Qwen |
| initial | n/l | 64 | 0.906 | **0.922** | Qwen |
| initial | d/t | 64 | 0.922 | **0.984** | Qwen |
| initial | r/l | 48 | 0.854 | **0.938** | Qwen |
| initial | s/sh | 48 | **1.000** | **1.000** | 平 |
| initial | z/zh | 64 | **0.984** | 0.891 | ISE |
| final | üe/ie | 64 | 0.688 | **0.875** | Qwen |
| final | ei/ie | 64 | 0.734 | **0.906** | Qwen |
| final | o/uo | 64 | **0.953** | 0.812 | ISE |
| final | ia/ie | 64 | 0.828 | **0.969** | Qwen |
| final | in/ing | 64 | 0.625 | **0.969** | Qwen |
| final | en/eng | 64 | **1.000** | 0.922 | ISE |
| tone | 1/3 | 64 | **0.953** | 0.922 | ISE |
| tone | 2/3 | 64 | **0.875** | 0.781 | ISE |
| tone | 3/4 | 64 | **0.984** | 0.953 | ISE |

### 4.4 检测（256 一致 + 256 不一致 claim）

`inconsistency score = -total_score`；阈值 -25 在 calib 半区（偶数行）学习，eval 半区评估。

| 半区 | n | TP | FP | TN | FN | Error-F1 | FPR | FNR | Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calib | 256 | 123 | 24 | 104 | 5 | 0.895 | 0.188 | 0.039 | 0.887 |
| **eval** | 256 | **122** | **17** | **111** | **6** | **0.914** | **0.133** | **0.047** | **0.910** |

AUC（全 512） = **0.909**；分数分布：一致 claim 的 `-total` 中位 -33.3，
不一致 claim 中位 -25.0（越高越可能不一致）。

对照：

| 判定方式 | AUC | Eval F1 | Eval FPR |
|---|---:|---:|---:|
| 讯飞 ISE（阈值） | 0.909 | 0.914 | 0.133 |
| Qwen 派生检测（自由识别+比较） | **0.910** | **0.923** | 0.141 |
| SDHY P0-v2 派生检测（全量口径参考） | — | 0.731 | 0.701 |

> 检测上 ISE 与 Qwen 几乎打平；两者都远好于“模型直接判一致”的 0.72。

---

## 5. 复现

```bash
# 凭证（不要提交）：.env 中
#   XFYUN_APPID=...
#   XFYUN_APIKEY=...
#   XFYUN_APISECRET=...
set -a && . ./.env && set +a

# 1) 单条探针（打印 XML）
.venv/bin/python scripts/probe_xfyun_ise.py \
  --audio data/audio/ZH_TTS_A/ZH_TTS_A_ai4.wav --pinyin ai4

# 2) 完整评测（自动缓存/续跑；约 5.8k 次交互 ≈ 25 分钟）
.venv/bin/python scripts/eval_xfyun_ise.py --concurrency 6

# 3) 只重算指标（不再调用 API）
.venv/bin/python scripts/eval_xfyun_ise.py --reuse-only

# 4) 看 smoke
.venv/bin/python scripts/eval_xfyun_ise.py --smoke --concurrency 4
```

产物：

```
results/ise-baseline.json          # 全部指标
results/ise-baseline.tone.csv      # 512 音频逐条
results/ise-baseline.pairs.csv     # 1,728 行候选
results/ise-baseline.weak.csv      # 1,856 行
results/ise-baseline.detect.csv    # 512 claim
results/ise-vs-qwen.json           # 与 Qwen 的同音频对比
data/ise/cache.jsonl               # 调用缓存（gitignored）
data/ise/padded/                   # 静音填充缓存（gitignored）
```

---

## 6. 局限与结论

1. **不是自由识别**：ISE 需要参考文本；本文的“候选打分”只覆盖声调/最小对/检测，
   不能把 0.922 直接写成“自由识别准确率”。
2. **候选集有限**：声调 4 候选、最小对 2 候选；没有做 378 音节全候选（交互不够）。
3. **绝对分被稀释**：三字标注卷的两个占位字会计入完整度/漏读，只应比较相对排序。
4. **三声/B 声线弱**：tone3 0.750、voice B 0.805——商用评测在短 TTS 单音节上也存在
   明显短板。
5. **强项明确**：送气、n/l、平翘舌、ian/üan、声调最小对、en/eng、o/uo、z/zh、
   2/3 与 3/4 声调判别上优于 Qwen。
6. **剩余额度** 4,157 次：建议用于 in/ing、üe/ie、ei/ie、ia/ie 等弱对加样本复核，
   或对 846–1,000 条音频做 4 候选全量声调（约 3.4–4k 次）。
7. 凭证已在对话中出现，建议在讯飞控制台轮换；本地只存 `.env`（gitignore）。

---

## 7. 性价比：ISE vs Qwen（本任务口径）

> 价格来源：[讯飞语音评测服务页](https://www.xfyun.cn/services/ise)，2026-10-30 套餐价。
> Qwen 成本为本次真实 run 的 API 记账（`usage` 实算），非估算。

### 7.1 单价

| 项 | 单价 | 备注 |
|---|---:|---|
| Qwen C1 全任务平均 | **¥0.000116 / 次** | 29,096 次实测 ¥3.3839 |
| Qwen 自由识别单次（含 initial/final/tone） | **¥0.000160 / 次** | exp3 free，一次调用出三维 |
| 讯飞 ISE（20 万次套餐） | **¥0.006000 / 次** | 60 元/万次（¥1200/20 万次） |
| 讯飞 ISE（150 万次套餐） | ¥0.003870 / 次 | 38.7 元/万次 |
| 讯飞 ISE（1 亿次套餐） | **¥0.001500 / 次** | 15 元/万次（最便宜档） |

**单次调用 ISE 是 Qwen 的 9.4–37.5 倍**；按任务（需要多候选）差距更大。

### 7.2 任务级成本（同样本量）

| 任务 | Qwen 成本 | 讯飞 ISE（20 万套餐） | 讯飞 ISE（1 亿套餐） | 倍数（小套餐/大套餐） |
|---|---:|---:|---:|---:|
| 声调分类，512 音频（Qwen 1 次/音频；ISE 4 候选） | **¥0.082** | ¥12.29 | ¥3.07 | **150× / 37.5×** |
| 最小对 864 音频侧（Qwen 1 次/侧；ISE 2 候选） | **¥0.138** | ¥10.37 | ¥2.59 | **75× / 18.8×** |
| 检测 512 claim（Qwen 派生 0 额外调用；ISE 1 次/claim） | **¥0（复用自由识别）** | ¥3.07 | ¥0.77 | ∞ / ∞ |
| 本次 ISE 全部 5,843 次交互 | — | **¥35.06** | **¥8.76** | — |
| 对照：Qwen mini128 完整 4 条件 run | **¥16.78 / 79,692 次** | — | — | Qwen 更便宜且覆盖多条件 |

> 注：本次 ISE 交互余额来自已有额度（若为免费包则现金成本 ¥0）；上表按**替换价格**
> 计算，用于判断规模化成本。

### 7.3 精度对比（同子集）

| 指标 | Qwen | 讯飞 ISE | 差异 |
|---|---:|---:|---|
| Tone（512 音频） | 0.904 | **0.922** | +1.8pp，McNemar p=0.328（不显著） |
| 7 类最小对（864 侧） | 0.959（加权） | **0.962** | +0.3pp |
| 检测 AUC | **0.910** | 0.909 | ≈0 |
| 检测 Eval F1 / FPR | **0.923 / 0.141** | 0.914 / **0.133** | Qwen F1 略高，ISE FPR 略低 |
| 自由识别（initial/final/tone） | **有**（一次调用） | **不支持**（需参考文本） | Qwen 独有 |

**精度基本打平**，没有一项是 ISE 显著优于 Qwen；Qwen 还免费附带自由识别与派生检测。

### 7.4 结论

1. **本任务（短单音节 + 识别/检测）Qwen 性价比高约 1–2 个数量级**：
   单次便宜 9–37 倍，任务级便宜 19–150 倍，精度还基本持平。
2. **ISE 的合理定位**：小样本“商用参考/校准”、reference-known 的发音打分、
   或作为某些弱对（en/eng、o/uo、2/3 声、in/ing 等）的**交叉验证**；不适合作为
   大规模自由识别 baseline。
3. **规模化的盈亏平衡**：声调 4 候选场景下，ISE 需降到 **¥0.00004/次** 才与
   Qwen 单次成本持平（现最低 ¥0.0015，仍差 37.5 倍）；最小对 2 候选需
   **¥0.00008/次**（现差 18.8 倍）。
4. **ISE 可能更划算的场景**：长音频/长试卷（ISE 按次固定价，Qwen 输入 token 随
   音频时长线性增长）——例如 5 分钟音频 Qwen 音频输入约 2100 token（≈¥0.0017
   输入成本）已接近 ISE 最便宜档；但本数据集的 0.2–0.6s 单音节完全不在该区间。
5. 已有 4,157 次余额建议只用于**定向弱对复核**（按替换价约 ¥25，但现金已沉没），
   不再为规模化买 ISE 额度。
