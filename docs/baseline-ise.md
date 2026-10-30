# 基线：讯飞语音评测（ISE 流式版）在 prosodic-abx 子集上的结果

> 目的：用讯飞开放平台「语音评测（流式版）」作为**第二个外部基线**，与
> Qwen3.8-Omni、SDHY P0-v2 在最小化实验同一批音频上对比。
> 日期：2026-10-30 · 评测脚本：[scripts/eval_xfyun_ise.py](../scripts/eval_xfyun_ise.py)
> · 协议探针：[scripts/probe_xfyun_ise.py](../scripts/probe_xfyun_ise.py)

## 1. 接口与鉴权（已实测打通）

| 项 | 内容 |
|---|---|
| 文档 | [讯飞语音评测（流式版）API](https://static.xfyun.cn/doc/Ise/IseAPI.html) |
| 端点 | `wss://ise-api.xfyun.cn/v2/open-ise` |
| 鉴权 | 握手 URL 上带 `authorization`（HMAC-SHA256 over `host/date/request-line`）、`host`、`date`；APIKey/APISecret 从 `.env` 读取，不入库 |
| 参数上传 | `business{cmd=ssb, sub=ise, ent=cn_vip, category=read_syllable, aue=raw, auf=audio/L16;rate=16000, tte=utf-8, ttp_skip=true, rst=entirety, ise_unite=1}` + `data.status=0` |
| 音频上传 | 16 kHz / 16-bit / 单声道 PCM，base64；首帧 `cmd=auw, aus=1, status=1`，中间 `aus=2`，末帧 `aus=4, status=2` |
| 结果 | `data.data` base64 → XML；取 `<read_syllable total_score/tone_score/phone_score>` 与逐 phone 的 `perr_msg/dp_message` |
| 实测 | smoke 已验证 200/0 错误；正式评测 **5,843 次成功交互**（余额 10,000，剩余约 4,157） |

## 2. 适配：把打分 API 变成可比较的基线

ISE 不是自由识别 API，而是“给音频 + 参考试卷 → 打分”。我们采用**候选打分**：

1. **静音填充**：原始单音节音频仅 0.2–0.6 s，直接送评会出现退化分数；在前后各补
   300 ms 静音（实测 `an1` 对正确候选从 0 分恢复到 100 分）；
2. **拼音标注试卷**：单字+拼音标注会被拒绝（标注汉字数须 ≤ 全卷 1/3），所以送
   `\ufeff一<ai4>一一`（3 个字、仅第 1 个标注）——ISE 按标注读音评测第一个音节；
   两个占位字被记为漏读，绝对分被稀释，但候选间排序稳定；
3. **候选选择**：
   * **声调分类**：同一音节 4 个声调候选 → 取 `total_score` 最高者；
   * **设计最小对**：每侧音频 2 个候选（本侧 vs 对侧成员，同声调）→ 取高者；
   * **SDHY 弱对**：同样 2 候选；
   * **检测**：参考=声称 target+声调，`inconsistency score = -total_score`；
     用一半样本校准阈值，另一半评估（AUC 与阈值无关）。
4. 候选 pinyin 按规范折算：`ü→u`（j/q/x）、`lü/nü→lv/nv`、零声母 `yi/ya/ye/yan/
   yue/yuan/wu/wo/…`。

## 3. 结果

### 3.1 声调分类（512 音频，32/voice/tone，4 候选）

| 模型 | Tone accuracy |
|---|---:|
| **讯飞 ISE（候选打分）** | **0.922** |
| Qwen3.8-Omni（C1 自由识别） | 0.904 |

> ISE 逐 voice：A 0.969 / B **0.805** / C 0.938 / D 0.977——B 声线明显偏弱。

### 3.2 设计里的 7 类最小对（free 条件，同一批音频侧）

| 最小对类型 | n | 讯飞 ISE | Qwen（C1） | 胜者 |
|---|---:|---:|---:|---|
| 平翘舌 sh/zh | 128 | **0.992** | **0.992** | 平 |
| 送气 aspiration | 128 | **1.000** | 0.969 | ISE |
| n/l nasal_lateral | 128 | **0.969** | 0.945 | ISE |
| 前后鼻 an/ang 等 | 128 | 0.906 | **0.961** | Qwen |
| i/ü | 128 | 0.938 | **0.984** | Qwen |
| ian/üan | 96 | **0.948** | 0.906 | ISE |
| 声调最小对 | 128 | **0.977** | 0.945 | ISE |

### 3.3 SDHY 基线的弱对（每对 64 音频，同口径）

| 类型 | 对 | n | 讯飞 ISE | Qwen |
|---|---|---:|---:|---:|
| initial | m/n | 64 | 0.812 | **0.859** |
| initial | n/l | 64 | 0.906 | **0.922** |
| initial | d/t | 64 | 0.922 | **0.984** |
| initial | r/l | 48 | 0.854 | **0.938** |
| initial | s/sh | 48 | **1.000** | **1.000** |
| initial | z/zh | 64 | **0.984** | 0.891 |
| final | üe/ie | 64 | 0.688 | **0.875** |
| final | ei/ie | 64 | 0.734 | **0.906** |
| final | o/uo | 64 | **0.953** | 0.812 |
| final | ia/ie | 64 | 0.828 | **0.969** |
| final | in/ing | 64 | 0.625 | **0.969** |
| final | en/eng | 64 | **1.000** | 0.922 |
| tone | 1/3 | 64 | **0.953** | 0.922 |
| tone | 2/3 | 64 | **0.875** | 0.781 |
| tone | 3/4 | 64 | **0.984** | 0.953 |

### 3.4 检测（256 一致 + 256 不一致 claim；同批 claim 对比）

| 指标 | 讯飞 ISE | Qwen 派生检测 | SDHY 派生检测（参考） |
|---|---:|---:|---:|
| AUC | **0.909** | 0.910 | — |
| Eval half Error-F1 | 0.914 | **0.923** | 0.731（全量口径） |
| Eval half FPR | **0.133** | 0.141 | 0.701 |
| Eval half FNR | 0.047 | **0.023** | — |
| 阈值 | `-total_score ≥ -25`（校准半区 F1 0.895） | 规则判定（无需阈值） | 规则判定 |

> 结论：ISE 与 Qwen 派生检测 AUC 几乎相同（0.909 vs 0.910），F1 差 1pp；
> 两者都显著优于 SDHY 的 P0-v2 派生口径。

## 4. 解读

* **ISE 在“成对/候选判别”上非常强**：送气、平翘舌、n/l、ian/üan、声调最小对
  都与 Qwen 持平或更高；对 SDHY 弱对中的 en/eng、o/uo、z/zh、2/3 声也明显更好。
* **ISE 的弱点**：in/ing（0.625）、üe/ie（0.688）、ei/ie、ia/ie 以及 B 声线的声调
  （0.805）——这些恰好是短音节+近音韵母，候选打分也难区分。
* **Qwen 的优势**：in/ing、üe/ie、ei/ie、前后鼻等更稳；检测派生口径略好（F1/FNR）。
* **注意事项（重要）**：
  1. ISE 是**参考打分**，本文用“2–4 候选强制选择”适配；它比自由识别容易，
     不能把分数直接当作“自由识别准确率”与 Qwen 等同比较；
  2. 候选集合只覆盖声调/最小对，不是全 378 音节；
  3. 绝对分数被 3 字试卷稀释，论文中只应引用**相对比较/胜率**；
  4. 交互余额 10,000，已用 5,843，剩余约 4,157（可做补充实验）。

## 5. 复现

```bash
# 凭证写在 .env（XFYUN_APPID / XFYUN_APIKEY / XFYUN_APISECRET；已被 gitignore）
set -a && . ./.env && set +a

# 单条探针
.venv/bin/python scripts/probe_xfyun_ise.py \
  --audio data/audio/ZH_TTS_A/ZH_TTS_A_ai4.wav --pinyin ai4

# 完整基线（续跑自动复用 data/ise/cache.jsonl）
.venv/bin/python scripts/eval_xfyun_ise.py --concurrency 6
# 只重算指标、不再调用 API：
.venv/bin/python scripts/eval_xfyun_ise.py --reuse-only
```

产物：`results/ise-baseline.json` + `.tone.csv/.pairs.csv/.weak.csv/.detect.csv`，
对比汇总 `results/ise-vs-qwen.json`（results/ 已 gitignore）。
