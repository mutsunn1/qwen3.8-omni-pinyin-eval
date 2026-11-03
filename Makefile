# 统一环境与常用命令（所有 Python 操作都通过 uv，无需手动激活 venv）
UV ?= uv
CONFIG ?= configs/base.yaml

.PHONY: help setup check data tasks mock test lint ci clean \
	sentence-data sentence-check sentence-estimate sentence-mock \
	sentence-generate sentence-score sentence-report

help:                       ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

setup:                      ## 创建/同步 Python 3.12 环境，并生成 .env（若不存在）
	@command -v $(UV) >/dev/null || { echo "需要 uv（https://docs.astral.sh/uv/）"; exit 1; }
	$(UV) sync --locked
	@test -f .env || (cp .env.example .env && echo "已从 .env.example 创建 .env，请填入 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL")

check:                      ## 环境 / 数据 / 模型配置自检
	$(UV) run pinyin-eval check

data:                       ## 下载数据 + 导出音频 + 构造任务（离线可重复）
	$(UV) run pinyin-eval prepare-data

tasks: data                 ## 构造实验任务并校验最小对计数
	$(UV) run pinyin-eval build-tasks --validate

mock: tasks                 ## 无 API key 的全链路冒烟（mock client）
	$(UV) run pinyin-eval all-mock --audios 40

test:                       ## 运行全部离线测试
	$(UV) run pytest

lint:                       ## ruff 静态检查
	$(UV) run ruff check src tests

ci: lint test               ## CI 等价命令

# ---------------------------------------------------------------------------
# 整句 pilot（docs/sentence-experiment-design.md；模型产物在 results/sentence/）
# ---------------------------------------------------------------------------
SENTENCE_CONFIG ?= configs/sentence.yaml
SENTENCE_RUN ?= sentence-v1

sentence-data:              ## 整句 pilot：AISHELL-3 抽样 + OMPAL 清洗 + 音频下载
	$(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) prepare

sentence-check:             ## 整句 pilot：数据 / 模型 / 计划调用自检
	$(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) check

sentence-estimate:          ## 整句 pilot：预计调用量与成本（expected / worst-case）
	$(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) estimate

sentence-mock:              ## 整句 pilot：无 API key 的离线全链路（mock 7,796 次调用）
	$(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) generate --mock --run-id mock-$(SENTENCE_RUN) \
		&& $(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) score --run-id mock-$(SENTENCE_RUN) \
		&& $(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) cost --run-id mock-$(SENTENCE_RUN) \
		&& $(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) report --run-id mock-$(SENTENCE_RUN)

sentence-generate:          ## 整句 pilot：真实 C1/C2 生成（可续跑 + 硬预算）
	$(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) generate --run-id $(SENTENCE_RUN)

sentence-score:             ## 整句 pilot：解析 + 逐项评分
	$(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) score --run-id $(SENTENCE_RUN)

sentence-report:            ## 整句 pilot：指标 / 表格 / SVG / 验收报告
	$(UV) run pinyin-eval sentence --config $(SENTENCE_CONFIG) report --run-id $(SENTENCE_RUN)

clean:                      ## 清理可再生产物与缓存（保留 data/raw 与 data/mapping）
	rm -rf results/runs data/audio data/noisy data/manifests .pytest_cache .ruff_cache
