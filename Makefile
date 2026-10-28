# 统一环境与常用命令（所有 Python 操作都通过 uv，无需手动激活 venv）
UV ?= uv
CONFIG ?= configs/base.yaml

.PHONY: help setup check data tasks mock test lint ci clean

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

clean:                      ## 清理可再生产物与缓存（保留 data/raw 与 data/mapping）
	rm -rf results/runs data/audio data/noisy data/manifests .pytest_cache .ruff_cache
