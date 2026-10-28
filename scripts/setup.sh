#!/usr/bin/env bash
# 统一环境引导：安装 uv（如需）-> 同步依赖 -> 生成 .env -> 自检
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] 未检测到 uv，正在安装 ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] python: $(uv python find 3.12 2>/dev/null || echo '将由 uv 自动下载 3.12')"
uv sync --locked

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[setup] 已创建 .env（请填写 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL）"
fi

uv run pinyin-eval check || true
echo "[setup] 完成。下一步：编辑 .env 后运行 \`uv run pinyin-eval smoke\`"
