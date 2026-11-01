"""Sync the Obsidian ``study/`` notes into the repository's ``study/`` folder.

The Obsidian vault is the authoring surface; GitHub is the shared readable copy.
This script copies the five experiment notes verbatim (except link rewrites) and
regenerates ``study/README.md`` as an index.

Link rewrites:

* Obsidian wiki links ``[[Qwen 3.8 omni X]]`` -> relative Markdown links to the
  corresponding file in this folder (percent-encoded spaces);
* repo-relative links that would break inside ``study/`` are pointed back to the
  canonical ``docs/`` file (e.g. ``baseline-ise.md`` -> ``../docs/baseline-ise.md``).

Usage::

    .venv/bin/python scripts/sync_study_docs.py \
        --vault-dir "$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/study"

The vault path defaults to ``$OBSIDIAN_STUDY_DIR`` so it can be exported once.
"""

from __future__ import annotations

import argparse
import os
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: vault file name -> (canonical repo doc, one-line description for the index)
DOCS: dict[str, tuple[str, str]] = {
    "Qwen 3.8 omni 音素识别实验设计.md": (
        "docs/experiment-design.md",
        "完整实验设计（v3 设计快照，2026-10-28）",
    ),
    "Qwen 3.8 omni 最小化实验设计.md": (
        "docs/experiment-design-minimal.md",
        "¥70 最小化实验设计（v3，含两档样本与成本估算）",
    ),
    "Qwen 3.8 omni 最简实验报告.md": (
        "docs/report-minimal.md",
        "一页结论：主结果、消融、双外部基线、成本（2026-10-30）",
    ),
    "Qwen 3.8 omni 讯飞 ISE 基线详情.md": (
        "docs/baseline-ise-detail.md",
        "讯飞 ISE 详细基线：鉴权/协议/适配/交互账目/全量结果/性价比",
    ),
    "Qwen 3.8 omni 三模型弱项对比.md": (
        "docs/model-weakness-comparison.md",
        "Qwen / SDHY / 讯飞 ISE 三模型弱项与逐对对比",
    ),
}

#: short label used when rewriting a wiki link
SHORT = {
    "Qwen 3.8 omni 音素识别实验设计": "完整实验设计",
    "Qwen 3.8 omni 最小化实验设计": "最小化实验设计",
    "Qwen 3.8 omni 最简实验报告": "最简实验报告",
    "Qwen 3.8 omni 讯飞 ISE 基线详情": "讯飞 ISE 基线详情",
    "Qwen 3.8 omni 三模型弱项对比": "三模型弱项对比",
}

#: relative links inside the vault notes that must point back into docs/
LINK_FIXES = {
    "(experiment-design.md)": "(Qwen%203.8%20omni%20音素识别实验设计.md)",
    "(baseline-ise.md)": "(../docs/baseline-ise.md)",
}


def rewrite(text: str) -> str:
    for name, label in SHORT.items():
        target = urllib.parse.quote(f"{name}.md")
        text = text.replace(f"[[{name}]]", f"[{label}]({target})")
    for old, new in LINK_FIXES.items():
        text = text.replace(old, new)
    return text


def index(dest_files: list[str]) -> str:
    lines = [
        "# study/ 文档镜像",
        "",
        "本目录是 Obsidian Vault `study/` 笔记的只读镜像，便于在 GitHub 上直接阅读。",
        "**权威版本**仍以 `docs/` 下的对应文件为准（表格格式可能因 Obsidian 插件不同而略有差异）。",
        "",
        "| 文档 | 内容 | 仓库权威版 |",
        "|---|---|---|",
    ]
    for name, (canonical, desc) in DOCS.items():
        target = urllib.parse.quote(name)
        lines.append(f"| [{name}]({target}) | {desc} | [`{canonical}`](../{canonical}) |")
    lines += [
        "",
        "同步方式：`python scripts/sync_study_docs.py --vault-dir <Obsidian study/ 目录>`。",
        "脚本只做复制与链接改写（Obsidian `[[wiki]]` → Markdown 相对链接），不改正文。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vault-dir",
        default=os.environ.get("OBSIDIAN_STUDY_DIR", ""),
        help="Obsidian study/ directory (or set OBSIDIAN_STUDY_DIR)",
    )
    parser.add_argument("--dest", default="study", help="repo destination directory")
    args = parser.parse_args()
    if not args.vault_dir:
        raise SystemExit("pass --vault-dir or set OBSIDIAN_STUDY_DIR")
    vault = Path(args.vault_dir).expanduser()
    dest = ROOT / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    missing = [name for name in DOCS if not (vault / name).is_file()]
    if missing:
        raise SystemExit(f"missing vault documents: {missing}")
    written = []
    for name in DOCS:
        text = (vault / name).read_text(encoding="utf-8")
        (dest / name).write_text(rewrite(text), encoding="utf-8")
        written.append(name)
    (dest / "README.md").write_text(index(written), encoding="utf-8")
    print(f"synced {len(written)} documents to {dest.relative_to(ROOT)}:")
    for name in written:
        print(" -", name)
    print(" - README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
