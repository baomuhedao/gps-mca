"""
下载公开文本语料库，供 GPS-MCA v3.0 学习使用。

默认下载 WikiText-2 (维基百科精选文章, ~5MB)。
可选: tiny-textbooks (合成教科书, ~420K条)。

用法:
  python download_corpus.py                          # 下载 WikiText-2
  python download_corpus.py --corpus tiny-textbooks  # 下载 tiny-textbooks
  python download_corpus.py --max-articles 200       # 限制文章数
  python download_corpus.py --output ./my_data       # 指定输出目录

下载完成后直接用 train.py 学习:
  python train.py --input ./corpus/wikitext2/ --epochs 3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

AVAILABLE_CORPORA = {
    "wikitext2": {
        "hf_path": "Salesforce/wikitext",
        "hf_name": "wikitext-2-raw-v1",
        "split": "train",
        "desc": "WikiText-2 (维基百科精选文章, ~5MB, 36K samples)",
    },
    "tiny-textbooks": {
        "hf_path": "nampdn-ai/tiny-textbooks",
        "hf_name": None,
        "split": "train",
        "desc": "Tiny Textbooks (合成教科书, 420K 条, ~800MB)",
    },
}


def download_and_save(corpus_key: str, output_dir: str, max_articles: int | None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency. Install with:")
        print("  pip install datasets")
        sys.exit(1)

    info = AVAILABLE_CORPORA[corpus_key]
    print(f"Corpus: {info['desc']}")
    print(f"HuggingFace: {info['hf_path']}")
    print(f"Downloading...")

    ds = load_dataset(info["hf_path"], info["hf_name"], split=info["split"])
    print(f"  Raw samples: {len(ds)}")

    out_path = Path(output_dir) / corpus_key
    out_path.mkdir(parents=True, exist_ok=True)

    if corpus_key == "wikitext2":
        _save_wikitext(ds, out_path, max_articles)
    elif corpus_key == "tiny-textbooks":
        _save_textbooks(ds, out_path, max_articles)
    else:
        _save_generic(ds, out_path, max_articles)

    total_files = len(list(out_path.glob("*.txt")))
    total_bytes = sum(f.stat().st_size for f in out_path.glob("*.txt"))
    print(f"\nDone!")
    print(f"  Output: {out_path}")
    print(f"  Files:  {total_files}")
    print(f"  Size:   {total_bytes / 1024:.1f} KB")
    print(f"\nTrain with:")
    print(f"  python train.py --input {out_path} --epochs 3")


def _save_wikitext(ds, out_path: Path, max_articles: int | None):
    """WikiText-2 按文章分割保存，每篇文章一个 .txt 文件"""
    articles: list[tuple[str, list[str]]] = []
    current_title = None
    current_lines: list[str] = []

    for row in ds:
        line = row["text"].strip()
        if not line:
            continue

        title_match = re.match(r"^= ([^=]+) =$", line)
        if title_match:
            if current_title and current_lines:
                articles.append((current_title, current_lines))
            current_title = title_match.group(1).strip()
            current_lines = []
        else:
            if line.startswith("= =") or line.startswith("= = ="):
                continue
            current_lines.append(line)

    if current_title and current_lines:
        articles.append((current_title, current_lines))

    articles = [a for a in articles if len(a[1]) >= 3]

    if max_articles:
        articles = articles[:max_articles]

    print(f"  Articles with 3+ paragraphs: {len(articles)}")

    for i, (title, lines) in enumerate(articles):
        safe_name = re.sub(r'[^\w\s-]', '', title)[:60].strip().replace(' ', '_')
        filename = f"{i:04d}_{safe_name}.txt"
        content = f"{title}\n\n" + "\n\n".join(lines)
        (out_path / filename).write_text(content, encoding="utf-8")

    print(f"  Saved {len(articles)} articles as .txt files")


def _save_textbooks(ds, out_path: Path, max_articles: int | None):
    """Tiny Textbooks 每条保存为一个 .txt 文件"""
    limit = max_articles or 500
    count = 0

    for i, row in enumerate(ds):
        if count >= limit:
            break
        text = row.get("textbook", row.get("text", ""))
        if not text or len(text.strip()) < 100:
            continue
        filename = f"textbook_{i:05d}.txt"
        (out_path / filename).write_text(text.strip(), encoding="utf-8")
        count += 1

    print(f"  Saved {count} textbook entries as .txt files")


def _save_generic(ds, out_path: Path, max_articles: int | None):
    """通用格式：提取 text 字段"""
    limit = max_articles or 1000
    count = 0

    for i, row in enumerate(ds):
        if count >= limit:
            break
        text = row.get("text", "")
        if not text or len(text.strip()) < 50:
            continue
        (out_path / f"doc_{i:05d}.txt").write_text(text.strip(), encoding="utf-8")
        count += 1

    print(f"  Saved {count} documents as .txt files")


def main():
    parser = argparse.ArgumentParser(description="Download text corpus for GPS-MCA learning")
    parser.add_argument(
        "--corpus",
        choices=list(AVAILABLE_CORPORA.keys()),
        default="wikitext2",
        help="Corpus to download (default: wikitext2)",
    )
    parser.add_argument("--output", type=str, default="./corpus", help="Output directory")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles to save")
    args = parser.parse_args()

    print("=" * 60)
    print("  GPS-MCA Corpus Downloader")
    print("=" * 60)

    print(f"\nAvailable corpora:")
    for k, v in AVAILABLE_CORPORA.items():
        marker = " <--" if k == args.corpus else ""
        print(f"  [{k}] {v['desc']}{marker}")

    print()
    download_and_save(args.corpus, args.output, args.max_articles)


if __name__ == "__main__":
    main()
