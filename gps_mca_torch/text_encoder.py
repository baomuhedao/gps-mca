"""
文本编码器 — sentence-transformers 封装

将原始文本转换为 384 维嵌入向量，供 GPS-MCA 意识引擎处理。

可用模型:
  - paraphrase-multilingual-MiniLM-L12-v2  (默认, 384维, 50+语言含中文)
  - all-MiniLM-L6-v2                       (英文专用, 384维, 更快)
"""

from __future__ import annotations

import re
from typing import Iterator

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class TextEncoder:
    """文本 → 嵌入向量 (支持中英文及50+语言)"""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | torch.device = "cpu"):
        self.device = str(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embed_dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """单条文本 → (embed_dim,) tensor (detached, normal mode)"""
        emb = self.model.encode(text, convert_to_tensor=True)
        return emb.to(self.device).float().clone()

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """批量文本 → (batch, embed_dim) tensor"""
        emb = self.model.encode(texts, convert_to_tensor=True, batch_size=32)
        return emb.to(self.device).float().clone()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    将长文本按句分割为重叠的文本块。
    chunk_size: 每块大约的字符数
    overlap: 块之间的重叠字符数
    """
    sentences = re.split(r'(?<=[.!?\n。！？])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        current.append(sent)
        current_len += len(sent)

        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # 保留尾部作为重叠
            keep = []
            keep_len = 0
            for s in reversed(current):
                if keep_len + len(s) > overlap:
                    break
                keep.insert(0, s)
                keep_len += len(s)
            current = keep
            current_len = keep_len

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown",
    ".json", ".jsonl",
    ".csv", ".tsv",
    ".html", ".htm",
    ".xml",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".log", ".tex", ".rst",
    ".pdf",
    ".docx",
}


def _is_temp_file(path: str) -> bool:
    """检测操作系统或应用程序生成的临时文件"""
    from pathlib import Path
    name = Path(path).name
    return (
        name.startswith("~$")      # Word / Excel 锁文件
        or name.startswith("~")     # 通用临时文件
        or name.startswith(".")     # 隐藏文件
        or name.endswith(".tmp")
        or name == "Thumbs.db"
        or name == "desktop.ini"
    )


def read_file(path: str) -> str:
    """
    通用文件读取器，根据扩展名自动选择解析方式。
    支持: txt/md/json/csv/html/代码/pdf/docx 等。
    自动跳过临时文件，读取失败时返回空字符串而非崩溃。
    """
    from pathlib import Path

    if _is_temp_file(path):
        return ""

    ext = Path(path).suffix.lower()

    try:
        if ext == ".pdf":
            return _read_pdf(path)
        if ext == ".docx":
            return _read_docx(path)
        if ext in (".html", ".htm"):
            return _read_html(path)
        if ext in (".json", ".jsonl"):
            return _read_json(path)
        if ext in (".csv", ".tsv"):
            return _read_csv(path)
        return _read_text(path)
    except Exception as e:
        print(f"  [!] Failed to read {Path(path).name}: {e}")
        return ""


def _read_text(path: str) -> str:
    """纯文本读取，自动检测编码"""
    raw = open(path, "rb").read()

    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", errors="replace")

    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    return raw.decode("utf-8", errors="replace")


def _read_pdf(path: str) -> str:
    """读取 PDF (需要 PyMuPDF 或降级为提示)"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except ImportError:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print(f"  [!] PDF support requires: pip install pymupdf  or  pip install pypdf")
            print(f"      Skipping: {path}")
            return ""


def _read_docx(path: str) -> str:
    """读取 Word docx"""
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        print(f"  [!] DOCX support requires: pip install python-docx")
        print(f"      Skipping: {path}")
        return ""


def _read_html(path: str) -> str:
    """读取 HTML，提取纯文本"""
    raw = _read_text(path)
    try:
        from html.parser import HTMLParser

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: list[str] = []
            def handle_data(self, d: str):
                self.parts.append(d)

        s = _Stripper()
        s.feed(raw)
        return " ".join(s.parts)
    except Exception:
        return re.sub(r"<[^>]+>", " ", raw)


def _read_json(path: str) -> str:
    """读取 JSON/JSONL，提取所有字符串值"""
    import json
    raw = _read_text(path)
    texts = []

    def _extract(obj):
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _extract(v)
        elif isinstance(obj, list):
            for v in obj:
                _extract(v)

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            _extract(json.loads(line))
        except json.JSONDecodeError:
            pass

    if not texts:
        try:
            _extract(json.loads(raw))
        except json.JSONDecodeError:
            texts = [raw]

    return "\n".join(texts)


def _read_csv(path: str) -> str:
    """读取 CSV/TSV，每行拼接为一句"""
    import csv
    from pathlib import Path
    delimiter = "\t" if Path(path).suffix.lower() == ".tsv" else ","
    raw = _read_text(path)
    lines = []
    for row in csv.reader(raw.splitlines(), delimiter=delimiter):
        lines.append(" ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(lines)


# 向后兼容
read_text_file = read_file
