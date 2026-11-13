"""RAG 运行时资源：embedding 模型、FAISS 索引、solid 数据、文档元信息。

这里的所有加载都是**懒加载**：import 本模块不会读磁盘、也不会加载 bge 模型，
因此单元测试和 `--help` 都能在无数据/无模型的环境下正常运行。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

# 必须在导入 sentence_transformers 之前设置；默认用本机缓存离线加载。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ..paths import (  # noqa: E402
    CHUNKS_FILE,
    CHUNKS_INDEX_FILE,
    PROJECT_ROOT,
    REF_FILES_DIR,
    SENTENCES_FILE,
    SENTENCES_INDEX_FILE,
    resolve_project_path,
)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class RagDataError(RuntimeError):
    """数据/索引缺失或损坏，调用方应转成给模型看的工具失败信息。"""


@dataclass
class RagResources:
    sentences: list[dict]
    chunks: list[dict]
    index_sentences: Any
    index_chunks: Any
    document_meta: dict[int, dict]

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def document_count(self) -> int:
        return len(self.document_meta)


_resources: RagResources | None = None
_embedding_model: Any = None


def get_embedding_model() -> Any:
    """懒加载 bge 向量模型（首次调用会花几秒）。"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        # 模型已在本机缓存；local_files_only 避免离线环境下反复探测 Hub。
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, local_files_only=True
        )
    return _embedding_model


def _build_document_meta(sentences: list[dict], chunks: list[dict]) -> dict[int, dict]:
    """从 solid 数据建立 document_id 到文档元信息的映射。"""
    meta: dict[int, dict] = {}
    for record in list(sentences) + list(chunks):
        doc_id = record.get("id")
        if doc_id is None or doc_id in meta:
            continue
        meta[doc_id] = {
            "path": record.get("path"),
            "name": record.get("name"),
            "title": record.get("title"),
            "version_date": record.get("version_date"),
            "mtime": record.get("mtime"),
        }
    return meta


def _read_json_list(path: Any, label: str) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError, OSError, json.JSONDecodeError) as exc:
        raise RagDataError(
            f"{label} 缺失或损坏（{path}）。请先运行 `python run.py build-index` 生成数据与索引。"
        ) from exc
    if not isinstance(data, list):
        raise RagDataError(f"{label} 格式不正确（{path}）。请重新运行 `python run.py build-index`。")
    return data


def load_resources(force: bool = False) -> RagResources:
    """加载并缓存 solid 数据 + 两个 FAISS 索引。"""
    global _resources
    if _resources is not None and not force:
        return _resources

    missing = [
        str(path)
        for path in (SENTENCES_FILE, CHUNKS_FILE, SENTENCES_INDEX_FILE, CHUNKS_INDEX_FILE)
        if not path.exists()
    ]
    if missing:
        raise RagDataError(
            "RAG 数据/索引不完整，缺少："
            + "、".join(missing)
            + "。请先运行 `python run.py build-index`。"
        )

    import faiss
    import numpy as np

    sentences = _read_json_list(SENTENCES_FILE, "solid_sentences.json")
    chunks = _read_json_list(CHUNKS_FILE, "solid_chunks.json")
    try:
        index_sentences = faiss.deserialize_index(
            np.frombuffer(SENTENCES_INDEX_FILE.read_bytes(), dtype="uint8")
        )
        index_chunks = faiss.deserialize_index(
            np.frombuffer(CHUNKS_INDEX_FILE.read_bytes(), dtype="uint8")
        )
    except Exception as exc:  # noqa: BLE001 - faiss 的 C++ 异常类型不稳定
        raise RagDataError(f"FAISS 索引损坏，请重新运行 `python run.py build-index --force`：{exc}") from exc

    _resources = RagResources(
        sentences=sentences,
        chunks=chunks,
        index_sentences=index_sentences,
        index_chunks=index_chunks,
        document_meta=_build_document_meta(sentences, chunks),
    )
    return _resources


def invalidate_resources() -> None:
    """索引重建后清空缓存，下一次工具调用会重新加载。"""
    global _resources
    _resources = None


def resolve_document_path(meta: dict) -> Any:
    """解析文档路径；数据里可能是相对仓库根的路径，也可能是旧版绝对路径。"""
    raw = meta.get("path")
    if not raw:
        return None
    path = resolve_project_path(raw)
    if path.exists():
        return path
    # 兼容：数据里只记录了文件名 / 相对路径过时，退回到当前 ref_files 目录。
    name = meta.get("name")
    if name:
        fallback = REF_FILES_DIR / name
        if fallback.exists():
            return fallback
    return path


def data_status() -> dict[str, Any]:
    """给 `run.py check` 用的只读状态摘要（不会加载模型）。"""
    status: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "ref_files_dir": str(REF_FILES_DIR),
        "ref_file_count": 0,
        "sentences_file": SENTENCES_FILE.exists(),
        "chunks_file": CHUNKS_FILE.exists(),
        "sentences_index": SENTENCES_INDEX_FILE.exists(),
        "chunks_index": CHUNKS_INDEX_FILE.exists(),
        "document_count": None,
        "sentence_count": None,
        "chunk_count": None,
        "error": None,
    }
    try:
        status["ref_file_count"] = len(
            [p for p in REF_FILES_DIR.iterdir() if p.suffix.lower() in {".md", ".txt"}]
        )
    except OSError:
        pass
    try:
        resources = load_resources()
    except RagDataError as exc:
        status["error"] = str(exc)
        return status
    status.update(
        {
            "document_count": resources.document_count,
            "sentence_count": resources.sentence_count,
            "chunk_count": resources.chunk_count,
        }
    )
    return status
