"""索引构建管线：solid 数据 → FAISS 索引 → hash 记录。"""

from __future__ import annotations

from typing import Any

from ..paths import CHUNKS_INDEX_FILE, SENTENCES_INDEX_FILE
from ..validators import clear_library_cache
from .chunking import split_it_up
from .documents import do_we_have_file_change, return_pipeline_list, save_cur_hash
from .resources import get_embedding_model, invalidate_resources
from .tools import reset_query_state


def _write_index(path: Any, texts: list[str], model: Any) -> None:
    import faiss

    dimension = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    index = faiss.IndexFlatIP(dimension)
    if texts:
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        index.add(vectors)
    # faiss.write_index 的 C++ fopen 不支持中文路径，改用 Python 写序列化字节。
    path.write_bytes(faiss.serialize_index(index).tobytes())


def build_index(force: bool = False) -> dict[str, Any]:
    """文档有变化（或 force=True）时重建 solid 数据与两个索引。

    返回摘要字典；未重建时 ``rebuilt=False``。
    """
    if not force and not do_we_have_file_change():
        return {"rebuilt": False, "reason": "文档集合未变化，跳过重建"}

    pipeline = return_pipeline_list()
    if not pipeline:
        raise RuntimeError("data/ref_files/ 下没有可用的 .md / .txt 文档，无法构建索引。")

    sentences, chunks = split_it_up(pipeline)
    model = get_embedding_model()
    _write_index(SENTENCES_INDEX_FILE, [item["content"] for item in sentences], model)
    _write_index(CHUNKS_INDEX_FILE, [item["content"] for item in chunks], model)
    save_cur_hash()

    # 清掉运行期缓存，确保后续工具和引用校验读到新数据。
    invalidate_resources()
    clear_library_cache()
    reset_query_state()

    return {
        "rebuilt": True,
        "documents": len(pipeline),
        "sentences": len(sentences),
        "chunks": len(chunks),
    }


def get_index_ready() -> dict[str, Any]:
    """Guard 的 init_turn 使用的入口：hash 变了才重建。"""
    return build_index(force=False)
