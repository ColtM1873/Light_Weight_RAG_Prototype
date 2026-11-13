"""项目路径的唯一来源。

所有模块都从这里取路径，避免再出现"代码在 v1/、数据在上一级"这类隐式约定。
项目根目录为 ``src/`` 的上一级，即仓库根（含 README.md / data/ / examples/）。
"""

from __future__ import annotations

from pathlib import Path

# .../src/knowledge_agent/paths.py -> .../src/knowledge_agent -> .../src -> 仓库根
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"
REF_FILES_DIR = DATA_DIR / "ref_files"
STATIC_DIR = PACKAGE_DIR / "static"
ENV_FILE = PROJECT_ROOT / ".env"

SENTENCES_FILE = DATA_DIR / "solid_sentences.json"
CHUNKS_FILE = DATA_DIR / "solid_chunks.json"
SENTENCES_INDEX_FILE = DATA_DIR / "index_sentences.faiss"
CHUNKS_INDEX_FILE = DATA_DIR / "index_chunks.faiss"
NON_IDS_MAP_FILE = DATA_DIR / "non_ids_map.json"
HASH_FILE = DATA_DIR / "ref_files_hash.hash"


def resolve_project_path(value: str | Path) -> Path:
    """把数据文件里记录的路径解析成绝对路径。

    新生成的 solid 数据保存的是相对仓库根的路径（如 ``data/ref_files/D001_x.md``），
    这样仓库整体搬家后仍然可用；旧数据里的绝对路径也兼容。
    """
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
