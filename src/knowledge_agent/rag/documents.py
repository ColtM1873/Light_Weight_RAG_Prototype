"""文档扫描、ID 分配、版本日期解析与文档集合 hash。

产出的路径统一存成**相对仓库根**的形式（如 ``data/ref_files/D001_x.md``），
这样仓库整体移动/提交后仍然可用。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from ..paths import (
    HASH_FILE,
    NON_IDS_MAP_FILE,
    PROJECT_ROOT,
    REF_FILES_DIR,
)

FILE_SUFFIXES = {".txt", ".md"}
NON_ID_STARTER = 1000

RE_FORMAT_REQUIRED = re.compile(r"^D(\d+)_(.+)$")
RE_VERSION_DATE = re.compile(r"^版本日期\s*[:：]\s*(\d{4}-\d{2}-\d{2})")


def _iter_ref_files(docs_dir: Path = REF_FILES_DIR):
    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"文档目录不存在：{docs_dir}。请先在 data/ref_files/ 放入 .md / .txt 文档。"
        )
    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in FILE_SUFFIXES:
            yield path


def validate_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_version_date(path: Path) -> str | None:
    """第一行形如 ``版本日期：YYYY-MM-DD`` 时返回日期，否则 None。"""
    try:
        with path.open(encoding="utf-8") as file:
            first_line = file.readline().strip().lstrip("\ufeff")
    except OSError:
        return None
    match = RE_VERSION_DATE.match(first_line)
    if not match:
        return None
    value = match.group(1)
    return value if validate_date(value) else None


def get_sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.digest()


def calc_total_hash(docs_dir: Path = REF_FILES_DIR) -> str:
    """文档集合 hash：文件名 + 内容都参与，改名或改内容都会触发重建。"""
    digest = hashlib.sha256()
    for path in _iter_ref_files(docs_dir):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(get_sha256_of_file(path))
        digest.update(b"\0")
    return digest.hexdigest()


def do_we_have_file_change() -> bool:
    """hash 文件不存在、或文档集合发生变化时返回 True。"""
    if not HASH_FILE.exists():
        return True
    current = calc_total_hash()
    old = HASH_FILE.read_text(encoding="utf-8").strip().lstrip("\ufeff")
    return current != old


def save_cur_hash() -> None:
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(calc_total_hash(), encoding="utf-8")


def scan_files(docs_dir: Path = REF_FILES_DIR) -> list[dict]:
    """扫描文档，返回 path/name/mtime/version_date。"""
    files: list[dict] = []
    for path in _iter_ref_files(docs_dir):
        files.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "name": path.name,
                "mtime": path.stat().st_mtime,
                "version_date": get_version_date(path),
            }
        )
    return files


def _assign_ids(files_without_ids: list[dict], occupied: set[int]) -> None:
    current_id = NON_ID_STARTER
    id_map: dict[str, str] = {}
    for file in files_without_ids:
        while current_id in occupied:
            current_id += 1
        file["id"] = current_id
        file["file_prefix"] = f"D{current_id}"
        id_map[str(current_id)] = file["path"]
        current_id += 1
    NON_IDS_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    NON_IDS_MAP_FILE.write_text(
        json.dumps(id_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def classify(files: list[dict]) -> list[dict]:
    """按文件名分配文档 ID / 标题；没有 Dxxx 前缀的文档从 1000 开始编号。"""
    files_with_ids: list[dict] = []
    files_without_ids: list[dict] = []
    occupied: set[int] = set()

    for file in files:
        stem = Path(file["name"]).stem
        match = RE_FORMAT_REQUIRED.match(stem)
        if match:
            file["file_prefix"] = f"D{match.group(1)}"
            file["title"] = match.group(2)
            file["id"] = int(match.group(1))
            occupied.add(file["id"])
            files_with_ids.append(file)
        else:
            file["title"] = stem
            files_without_ids.append(file)

    _assign_ids(files_without_ids, occupied)
    pipeline = files_with_ids + files_without_ids
    pipeline.sort(key=lambda item: item["id"])
    return pipeline


def return_pipeline_list(docs_dir: Path = REF_FILES_DIR) -> list[dict]:
    """扫描 + 分类，返回按 ID 排序的文档列表（不写 hash）。"""
    return classify(scan_files(docs_dir))
