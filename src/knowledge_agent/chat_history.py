"""浏览器会话持久化：每个 thread 一份 JSON + Markdown，落在根目录 ``Chat-History/``。

服务端每完成一轮问答就追加写入，不依赖前端；即使页面关掉，历史也已经落盘。
原始记录默认不提交到 git（见 .gitignore），用于后续人工整理 Trace / 复盘。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT
from .tracing import TraceRecorder

CHAT_HISTORY_DIR = PROJECT_ROOT / "Chat-History"
_SAFE_NAME = re.compile(r"[^0-9A-Za-z_.-]+")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_thread_name(thread_id: str) -> str:
    """把 thread_id 变成安全文件名；非 ASCII / 特殊字符加短 hash 防碰撞。"""
    cleaned = _SAFE_NAME.sub("_", thread_id).strip("._")[:80] or "thread"
    if cleaned != thread_id:
        digest = hashlib.sha1(thread_id.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned}-{digest}"
    return cleaned


def _render_markdown(data: dict[str, Any]) -> str:
    recorder = TraceRecorder(
        case_id=data.get("thread_id") or "web-session",
        title="浏览器会话记录",
        mode=data.get("source") or "web",
        model=data.get("model") or {},
        acceptance_points=data.get("acceptance_points") or [],
    )
    recorder.created_at = data.get("created_at") or recorder.created_at
    recorder.turns = data.get("turns") or []
    lines = [
        "> 本文件由服务端自动生成，原始 JSON 同目录同名。",
        "> 用于复盘真实浏览器会话，不一定覆盖全部验收点。",
        "",
        recorder.to_markdown(),
    ]
    return "\n".join(lines)


class ConversationStore:
    """按 thread 追加保存对话；线程安全、原子写入。"""

    def __init__(self, base_dir: Path = CHAT_HISTORY_DIR) -> None:
        self.base_dir = Path(base_dir)
        self._lock = threading.Lock()

    def _path(self, thread_id: str) -> Path:
        return self.base_dir / f"{safe_thread_name(thread_id)}.json"

    def append_turn(
        self,
        thread_id: str,
        recorder: TraceRecorder,
        *,
        source: str = "web",
        error: str | None = None,
        model: dict[str, Any] | None = None,
    ) -> Path:
        """把 recorder 中的一轮（或多轮）追加进该 thread 的 JSON/Markdown。"""
        with self._lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            path = self._path(thread_id)
            data = self._load(path, thread_id)
            data["source"] = source
            data["updated_at"] = _now()
            if model:
                data["model"] = model
            data.setdefault("model", {})

            existing = data.setdefault("turns", [])
            for turn in recorder.turns:
                turn = dict(turn)
                turn["turn"] = len(existing) + 1
                turn["saved_at"] = _now()
                if error:
                    turn["error"] = error
                existing.append(turn)

            self._atomic_write_json(path, data)
            path.with_suffix(".md").write_text(_render_markdown(data), encoding="utf-8")
            self._append_index(thread_id, path, existing[-1] if existing else {})
            return path

    def _load(self, path: Path, thread_id: str) -> dict[str, Any]:
        if not path.exists():
            return {
                "thread_id": thread_id,
                "source": "web",
                "created_at": _now(),
                "updated_at": _now(),
                "model": {},
                "turns": [],
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            backup = path.with_suffix(f".corrupt-{int(datetime.now().timestamp())}.json")
            try:
                path.replace(backup)
            except OSError:
                pass
            return {
                "thread_id": thread_id,
                "source": "web",
                "created_at": _now(),
                "updated_at": _now(),
                "model": {},
                "turns": [],
            }
        if not isinstance(data, dict):
            data = {"thread_id": thread_id, "created_at": _now(), "turns": []}
        data.setdefault("thread_id", thread_id)
        data.setdefault("created_at", _now())
        data.setdefault("turns", [])
        return data

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _append_index(
        self, thread_id: str, path: Path, last_turn: dict[str, Any]
    ) -> None:
        index_path = self.base_dir / "index.jsonl"
        summary = {
            "time": _now(),
            "thread_id": thread_id,
            "file": path.name,
            "turn": last_turn.get("turn"),
            "question": last_turn.get("question"),
            "completed": last_turn.get("completed"),
            "tool_calls": last_turn.get("tool_calls"),
            "task_status": last_turn.get("task_status"),
            "citations": len(last_turn.get("citations") or []),
            "final_answer": (last_turn.get("final_answer") or "")[:600],
            "error": last_turn.get("error"),
        }
        with index_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # -- 供人工/脚本读取 ---------------------------------------------------
    def load_thread(self, thread_id: str) -> dict[str, Any] | None:
        path = self._path(thread_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_files(self) -> list[Path]:
        if not self.base_dir.exists():
            return []
        return sorted(self.base_dir.glob("*.json"))
