"""pytest 引导：把 ``src/`` 加入 import 路径，并保证测试环境不注入工具故障。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 单元测试不应受到外部故障注入环境影响。
os.environ.pop("RAG_FAULT_TOOL", None)
os.environ.pop("RAG_FAULT_TIMES", None)
