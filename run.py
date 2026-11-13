"""仓库根入口：把 ``src/`` 加进 import 路径后转交给 CLI。

用法::

    python run.py check
    python run.py build-index
    python run.py serve
    python run.py ask -q "出差住宿标准是多少？"
    python run.py demo --case normal
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from knowledge_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
