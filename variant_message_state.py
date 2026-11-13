"""主图使用的扩展 state。

在 LangGraph 的 MessagesState 基础上，增加 Guard 层需要的控制字段。
所有字段都可以通过 ``state.get(key)`` 安全读取（首次运行时可能不存在）。
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from langgraph.graph.message import MessagesState


class ExtendedMessagesState(MessagesState):
    """messages 自动累积；其余字段由各节点按需写入。"""

    # 步数：每个节点执行后 +1（reducer 自动累加，节点返回 {"step_count": 1}）
    step_count: Annotated[int, operator.add]

    # 当前阶段：draft = 正常主循环；final = finalize 整理后的复检阶段
    phase: Literal["draft", "final"]

    # Model C 是否判定过 unsupported。
    # True 表示发生过辩论，最终支持后必须走 finalize + 复检。
    debate_flag: bool

    # 本轮是否注入过任何 Guard 反馈（missing_fields / bad_refs / unsupported）。
    # 没有反馈时，Model A 的最后一条消息就是最终答复，直接 END。
    feedback_injected: bool

    # 本轮真实用户提问（START 时确定，之后只读）
    user_question: str
    user_question_msg_id: str

    # hash 检查相关
    hash_changed: bool
    history_cleared: bool
    index_rebuild_error: str | None

    # Guard 解析结果（调试 / 复用）
    parsed_answer: dict | None
    parse_attempts: int
    support_verdict: dict | None
    support_attempts: int

    # Model C 的证据包（调试）
    evidence_pack: str | None
