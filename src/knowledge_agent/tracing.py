"""执行路径记录：把图上产生的消息整理成可提交的 JSON / Markdown Trace。

记录内容包括：
- 用户问题、Model A 的每条回复与工具调用参数；
- 工具返回全文；
- Guard 1 的 ``parsed_answer``（自然语言回复 / 引用 / 资料缺口 / 任务状态）；
- Guard 2 的 ``support_verdict``（support / unsupported + 理由）；
- 每轮最终的结构化任务状态与步数统计。

"下一步 / 决策依据"只根据相邻事件的事实推导，不记录模型内部思维过程。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .validators import content_to_text

SCHEMA_VERSION = 2

_GUARD_1_KINDS = {"missing_fields", "bad_refs"}


def _one_line(text: Any, limit: int = 260) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[:limit] + "…"


def _md_cell(text: Any, limit: int = 260) -> str:
    value = _one_line(text, limit).replace("|", "\\|")
    return value or "（空）"


def _doc_label(document_id: Any) -> str:
    if isinstance(document_id, int):
        return f"D{document_id:03d}"
    return str(document_id)


def _format_citation(citation: dict[str, Any], quote_limit: int = 160) -> str:
    return f"{_doc_label(citation.get('document_id'))}：「{_one_line(citation.get('quoted_content'), quote_limit)}」"


def _quote_block(text: Any) -> str:
    """把多行文本转成 Markdown 引用块；空行不产生行尾空格。"""
    lines = str(text or "（无）").splitlines()
    if not lines:
        return ">"
    return "\n".join(">" if not line else f"> {line}" for line in lines)


def _status_label(status: dict[str, Any]) -> str:
    if status.get("conflict"):
        return "文档冲突，待用户确认"
    if status.get("no_support"):
        return "资料不足，未给出确定答案"
    if status.get("done"):
        return "已完成"
    return "未完成 / 状态未知"


class TraceRecorder:
    """收集一轮或多轮对话的节点消息，并补充"下一步/决策依据"和结构化状态。"""

    def __init__(
        self,
        *,
        case_id: str,
        title: str,
        mode: str,
        model: dict[str, Any],
        acceptance_points: list[str] | None = None,
    ) -> None:
        self.case_id = case_id
        self.title = title
        self.mode = mode
        self.model = model
        self.acceptance_points = acceptance_points or []
        self.created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.turns: list[dict[str, Any]] = []
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}

    # -- 单条消息 -----------------------------------------------------------
    def add_message(self, node: str, message: BaseMessage) -> dict[str, Any] | None:
        if isinstance(message, HumanMessage):
            guard_kind = (message.additional_kwargs or {}).get("guard_kind")
            content = content_to_text(message.content)
            if not content:
                return None
            if guard_kind:
                if guard_kind == "finalize":
                    guard_scope = "finalize"
                elif guard_kind in _GUARD_1_KINDS:
                    guard_scope = "parse"
                else:
                    guard_scope = "support"
                return {
                    "kind": "guard",
                    "node": node,
                    "label": f"Guard 反馈：{guard_kind}",
                    "guard_kind": guard_kind,
                    "guard": guard_scope,
                    "ok": guard_kind == "finalize",
                    "content": content,
                }
            return {
                "kind": "user",
                "node": node,
                "label": "用户",
                "content": content,
            }

        if isinstance(message, AIMessage):
            tool_calls = [
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("args", {}),
                }
                for call in (getattr(message, "tool_calls", None) or [])
                if isinstance(call, dict)
            ]
            batch_id = f"batch-{uuid.uuid4().hex[:8]}"
            for call in tool_calls:
                if call.get("id"):
                    self._pending_tool_calls[str(call["id"])] = {
                        **call,
                        "batch_id": batch_id,
                    }
            content = content_to_text(message.content)
            if not content and tool_calls:
                content = "→ 调用工具：" + ", ".join(str(c["name"]) for c in tool_calls)
            if not content:
                return None
            return {
                "kind": "ai",
                "node": node,
                "label": "AI（Model A）",
                "content": content,
                "tool_calls": tool_calls,
                "batch_id": batch_id,
            }

        if isinstance(message, ToolMessage):
            call = self._pending_tool_calls.get(str(message.tool_call_id), {})
            return {
                "kind": "tool",
                "node": node,
                "label": f"工具：{getattr(message, 'name', None) or 'unknown'}",
                "tool_name": getattr(message, "name", None),
                "tool_args": call.get("args", {}),
                "tool_call_id": message.tool_call_id,
                "batch_id": call.get("batch_id"),
                "content": content_to_text(message.content),
            }

        return None

    # -- 一轮 ---------------------------------------------------------------
    def start_turn(self, index: int, question: str) -> None:
        self.turns.append(
            {
                "turn": index,
                "question": question,
                "events": [
                    {
                        "kind": "user",
                        "node": "input",
                        "label": "用户",
                        "content": question,
                    }
                ],
            }
        )

    def add_event(self, node: str, message: BaseMessage) -> dict[str, Any] | None:
        event = self.add_message(node, message)
        if event is not None:
            self.turns[-1]["events"].append(event)
        return event

    def add_node_event(self, node: str, update: dict[str, Any]) -> dict[str, Any] | None:
        """记录没有产生消息、但执行了工作的节点（Guard 判定、初始化等）。

        这是"两个 Guard 是否真的运转"的关键证据：把 Model B 的解析结果和
        Model C 的判定结果原样写进 Trace。
        """
        event: dict[str, Any] = {
            "kind": "node",
            "node": node,
            "label": f"节点：{node}",
        }

        if node == "init_turn":
            event["content"] = (
                f"初始化完成；hash_changed={bool(update.get('hash_changed'))}；"
                f"history_cleared={bool(update.get('history_cleared'))}；"
                f"index_rebuild_error={update.get('index_rebuild_error')}"
            )
            event["state"] = {
                "hash_changed": bool(update.get("hash_changed")),
                "history_cleared": bool(update.get("history_cleared")),
                "index_rebuild_error": update.get("index_rebuild_error"),
            }
        elif node == "guard_parse":
            parsed = update.get("parsed_answer")
            if parsed:
                status = parsed.get("status") or {}
                citations = parsed.get("citations") or []
                event.update(
                    {
                        "guard": "parse",
                        "ok": True,
                        "parsed_answer": parsed,
                        "parse_attempts": update.get("parse_attempts"),
                        "content": (
                            f"Guard 1 通过：引用 {len(citations)} 条；"
                            f"任务状态 done={status.get('done')}, "
                            f"conflict={status.get('conflict')}, "
                            f"no_support={status.get('no_support')}"
                        ),
                    }
                )
            else:
                event.update(
                    {
                        "guard": "parse",
                        "ok": False,
                        "parse_attempts": update.get("parse_attempts"),
                        "content": "Guard 1 未通过：已注入反馈，回到 Model A 修正。",
                    }
                )
        elif node == "guard_support":
            verdict = update.get("support_verdict")
            if verdict:
                event.update(
                    {
                        "guard": "support",
                        "ok": verdict.get("support") == "support",
                        "support_verdict": verdict,
                        "support_attempts": update.get("support_attempts"),
                        "evidence_pack_chars": len(update.get("evidence_pack") or ""),
                        "content": (
                            f"Guard 2 判定：{verdict.get('support')}"
                            f"（error_type={verdict.get('error_type')}）；"
                            f"理由：{_one_line(verdict.get('reason'), 220)}"
                        ),
                    }
                )
            else:
                event.update(
                    {
                        "guard": "support",
                        "ok": False,
                        "support_attempts": update.get("support_attempts"),
                        "content": "Guard 2 未通过：已注入反馈，回到 Model A 修正。",
                    }
                )
        elif node == "feedback":
            event.update(
                {
                    "content": f"反馈已注入，回到 Model A 继续处理；phase={update.get('phase')}",
                    "state": {
                        "phase": update.get("phase"),
                        "feedback_injected": True,
                    },
                }
            )
        else:
            return None

        self.turns[-1]["events"].append(event)
        return event

    def end_turn(self, state_values: dict[str, Any]) -> dict[str, Any]:
        turn = self.turns[-1]
        events = turn["events"]
        self._annotate(events)

        final_answer = None
        for event in reversed(events):
            if event["kind"] == "ai" and not event.get("tool_calls"):
                final_answer = event["content"]
                break
        hit_step_limit = any(event.get("node") == "step_limit" for event in events)
        tool_calls = sum(1 for event in events if event["kind"] == "tool")

        parsed = self._last_payload(events, "parsed_answer")
        verdict = self._last_payload(events, "support_verdict")
        status = (parsed or {}).get("status") or {}

        guard1 = "not_run"
        guard2 = "not_run"
        for event in events:
            if event.get("guard") == "parse":
                guard1 = "pass" if event.get("ok") else "feedback"
            elif event.get("guard") == "support":
                if event.get("ok"):
                    guard2 = "support"
                else:
                    guard2 = "unsupported"

        task_status = {
            "done": status.get("done"),
            "conflict": status.get("conflict"),
            "no_support": status.get("no_support"),
            "label": (
                "达到步数上限，本轮未完成"
                if hit_step_limit
                else _status_label(status)
            ),
            "guard1": guard1,
            "guard2": guard2,
            "feedback_injected": bool(state_values.get("feedback_injected")),
            "debate_flag": bool(state_values.get("debate_flag")),
            "step_count": state_values.get("step_count"),
            "tool_calls": tool_calls,
            "completed": bool(final_answer) and not hit_step_limit,
            "hit_step_limit": hit_step_limit,
            "index_rebuild_error": state_values.get("index_rebuild_error"),
        }

        turn.update(
            {
                "final_answer": final_answer,
                "step_count": state_values.get("step_count"),
                "tool_calls": tool_calls,
                "completed": task_status["completed"],
                "hit_step_limit": hit_step_limit,
                "feedback_injected": task_status["feedback_injected"],
                "debate_flag": task_status["debate_flag"],
                "index_rebuild_error": state_values.get("index_rebuild_error"),
                "task_status": task_status,
                "citations": (parsed or {}).get("citations") or [],
                "gaps": (parsed or {}).get("gaps") or [],
                "support_verdict": verdict,
            }
        )
        return turn

    @staticmethod
    def _last_payload(
        events: list[dict[str, Any]], key: str
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            payload = event.get(key)
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _annotate(events: list[dict[str, Any]]) -> None:
        """根据相邻事件补充"下一步"和可观察到的决策依据（不涉及模型内部思维）。"""
        for index, event in enumerate(events):
            next_event = events[index + 1] if index + 1 < len(events) else None
            event["next_action"] = _describe_next(event, next_event)
            event["decision_basis"] = _describe_basis(event)

    # -- 给 UI / CLI 的状态卡 ----------------------------------------------
    def status_payload(self) -> dict[str, Any]:
        turn = self.turns[-1] if self.turns else {}
        task_status = turn.get("task_status") or {}
        citations = turn.get("citations") or []
        gaps = turn.get("gaps") or []

        lines = [f"任务状态：{task_status.get('label') or '状态未知'}"]
        lines.append(
            "Guard 1："
            + {"pass": "通过", "feedback": "未通过（已反馈）"}.get(
                task_status.get("guard1"), "未执行"
            )
            + "；Guard 2："
            + {
                "support": "support",
                "unsupported": "unsupported（已反馈）",
            }.get(task_status.get("guard2"), "未执行")
        )
        verdict = turn.get("support_verdict") or {}
        if verdict.get("reason"):
            lines.append("Guard 2 理由：" + _one_line(verdict.get("reason"), 220))
        if any(
            event.get("guard_kind") == "finalize"
            for event in (turn.get("events") or [])
        ):
            lines.append("注：finalize 后的整理稿未再次做结构化校验。")
        if citations:
            lines.append("引用：")
            for citation in citations[:6]:
                lines.append("  - " + _format_citation(citation))
            if len(citations) > 6:
                lines.append(f"  - ……（共 {len(citations)} 条）")
        if gaps:
            lines.append("资料缺口：")
            for gap in gaps[:6]:
                lines.append("  - " + _one_line(gap, 160))
            if len(gaps) > 6:
                lines.append(f"  - ……（共 {len(gaps)} 条）")
        lines.append(
            f"步数 {task_status.get('step_count')}；"
            f"工具调用 {task_status.get('tool_calls')} 次；"
            f"Guard 反馈 {'有' if task_status.get('feedback_injected') else '无'}；"
            f"Model C 辩论 {'有' if task_status.get('debate_flag') else '无'}"
        )
        return {
            "type": "status",
            "label": "任务状态",
            "content": "\n".join(lines),
            "status": task_status,
            "citations": citations,
            "gaps": gaps,
        }

    # -- 导出 ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": self.case_id,
            "title": self.title,
            "mode": self.mode,
            "model": self.model,
            "created_at": self.created_at,
            "acceptance_points": self.acceptance_points,
            "questions": [turn["question"] for turn in self.turns],
            "turns": self.turns,
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def to_markdown(self) -> str:
        model_name = self.model.get("name") or self.model.get("llm_provider_name") or "unknown"
        lines = [
            f"# {self.case_id} · {self.title}",
            "",
            f"- 运行模式：`{self.mode}`",
            f"- 模型：`{model_name}`",
            f"- 生成时间：{self.created_at}",
        ]
        if self.acceptance_points:
            lines.append("- 覆盖验收点：" + "、".join(self.acceptance_points))
        lines.append("")

        for turn in self.turns:
            lines += [f"## Turn {turn['turn']}", "", f"**问题**：{turn['question']}", ""]
            lines += [
                "| # | 节点 | 类型 | 动作 / 观察 | 下一步 |",
                "|---|---|---|---|---|",
            ]
            for index, event in enumerate(turn["events"], 1):
                content = _md_cell(event.get("content"))
                if event["kind"] == "ai" and event.get("tool_calls"):
                    calls = ", ".join(
                        f"{call['name']} {json.dumps(call.get('args', {}), ensure_ascii=False)}"
                        for call in event["tool_calls"]
                    )
                    content = _md_cell(f"调用工具：{calls}")
                lines.append(
                    "| {idx} | {node} | {kind} | {content} | {next_action} |".format(
                        idx=index,
                        node=event.get("node", ""),
                        kind=event.get("kind", ""),
                        content=content,
                        next_action=_md_cell(event.get("next_action", "")),
                    )
                )
            lines += self._markdown_status(turn)
            lines += [
                "",
                "**最终答复**：",
                "",
                _quote_block(turn.get("final_answer")),
                "",
                (
                    f"**本轮状态**：步数 {turn.get('step_count')}；"
                    f"工具调用 {turn.get('tool_calls')} 次；"
                    f"完成={turn.get('completed')}；"
                    f"Guard 反馈={'有' if turn.get('feedback_injected') else '无'}；"
                    f"Model C 辩论={'有' if turn.get('debate_flag') else '无'}"
                ),
                "",
            ]
        return "\n".join(lines)

    @staticmethod
    def _markdown_status(turn: dict[str, Any]) -> list[str]:
        task_status = turn.get("task_status") or {}
        if not task_status:
            return []
        citations = turn.get("citations") or []
        gaps = turn.get("gaps") or []
        lines = [
            "",
            "**结构化任务状态**：",
            "",
            f"- Guard 1：{'通过' if task_status.get('guard1') == 'pass' else task_status.get('guard1')}",
            f"- Guard 2：{task_status.get('guard2')}",
            f"- 任务状态：{task_status.get('label')}",
            f"- done / conflict / no_support："
            f"{task_status.get('done')} / {task_status.get('conflict')} / {task_status.get('no_support')}",
        ]
        if citations:
            lines.append("- 引用：")
            lines += ["  - " + _format_citation(citation) for citation in citations[:8]]
            if len(citations) > 8:
                lines.append(f"  - ……（共 {len(citations)} 条，完整内容见同目录 JSON）")
        if gaps:
            lines.append("- 资料缺口：")
            lines += ["  - " + _one_line(gap, 180) for gap in gaps[:8]]
        return lines

    def write_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target

    def summary(self) -> dict[str, Any]:
        turns = self.turns
        last = turns[-1] if turns else {}
        return {
            "case_id": self.case_id,
            "turns": len(turns),
            "completed": last.get("completed"),
            "task_status": last.get("task_status"),
            "final_answer": last.get("final_answer"),
        }


def _describe_next(
    current_event: dict[str, Any], next_event: dict[str, Any] | None
) -> str:
    if next_event is None:
        return "本轮结束"
    node = next_event.get("node")
    kind = next_event.get("kind")
    if kind == "tool":
        if (
            current_event.get("kind") == "tool"
            and current_event.get("batch_id")
            and current_event.get("batch_id") == next_event.get("batch_id")
        ):
            return "继续接收同一批工具结果"
        names = [
            str(call.get("name"))
            for call in (current_event.get("tool_calls") or [])
        ]
        if len(names) > 1:
            return f"执行本批 {len(names)} 个工具：{', '.join(names)}"
        return f"执行工具 {next_event.get('tool_name') or next_event.get('label')}"
    if kind == "ai":
        if next_event.get("tool_calls"):
            names = ", ".join(str(call.get("name")) for call in next_event["tool_calls"])
            return f"模型继续调用工具：{names}"
        return "模型生成答复"
    if kind == "guard":
        return f"进入 Guard 反馈（{next_event.get('guard_kind')}）"
    if kind == "node":
        return next_event.get("content") or f"进入节点 {node}"
    if node == "guard_parse":
        return "进入引用 / 字段校验"
    if node == "guard_support":
        return "进入证据支持性判定"
    if node == "finalize":
        return "进入最终整理"
    if node == "step_limit":
        return "达到步数上限，输出未完成状态"
    return f"进入节点 {node}"


def _describe_basis(event: dict[str, Any]) -> str:
    kind = event.get("kind")
    if kind == "user":
        return "接收用户问题"
    if kind == "tool":
        content = str(event.get("content") or "")
        if "工具调用失败" in content or "失败" in content[:40]:
            return "观察到工具失败，需要换工具或换查询"
        return f"观察到工具返回 {len(content)} 字的结果"
    if kind == "ai":
        if event.get("tool_calls"):
            return "根据问题与已有观察，选择工具补充证据"
        return "认为已有依据足够，停止检索并给出答复"
    if kind == "guard":
        if event.get("guard") == "finalize":
            return "Guard 通过后进入最终整理"
        return "Guard 校验未通过，注入反馈要求修正"
    if kind == "node":
        node = event.get("node")
        if node == "guard_parse":
            return "Guard 1 通过后进入支持性判定"
        if node == "guard_support":
            return "Guard 2 通过后结束或进入最终整理"
        if node == "init_turn":
            return "完成本轮初始化"
        if node == "feedback":
            return "注入反馈并回到 Model A"
    return ""
