"""离线 Mock 入口：用脚本化模型驱动真实图与真实 RAG 工具。

用途：没有 API key / 模型不可用时，仍能演示：
  - 模型选择工具 → 执行工具 → 观察结果 → 决定下一步 的循环；
  - Guard 1 / Guard 2 的通过路径；
  - 工具失败后的调整。

边界（务必在评审文档中说明）：
  - 这里的主模型回复是**预设脚本**，用于验证流程与工具接线；
  - 它**不能证明真实模型的决策效果**；真实效果请看 ``examples/cases/`` 下的在线 Trace。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from .config import SYSTEM_PROMPT
from .schemas import AnswerStatus, Citation, StructuredAnswer, SupportVerdict


class _StructuredScript:
    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.index = 0

    async def ainvoke(self, messages: Any) -> Any:
        item = self.items[min(self.index, len(self.items) - 1)]
        self.index += 1
        return item


class ScriptedModel:
    """按预设序列返回 AIMessage / StructuredAnswer / SupportVerdict。"""

    def __init__(
        self,
        main_items: list[Any],
        parse_items: list[StructuredAnswer],
        verdict_items: list[SupportVerdict],
    ) -> None:
        self.main_items = list(main_items)
        self.parse_items = list(parse_items)
        self.verdict_items = list(verdict_items)
        self.main_index = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def with_structured_output(self, schema: Any) -> _StructuredScript:
        if schema is StructuredAnswer:
            return _StructuredScript(self.parse_items)
        return _StructuredScript(self.verdict_items)

    async def ainvoke(self, messages: Any) -> AIMessage:
        item = self.main_items[min(self.main_index, len(self.main_items) - 1)]
        self.main_index += 1
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=str(item))


def _support(reason: str = "文档内容支持该回复。") -> SupportVerdict:
    return SupportVerdict(support="support", error_type="none", reason=reason)


@dataclass(frozen=True)
class OfflineScenario:
    key: str
    title: str
    description: str
    questions: list[str]
    main_items: list[Any]
    parse_items: list[StructuredAnswer]
    verdict_items: list[SupportVerdict]
    acceptance_points: list[str]
    fault_tool: str | None = None
    fault_times: int = 0

    def build_model(self) -> ScriptedModel:
        return ScriptedModel(self.main_items, self.parse_items, self.verdict_items)


def _ai_tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


_ACCOMMODATION_REPLY = (
    "根据 D001《员工差旅报销管理制度》第五条：『一线城市（北京、上海、广州、深圳）"
    "每晚不超过 500 元，其他城市每晚不超过 350 元。』出差住宿按城市级别执行该标准，"
    "超标部分由个人承担。"
)
_CONFLICT_REPLY = (
    "检索到两份文档对报销期限的规定不一致：\n"
    "1. D001《员工差旅报销管理制度》（版本日期 2024-06-30）第十一条："
    "『员工应于出差结束后三十天内提交全部报销材料。』\n"
    "2. D002《差旅报销常见问题 FAQ》（版本日期 2024-11-18）Q1："
    "『出差结束后十五天内提交报销材料即可。』\n"
    "两份文档都没有说明哪一份可以覆盖另一份，因此无法确定适用哪一版。"
    "请确认你们适用的是 D001 还是 D002，我再给出对应期限。"
)
_NO_SUPPORT_REPLY = (
    "我在现有文档库里没有找到与“公司年会举办地点”相关的资料。\n"
    "资料缺口：年会地点、时间及组织方。现有文档覆盖差旅报销、年终奖、云办公、"
    "招聘入职、薪酬福利、CRM 和信息安全，无法回答这个问题。"
)
_TRANSPORT_REPLY = (
    "根据 D001《员工差旅报销管理制度》第四条：『出差地市内交通按每人每天 80 元标准包干，』"
    "凭票据实报实销，超出部分自理。"
)

# finalize 统一整理后的"最终版"（离线 Mock 用；真实模型会自己重写）
_ACCOMMODATION_FINAL = _ACCOMMODATION_REPLY + "\n\n如需报销，请按上述标准准备发票与行程凭证。"
_CONFLICT_FINAL = _CONFLICT_REPLY + "\n\n请确认适用 D001 还是 D002 后，我再按对应期限给出结论。"
_NO_SUPPORT_FINAL = _NO_SUPPORT_REPLY + "\n\n如果你能提供年会相关文件，我可以再检索。"
_TRANSPORT_FINAL = _TRANSPORT_REPLY + "\n\n保留好市内交通票据，在标准内凭票报销。"


_ACCOMMODATION_PARSE = StructuredAnswer(
    natural_language_reply=_ACCOMMODATION_REPLY,
    citations=[Citation(document_id=1, quoted_content="其他城市每晚不超过 350 元。")],
    gaps=[],
    status=AnswerStatus(done=True, conflict=False, no_support=False),
)
_CONFLICT_PARSE = StructuredAnswer(
    natural_language_reply=_CONFLICT_REPLY,
    citations=[
        Citation(
            document_id=1,
            quoted_content="第十一条 员工应于出差结束后三十天内提交全部报销材料。",
        ),
        Citation(document_id=2, quoted_content="出差结束后十五天内提交报销材料即可。"),
    ],
    gaps=["两份文档都没有说明优先级或适用范围"],
    status=AnswerStatus(done=False, conflict=True, no_support=False),
)
_NO_SUPPORT_PARSE = StructuredAnswer(
    natural_language_reply=_NO_SUPPORT_REPLY,
    citations=[],
    gaps=["年会地点、时间及组织方"],
    status=AnswerStatus(done=False, conflict=False, no_support=True),
)
_TRANSPORT_PARSE = StructuredAnswer(
    natural_language_reply=_TRANSPORT_REPLY,
    citations=[
        Citation(document_id=1, quoted_content="出差地市内交通按每人每天 80 元标准包干，")
    ],
    gaps=[],
    status=AnswerStatus(done=True, conflict=False, no_support=False),
)


SCENARIOS: dict[str, OfflineScenario] = {
    "normal": OfflineScenario(
        key="normal",
        title="明确答案（离线 Mock）",
        description="模型检索住宿标准后，直接给出带引用的明确答复。",
        questions=["出差住宿标准是多少？"],
        main_items=[
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "出差住宿标准", "number_of_retrieval": 4},
                "call-normal-1",
            ),
            AIMessage(content=_ACCOMMODATION_REPLY),
            AIMessage(content=_ACCOMMODATION_FINAL),
        ],
        parse_items=[_ACCOMMODATION_PARSE],
        verdict_items=[_support()],
        acceptance_points=["明确答案"],
    ),
    "conflict": OfflineScenario(
        key="conflict",
        title="文档冲突（离线 Mock）",
        description="模型检索到两份期限规定不同的文档，列出冲突并请求用户确认。",
        questions=["差旅报销期限是多久？"],
        main_items=[
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "差旅报销期限", "number_of_retrieval": 6},
                "call-conflict-1",
            ),
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "报销 十五天 三十天", "number_of_retrieval": 6},
                "call-conflict-2",
            ),
            AIMessage(content=_CONFLICT_REPLY),
            AIMessage(content=_CONFLICT_FINAL),
        ],
        parse_items=[_CONFLICT_PARSE],
        verdict_items=[_support()],
        acceptance_points=["文档冲突"],
    ),
    "no-support": OfflineScenario(
        key="no-support",
        title="资料外问题（离线 Mock）",
        description="检索不到相关文档时，明确说明资料缺口，不编造答案。",
        questions=["公司年会在哪里举办？"],
        main_items=[
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "公司年会 举办地点", "number_of_retrieval": 4},
                "call-nosupport-1",
            ),
            AIMessage(content=_NO_SUPPORT_REPLY),
            AIMessage(content=_NO_SUPPORT_FINAL),
        ],
        parse_items=[_NO_SUPPORT_PARSE],
        verdict_items=[_support()],
        acceptance_points=["资料外问题"],
    ),
    "paraphrase": OfflineScenario(
        key="paraphrase",
        title="改写后的相似问题（离线 Mock）",
        description="换一种说法询问报销期限，仍然检索到同样的冲突文档。",
        questions=["差旅费需要在几天内提交报销？"],
        main_items=[
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "差旅费 提交报销 期限 几天", "number_of_retrieval": 6},
                "call-paraphrase-1",
            ),
            AIMessage(content=_CONFLICT_REPLY),
            AIMessage(content=_CONFLICT_FINAL),
        ],
        parse_items=[_CONFLICT_PARSE],
        verdict_items=[_support()],
        acceptance_points=["改写后的相似问题", "文档冲突"],
    ),
    "followup": OfflineScenario(
        key="followup",
        title="上下文追问（离线 Mock）",
        description="同一会话内先问住宿标准，再追问交通费，模型分别检索并回答。",
        questions=["出差住宿标准是多少？", "那交通费怎么报？"],
        main_items=[
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "出差住宿标准", "number_of_retrieval": 4},
                "call-followup-1",
            ),
            AIMessage(content=_ACCOMMODATION_REPLY),
            AIMessage(content=_ACCOMMODATION_FINAL),
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "市内交通费 报销标准", "number_of_retrieval": 4},
                "call-followup-2",
            ),
            AIMessage(content=_TRANSPORT_REPLY),
            AIMessage(content=_TRANSPORT_FINAL),
        ],
        parse_items=[_ACCOMMODATION_PARSE, _TRANSPORT_PARSE],
        verdict_items=[_support(), _support()],
        acceptance_points=["上下文追问", "明确答案"],
    ),
    "tool-failure": OfflineScenario(
        key="tool-failure",
        title="工具失败（离线 Mock）",
        description="第一次检索被故障注入打断，模型换一种查询重新检索并完成回答。",
        questions=["差旅住宿标准是多少？"],
        main_items=[
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "出差住宿标准", "number_of_retrieval": 4},
                "call-fault-1",
            ),
            _ai_tool_call(
                "query_the_docs",
                {"query_str": "住宿费 每晚 标准", "number_of_retrieval": 4},
                "call-fault-2",
            ),
            AIMessage(content=_ACCOMMODATION_REPLY),
            AIMessage(content=_ACCOMMODATION_FINAL),
        ],
        parse_items=[_ACCOMMODATION_PARSE],
        verdict_items=[_support()],
        acceptance_points=["工具失败", "明确答案"],
        fault_tool="query_the_docs",
        fault_times=1,
    ),
}


def get_scenario(key: str) -> OfflineScenario:
    try:
        return SCENARIOS[key]
    except KeyError as exc:
        available = "、".join(SCENARIOS)
        raise KeyError(f"未知的离线场景：{key}；可选：{available}") from exc


__all__ = ["ScriptedModel", "OfflineScenario", "SCENARIOS", "get_scenario", "SYSTEM_PROMPT"]
