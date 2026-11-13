> **历史设计记录**：保留 Guard 方案的迭代与取舍过程，部分路径和行为（例如“两个 Guard 一次过就不 finalize”）已经改变。
> 当前行为以 README.md、docs/02-solution.md 和源码为准。

# Guard 层设计方案（Human_Code v0.1 迭代草案）

> 版本：v0.2 草案（按 2026-11-14 反馈修订）
> 范围：Model A 主循环 + 双 Guard（解析/引用校验、支持性判定）+ 复检 + step 上限 + hash 记忆失效
> 状态：**纯设计文档，不含实现代码**。本轮不改任何 `.py` 文件。

---

## 0. 一页速览（决策清单）

| 设计点 | 结论 |
|---|---|
| Guard 触发点 | Model A 的 `state["messages"][-1].tool_calls == []` 时进入 Guard 链，而不是直接 `END` |
| Guard 1 分工 | **Model B 只做"抽取"**（把最后一条 AI 消息解析成结构化 schema）；**引用校验用代码查表**（确定性比对 `solid_sentences` / `solid_chunks`） |
| Guard 2 | Model C 用 `with_structured_output` 判定"文档是否支持该结论"，输出 `support / unsupported` + `error_type` + `reason` |
| 反馈注入 | 统一构造成 `HumanMessage`，打 `guard_feedback` 标记，写回 `messages`，**只有 Model A 有记忆**；Model B/C 每次用独立消息调用 |
| 最终整理 | 仅当**本轮注入过反馈**（`missing_fields` / `bad_refs` / `unsupported`）时才进入 `finalize`；两个 Guard 都一次过则**直接 END**。整理之后不再做任何校验，见 §9 |
| 循环上限 | `step_count` 每个 node +1，100 步封顶返回人话；**不设 guard_round**（避免损伤回复完整性）；**必须同时调高 `recursion_limit`**（默认 25 会先炸），见 §11 |
| hash 失效 | `START` 检查 `do_we_have_file_change()`，变了：清空历史、保留本轮提问、**重建索引**（见 §12 的强制项） |
| 提问定位 | `START` 节点扫描"最后一条真实 HumanMessage"，写入 `state["user_question"] / state["user_question_msg_id"]`，排除 guard 构造的消息与上一轮提问 |
| Model B 重试 | 缺字段：节点内重试 1 次（共 2 次），仍缺 → 人话反馈回 Model A |
| Model C 重试 | 解析失败：节点内重试，建议硬上限 5 次（"直到成功"需要兜底，避免死循环） |

---

## 1. 现状盘点

### 1.1 项目文件（截至本轮）

```
C:\Users\A\Desktop\王者荣耀\v1\
├── .gitignore
├── LICENSE
├── assign_ids_and_versiondate.py     # 文件扫描 / ID 分配 / 版本日期 / hash
├── split_into_chunks_and_save.py     # 分句 / 分块 / solid_*.json 落盘
├── get_index_and_save.py             # FAISS 索引构建（需 torch）
├── query_it_up.py                    # RAG 工具（模块导入时实例化 SentenceTransformer）
├── build_the_world.py                # 主图（目前 system_prompt 已写，其余仍是 Under Construction 注释）
└── guard_plan.md                     # 本文档
```

### 1.2 `build_the_world.py` 的当前状态

| 位置 | 内容 |
|---|---|
| `build_the_world.py:17-25` | 已写入 `system_prompt`（企业知识问答、必须文档完全支持、冲突列举、输出"自然语言回复 + 结构化状态"） |
| `build_the_world.py:27-172` | 整段仍在 `""" --- Under Construction --- """` 三引号注释里，不会执行 |
| `route_to_func` | 函数体只写了 `for` 循环开头，未实现（`build_the_world.py:82-85`） |
| 已有节点 | `we_dont_want_unresponded_human_msg`、`call_main_llm`、`tool_node_front`、`tool_node`、`should_continue_main`、`delegate_instructions`、`consume_submitted_reports`、`produce_html_call_llm` |

> 注意：图里还引用了 `main_nonagent_tools_by_name`、`sub_agent_dict`、`sub_agent_tools`、`query_to_produce_html`、`delegate_instructions`、`consume_submitted_reports`、`store` 等，都尚未定义。本轮 Guard 方案**不处理** sub-agent / HTML 报告分支，只挂在"Model A 无 tool_calls 的出口"。

### 1.3 缺失模块（当前一 import 就 `ModuleNotFoundError`）

| 模块 | 被谁 import | 用途 |
|---|---|---|
| `variant_message_state.py` | `build_the_world.py:11` | `ExtendedMessagesState` |
| `various_prompts.py` | `build_the_world.py:12` | `summery_promt`、`ReAct_system_prompt` |
| `sub_agent_sub_graph_factory_new.py` | `build_the_world.py:13` | `build_sub_agent` |

### 1.4 环境约束（本轮实测）

| 项 | 现状 | 影响 |
|---|---|---|
| provider 集成包 | **一个都没装**（`langchain-openai` / `langchain-deepseek` / ...） | `init_chat_model` 运行到初始化时才抛 `ImportError` |
| torch | 已从兄弟 venv 复制 **2.13.0+cpu**（Python 3.13.4，同版本直接复用；`sentence_transformers` 已验证可 import） | RAG 链路与索引重建不再被阻塞 |
| `transformers` | 5.23.0 | 属 `sentence-transformers 6.3` 允许范围，但较新，API 兼容性待验证 |
| `data/` | **不存在**（本轮不生成，按你的意思先不管） | `solid_sentences.json` / `solid_chunks.json` / FAISS 索引都还没有 |
| data 路径 | 脚本里 `project_dir = Path(__file__).parent.parent` → `王者荣耀/data`，**不是 `v1/data`** | 方案里涉及数据路径的地方要按这个约定 |

### 1.5 与版本相关的 API 事实（实测，写实现时按这个来）

| API | 事实 |
|---|---|
| `REMOVE_ALL_MESSAGES` | 在 **`langgraph.graph.message`**（值 `"__remove_all__"`），`langchain_core.messages` 里**没有**。用法：`[RemoveMessage(id=REMOVE_ALL_MESSAGES), *保留的消息]`，reducer 取该消息**之后**的列表 |
| `RemoveMessage` | `langchain_core.messages.RemoveMessage`，必须传 `id`；删不存在的 id 会抛 `ValueError` |
| `with_structured_output` | 签名 `(schema, *, include_raw=False, **kwargs)`；基类实现**丢弃 `method`/`strict`**，固定走 `bind_tools(..., tool_choice="any")`；`method="json_mode"` 只有在 provider 覆盖实现时才生效 |
| `bind` | `BaseChatModel.bind(**kwargs)` → `_ChatModelBinding`（`RunnableBinding` 子类），`.invoke/.ainvoke` 可用，但**没有** `.bind_tools` |
| `Command` | `Command(graph=None, update=None, resume=None, goto=...)`，节点可返回 `Command(goto=..., update=...)` |
| `recursion_limit` | LangGraph 默认 **25** 个 super-step；100 step 的方案必须显式调高 |
| `MessagesState` / `add_messages` | `langgraph.graph.message`（或 `langgraph.graph` 再导出） |

---

## 2. 图拓扑

```
START
  |
  v
init_turn（hash 检查 / 提问标记 / step 清零）
  |
  v
call_main_llm (Model A)  <─────────────────────────────────────────────┐
  |                                                                    |
  ├─ tool_calls 非空 ──► tool_node ─────────────────────────────────────┤
  |                                                                    |
  └─ tool_calls 为空                                                   |
        |                                                              |
        v                                                              |
   guard_parse（Model B 抽取 + 代码引用校验）                             |
        ├─ 缺字段 / 引用错 ──► feedback ────────────────────────────────┤
        |                                                              |
      pass                                                             |
        v                                                              |
   guard_support（Model C 判定"文档是否支持结论"）
        ├─ unsupported ──► feedback（debate_flag = True）───────────────┤
        |                                                              |
      support                                                          |
        v                                                              |
   finalize（Model A 输出"结构化内容 + 自然说明"的最终回复）
        v
       END（之后不再做任何校验）

任意节点执行后 step_count >= 100 ──► step_limit（输出人话）──► END
```

要点：

1. `init_turn` 是唯一入口（替代原来的 `we_dont_want_unresponded_human_msg`，或与之合并）。
2. `tool_node` 执行完**回到 `call_main_llm`**（`add_edge("tool_node", "call_main_llm")`），循环直到 Model A 不再产生 `tool_calls`。
3. `guard_parse` / `guard_support` / `finalize` 等节点，建议用条件边连接（可读性、可测试性更好），`Command` 留给"同一节点内生成了反馈消息还要跳转"的场景。
4. `feedback` 节点统一构造 HumanMessage，`goto="call_main_llm"`。
5. 任何路由点先查 `step_count`，超限走 `step_limit`。

---

## 3. State 设计

`ExtendedMessagesState` 由 `variant_message_state.py` 定义（该文件目前不存在，需新建）。建议字段：

```python
from typing import Annotated, Literal
import operator
from langgraph.graph.message import MessagesState
from langgraph.graph import add_messages

class ExtendedMessagesState(MessagesState):
    # 现有
    messages: Annotated[list, add_messages]

    # 本轮控制
    step_count: Annotated[int, operator.add]      # 每个 node 返回 {"step_count": 1}
    phase: Literal["draft", "final"]              # draft=正常主循环；final=最终整理阶段
    debate_flag: bool                             # Model C 是否判定过 unsupported（决定要不要复检）

    # 本轮提问（START 时确定，之后只读）
    user_question: str
    user_question_msg_id: str

    # hash 失效
    hash_changed: bool
    history_cleared: bool

    # Guard 解析结果（调试 / 兼容）
    parsed_answer: dict | None
    parse_attempts: int
    support_verdict: dict | None
    support_attempts: int

    # 供 Model C 使用的工具证据缓存（可选，避免重复拼装）
    evidence_pack: str | None
```

> `step_count` 用 `operator.add` reducer：节点只需返回 `{"step_count": 1}`，不读旧值，最不容易写错。注意 `init_turn` 要显式给初值（例如返回 `{"step_count": 0, ...}`），保证 channel 存在。

---

## 4. 消息与记忆约定

### 4.1 三类消息的标记

| 来源 | 类型 | 标记（`additional_kwargs`） | 说明 |
|---|---|---|---|
| 真实用户（本轮） | `HumanMessage` | 无（由 `init_turn` 记录 id 到 state） | 每轮仅一条 |
| Guard 反馈 | `HumanMessage` | `{"guard_feedback": True, "guard_kind": "missing_fields" \| "bad_refs" \| "unsupported" \| "finalize"}` | 只给 Model A 看 |
| Model A 草稿 | `AIMessage` | 无 | Guard 的输入 |
| 工具结果 | `ToolMessage` | 无 | Model C 的证据来源 |

### 4.2 "只有 Model A 有记忆"的落实

- `call_main_llm`：`[SystemMessage(system_prompt)] + state["messages"]`。
- `guard_parse` / `guard_support`：**不传 `state["messages"]`**，每次用节点内临时构造的消息列表（`SystemMessage(guard_prompt) + HumanMessage(打包内容)`），因此 Model B/C 天然无记忆。
- Guard 的判定结果**不写回** `messages`，只写结构化字段（`parsed_answer` / `support_verdict`），避免污染 Model A 的上下文。

### 4.3 历史清理的写法

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.messages import RemoveMessage

def clear_except(state, keep_ids: set[str]):
    keep = [m for m in state["messages"] if m.id in keep_ids]
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *keep]}
```

注意：`add_messages` 对 `REMOVE_ALL_MESSAGES` 的处理是"只保留该消息之后的列表"，所以旧历史被整体替换，`keep` 必须显式放入右侧。

---

## 5. 节点详细设计

### 5.1 `init_turn`（START 入口）

职责：初始化 + hash 检查 + 提问定位 + 未回复消息清理。

伪代码：

```python
async def init_turn(state, config) -> dict:
    msgs = state.get("messages", [])
    update = {"step_count": 0, "phase": "draft", "debate_flag": False}

    # 1) hash 检查
    changed = do_we_have_file_change()
    update["hash_changed"] = changed

    # 2) 定位本轮真实提问（从后往前）
    prev_id = state.get("user_question_msg_id")
    cur = None
    for m in reversed(msgs):
        if not isinstance(m, HumanMessage):
            continue
        if (m.additional_kwargs or {}).get("guard_feedback"):
            continue          # 排除 Guard 构造的消息
        if m.id == prev_id:
            break             # 撞到上一轮提问，停止
        cur = m
        break

    if cur is None:
        # 调用方没有追加新消息 → 无法确定提问，直接返回人话
        return {**update, "messages": [AIMessage(content="没有检测到新的用户提问……")],
                "step_count": 0, "goto_end": True}

    update["user_question"] = cur.content if isinstance(cur.content, str) else str(cur.content)
    update["user_question_msg_id"] = cur.id

    # 3) hash 变了 → 清空历史，只留本轮提问
    if changed:
        update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), cur]
        update["history_cleared"] = True
        # 必须：重建索引（见 §12）
        # get_index_ready()   ← 同步阻塞，先确认 torch 环境
    else:
        update["history_cleared"] = False

    # 4) 可选：清理"上一条没人回复的 HumanMessage"
    #    （原 we_dont_want_unresponded_human_msg 的逻辑可以合并到这里）
    return update
```

要点：

- **提问定位不依赖"最后一条消息"**：必须从后往前跳过 `guard_feedback` 的 HumanMessage，否则 Guard 反馈会被误当成本轮提问。
- 若找不到（例如被 `aupdate_state` 直接塞入内容、或重复调用），应返回明确人话并结束，而不是猜。
- `user_question_msg_id` 会随 checkpointer 持久化，下一轮 START 时用于排除上一轮提问。

### 5.2 `call_main_llm`（Model A）

保持现有结构：

```python
async def call_main_llm(state):
    feed = [SystemMessage(content=main_system_prompt)] + state["messages"]
    resp = await main_model_with_tools.ainvoke(feed)
    return {"messages": [resp], "step_count": 1}
```

`main_model_with_tools = main_model.bind_tools(rag_tools)`，保持现有写法。

### 5.3 路由 `route_after_model_a`

```python
def route_after_model_a(state):
    if state["step_count"] >= MAX_STEPS:
        return "step_limit"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"        # tool_node 之后 add_edge 回 call_main_llm
    if state.get("phase") == "final":
        return END                # finalize 后的最终回复：不再校验
    return "guard_parse"          # ← 普通出口：走 Guard 1，不再直接 END
```

### 5.4 `tool_node` / 子图分支

沿用现有 `tool_node` / `tool_node_front` 逻辑；每个节点返回 `{"messages": [...], "step_count": 1}`。工具名白名单、异常转 `ToolMessage` 的现有做法保留。

`tool_node` 执行完**回到 `call_main_llm`**（`add_edge("tool_node", "call_main_llm")`），形成"调用工具 → Model A 继续推理"的循环，直到某次 `tool_calls` 为空才进入 `guard_parse`。

### 5.5 `guard_parse`（Guard 1）

**分工原则（重要）**：Model B **只负责抽取**，不做"这个内容是否对应文档 ID"的判断；后者用代码查表，确定性强、可测试、无幻觉。

步骤：

```python
async def guard_parse(state):
    draft = state["messages"][-1]
    if not isinstance(draft, AIMessage) or draft.tool_calls:
        return Command(goto=END, update={"step_count": 1})   # 防御：状态异常，直接收尾
    text = draft.content if isinstance(draft.content, str) else str(draft.content)

    # 1) Model B 抽取（最多 2 次：首次 + 重试 1 次）
    parsed = None
    for attempt in range(2):
        parsed = await model_b_structured.ainvoke(
            [SystemMessage(content=GUARD_PARSE_PROMPT), HumanMessage(content=text)]
        )
        if parsed and not missing_required_fields(parsed):
            break

    # 2) 缺字段 → 人话反馈回 Model A
    if parsed is None or missing_required_fields(parsed):
        return Command(
            goto="feedback",
            update={"messages": [make_feedback("missing_fields", missing_required_fields(parsed))],
                    "parse_attempts": 2, "step_count": 1},
        )

    # 3) 引用校验（纯代码）
    errors = validate_citations(parsed.citations)   # 见 §5.5.1
    if errors:
        return Command(
            goto="feedback",
            update={"messages": [make_feedback("bad_refs", errors)],
                    "parsed_answer": parsed.model_dump(), "step_count": 1},
        )

    return Command(goto="guard_support",
                   update={"parsed_answer": parsed.model_dump(), "step_count": 1})
```

#### 5.5.1 引用校验（确定性代码，不是 LLM）

对 `parsed.citations` 逐条（**只认 document_id + 原文片段**）：

| 检查 | 错误示例 |
|---|---|
| `document_id` 存在于库中 | `9999` → 不存在 |
| `citation.quoted_content` 能在该文档的任意片段（单句或小段落）里找到（先精确包含，失败退化为片段相似度） | 引文在该文档里找不到 → 不匹配 |
| `status.no_support == False` 且 `citations` 为空 → 缺字段；`status.conflict == True` 时 citations **必须非空**（用来证明冲突的是什么） | 见 §10 |

**不再校验 `piece_id` / `piece_type`**（按你的要求：引用只需要文档 ID + 原文内容）。

错误信息要能直接变成人话，例如：

```
- 引用【document_id=9999】不存在，库中没有这个文档 ID；
- 引用【document_id=1】的原文片段在该文档中找不到，请核对引用的内容。
```

### 5.6 `guard_support`（Guard 2）

输入打包（每条 ToolMessage 都要有"字段解释"）：

```
[本轮用户问题]
{user_question}

[文档检索记录]
（说明：query_the_docs 返回 list[dict]，字段 document_id=文档 ID，piece_content=原文片段，
  piece_id=片段序号，piece_type=sentence/chunk）
1. 工具：query_the_docs；参数：{...}
   结果：...
2. 工具：retrieve_piece_context；参数：{...}
   结果：...

[待判定的最终回复]
{Model A 的回复原文}

请判断：上述文档内容是否完全支持该回复中的结论？
```

Schema：

```python
class SupportVerdict(BaseModel):
    support: Literal["support", "unsupported"]
    error_type: Literal["none", "overclaim", "underanswer"]  # 无错/过度断言/文档不足却给明确答复
    reason: str
```

判定逻辑：

```python
verdict = None
for attempt in range(MAX_SUPPORT_PARSE_ATTEMPTS):   # 建议 5
    try:
        verdict = await model_c_structured.ainvoke([...])
        if verdict is not None:
            break
    except Exception:
        continue
if verdict is None:
    # 解析始终失败 → 保守处理：按"不支持"走，但在反馈里说明是"系统无法确认依据"
    ...

if verdict.support == "unsupported":
    return Command(goto="feedback", update={
        "messages": [make_feedback("unsupported", verdict.reason)],
        "support_verdict": verdict.model_dump(),
        "debate_flag": True,        # 发生过辩论：最终支持后必须整理 + 复检
        "step_count": 1,
    })
# 判定支持后一律进入 finalize；finalize 之后不再做任何校验
return Command(goto="finalize", update={"support_verdict": verdict.model_dump(), "step_count": 1})
```

> 不设 `guard_round` 上限：辩论轮次不额外限制，避免为了收尾而牺牲回复完整性；循环由 `step_count` 兜底。

### 5.7 `finalize`（最终整理，之后不再校验）

触发条件：Model C 判定 `support`，**且本轮注入过 Guard 反馈**（`feedback_injected == True`）。两个 Guard 都一次过时不触发，直接 `END`。

原因：Model A 是有记忆的，它最后一条 AI 消息可能是在"回应 Model B/C 的质疑"（比如"你的核对是对的，我上一轮标注错了……"），而不是给用户看的最终答复。加这一步后，用户拿到的永远是"结构化内容 + 自然说明"的成品。

流程：

1. `finalize` 节点发一条 `HumanMessage`（`guard_kind="finalize"`）：

   ```
   现在请给出给用户的最终回复。
   要求：
   1. 保留结构化的内容：引用的文档 ID 与原文片段、资料缺口、任务状态；
   2. 把结构化内容和流利、自然的文字说明放在一起，读起来像一段完整的答复；
   3. 这是面向用户的最终版本，不要再提我们之间的核对或讨论过程，也不要用"更正说明"的口吻；
   4. 直接给出结论、依据和必要的说明即可。
   ```

   同时设 `phase="final"`，`goto="call_main_llm"`。
2. `call_main_llm` 生成整理稿后，`route_after_model_a` 看到 `phase=="final"` → 直接 `END`。
3. **之后不再做任何校验**：不跑 Model B，不跑 Model C。

防循环：这一步只发生一次（`phase` 已置为 `final`），不会重复；若此时已接近 `step_count` 上限，路由会先走 `step_limit`。

### 5.8 `feedback`（反馈注入，唯一"回 Model A"的入口）

```python
async def feedback(state):
    return Command(goto="call_main_llm",
                   update={"phase": "draft", "step_count": 1})
```

设计上把"构造 HumanMessage"和"跳回 Model A"分开：构造在 `make_feedback(...)` 纯函数里（可单测），跳转固定。

### 5.9 `step_limit`

```python
async def step_limit(state):
    return {"messages": [AIMessage(content=f"本轮已经进行了 {MAX_STEPS} 步……")],
            "step_count": 1}
```

路由在任意检查点发现 `step_count >= MAX_STEPS` 时 `goto="step_limit"`，`step_limit` 后接 `END`。

---

## 6. Schema 定义（Pydantic 草案）

```python
from typing import Literal
from pydantic import BaseModel, Field

class Citation(BaseModel):
    document_id: int = Field(description="引用片段所属文档 ID")
    quoted_content: str = Field(description="引用到的原文片段，需与库中内容一致")

class AnswerStatus(BaseModel):
    done: bool = Field(description="任务是否完成")
    conflict: bool = Field(description="文档之间是否存在明确冲突")
    no_support: bool = Field(description="是否没有任何文档能支撑回答")

class StructuredAnswer(BaseModel):
    natural_language_reply: str = Field(description="给用户的自然语言回复全文（必填）")
    citations: list[Citation] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="资料缺口")
    status: AnswerStatus

class SupportVerdict(BaseModel):
    support: Literal["support", "unsupported"]
    error_type: Literal["none", "overclaim", "underanswer"]
    reason: str = Field(description="判断理由，需引用具体文档内容")
```

注意：

- 字段是否"缺失"由代码判断（`None`、`natural_language_reply` 为空、空 `citations` 且不满足 §10 的豁免），不要给 schema 加 `missing_fields` 让 Model B 自己报。
- `natural_language_reply` 是必填项：Model B 必须把回复里面向用户的自然语言部分抽出来。
- `with_structured_output(StructuredAnswer)` 返回 pydantic 实例；`include_raw=True` 便于记录 `parsing_error`。

---

## 7. Prompt 草案

### 7.1 Model B（抽取器）

```
你是信息抽取器，不是评审员。你的唯一任务是把助手回复里的内容原样抽出来，
分成 natural_language_reply（给用户的自然语言回复）、citations（引用）、
gaps（资料缺口）、status（任务状态）四部分。
要求：
1. 只抽取回复中明确出现的内容，不要补全、不要推测、不要修正；
2. natural_language_reply 必须是回复里面向用户的自然语言部分，逐字保留；
3. 引用只需要给出 document_id（文档 ID）和 quoted_content（原文片段），不需要其它字段；
4. 如果回复里没有给出某类信息，对应字段留空，不要编造；
5. 不要评价回复好坏，不要输出额外解释。
```

### 7.2 Model C（辩论对手，不是判官）

```
你是另一位资深知识库助手，正在和同事讨论"这个回答是否有充分文档依据"。
请只做判断：文档内容是否完全支持回复里的结论？输出 support / unsupported 和理由。
判断标准：
- 文档充足但回复说"无法回答" → underanswer；
- 文档不足却给出明确答复 → overclaim；
- 文档之间存在冲突而回复未列明 → overclaim 并说明冲突；
- 文档确实完全支持 → support。
说话方式是讨论，不是命令：指出依据、指出缺口，不要替同事重写答案。
```

### 7.3 反馈话术模板（委婉，见 §8）

---

## 8. 反馈话术（关键：不能把 Model A 压服）

原则：

1. **不否定人格**，只摆事实 + 提问；
2. 给 Model A 留出"反驳/举证"的空间；
3. 明确说"如果你有别的依据，可以说出来"。

模板（`missing_fields`）：

```
我这边整理的时候发现，你的回复里好像还没有把「{缺失字段}」说清楚。
方便的话，你再看看呢？如果你觉得这些内容不需要写，也可以告诉我你的理由，
我们可以一起对一下。
```

模板（`bad_refs`）：

```
我对了一下你提到的引用，有几个地方跟库里的记录对不上：
{逐条：预期 vs 实际}
有可能是我的核对方式有问题，也可能是引用时记混了。你再看看呢？
如果这些内容确实出自别处，可以把你看到的原文贴出来，我们一起确认。
```

模板（`unsupported`）：

```
我根据你调用工具拿到的文档，试着核对了一下你的结论：{reason}
也许多看几条检索结果会不一样，你觉得有没有道理？
如果我的理解有偏差，你可以把支持这个结论的原文指出来，我们讨论一下。
```

模板（`finalize`）：

```
刚才的讨论很有帮助，谢谢。现在麻烦你把最终结论重新整理一下，
写成直接给用户的答复（自然语言回复 + 该有的结构化状态）。
```

模板（`step_limit`）：

```
抱歉，这一轮我们来回补充和核对已经达到 {MAX_STEPS} 步上限，先停在这里。
目前能确认的是：{当前已知信息摘要}
如果你需要继续，请补充更具体的问题或缩小范围，我们下一轮接着来。
```

---

## 9. 最终整理与"之后不再校验"

修正后的规则（最终版）：

| 场景 | 行为 |
|---|---|
| 两个 Guard 都一次过（`feedback_injected == False`） | Model A 的最后一条消息就是最终答复，**直接 END**，不注入 `finalize` |
| 本轮注入过反馈（`feedback_injected == True`） | 进入 `finalize`：Model A 输出"结构化内容 + 自然说明"的最终回复，然后**直接 END** |
| `finalize` 之后 | **不再做任何校验**（不跑 Model B、不跑 Model C） |

为什么这样改：

- 不管有没有辩论，Model A 的最后一条 AI 消息都可能带着"回应质疑"的口吻（例如"你的核对是对的，我上一轮标注错了…"），不一定是最终成品；
- 加一次 `finalize`，用户拿到的永远是面向用户的完整答复；
- 之后不再校验，是为了避免"整理稿又被判不合格 → 再改 → 再整理"的无限循环。

代价：`finalize` 这一步只由 prompt 保证质量，如果 Model A 整理时改了结论或引用，不会再被自动校验。这是明确接受的取舍（只发生一次，且由 `step_count` 兜底）。

---

## 10. "合法无引用"的边界（按你的修正）

规则收紧为：

```
允许 citations 为空，当且仅当：
1. status.no_support == True（明确说明没有任何文档能支撑本问题）。

否则：
2. status.conflict == True → citations 必须非空（必须引用冲突双方各自的原文，
   才能说清楚"冲突的是什么"）；
3. 其余情况（尤其 done=True 且 citations=[]）→ 判定"缺字段"。
```

也就是说：

- "文档没有支撑"是**唯一**的合法无引用场景；
- "文档冲突"恰恰**需要有引用**来证明冲突的具体内容；
- "任务未完成"如果是因为缺资料，应当通过 citations/gaps 说明缺什么，而不是空引用。

这条规则写进 `missing_required_fields()` 并单测覆盖，是 Guard 1 最容易出错的地方。

---

## 11. step 计数与上限

### 11.1 计数

- `step_count: Annotated[int, operator.add]`，每个节点返回 `{"step_count": 1}`；
- 用装饰器统一处理，避免每个节点手写：

```python
def counted(fn):
    async def wrapper(state, config=None):
        out = await fn(state) if config is None else await fn(state, config)
        if isinstance(out, Command):
            out.update = {**(out.update or {}), "step_count": 1}
            return out
        return {**(out or {}), "step_count": 1}
    return wrapper
```

> 注意：`Command.update` 是可变字段，包装时不要修改原对象（可 `Command(goto=..., update={...})` 重建）。

### 11.2 上限

- `MAX_STEPS = 100`；
- 每个路由函数先判断 `state["step_count"] >= MAX_STEPS` → `goto="step_limit"`；
- `step_limit` 输出人话后 `END`。

### 11.3 必须同步调高的 `recursion_limit`（易踩坑）

LangGraph 默认 `recursion_limit = 25`。你的 step 上限是 100，**如果不调高，会在第 25 个 super-step 先抛 `GraphRecursionError`**，根本走不到 `step_limit`。

```python
graph.ainvoke(input, config={
    "configurable": {"thread_id": thread_id},
    "recursion_limit": MAX_STEPS + 20,
})
```

---

## 12. hash 检查与记忆清理

### 12.1 检查

`START` 的 `init_turn` 调用 `assign_ids_and_versiondate.do_we_have_file_change()`：

- 首次运行（hash 文件不存在）→ 返回 `True`；
- 文件集合哈希不一致 → `True`。

### 12.2 清理

```python
[RemoveMessage(id=REMOVE_ALL_MESSAGES), 本轮 user message]
```

注意保留 `user_question_msg_id` 对应消息，否则 Model A 会丢失本轮问题。

### 12.3 强制项：hash 变了必须重建索引（你的方案里没提）

只删历史是不够的：`solid_sentences.json` / `solid_chunks.json` / FAISS 索引若还是旧数据，Guard 1 的引用校验和 Guard 2 的文档证据都会基于**过期库**，直接导致误判。

建议在 `init_turn` 里：

```python
if changed:
    update["messages"] = [...清历史...]
    get_index_ready()        # 内部：do_we_have_file_change → return_pipeline_list（会写新 hash）→ split → 建 FAISS
```

注意：

- `get_index_ready()` 依赖 torch（`SentenceTransformer`）——已从兄弟 venv 复制 `torch 2.13.0+cpu`，`sentence_transformers` 已验证可 import；
- 你已确认"同步/后台随意"，**本轮选同步**：实现简单，START 多花一次建索引时间可接受；后续再考虑后台化。

### 12.4 数据路径提醒

`assign_ids_and_versiondate.py` / `split_into_chunks_and_save.py` / `get_index_and_save.py` / `query_it_up.py` 里都是：

```python
project_dir = Path(__file__).resolve().parent.parent   # → C:\Users\A\Desktop\王者荣耀
```

所以 `data/` 实际在 **`王者荣耀\data`**，不在 `v1\data`。`query_it_up.py` 里还有模块级 `solid_sentences = json.loads(path_sentences.read_text())`，**文件不存在时 import 就崩**。写实现前要先确认数据落在哪、是否已生成。

---

## 13. 与现有代码的接入点

| 现有代码 | 改动建议 |
|---|---|
| `build_the_world.py:66-70` `we_dont_want_unresponded_human_msg` | 与 `init_turn` 合并（清理未回复消息的逻辑保留） |
| `build_the_world.py:72-80` `call_main_llm` | 加 `step_count: 1`；其余不变 |
| `build_the_world.py:141-145` `should_continue_main` | 改为 `route_after_model_a`：`tool_calls` 空 → `guard_parse`，不直接 `END` |
| `build_the_world.py:149-166` 的节点/边注册 | Guard 节点与边在此追加 |
| `build_the_world.py:168` `compile(checkpointer=...)` | 不变，但调用方要传 `recursion_limit` |
| `query_it_up.py:get_tools()` | 工具列表直接复用；`already_queried_accumulate_times` 是全局变量，多 thread 并发会串味（见 §14） |
| `various_prompts.py`（缺失） | Guard 的 prompt 建议放在新文件 `guard_prompts.py`，不要塞进这个现有模块 |
| `variant_message_state.py`（缺失） | 按 §3 补字段 |

建议新增文件（实现阶段）：

```
guard_schemas.py      # Pydantic schema（§6）
guard_prompts.py      # Model B/C prompt 与反馈话术模板（§7、§8）
guard_validators.py   # validate_citations / missing_required_fields / make_feedback（纯函数，可单测）
guard_nodes.py        # init_turn / guard_parse / guard_support / finalize / feedback / step_limit
```

---

## 14. 风险与开放问题

| # | 风险 / 问题 | 说明 | 建议 |
|---|---|---|---|
| 1 | 合法无引用被误判为缺字段 | 会逼 Model A 编造引用 | 规则已定（§10）：只有 `no_support=True` 允许空引用 |
| 2 | 过度纠错导致 Model A "屈服" | 你的判断很对，绝对化措辞会压掉正确答案 | 话术模板 + 只在"证据明确"时才反馈 |
| 3 | Guard 无限循环 | Model A 补了又错、Model C 始终不支持 | 仅靠 `step_count`（100）兜底；不设 guard_round，避免损伤回复完整性 |
| 4 | Model C 解析始终失败 | "直到成功"没有兜底 | 硬上限 5 次，失败按保守路径（说明"系统无法确认"） |
| 5 | Model A 输出格式不稳定 | 自由文本里"结构化状态"的形态每次不同，Model B 抽取率低 | 接受这个风险：相信 Model B，抽取失败就反馈让 Model A 改，多改几轮（step 上限兜底）；Schema 已包含 `natural_language_reply` |
| 6 | token 成本与延迟 | 每轮至少 3~4 次 LLM 调用；Model C 要吞全部 ToolMessage（现为全文，不截断） | `finalize` 只执行一次、之后不再校验（§9）；必要时把 Guard 2 做成可开关 |
| 7 | `transformers 5.23` 兼容性 | 较新，`HuggingFaceEmbeddings` 可能有 API 变化 | 跑通 embedding 前先验证；必要时降到 4.x |
| 8 | ~~未装 torch~~ | 已从兄弟 venv 复制 `torch 2.13.0+cpu`（同 Python 3.13.4） | `sentence_transformers` 已验证可 import |
| 9 | provider 包未装 | `init_chat_model` 直接 ImportError | 确定用哪家模型后补装 `langchain-*` |
| 10 | `init_chat_model(model=...)` 传参 | 现有代码把 `llm_provider_name` 当模型名传 | 用 `model="deepseek-chat" + model_provider="deepseek"`（或 `"deepseek:deepseek-chat"`） |
| 11 | 数据路径在上级目录 | `project_dir = parent.parent` | 确认后统一，或改为可配置 |
| 12 | 全局 `already_queried_accumulate_times` | 多 thread 并发会串 | 从 state/config 透传，或按 thread_id 分桶 |
| 13 | `ToolMessage` 字段解释 | 模型看到 `list[dict]` 不知道含义 | 在 Model C 输入里固定附字段说明 |
| 14 | Guard 失败时的"最终兜底" | 100 步用完还没有"支持"结论 | `step_limit` 输出"已知信息摘要 + 未确认点"，不要给假结论 |

---

## 15. 测试与验收

### 15.1 可单测的部分（不依赖 LLM）

| 函数 | 用例 |
|---|---|
| `missing_required_fields(parsed)` | 正常 / no_support 合法无引用 / conflict 缺引用 / done 无引用 / natural_language_reply 为空 |
| `validate_citations(citations)` | 越界 / document_id 不匹配 / 片段内容不匹配 / 全通过 |
| `make_feedback(kind, detail)` | 三类模板输出、带 `guard_feedback` 标记、长度上限 |
| `init_turn` 的提问定位 | 历史中混入 guard 反馈、上一轮提问、无新提问 |
| `clear_except` | 只保留指定消息 + 清空历史 |

### 15.2 假模型联调

`langchain_core.language_models.fake_chat_models.FakeMessagesListChatModel` 可用于跑通图结构（注意它对 `bind_tools` / `with_structured_output` 的支持不完整，可能需要自写一个最小 stub chat model，实现 `bind_tools` 返回预设 AIMessage、`with_structured_output` 返回预设解析结果）。

### 15.3 端到端验收清单

- [ ] 无工具调用时进入 `guard_parse`，而不是直接 END
- [ ] 缺字段 → 重试 1 次 → 仍缺 → 生成 HumanMessage 且带 `guard_feedback` 标记
- [ ] 引用错误（document_id 不存在 / 原文片段在该文档中找不到）能被代码捕获
- [ ] Model C 判定不支持 → 委婉 HumanMessage → Model A 重新生成（`debate_flag=True`）
- [ ] Model C 判定支持（无论是否辩论）→ `finalize` 整理 → 直接 END，不再校验
- [ ] `finalize` 后的最终消息是"结构化内容 + 自然说明"，且没有"更正说明"口吻
- [ ] 100 步上限触发时返回人话，且 `recursion_limit` 未先爆
- [ ] hash 变化时历史被清空，只剩本轮 user message
- [ ] 本轮提问识别正确（Guard 反馈不被误当成提问）
- [ ] Model B/C 每次调用都不携带历史（只有 Model A 有记忆）

---

## 16. 分阶段实施

| 阶段 | 内容 | 前置条件 |
|---|---|---|
| P0 | `guard_schemas.py` / `guard_validators.py` / `make_feedback` + 单测 | 无（纯 Python） |
| P1 | `init_turn`（提问定位 + hash 检查 + step 初始化） | 无数据也能测定位逻辑 |
| P2 | `guard_parse`（Model B + 代码校验） | provider 包 + API key |
| P3 | `guard_support`（Model C）+ 反馈回 Model A | 同上 |
| P4 | `finalize` + `step_limit` + `recursion_limit` 调整 | 同上 |
| P5 | hash → 重建索引 + 端到端联调 | torch（已就绪）+ data |

> 建议 P0 先做：校验器和反馈话术是纯函数，不依赖模型，先把"什么算错、错了怎么说"固定下来，再接线。

---

## 17. 一句话总结

**Model A 负责"答"，Model B 负责"把答案拆成可校验的结构"，代码负责"引用对不对"，Model C 负责"证据够不够"，反馈消息负责"把质疑变成讨论而不是判决"。** 辩论过的结论必须先让 Model A 整理成"给用户的稿子"再交付；剩下全部交给 `step_count` 兜底。

---

## 18. 实现落地记录（2026-11-14 动工）

### 18.1 已实现文件

| 文件 | 内容 |
|---|---|
| `guard_schemas.py` | `Citation` / `AnswerStatus` / `StructuredAnswer` / `SupportVerdict` |
| `guard_prompts.py` | Model B/C prompt、工具字段说明、反馈话术模板 |
| `guard_validators.py` | `missing_required_fields` / `validate_citations` / `make_feedback`（纯函数） |
| `guard_nodes.py` | `make_guard_nodes` 工厂 + 8 个节点 + `route_after_model_a` |
| `variant_message_state.py` | `ExtendedMessagesState` |
| `build_the_world.py` | 接入 Guard 链（`TheWorld` / `assemble_graph`） |
| `tests/test_guard.py` | 25 个单测 / 集成测试 |

### 18.2 与设计的差异（实现时确认）

1. **`finalize` 是最终整理节点**：Model C 判定支持后一律进入，Model A 生成最终回复后直接 `END`，之后不再校验（不再有 `guard_final`）。
2. **节点用 `Command(goto=...)` 路由**，没有全部用条件边；只有 `call_main_llm` 用条件边（因为要看 `tool_calls`）。
3. **`step_count` 用 `operator.add` reducer**（实测缺失初值时也能从 0 开始），节点统一返回 `{"step_count": 1}`。
4. **`query_it_up` 在数据未生成时导入会崩**，`TheWorld` 里做了 `try/except` 降级为空工具列表。
5. **`init_turn` 的 hash 检查失败不中断问答**（如 `data/` 不存在），错误记录在 `index_rebuild_error`。

### 18.3 验证

```
.venv\Scripts\python.exe -m pytest tests -q
25 passed
```

### 18.4 数据处理链路修复（2026-11-14，data 导入后）

| 文件 | 问题 | 修复 |
|---|---|---|
| `assign_ids_and_versiondate.py` | `re_version_date.match` 未调用 | 改为 `match(first_line)` |
| `assign_ids_and_versiondate.py` | 文件对象上调用 `read_text` / `write_text` | 改用 `read()` / `write()` |
| `split_into_chunks_and_save.py` | `dict + dict` | 改为 `{**file_here, "content": ...}` |
| `split_into_chunks_and_save.py` | 字符串上调用 `read_text()` | 改为 `Path(...).read_text()` |
| `get_index_and_save.py` / `query_it_up.py` | faiss 的 C++ fopen 不支持中文路径 | 改用 `serialize_index` / `deserialize_index` |
| `query_it_up.py` | 缺 `import numpy as np` | 补上 |

模型：`bge-small-zh-v1.5` 已下载进本机 HF 缓存，默认 `HF_HUB_OFFLINE=1`（需要联网时设 `HF_HUB_OFFLINE=0`）。
产物（父目录 `data/`）：495 句 / 29 chunk / FAISS 索引 / hash。

仍未解决：无（provider 已装、工具已补；端到端只需提供 API key）。

### 18.5 provider 与工具补全（2026-11-14）

- 已安装 `langchain-deepseek 1.1.1`、`langchain-openai 1.8.0`（`init_chat_model` 可识别 `deepseek` / `openai` 两个 provider）。
- `read_full_document(document_id)`：按 document_id 读取 `ref_files` 全文，返回时附带文档标题与版本日期；若文件 mtime 与建索引时记录不一致，在开头加"文件可能已变动"提示。
- `get_version_date(document_id)`：从 solid 数据返回文档版本日期；未声明时明确说明。
- 验证：`read_full_document(1)` 返回 1139 字全文；`get_version_date(1/2)` 返回 2024-06-30 / 2024-11-18；未知 id 返回明确提示。

### 18.6 运行配置（.env）

项目根目录的 `.env`（模板见 `.env.example`，`.env` 已被 gitignore）：

```
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-...
# LLM_BASE_URL=            # 可选：自定义接口地址
```

`build_the_world.load_agent_specs()` 的读取规则：

- 先加载 `.env`（python-dotenv），**已存在的环境变量优先**；
- `LLM_API_KEY` 为空时回退到 `{PROVIDER}_API_KEY`（如 `DEEPSEEK_API_KEY`）；
- 都没有时抛出明确错误，提示 `.env` 路径。

`TheWorld()` 第二个参数可以省略，省略时自动调用 `load_agent_specs()`。

### 18.7 最终整理节点的加入（2026-11-14）

按"所有情况都加一条最终整理消息"的要求调整：

- `guard_support` 判定 `support` 后**一律** `goto="finalize"`（不再区分是否辩论过）；
- `finalize` 发一条 HumanMessage，要求 Model A 输出"结构化内容 + 自然说明"的最终回复；
- `route_after_model_a` 看到 `phase=="final"` → 直接 `END`，**之后不再校验**；
- 删除 `guard_parse` 里的 `phase=="final"` 复检分支与 `debate_flag` 路由。

端到端验证（deepseek-chat，"出差住宿标准是多少？"）：

- 最终消息是整理稿（结论 + 引用表格），没有"更正说明"口吻 ✅
- 但本轮出现 9 次 `bad_refs` 循环（`step_count=56`），说明引用反馈还可以更"一次到位"（建议反馈里直接给出库中该 piece 的真实 document_id 与内容摘要）。

### 18.8 引用口径简化（2026-11-14）

按你的要求，引用只保留 `document_id` + 原文片段（piece_content）：

- `Citation` 去掉 `piece_id` / `piece_type`；
- `validate_citations` 改为：`document_id` 必须存在 + 原文片段能在该文档任意片段中找到；
- Model B 的抽取 prompt 同步简化。

同一问题「出差住宿标准是多少？」的对比：

| 版本 | step_count | bad_refs 循环 |
|---|---|---|
| 旧（校验 piece_id/piece_type） | 56 | 9 次 |
| 新（只校验 document_id + 内容） | **8** | **0 次** |

最终回复是干净的整理稿（结论 + 引用文档表格），只有一次 `finalize`。

### 18.9 finalize 条件化（2026-11-14）

按你的要求：两个 Guard 都一次过时，不注入 finalize，直接 `END`。

- State 增加 `feedback_injected`（`init_turn` 重置为 `False`）；
- 所有反馈注入都经过 `feedback` 节点，因此在该节点置 `True`；
- `guard_support` 判定 `support` 后：`feedback_injected == True` → `finalize`；否则 → `END`。

实测（同一问题，本次两个 Guard 都一次过）：SSE 只输出 `AI → Tool → Tool → AI → done`，没有 `guard:finalize` 消息。

### 18.10 证据包改为全文（2026-11-14）

三层截断的确认结果：

| 层 | 行为 |
|---|---|
| 前端显示 | 截断到 1000 字（仅 UI，`static/index.html` 的 `MAX_TOOL_CHARS`），不影响模型 |
| Model A | 收到完整 `ToolMessage`，无截断 |
| Model C 证据包 | 原来每条工具结果被 `_truncate` 截到 4000 字 |

按你的要求，证据包改为**全文**（`_truncate` 不再用于证据包）。验证：12000 字的工具结果完整进入证据包（证据包 12251 字），无截断标记。
