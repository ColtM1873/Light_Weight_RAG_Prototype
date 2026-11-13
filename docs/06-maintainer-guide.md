# 06 · 维护者指南

> 面向后续修改代码的人。快速开始、验收覆盖见根目录 [README.md](../README.md)；历史设计过程见 [07-guard-design-log.md](07-guard-design-log.md)（已不保证与当前行为一致）。

## 1. 模块地图

| 文件 | 职责 | 修改时注意 |
|---|---|---|
| `run.py` | 把 `src/` 加入 `sys.path`，转交 CLI | 只做引导，不放业务 |
| `src/knowledge_agent/paths.py` | 所有路径的唯一来源 | 数据/静态资源路径都从这里取 |
| `config.py` | `.env` 读取、Model A 系统提示、可安全打印的配置摘要 | 不要把 key 打进日志 |
| `state.py` | LangGraph 扩展状态 | 新字段要在这里声明，并在 `init_turn` 给初值 |
| `schemas.py` | `Citation` / `StructuredAnswer` / `SupportVerdict` | Guard B/C 的结构化输出契约 |
| `prompts.py` | Model B/C prompt、反馈话术、步数上限话术 | 改话术要补测试 |
| `validators.py` | 缺字段、引用校验、反馈构造 | 纯函数，不依赖 LLM；最适合单测 |
| `nodes.py` | Model A、工具节点、Guard 1/2、finalize、feedback、step_limit | 路由/步数/故障注入都在这里 |
| `graph.py` | `TheWorld`、`build_graph_with_model`、`assemble_graph` | 图拓扑改动要同步文档/测试 |
| `server.py` | FastAPI + SSE + Guard/状态卡 + `Chat-History` 落盘 | 新增 SSE 类型要同步前端 |
| `cli.py` | `serve` / `ask` / `demo` / `build-index` / `check` | 示例生成入口 |
| `tracing.py` | Trace 事件、`task_status`、Markdown/JSON 导出 | 新事件字段要兼容旧记录 |
| `chat_history.py` | 浏览器会话 JSON/Markdown/索引 | 线程安全、原子写 |
| `offline.py` | 脚本化模型 + 离线场景 | 主模型回复是预设脚本，明确标注边界 |
| `static/index.html` | 单页前端（纯文本） | SSE 事件类型见 `server.py` |
| `rag/resources.py` | 懒加载 bge/FAISS/solid 数据 | import 时不要读大文件 |
| `rag/tools.py` | 四个 RAG 工具 + 会话翻页状态 | 工具异常必须转字符串 |
| `rag/documents.py` | 扫描/ID/版本日期/hash | hash 变化触发重建 |
| `rag/chunking.py` | 分句/分块 | 上线前看 chunk 边界 |
| `rag/indexing.py` | solid + FAISS 重建 | 重建后清缓存 |

## 2. 一次问答的调用栈

```text
CLI/HTTP
  → graph.astream({"messages":[HumanMessage]}, config)
  → init_turn
  → call_main_llm → tool_node → ...
  → guard_parse → (feedback) → call_main_llm
  → guard_support → (feedback) → call_main_llm
  → finalize → call_main_llm (phase=final) → END
```

关键状态字段：

| 字段 | 写入者 | 用途 |
|---|---|---|
| `messages` | 各节点 | 对话历史 |
| `step_count` | 每个节点 +1 | 100 步上限 |
| `phase` | init/finalize/feedback | `final` 时路由直接 END |
| `feedback_injected` | feedback / init | 统计是否发生过 Guard 反馈（不再决定是否 finalize） |
| `user_question` / `user_question_msg_id` | init | 本轮真实提问，供 Model C 和下一轮排除 |
| `hash_changed` / `history_cleared` / `index_rebuild_error` | init | 文档变化与重建结果 |
| `parsed_answer` / `support_verdict` | Guard 1/2 | 调试与 Trace |
| `parse_attempts` / `support_attempts` | Guard 1/2 | 重试统计 |

## 3. 常见修改

### 3.1 加一个工具

1. 在 `rag/tools.py` 用 `@tool` 定义 async 函数（保留类型注解 + docstring，模型靠它理解）；
2. 加入 `get_tools()` 的返回列表；
3. `TheWorld` 自动 `bind_tools`，`tool_node` 无需改；
4. 若工具需要 `thread_id`，用 `config: Annotated[RunnableConfig, InjectedToolArg]`；
5. 补单测：schema 可见参数、异常转字符串、会话隔离。

### 3.2 换模型 / provider

- 改 `.env` 的 `LLM_PROVIDER` / `LLM_MODEL`；
- 装对应 `langchain-*` 包；
- 自定义 gateway 填 `LLM_BASE_URL`；
- 重启服务（图是单例）。

### 3.3 改引用校验规则

编辑 `validators.py`：

- `missing_required_fields`：必填字段、合法无引用条件；
- `validate_citations`：错误口径、相似度阈值（当前 0.75）；
- 补 `tests/test_guard.py` 用例，并确保 `examples/` 的引用仍能通过。

### 3.4 加 Guard 节点 / 改路由

1. 在 `nodes.make_guard_nodes` 加 async 节点（建议返回 `Command(goto=...)`）；
2. 在返回字典注册；
3. 在 `graph.assemble_graph` `add_node`；
4. 如需条件边，加 `add_conditional_edges`；
5. 新 state 字段加到 `state.py`，并在 `init_turn` 给初值；
6. 更新 `tracing.py` 的 `task_status`/状态卡。

### 3.5 改数据/索引

```powershell
python run.py build-index --force
python run.py check
```

- `data/ref_files/` 是文档源；
- `data/solid_*.json` / `index_*.faiss` / hash 是生成物；
- solid 数据里的路径是相对仓库根；
- 文档增删改后必须重建；`force` 跳过 hash 检查。

### 3.6 改前端/SSE

`server.py` 当前发送：

| event | data |
|---|---|
| `message` | `{type: human/ai/tool/guard/status, label, content, ...}` |
| `saved` | `{file}`（Chat-History 文件名） |
| `done` | `{thread_id}` |
| `failed` | `{message}` |

前端在 `static/index.html` 的 `es.onmessage` 里按 `type` 加 class；新增类型记得加 CSS。

## 4. 调试

```powershell
python run.py check                 # 配置/数据/索引/hash
python run.py build-index --force   # 强制重建
python run.py demo --case normal    # 离线看图/工具
python run.py ask -q "..." --trace examples/cases/debug.json
.venv\Scripts\python.exe -m pytest tests -q
```

- 看某轮 Trace：`examples/cases/*.md` 或 `Chat-History/*.md`；
- 看原始事件：同名 `.json` 的 `turns[].events`；
- 看 Guard 输入：`guard_support` 事件有 `evidence_pack_chars`；工具返回全文在 `tool` 事件里；
- 工具失败：`RAG_FAULT_TOOL` / `RAG_FAULT_TIMES`，或 CLI `--fault-tool`。

## 5. 编码约定

- 中文注释/文档为主；代码标识符英文；
- 所有路径从 `paths.py` 取；
- 可预期异常转成用户/模型可读字符串，不要直接让图崩；
- 纯函数放 `validators.py`，LLM/图相关放 `nodes.py`；
- 新增行为必须补测试；示例数字必须来自实际运行。
