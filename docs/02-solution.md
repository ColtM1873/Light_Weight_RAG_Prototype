# 02 · 方案说明

## 1. 总体思路

把“回答”拆成三个可检查的环节：

1. **Model A 负责找证据和作答**：通过工具检索、读上下文/全文、查版本日期，再给出自然语言回复；
2. **Guard 1 负责“引用是否真实”**：Model B 把回复抽成结构化结果，代码逐条校验 `document_id` 是否存在、`quoted_content` 是否能在该文档中找到；
3. **Guard 2 负责“证据是否支持结论”**：Model C 对照工具返回的原文，判断 support / unsupported 并说明理由。

只有两道 Guard 都通过，才进入 `finalize`，由 Model A 输出面向用户的最终整理稿。反馈不是命令，而是“把疑点、原文和缺口摆出来，请模型再核对”，避免把本来正确的回答压服。

## 2. 主图流程

```text
START
  ↓
init_turn
  ├─ 检查文档集合 hash（变了 → 清历史 + 重建索引）
  ├─ 定位本轮真实提问（排除 guard 反馈、上一轮提问）
  └─ 初始化各控制字段
  ↓
call_main_llm（Model A）
  ├─ tool_calls 非空 → tool_node（执行工具）→ 回 call_main_llm
  └─ tool_calls 为空
        ↓
   guard_parse（Model B + 代码校验）
        ├─ 缺字段 / 引用不匹配 → feedback → 回 call_main_llm
        └─ 通过
              ↓
        guard_support（Model C）
              ├─ unsupported → feedback → 回 call_main_llm
              └─ support
                    ↓
                 finalize（注入“给出最终回复”，phase=final）
                    ↓
                 Model A 最终整理 → END
```

- 任意节点后 `step_count >= 100` → `step_limit`，输出“本轮未完成”；
- `finalize` 之后不再做 Guard 校验（明确取舍，状态卡会标注）；
- 两个 Guard 一次通过也必须 `finalize`（2026-11-14 调整）。

## 3. 动态决策

模型不是固定脚本；它根据工具观察选择动作：

| 观察 | 动作 |
|---|---|
| 检索结果零散、缺少上下文 | `retrieve_piece_context` |
| 需要完整条款/适用范围 | `read_full_document` |
| 怀疑文档冲突 | `get_version_date` + 阅读冲突双方 |
| 查询词效果不好 | 换查询词再次 `query_the_docs` |
| 工具调用失败 | 换查询/换工具；仍失败则如实说明无法检索 |
| 已有足够原文 | 停止检索，给出带引用的答复 |

工具实现要点：

- `query_the_docs` 同时检索单句与 300 字小段落，合并后去掉被段落包含的单句；
- 同一 `thread_id` 内的翻页状态隔离，避免不同会话互相污染；
- `retrieve_piece_context` 取同文档前后文，按剩余空间从另一侧补齐，并标注是否到达文档边界；
- `read_full_document` 解析相对仓库根的路径，兼容旧绝对路径；mtime 变化时提示“可能已变动”；
- 所有可预期异常都转成 ToolMessage 字符串，交给模型处理。

## 4. Guard 细节

### 4.1 Guard 1：结构化 + 引用校验

- Model B 只抽取、不评价、不补全：
  ```python
  StructuredAnswer(
      natural_language_reply: str,
      citations: list[Citation],   # document_id + quoted_content
      gaps: list[str],
      status: AnswerStatus(done, conflict, no_support),
  )
  ```
- 代码校验（`validators.py`）：
  - `natural_language_reply` 必填；
  - 引用为空只有在 `no_support=True` 时合法；
  - `conflict=True` 必须有引用；
  - `document_id` 必须存在于 solid 数据；
  - `quoted_content` 必须能在该文档任意单句/段落中找到（去标点后包含，或相似度 ≥ 0.75）。
- 校验失败 → `make_feedback("missing_fields"/"bad_refs")` 注入 HumanMessage，回 Model A 修正。

### 4.2 Guard 2：支持性判定

- Model C 拿到“本轮用户问题 + 全部工具返回全文 + 最后一条 AI 回复”；
- 判断标准：
  - 文档充足但回答“无法回答” → `underanswer`；
  - 文档不足却给明确结论 → `overclaim`；
  - 文档冲突未列明 → `overclaim`；
  - 文档确实支持 → `support`。
- `unsupported` → 反馈 Model A；连续解析失败 5 次后按证据不足处理（防死循环）。

### 4.3 反馈话术

- 只指出疑点、原文和缺口，不替模型重写答案；
- 三种反馈：`missing_fields` / `bad_refs` / `unsupported`；
- 所有反馈带 `guard_feedback=True`、`guard_kind`，下一轮 `init_turn` 会跳过它们，避免把反馈误认成用户提问。

## 5. 最终整理（finalize）

无论是否发生过反馈，Guard 2 通过后都进入 `finalize`：

```text
finalize → 注入 FEEDBACK_FINALIZE（HumanMessage, guard_kind=finalize）
         → phase=final
         → Model A 输出“结论 + 引用 + 缺口 + 状态”的最终稿
         → route 直接 END（不再校验）
```

好处：用户看到的始终是一版经过整理的成品，而不是“草稿 + 反馈 + 草稿”的中间态。
代价：整理稿如果改写结论或引用，不会被自动复查；这一点在状态卡和历史记录中明确标注。

## 6. 框架与技术栈

| 组件 | 版本（验证环境） | 使用的功能 | 承担职责 |
|---|---|---|---|
| Python | 3.13.4 | — | 运行环境 |
| LangGraph | 1.2.21 | `StateGraph` / `Command(goto=...)` / conditional edges / `InMemorySaver` / `astream` | 主图、循环、检查点、流式事件 |
| LangChain | 1.5.0 | `init_chat_model` / `bind_tools` / `with_structured_output` | 模型接入、工具绑定、结构化输出 |
| langchain-core | 1.8.0 | Messages / `@tool` / `InjectedToolArg` | 消息与工具协议 |
| langchain-deepseek | 1.1.1 | DeepSeek provider | `deepseek-chat` |
| langchain-openai | 1.8.0 | OpenAI 兼容 provider | 可选 provider |
| FastAPI | 0.162.7 | 路由、`StreamingResponse` | HTTP API + SSE |
| uvicorn | 0.54.0 | ASGI server | 本地服务 |
| sentence-transformers | 6.3.0 | `BAAI/bge-small-zh-v1.5` | 向量编码 |
| faiss-cpu | 1.16.0 | `IndexFlatIP` / `serialize_index` | 向量检索 |
| pydantic | 2.14.3 | `BaseModel` | 结构化 schema |
| pytest | 8.x | — | 单元/集成测试 |

> 框架本身不带来分数；真正的工作量在 Guard 设计、引用校验、工具循环和失败处理。

## 7. 数据管线

```text
data/ref_files/*.md
  → documents.scan_files()        # 文件名 Dxxx、版本日期、mtime、相对路径
  → documents.classify()          # ID 分配（无 ID 的从 1000 开始）
  → chunking.split_it_up()        # 按中文标点分句（≤512 字），合并成 ~300 字块
  → data/solid_sentences.json
  → data/solid_chunks.json
  → indexing.build_index()        # bge 编码 + FAISS IndexFlatIP
  → data/index_sentences.faiss / index_chunks.faiss
  → data/ref_files_hash.hash      # 文件名 + 内容 hash，变化时重建
```

工程细节：

- solid 数据中保存**相对仓库根**的路径，仓库搬家/提交后仍可用；
- FAISS 的 C++ `fopen` 不支持中文路径，改用 `serialize_index` + Python 写文件；
- 资源懒加载：import 不加载模型/大文件，测试和 `--help` 不受影响；
- 索引重建后清空引用校验缓存、资源缓存和查询翻页状态。

## 8. 技术取舍

| 取舍 | 选择 | 理由 |
|---|---|---|
| 引用校验 | 代码校验 + LLM 抽取 | ID/原文匹配是确定性的，交给 LLM 不稳定 |
| Guard 2 | 独立 prompt 判定 | 与 Model A 视角分离，能指出 overclaim/underanswer |
| Model B/C 模型 | 与 Model A 共用底层模型 | 成本低；后续可替换为更强/更便宜的模型 |
| 是否 finalize | **所有路径都 finalize** | 用户看到的是整理稿；代价是不再复查 |
| 会话持久化 | `InMemorySaver` + `Chat-History` 落盘 | 原型足够；重启不丢原始记录，但 LangGraph 内存历史会丢 |
| 检索 | bge + FAISS 单路向量检索 | 依赖轻、可离线；没有 reranker/混合检索是已知上限 |
| 前端 | 原生 JS 单页 | 快速验证 SSE；不渲染 Markdown 避免 XSS |
| 离线入口 | 脚本化模型驱动真实图/工具 | 无 API key 也能验证接线；明确标注“不证明真实决策效果” |

## 9. 错误处理

| 场景 | 处理 |
|---|---|
| `.env` 缺 key | `load_agent_specs` 抛明确错误，CLI 返回 2 |
| 数据/索引缺失 | 工具返回“请先 build-index”，图仍可运行 |
| hash 检查失败 | 记录 `index_rebuild_error`，不阻塞问答 |
| 索引重建失败 | 记录错误，继续用旧索引/无工具路径 |
| 工具异常 | ToolMessage 返回失败信息，模型决定重试/换工具/说明 |
| Model B 解析失败 | 按缺字段处理，最多 2 次 |
| Model C 解析失败 | 最多 5 次，仍失败按 unsupported 反馈 |
| 模型无限循环 | `step_count >= 100` → `step_limit`，输出本轮未完成 |
| SSE 异常 | 推送 `failed` 事件，不直接断连接 |
| Chat-History 写入失败 | 打印日志，不影响主流程 |
