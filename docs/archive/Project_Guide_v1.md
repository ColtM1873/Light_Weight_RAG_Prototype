> **归档说明**：本文档对应 v1 原始结构（代码平铺在仓库根目录、data/ 在上一级）。
> 当前可运行的结构、命令和行为以根目录 README.md、docs/06-maintainer-guide.md 为准。

# Project Guide · 轻量 RAG + 双 Guard 原型

> 面向后续维护者 / 开发者的项目全貌指南。
> 配套设计文档：[guard_plan.md](guard_plan.md)（更细的取舍与讨论过程）。

---

## 0. 一句话

一个基于 **LangGraph** 的企业文档问答原型：Model A 负责检索和作答，两道 Guard（Model B 做结构化解析 + 代码校验、Model C 做证据支持性判定）负责拦截"编造引用"和"文档不足却硬答"，最终通过 FastAPI + SSE 在浏览器里逐条打印 Human / Tool / AI 消息。

---

## 1. 它解决什么问题

普通 RAG 的失败模式主要有两类：

1. **引用不实**：模型声称"根据文档 X"，但引用的原文片段根本不存在、或不属于那个文档；
2. **过度断言**：检索到的文档不足以支撑结论，模型却给了明确答复。

本项目的做法：

| 角色 | 职责 | 是否看对话历史 |
|---|---|---|
| **Model A**（主模型） | 调工具检索、生成答复（自然语言 + 结构化状态） | 有记忆（`state["messages"]`） |
| **Model B**（解析器） | 把 Model A 的答复抽成 `StructuredAnswer`（回复全文 / 引用 / 缺口 / 状态） | 无记忆（每次独立消息） |
| **代码** | 校验引用：`document_id` 是否存在、原文片段能否在该文档中找到 | — |
| **Model C**（辩论对手） | 判断"文档是否完全支持该结论"，输出 support / unsupported + 理由 | 无记忆 |
| **反馈消息** | 把校验失败/证据不足包装成委婉的 HumanMessage 回给 Model A | 只进 Model A 的历史 |

关键设计原则：

- **确定性的事交给代码**（引用比对），**模糊判断才交给 LLM**；
- 反馈是"讨论 + 举证"，不是"判决 + 命令"，避免把本来正确的 Model A 压服；
- 两 Guard 都一次过 → 直接返回；有过反馈 → 让 Model A 整理一版"给用户的最终回复"，之后不再校验。

---

## 2. 快速开始

### 2.1 环境要求

- Windows（本项目主要在中文 Windows 上开发，数据路径含中文）
- Python **3.13**（`.venv` 基于 `C:\Python313`）
- [uv](https://docs.astral.sh/uv/)（依赖管理）
- 可用的 LLM API key（默认 `deepseek`，也支持 `openai`）

### 2.2 安装依赖

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe `
    langchain langgraph langchain-core langchain-huggingface typing_extensions `
    langchain-deepseek langchain-openai `
    faiss-cpu numpy transformers scikit-learn scipy pillow `
    python-dotenv fastapi uvicorn pytest
uv pip install --python .venv\Scripts\python.exe --no-deps sentence-transformers
```

> **torch 的特殊来源**：`sentence-transformers` 硬依赖 torch，但本项目环境里没有走网络下载。当前 `.venv` 的 `torch 2.13.0+cpu`（以及 sympy / networkx / jinja2 / markupsafe / mpmath / setuptools）是从**同级目录 `..\venv of another repository`** 直接复制过来的（对方同为 Python 3.13.4）。换机器部署时，需要自行安装 torch（CPU 版即可）：
>
> ```powershell
> uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 2.3 配置 `.env`

复制模板后填写：

```powershell
Copy-Item .env.example .env
```

```ini
LLM_PROVIDER=deepseek          # deepseek 或 openai
LLM_MODEL=deepseek-chat        # deepseek-chat / deepseek-reasoner / gpt-4o-mini ...
LLM_API_KEY=sk-...             # 你自己的 key
# LLM_BASE_URL=                # 可选：自定义接口地址（中转/代理）
```

`build_the_world.load_agent_specs()` 的读取规则：

- 先加载 `.env`（python-dotenv），**已存在的环境变量优先**；
- `LLM_API_KEY` 为空时回退 `{PROVIDER}_API_KEY`（如 `DEEPSEEK_API_KEY`）；
- 都没有则抛出明确错误，提示 `.env` 路径。

> `.env` 含有 API key，已在 `.gitignore` 中排除，**不要提交**。

### 2.4 准备数据

文档放在**项目上一级目录**的 `data/ref_files/`（注意：不是 `v1/data/`，原因见 §6.1），支持 `.md` / `.txt`，文件名建议 `D001_标题.md`，首行可写 `版本日期：YYYY-MM-DD`。

```powershell
# 生成 solid_sentences.json / solid_chunks.json / FAISS 索引 / hash
.venv\Scripts\python.exe -c "from get_index_and_save import get_index_ready; get_index_ready()"
```

### 2.5 启动

最简单：双击 **`start.bat`**（启动 uvicorn + 等待就绪 + 打开默认浏览器）。

手动启动：

```powershell
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8010
```

浏览器打开 **http://127.0.0.1:8010**，输入问题即可。

### 2.6 跑测试

```powershell
.venv\Scripts\python.exe -m pytest tests -q
# 26 passed
```

---

## 3. 目录结构

```
v1/                                  # 项目根（代码区）
├── build_the_world.py               # 主图装配：TheWorld() / assemble_graph() / .env 读取
├── variant_message_state.py         # ExtendedMessagesState（图状态字段）
├── guard_nodes.py                   # Guard 节点工厂 + Model A/工具节点 + 路由
├── guard_schemas.py                 # Pydantic schema：Citation / StructuredAnswer / SupportVerdict
├── guard_validators.py              # 纯函数：缺字段判定、引用校验、反馈消息构造
├── guard_prompts.py                 # Model B/C prompt、反馈话术、字段说明
├── server.py                        # FastAPI：GET / 页面、GET /api/chat SSE 流
├── static/
│   └── index.html                   # 单页前端（纯文本打印消息，不渲染 Markdown）
├── start.bat                        # 一键启动（GBK + CRLF 编码）
├── .env.example                     # 配置模板（可提交）
├── .env                             # 本地配置（含 key，已 gitignore）
├── tests/
│   └── test_guard.py                # 26 个单测 + 集成测试
├── guard_plan.md                    # 设计文档（决策过程、取舍）
├── Project_Guide.md                 # 本文档
│
│   —— 以下为数据处理/检索链路（早期脚本，非 Guard 层） ——
├── assign_ids_and_versiondate.py    # 扫描文档、分配 ID、解析版本日期、计算 hash
├── split_into_chunks_and_save.py    # 分句 / 分块，生成 solid_*.json
├── get_index_and_save.py            # 加载 bge 模型，构建 FAISS 索引
├── query_it_up.py                   # RAG 工具：query_the_docs / retrieve_piece_context /
│                                    #   read_full_document / get_version_date
└── .gitignore

..\                                  # 项目上一级目录（数据区）
└── data/
    ├── ref_files/                   # 原始文档（8 个 md）
    ├── solid_sentences.json         # 495 条句子
    ├── solid_chunks.json            # 29 个小段落
    ├── index_sentences.faiss        # 句子向量索引
    ├── index_chunks.faiss           # 段落向量索引
    ├── non_ids_map.json             # 无 ID 文件 → 分配 ID 的映射
    └── ref_files_hash.hash          # 文档集合哈希
```

---

## 4. 系统架构

### 4.1 数据准备管线（离线）

```
ref_files/*.md
    │  assign_ids_and_versiondate.scan_files()
    │    - 解析文件名 D001_xxx
    │    - 解析首行"版本日期"
    │    - 计算 sha256
    ▼
split_into_chunks_and_save.split_it_up()
    │    - 按中文标点分句（≤512 字，超长再切）
    │    - 按 300 字合并成 chunk
    ▼
data/solid_sentences.json + data/solid_chunks.json
    │
    ▼
get_index_and_save.get_index_ready()
    │    - SentenceTransformer(BAAI/bge-small-zh-v1.5) 编码
    │    - 写 FAISS 索引（serialize 后由 Python 写文件，绕开中文路径问题）
    ▼
data/index_*.faiss + ref_files_hash.hash
```

`get_index_ready()` 的触发条件是 `do_we_have_file_change()`：

- hash 文件不存在 → 需要建；
- 文档集合的 sha256 与记录不一致 → 需要重建；
- 否则跳过。

### 4.2 在线问答管线（主图）

```
START
  │
  ▼
init_turn ────────────────────────────────────────────────┐
  │  ① hash 检查（变了 → 清历史 + 重建索引）               │
  │  ② 定位本轮真实提问（排除 guard 反馈 / 上一轮提问）    │
  ▼                                                       │
call_main_llm (Model A) ◄──────────────────────────────┐  │
  │                                                     │  │
  ├─ tool_calls 非空 ──► tool_node ─────────────────────┘  │
  │                                                     │  │
  └─ tool_calls 为空                                    │  │
        ▼                                               │  │
   guard_parse（Model B 抽取 + 代码引用校验）             │  │
        ├─ 缺字段 / 引用错 ──► feedback ─────────────────┘  │
        │                                                   │
      pass                                                  │
        ▼                                                   │
   guard_support（Model C 判定"文档是否支持结论"）           │
        ├─ unsupported ──► feedback（feedback_injected=True）┘
        │
      support
        ▼
   feedback_injected == False ──► END（一遍过，不整理）
        │
        └─ True ──► finalize（注入"给出最终回复"）──► Model A ──► END
                     （phase=final，route 直接 END，不再校验）

任意节点后 step_count >= 100 ──► step_limit（人话）──► END
```

### 4.3 图拓扑（代码视角）

| 从 | 到 | 类型 |
|---|---|---|
| `START` | `init_turn` | 固定边 |
| `init_turn` | `call_main_llm` 或 `END` | `Command(goto=...)` |
| `call_main_llm` | `tool_node` / `guard_parse` / `step_limit` / `END` | 条件边 `route_after_model_a` |
| `tool_node` | `call_main_llm` | 固定边 |
| `guard_parse` | `guard_support` / `feedback` / `step_limit` / `END` | `Command` |
| `guard_support` | `finalize` / `END` / `feedback` / `step_limit` | `Command` |
| `finalize` | `call_main_llm` | `Command` |
| `feedback` | `call_main_llm` | `Command` |
| `step_limit` | `END` | `Command` |

---

## 5. 核心模块

### 5.1 `build_the_world.py`

| 符号 | 作用 |
|---|---|
| `system_prompt` | Model A 的系统提示（企业问答、必须文档支持、冲突列举、结构化状态） |
| `ENV_FILE` | 指向项目根 `.env` |
| `load_agent_specs()` | 读取 `.env` / 环境变量，返回 `main_agent_specs` dict |
| `TheWorld(thread_checkpoint, main_agent_specs=None)` | 建模型、加载 RAG 工具、创建节点、装配并编译图 |
| `assemble_graph(nodes, checkpointer=None)` | 只负责把节点装配成图（与模型解耦，测试复用） |

内存要点：

- `rag_tools` 加载失败（例如 `data/` 未生成）会 `warnings.warn` 后降级为**空工具列表**，图仍可运行（只能走"无工具回答"路径）；
- `main_model_with_tools = main_model.bind_tools(rag_tools)`；
- Model B / Model C 是在 `make_guard_nodes` 内部用 `main_model.with_structured_output(...)` 创建的，**与 Model A 共用底层模型**。

### 5.2 `variant_message_state.py`（State 字段）

| 字段 | 类型 | 写入者 | 用途 |
|---|---|---|---|
| `messages` | `list[AnyMessage]` | 各节点 | 对话历史（`add_messages` reducer 自动累积） |
| `step_count` | `Annotated[int, operator.add]` | 每个节点 `+1` | 步数上限（100） |
| `phase` | `Literal["draft","final"]` | `init_turn` / `finalize` / `feedback` | `final` 表示"整理阶段"，路由直接 END |
| `debate_flag` | `bool` | `guard_support` | Model C 是否判过 unsupported（仅调试） |
| `feedback_injected` | `bool` | `feedback` / `init_turn` | 本轮是否注入过反馈 → 决定要不要 `finalize` |
| `user_question` | `str` | `init_turn` | 本轮真实提问（Model C 要用） |
| `user_question_msg_id` | `str` | `init_turn` | 本轮提问的消息 id（下一轮用于排除上一轮） |
| `hash_changed` | `bool` | `init_turn` | 文档是否变化 |
| `history_cleared` | `bool` | `init_turn` | 是否清过历史 |
| `index_rebuild_error` | `str \| None` | `init_turn` | hash/索引重建失败信息（不阻塞问答） |
| `parsed_answer` | `dict \| None` | `guard_parse` | Model B 解析结果（调试） |
| `parse_attempts` | `int` | `guard_parse` | Model B 重试次数 |
| `support_verdict` | `dict \| None` | `guard_support` | Model C 判定结果 |
| `support_attempts` | `int` | `guard_support` | Model C 重试次数 |
| `evidence_pack` | `str \| None` | `guard_support` | 给 Model C 的证据包（调试） |

### 5.3 `guard_nodes.py`

关键常量：

| 常量 | 值 | 含义 |
|---|---|---|
| `MAX_STEPS` | `100` | 单轮最多步数（超限输出人话） |
| `MAX_PARSE_ATTEMPTS` | `2` | Model B 缺字段重试次数（首次 + 1 次） |
| `MAX_SUPPORT_PARSE_ATTEMPTS` | `5` | Model C 解析失败硬上限 |
| `_TOOL_EVIDENCE_LIMIT` | `4000` | 历史遗留的截断上限（**证据包已改用全文**，此常量目前仅 `_truncate` 调试用） |

节点一览：

| 节点 | 行为 |
|---|---|
| `init_turn` | hash 检查、提问定位、必要时清历史 + 重建索引；`Command(goto="call_main_llm")` |
| `call_main_llm` | `SystemMessage(system_prompt) + messages` → `main_model_with_tools.ainvoke` |
| `tool_node` | 逐个执行 `tool_calls`（白名单 + 异常转 `ToolMessage`）→ 回 `call_main_llm` |
| `guard_parse` | Model B 抽取（最多 2 次）→ `missing_required_fields` → `validate_citations` |
| `guard_support` | 组装证据包 → Model C 判定（最多 5 次解析）→ `finalize` / `END` / `feedback` |
| `finalize` | 注入 `guard_kind="finalize"` 的 HumanMessage，`phase="final"` |
| `feedback` | 统一"回 Model A"入口，置 `feedback_injected=True` |
| `step_limit` | 输出步数上限人话并 END |

辅助函数：

- `build_evidence_pack(state)`：把**本轮用户问题 + 全部 ToolMessage 全文 + 最后一条 AI 回复**打包给 Model C；
- `route_after_model_a(state)`：`tool_calls` → `tool_node`；`phase=="final"` → `END`；否则 → `guard_parse`。

### 5.4 `guard_schemas.py`

```python
class Citation(BaseModel):
    document_id: int          # 只要求文档 ID + 原文片段
    quoted_content: str

class AnswerStatus(BaseModel):
    done: bool
    conflict: bool
    no_support: bool

class StructuredAnswer(BaseModel):
    natural_language_reply: str
    citations: list[Citation]
    gaps: list[str]
    status: AnswerStatus

class SupportVerdict(BaseModel):
    support: Literal["support", "unsupported"]
    error_type: Literal["none", "overclaim", "underanswer"]
    reason: str
```

> 引用**不要求** `piece_id` / `piece_type`（它们是内部实现细节，校验和用户回复里都不出现）。

### 5.5 `guard_validators.py`

| 函数 | 规则 |
|---|---|
| `missing_required_fields(parsed)` | `natural_language_reply` 必填；`citations` 为空的唯一合法条件是 `status.no_support == True`；`conflict == True` 时必须有引用 |
| `validate_citations(citations, library=None)` | ① `document_id` 必须存在于库中；② `quoted_content` 必须能在该文档的任意句子/段落中找到（精确包含或相似度 ≥ 0.75） |
| `make_feedback(kind, detail)` | 构造带 `guard_feedback=True` / `guard_kind=...` 的委婉 HumanMessage |
| `content_to_text(content)` | 兼容 `str` 与 `list[dict]` 的消息内容 |
| `load_solid_library()` | 懒加载 `solid_sentences.json` / `solid_chunks.json`，不存在时返回 `None`（跳过校验） |

### 5.6 `guard_prompts.py`

- `MODEL_B_SYSTEM_PROMPT`：抽取器，只抽取、不评价、不补全；
- `MODEL_C_SYSTEM_PROMPT`：辩论对手，判断 support/unsupported 并给理由，不下命令；
- `TOOL_RESULT_FIELD_DOC`：工具返回字段说明（Model C 需要知道 `list[dict]` 的含义）；
- 反馈话术：`FEEDBACK_MISSING_FIELDS` / `FEEDBACK_BAD_REFS` / `FEEDBACK_UNSUPPORTED` / `FEEDBACK_FINALIZE`；
- `STEP_LIMIT_MESSAGE`：步数上限人话。

### 5.7 `server.py`

| 路由 | 说明 |
|---|---|
| `GET /` | 返回 `static/index.html` |
| `GET /api/chat?message=...&thread_id=...` | SSE 流：逐条推送图上产生的消息 |

实现要点：

- 进程内**单例图 + `InMemorySaver`**（按 `thread_id` 隔离会话）；首次请求时才构建图（会加载 bge 模型，几秒）；
- `graph.astream(..., stream_mode="updates")`，遍历每个节点产出的 `messages` 推送；
- `RemoveMessage` / 空内容消息不推送；带 `tool_calls` 但 content 为空的消息显示为「→ 调用工具：xxx」；
- 异常通过 `failed` 事件推给前端，不直接断连接；
- `recursion_limit=120`（必须 ≥ step 上限 + 余量，否则到不了 `step_limit`）。

SSE 协议：

| event | data |
|---|---|
| `message` | `{type: human/ai/tool, label: str, content: str, id: str, node: str}` |
| `done` | `{thread_id: str}` |
| `failed` | `{message: str}` |

### 5.8 `static/index.html`

原生 JS 单页：

- 输入框 + 发送按钮，回车发送；
- 用 `EventSource` 连接 `/api/chat`，逐条追加气泡（纯文本 `textContent`，**不渲染 Markdown**）；
- 消息样式：Human 蓝、AI 绿、Tool 黄、错误红；
- Tool 内容超过 1000 字时**只截断显示**（`MAX_TOOL_CHARS`），并提示总长度；
- 页面生命周期内 `thread_id` 固定 → 支持多轮追问；刷新页面即开新会话。

### 5.9 数据处理四件套（早期脚本）

| 文件 | 说明 |
|---|---|
| `assign_ids_and_versiondate.py` | 扫描 `ref_files`、分配 ID（`D001` 或 1000+）、解析版本日期、计算整体 hash |
| `split_into_chunks_and_save.py` | 分句 / 分块，写 `solid_*.json` |
| `get_index_and_save.py` | 加载 bge 模型、构建 FAISS 索引；`get_index_ready()` 是总入口 |
| `query_it_up.py` | RAG 工具实现 + 文档元数据（`read_full_document` / `get_version_date`） |

工具说明：

| 工具 | 返回 |
|---|---|
| `query_the_docs(query_str, number_of_retrieval, first_try_on_same_query)` | `list[dict]`（document_id / piece_content / piece_id / piece_type）或"没有更多匹配项" |
| `retrieve_piece_context(piece_id, piece_type, number_of_context_to_retrieve)` | `list[str]` 上下文 |
| `read_full_document(document_id)` | 文档全文（带标题/版本日期头；mtime 变化时加"可能已变动"提示） |
| `get_version_date(document_id)` | 版本日期字符串 |

---

## 6. 数据与索引

### 6.1 为什么 `data/` 在上一级目录

数据处理脚本里统一使用：

```python
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
```

即 `C:\Users\A\Desktop\王者荣耀\data`。这是**早期脚本的既定约定**（代码在 `v1/`，数据在项目根）。不要移动到 `v1/data`，否则 `assign_ids` / `split` / `get_index` / `query_it_up` 四处的路径都要改。

### 6.2 产物

| 文件 | 内容 |
|---|---|
| `data/ref_files/*.md` | 原始文档（`.md` / `.txt`） |
| `data/solid_sentences.json` | 每条：`path / name / mtime / version_date / file_prefix / title / id / content` |
| `data/solid_chunks.json` | 同上，`content` 是合并后的 chunk |
| `data/index_sentences.faiss` | 句子向量索引（`serialize_index` 字节） |
| `data/index_chunks.faiss` | 段落向量索引 |
| `data/non_ids_map.json` | 无 ID 文件名 → 分配 ID 的映射 |
| `data/ref_files_hash.hash` | 文档集合 sha256，用于判断是否重建 |

当前数据规模：8 个文档 / 495 句 / 29 个 chunk。

### 6.3 hash 与重建

- 首次运行（hash 文件不存在）→ `changed=True`；
- 文档增删改 → 整体 hash 变化 → `changed=True`；
- `init_turn` 发现 `changed=True`：清空历史（只保留本轮提问）+ 调用 `get_index_ready()` 重建索引；
- 重建失败不会中断问答，错误记录在 `state["index_rebuild_error"]`。

### 6.4 模型缓存与离线

- 向量模型：`BAAI/bge-small-zh-v1.5`，已下载到本机 HF 缓存（`C:\Users\A\.cache\huggingface\hub`），维度 512；
- 两个脚本默认 `HF_HUB_OFFLINE=1`，避免每次启动都联网检查；需要联网时设 `HF_HUB_OFFLINE=0`；
- `HF_ENDPOINT=https://hf-mirror.com`（国内镜像）。

---

## 7. 配置参考

### 7.1 `.env`

| 变量 | 必填 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | 否 | `deepseek`（默认）或 `openai` |
| `LLM_MODEL` | 否 | 默认 `deepseek-chat` |
| `LLM_API_KEY` | 是 | API key；为空时回退 `{PROVIDER}_API_KEY` |
| `LLM_BASE_URL` | 否 | 自定义接口地址 |
| `AGENT_SYSTEM_PROMPT` | 否 | 覆盖内置 `system_prompt` |

### 7.2 端口

- 服务端口：**8010**（`start.bat` 和 `server.py` 文档注释一致；8000 曾被其他程序占用）；
- 改端口：改 `start.bat` 里的 `set "PORT=8010"` 和启动命令即可。

### 7.3 关键常量

| 位置 | 常量 | 值 |
|---|---|---|
| `guard_nodes.py` | `MAX_STEPS` | 100 |
| `guard_nodes.py` | `MAX_PARSE_ATTEMPTS` | 2 |
| `guard_nodes.py` | `MAX_SUPPORT_PARSE_ATTEMPTS` | 5 |
| `server.py` | `recursion_limit` | 120 |
| `static/index.html` | `MAX_TOOL_CHARS` | 1000（仅显示） |

---

## 8. 运行与调试

### 8.1 启动 / 停止

- 双击 `start.bat`：检测 8010 → 已占用则只开浏览器；否则新窗口启动 uvicorn、等待就绪、打开浏览器；
- 停止：关掉 "RAG Guard Server" 命令行窗口（或 `Ctrl+C`）；
- 改代码后**必须重启服务**（uvicorn 未开 `--reload`）。

### 8.2 观察事件流

浏览器开发者工具 → Network → `api/chat` 请求 → EventStream 里能看到每条 `message` 事件。也可以用命令行：

```powershell
curl.exe -N "http://127.0.0.1:8010/api/chat?message=%E4%BD%A0%E5%A5%BD&thread_id=test-1"
```

一次典型的"一遍过"流程：

```
AI    · 我会查询文档库…
Tool  · query_the_docs
Tool  · query_the_docs
AI    · （最终答复）
done
```

有反馈时会出现 `Human · guard:bad_refs` / `Human · guard:missing_fields` / `Human · guard:unsupported` / `Human · guard:finalize`，以及 `AI` 的修正稿。

### 8.3 常见问题排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 启动报 `WinError 10048` | 端口被占 | 关掉旧进程，或改 `start.bat` 的 PORT |
| `init_chat_model` 报 ImportError | 没装 provider 包 | `uv pip install langchain-deepseek` / `langchain-openai` |
| 401 / 认证失败 | key 无效或未填 | 检查 `.env` 的 `LLM_API_KEY` |
| `RuntimeError: 没有找到 API key` | `.env` 缺失或为空 | 按 §2.3 填写 |
| 回答一直不调用工具 | `data/` 未生成，工具降级为空 | 先跑 §2.4 的索引构建 |
| Model A 反复被反馈 | 引用错误 / 缺字段 | 看 SSE 里的 `guard:*` 消息；确认 `data/` 与 `guard_validators` 规则 |
| 模型完全没输出 | SSE 连接失败 / 服务未启动 | 检查服务窗口日志与浏览器 Network |
| `GraphRecursionError` | `recursion_limit` 太低 | 保持 ≥ 120（`server.py` 已设） |
| 中文路径写 FAISS 报错 | faiss 的 C++ `fopen` 不支持非 ASCII | 已改为 `serialize_index` + Python 写文件，勿回退 |
| HF 模型下载失败 | 网络/镜像问题 | 使用本机已缓存模型；必要时 `HF_HUB_OFFLINE=0` 重试 |
| `.env` 中文乱码 / bat 乱码 | 编码不匹配 | bat 必须是 GBK + CRLF；代码文件保持 UTF-8 |

---

## 9. 测试

运行：

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

26 个测试的分布：

| 类别 | 覆盖内容 |
|---|---|
| `missing_required_fields` | 完整 / `no_support` 允许空引用 / `conflict` 必须有引用 / 空回复 / `None` |
| `validate_citations` | 正确引用 / chunk 内容 / 未知 document_id / 内容不匹配 / 空库 |
| `make_feedback` | 标记字段、未知类型报错 |
| `route_after_model_a` | tool_calls / 普通回答 / 步数上限 / `phase=final` |
| `init_turn` | 提问定位、排除 guard 反馈、hash 变化清历史、hash 检查异常不崩 |
| 图编译 | 节点/边装配合法 |
| 集成（脚本化模型） | ① 一遍过直接 END；② 缺字段 → 反馈 → 补充 → END；③ 辩论 → finalize → 整理 → END |

集成测试用 `_ScriptedModel` 模拟 LLM（`bind_tools` / `with_structured_output` 返回预设结果），不消耗真实 API。

---

## 10. 已知限制与设计取舍

| 项 | 说明 |
|---|---|
| 单机原型 | 无鉴权、无多租户、无速率限制；不要直接暴露公网 |
| 会话在内存 | `InMemorySaver`，服务重启后历史丢失；换 `SqliteSaver` 可持久化 |
| 图单例 | 进程内只构建一张图；模型配置改动需重启 |
| 模型 B/C 与 A 同模型 | 成本低但能力同源；需要更强判定可以换成单独模型 |
| Guard 成本 | 每轮可能多 2~4 次 LLM 调用 + 循环；`step_count` 100 封顶 |
| `finalize` 后不校验 | 整理稿如果改了结论/引用，不会被自动复查（明确取舍） |
| 引用校验口径 | 只校验 `document_id + 原文片段`，不校验片段定位（`piece_id` 属于内部实现） |
| `read_full_document` 变动检测 | 用 mtime 而非内容 hash，理论上同秒修改可能漏检 |
| 前端截断 | Tool 内容超过 1000 字只截断显示，不影响模型 |
| 文档格式 | 只支持 `.md` / `.txt`；PDF/Word 需先转换 |
| `data/` 在上级目录 | 早期约定，移动会连带修改四个脚本 |

---

## 11. 扩展指南

### 11.1 加一个新工具

1. 在 `query_it_up.py` 里用 `@langchain_tool` 定义函数（保留类型注解 + docstring，模型靠它理解工具）；
2. 加入 `get_tools()` 的返回列表：

```python
async def get_tools():
    return [query_the_docs, retrieve_piece_context, read_full_document, get_version_date, my_new_tool]
```

3. `TheWorld` 会自动建 `tools_by_name` 并 `bind_tools`，`tool_node` 无需改动。

### 11.2 换模型 / provider

- 改 `.env`：`LLM_PROVIDER`、`LLM_MODEL`（如 `openai` + `gpt-4o-mini`）；
- 装对应包：`langchain-openai` / `langchain-deepseek` / 其他 `langchain-*`；
- 需要自定义 gateway 时填 `LLM_BASE_URL`；
- 重启服务生效。

### 11.3 改引用校验规则

编辑 `guard_validators.py`：

- `validate_citations`：错误口径、相似度阈值（当前 0.75）；
- `missing_required_fields`：必填字段、合法无引用条件。

改完补 `tests/test_guard.py` 的用例。

### 11.4 加 Guard 节点 / 改路由

1. 在 `guard_nodes.make_guard_nodes` 里加 async 节点（建议返回 `Command(goto=...)`）；
2. 在返回字典里注册；
3. 在 `build_the_world.assemble_graph` 里 `builder.add_node(...)`；
4. 若需要条件边，加 `add_conditional_edges`（参考 `route_after_model_a`）；
5. 若新增 state 字段，加到 `variant_message_state.py`，并在 `init_turn` 里给初值。

### 11.5 换持久化 / 接其他框架

- `TheWorld(checkpointer)` 接受任意 LangGraph checkpointer；`server.py` 当前传的是 `InMemorySaver()`，换成 `SqliteSaver.from_conn_string("checkpoints.db")` 即可持久化；
- 集成到其他服务时，复用 `TheWorld()` + `graph.ainvoke/astream` 即可，`server.py` 只是演示暴露方式。

### 11.6 前端渲染 Markdown

当前 `index.html` 用 `textContent` 纯文本显示。如需渲染，引入 `marked` 等库并把 `body.textContent = content` 改为 `innerHTML = marked.parse(content)`（注意 XSS，仅限可信内容）。

---

## 12. 术语表

| 术语 | 含义 |
|---|---|
| Model A | 主模型（LangGraph 节点 `call_main_llm`），有对话记忆，可调工具 |
| Model B | 解析模型（`guard_parse` 内），把 A 的答复抽成 `StructuredAnswer` |
| Model C | 判定模型（`guard_support` 内），判断文档是否支持 A 的结论 |
| Guard 1 | `guard_parse`：解析 + 代码引用校验 |
| Guard 2 | `guard_support`：Model C 证据支持性判定 |
| `feedback_injected` | 本轮是否注入过任何反馈；决定是否需要 `finalize` |
| `finalize` | 最终整理节点：要求 Model A 输出"结构化内容 + 自然说明"的成品 |
| `step_count` | 图执行步数，上限 100 |
| solid 数据 | `solid_sentences.json` / `solid_chunks.json`（带文档元信息的句子/段落） |
| hash 文件 | `ref_files_hash.hash`，判断文档集合是否变化 |

---

## 13. 相关文档

- [guard_plan.md](guard_plan.md)：设计动机、方案迭代、每轮反馈与取舍（含 §18 实现落地记录）；
- `.env.example`：配置模板；
- 测试：`tests/test_guard.py`（最贴近行为的"活文档"）。
