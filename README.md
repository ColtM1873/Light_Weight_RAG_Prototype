# 企业知识问答 Agent · 轻量 RAG + 双 Guard

一个基于 **LangGraph** 的企业文档问答原型：

- **Model A** 负责检索、读全文、查版本、作答；
- **Guard 1**（Model B + 确定性代码）把答复抽成结构化结果，并逐条校验引用是否真的来自文档；
- **Guard 2**（Model C）判断“文档是否完全支持该结论”，支持与模型辩论；
- 所有 Guard 通过后统一进入 **finalize**，由 Model A 输出一版面向用户的最终回复；
- 浏览器通过 FastAPI + SSE 逐条显示执行过程，并额外显示 Guard 判定和结构化任务状态；
- 每轮对话自动保存到 `Chat-History/`，便于复盘和整理 Trace。

## 1. 快速开始

### 1.1 环境

- Windows（项目主要在中文 Windows 上开发）
- Python **3.13**（验证版本 3.13.4）
- 依赖安装推荐 [uv](https://docs.astral.sh/uv/)，也可以直接用 pip
- 真实模型默认 **DeepSeek**（`deepseek-chat`），也支持 OpenAI 兼容接口

```powershell
uv venv .venv --python 3.13
.\.venv\Scripts\Activate.ps1
uv pip install --python .venv\Scripts\python.exe `
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
    -r requirements.txt
python run.py setup
```

最省事的方式是双击 **`setup.bat`**：它会创建 `.venv`、安装 `requirements.txt`、下载/加载向量模型并检查索引。它默认使用：

- pip 镜像：`https://pypi.tuna.tsinghua.edu.cn/simple`
- HuggingFace 镜像：`https://hf-mirror.com`

两个镜像都可以用同名环境变量覆盖（`PIP_INDEX_URL` / `HF_ENDPOINT`）。已经自己装好依赖时，只需执行 `python run.py setup`。

> **向量模型**：`BAAI/bge-small-zh-v1.5` 会在 `setup` / 首次运行时自动下载（默认走 `HF_ENDPOINT=https://hf-mirror.com`），**不需要预先准备 HF 缓存**。已缓存时直接本地加载；离线环境才需要预先放好缓存并显式设置 `HF_HUB_OFFLINE=1`。
>
> **torch**：`sentence-transformers` 需要 torch。网络慢或无 GPU 时，可以单独用 CPU 源安装：
>
> ```powershell
> .\.venv\Scripts\Activate.ps1
> uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 1.2 配置

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

```ini
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-...
# LLM_BASE_URL=          # 可选：中转/代理
# AGENT_SYSTEM_PROMPT=   # 可选：覆盖系统提示
```

`.env` 含密钥，已在 `.gitignore` 中排除，**不要提交**。

### 1.3 数据与索引

`data/` 已随仓库提交（8 份合成制度文档、solid 数据、FAISS 索引），开箱即可检索。

文档有增删改时重建：

```powershell
.\.venv\Scripts\Activate.ps1
python run.py build-index --force
python run.py check
```

> 如果服务已经在运行：修改 `data/ref_files/` 内的文档后，请在浏览器**刷新页面（Ctrl+F5）**再提问；服务端会在新一轮开始时检测文档 hash 并重建索引。刷新页面也会开启新的会话。

### 1.4 启动

```powershell
.\.venv\Scripts\Activate.ps1
python run.py serve
# 或双击 start.bat（GBK + CRLF）
```

浏览器打开 **http://127.0.0.1:8010**。

CLI 也可以直接跑：

```powershell
.\.venv\Scripts\Activate.ps1
# 在线（真实模型）
python run.py ask -q "出差住宿标准是多少？"

# 多轮（同一 thread）
python run.py ask -q "咱们工资一般是几号发啊？" -q "那年终奖一般什么时候发呢？"

# 离线 Mock（无需 API key；需要本地 bge 缓存）
python run.py demo --case normal

# 工具失败演示
python run.py ask -q "出差住宿标准是多少？" --fault-tool query_the_docs --fault-times 1
```

### 1.5 测试

```powershell
.\.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pytest tests -q
# 44 passed
```

覆盖：缺字段判定、引用校验、反馈消息、路由、init_turn、图装配、完整图流转（脚本化模型）、上下文档口、故障注入、Chat-History、Guard 结构化状态、步数上限。

### 1.6 Mock 模式（无 API key）

没有 API key、或不想消耗真实模型额度时，用 `demo` 子命令进入离线 Mock：

```powershell
.\.venv\Scripts\Activate.ps1
# 先确保向量模型就绪（只需一次）
python run.py setup

# 进入 Mock：不需要 .env / LLM_API_KEY
python run.py demo --case normal
```

可选场景：

| `--case` | 场景 |
|---|---|
| `normal` | 明确答案 |
| `conflict` | 文档冲突 |
| `no-support` | 资料外问题 |
| `paraphrase` | 改写后的相似问题 |
| `followup` | 上下文追问 |
| `tool-failure` | 工具失败（故障注入） |

也可以把 Mock 执行记录写成 Trace：

```powershell
.\.venv\Scripts\Activate.ps1
python run.py demo --case tool-failure --trace examples/offline/tool-failure.json
```

Mock 模式的行为：

- 使用**真实的主图、真实的 RAG 工具和真实的 Guard 节点**；
- 主模型回复是预设脚本（`src/knowledge_agent/offline.py`），**不调用 DeepSeek API**；
- 只验证流程、工具接线和 Guard 通过路径，**不能证明真实模型的决策效果**；真实效果见 `examples/cases/` 的在线 Trace；
- 仍然需要本地有向量模型（`python run.py setup` 会自动下载），因为 RAG 工具是真实执行的。

## 2. 项目结构

```text
v1/
├── README.md
├── requirements.txt
├── run.py                         # 统一入口：setup / serve / ask / demo / build-index / check
├── start.bat                      # 一键启动（GBK + CRLF）
├── setup.bat                      # 一键准备：venv + 依赖 + 向量模型
├── .env.example
├── src/knowledge_agent/
│   ├── paths.py                   # 项目路径唯一来源
│   ├── config.py                  # .env 读取 + Model A 系统提示
│   ├── state.py                   # LangGraph 扩展状态
│   ├── schemas.py                 # Citation / StructuredAnswer / SupportVerdict
│   ├── prompts.py                 # Model B / C prompt 与反馈话术
│   ├── validators.py              # 纯函数：缺字段、引用校验、反馈构造
│   ├── nodes.py                   # Model A / 工具 / Guard / finalize / feedback
│   ├── graph.py                   # 主图装配 + TheWorld
│   ├── server.py                  # FastAPI + SSE + Chat-History 落盘
│   ├── cli.py                     # setup / ask / demo / serve / build-index / check
│   ├── tracing.py                 # 执行路径 Trace（JSON / Markdown）
│   ├── chat_history.py            # 浏览器会话自动保存
│   ├── offline.py                 # 脚本化模型 + 离线场景
│   ├── static/index.html          # 单页前端
│   └── rag/
│       ├── resources.py           # 懒加载 bge / FAISS / solid 数据
│       ├── tools.py               # 检索、上下文、全文、版本日期
│       ├── documents.py           # 扫描、ID、版本日期、hash
│       ├── chunking.py            # 分句 / 分块
│       └── indexing.py            # solid 数据 + FAISS 重建
├── tests/
├── examples/                      # 8 个在线案例 + 4 个离线 Mock + Trace
├── data/                          # 合成文档、solid 数据、FAISS 索引
├── docs/                          # 需求 / 方案 / 评估 / AI 使用 / 复盘
└── Chat-History/                  # 浏览器会话原始记录（运行时生成，gitignore）
```

## 3. 系统架构

### 3.1 主图

```text
START
  ↓
init_turn（hash 检查 / 定位本轮提问 / 必要时清历史 + 重建索引）
  ↓
call_main_llm（Model A） ◄────────────────────────────┐
  ├─ tool_calls 非空 → tool_node ──────────────────────┘
  └─ tool_calls 为空
        ↓
   guard_parse（Model B 抽取 + 代码引用校验）
        ├─ 缺字段 / 引用不匹配 → feedback → 回 Model A
        └─ 通过
              ↓
        guard_support（Model C 判定）
              ├─ unsupported → feedback → 回 Model A
              └─ support
                    ↓
                 finalize（注入“给出最终回复”，phase=final）
                    ↓
                 Model A 最终整理 → END
```

任意节点后 `step_count >= 100` → `step_limit`，输出“本轮未完成”并 END。

### 3.2 三个角色

| 角色 | 职责 | 记忆 |
|---|---|---|
| Model A | 调工具、观察结果、决定下一步、作答；finalize 时重写最终答复 | 有（`state["messages"]`） |
| Model B | 只抽取：自然语言回复 / 引用 / 缺口 / 状态，不评价 | 无 |
| Model C | 只判定：文档是否完全支持结论 + 理由 | 无 |
| 代码 | `document_id` 存在性 + 引用原文匹配（精确包含或相似度 ≥ 0.75） | — |

设计原则：**确定性的事交给代码，模糊判断才交给 LLM**；反馈是“讨论 + 举证”，不是“判决 + 命令”。

### 3.3 动态决策

Model A 根据工具观察决定下一步，典型动作：

| 观察 | 可能的下一步 |
|---|---|
| 命中片段零散 / 缺少上下文 | `retrieve_piece_context` 查看前后文 |
| 需要完整条款、适用范围 | `read_full_document` 读全文 |
| 怀疑文档冲突 | `get_version_date` 核对版本，读冲突双方原文 |
| 检索词不合适 | 换查询词再次 `query_the_docs` |
| 工具失败 | 换查询 / 换工具；仍失败则如实说明无法检索 |
| 已有足够依据 | 停止检索，给出带引用的答复 |

工具：

| 工具 | 作用 |
|---|---|
| `query_the_docs(query_str, number_of_retrieval, first_try_on_same_query)` | 单句 + 段落双层级检索，合并去重 |
| `retrieve_piece_context(piece_id, piece_type, number_of_context_to_retrieve)` | 取同文档上下文，带边界提示 |
| `read_full_document(document_id)` | 读全文（版本日期 + mtime 变动提示） |
| `get_version_date(document_id)` | 查版本日期 |

### 3.4 结构化输出与任务状态

Guard 1 产出：

```python
StructuredAnswer(
    natural_language_reply: str,
    citations: [Citation(document_id: int, quoted_content: str)],
    gaps: [str],
    status: AnswerStatus(done: bool, conflict: bool, no_support: bool),
)
```

Guard 2 产出：

```python
SupportVerdict(support: Literal["support", "unsupported"],
               error_type: Literal["none", "overclaim", "underanswer"],
               reason: str)
```

浏览器里的蓝色“任务状态”卡和 `examples/` 里的 `task_status` 即由这两者汇总。

## 4. 示例与验收覆盖

完整的输入、实际输出、Trace 索引见 **[examples/README.md](examples/README.md)**。

| 验收点 | 示例 |
|---|---|
| 明确答案 | [01-explicit-answer](examples/cases/01-explicit-answer.md)、05、06、07 |
| 文档冲突 | [02-conflict-followup](examples/cases/02-conflict-followup.md)、04 |
| 资料外问题 | [03-no-support](examples/cases/03-no-support.md)、08 |
| 改写后的相似问题 | [04-paraphrase](examples/cases/04-paraphrase.md) |
| 上下文追问 | 02、05、08 |
| 工具失败 | [06-tool-failure](examples/cases/06-tool-failure.md) |
| 替换文档内容 | [07-doc-replacement](examples/cases/07-doc-replacement/README.md)（基线 350 → 替换后 600） |

## 5. 配置与关键常量

| 位置 | 常量 | 值 |
|---|---|---|
| `nodes.py` | `MAX_STEPS` | 100 |
| `nodes.py` | `MAX_PARSE_ATTEMPTS` | 2 |
| `nodes.py` | `MAX_SUPPORT_PARSE_ATTEMPTS` | 5 |
| `server.py` / `cli.py` | `recursion_limit` | 120 |
| `rag/chunking.py` | `CHUNK_SIZE` / `MAX_SIZE_CAPABLE` | 300 / 512 |
| `rag/resources.py` | `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` |
| `static/index.html` | `MAX_TOOL_CHARS` | 1000（仅显示截断） |

环境变量：

| 变量 | 说明 |
|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` | 模型配置；`LLM_API_KEY` 为空时回退 `{PROVIDER}_API_KEY` |
| `LLM_BASE_URL` | 自定义接口地址 |
| `AGENT_SYSTEM_PROMPT` | 覆盖 Model A 系统提示 |
| `RAG_FAULT_TOOL` / `RAG_FAULT_TIMES` | 工具故障注入（演示工具失败） |
| `HF_ENDPOINT` | 向量模型下载镜像，默认 `https://hf-mirror.com`；网络允许时可设为 `https://huggingface.co` |
| `HF_HUB_OFFLINE` | 可选。只有想强制离线、且本机已有模型缓存时才设置 `1` |

## 6. 已知限制

| 项 | 说明 |
|---|---|
| 单机原型 | 无鉴权、无多租户、无速率限制，不要直接暴露公网 |
| 会话在内存 | `InMemorySaver`，服务重启后 LangGraph 历史丢失；`Chat-History/` 的落盘记录仍保留 |
| 图单例 | 模型配置改动需重启 |
| Model B/C 与 A 同模型 | 成本低但能力同源；可换成独立模型 |
| finalize 后不复查 | 整理稿如果改写结论/引用，不会被自动再校验（状态卡已标注） |
| 引用校验口径 | 只校验 `document_id + 原文片段`，不校验片段定位（`piece_id` 属于内部实现） |
| 状态语义 | Model B 对“附条件的结论”可能给出 `done=False` 且未置 `no_support`，显示为“未完成 / 状态未知”；需要更细的状态枚举 |
| 前端纯文本 | 不渲染 Markdown；长工具结果只截断显示 |
| 文档格式 | 只支持 `.md` / `.txt`；PDF/Word 需先转换 |
| 检索 | 仅 bge + FAISS 向量检索，无 reranker / 关键词混合检索 |
| 首次准备 | `python run.py setup` 需要联网下载 bge 模型；完全离线部署需预先放好 HF 缓存 |
| Mock 边界 | 脚本化模型不根据工具结果动态决策，只验证流程/接线，不证明真实效果 |

## 7. 文档索引

- [docs/01-requirements.md](docs/01-requirements.md)：用户、目标、范围、成功标准
- [docs/02-solution.md](docs/02-solution.md)：调用流程、动态决策、技术取舍、框架职责
- [docs/03-evaluation.md](docs/03-evaluation.md)：测试预期、实际结果、失败与限制
- [docs/04-ai-usage.md](docs/04-ai-usage.md)：AI 辅助范围、个人修改、验证
- [docs/05-retrospective.md](docs/05-retrospective.md)：最难问题、学习收获、下一步
- [docs/06-maintainer-guide.md](docs/06-maintainer-guide.md)：模块说明与扩展点
- [docs/07-guard-design-log.md](docs/07-guard-design-log.md)：Guard 方案历史设计记录
- [docs/archive/Project_Guide_v1.md](docs/archive/Project_Guide_v1.md)：v1 原始结构归档

## 8. 运行记录与隐私

- `Chat-History/` 保存浏览器原始会话（含问题、工具结果、Guard 状态），已在 `.gitignore` 中排除；
- `examples/` 只提交经过挑选的案例，不含 API key；
- `.env`、`.venv/`、`Chat-History/` 均不进入 git。
