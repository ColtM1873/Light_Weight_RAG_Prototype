# 03 · 评估与结果

## 1. 测试方式

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

当前结果：**44 passed in 0.4s**（不调用真实 LLM、不加载 embedding 模型）。

| 测试文件 | 数量 | 覆盖内容 |
|---|---:|---|
| `tests/test_guard.py` | 27 | 缺字段、引用校验、反馈构造、路由、init_turn、图装配、完整图流转（脚本化模型）、finalize 必经、步数上限 |
| `tests/test_rag_pipeline.py` | 9 | 数据目录、工具 schema、分句/分块、上下文窗口边界、故障注入 |
| `tests/test_chat_history.py` | 5 | JSON/Markdown 落盘、追加、错误轮次、路径穿越、工具参数记录 |
| `tests/test_tracing_status.py` | 3 | Guard 1/2 结构化结果、失败标记、状态卡 |

集成测试用 `_ScriptedModel` 模拟 LLM（`bind_tools` / `with_structured_output` 返回预设结果），不消耗 API。

## 2. 真实模型案例结果

所有在线案例存于 [`examples/cases/`](../examples/cases/)，索引与覆盖矩阵见 [`examples/README.md`](../examples/README.md)。

| 案例 | 实际结果 | 关键证据 |
|---|---|---|
| 01 明确答案 | 回答一线 500/其他 350；Guard 1 通过、Guard 2 support、finalize | [01-explicit-answer.md](../examples/cases/01-explicit-answer.md) |
| 02 文档冲突 + 追问 | 两轮都识别 30 天 / 15 天（以及 10 个工作日）冲突，请用户确认；未擅自选边 | [02-conflict-followup.md](../examples/cases/02-conflict-followup.md) |
| 03 资料外 | 检索 4 次无年会资料，明确列出资料缺口；`no_support=True` | [03-no-support.md](../examples/cases/03-no-support.md) |
| 04 改写问题 | “差旅费需要在几天内提交报销？”仍命中 D001/D002，列出三种期限冲突 | [04-paraphrase.md](../examples/cases/04-paraphrase.md) |
| 05 上下文追问 | 工资发放 → 年终奖发放；同一 thread 两轮均完成 | [05-followup-salary.md](../examples/cases/05-followup-salary.md) |
| 06 工具失败 | 第一次 `query_the_docs` 返回超时；模型换查询词重试成功，继续给出带引用答复 | [06-tool-failure.md](../examples/cases/06-tool-failure.md) |
| 07 替换文档 | 同一问题：原始文档 350 元/晚 → 替换后 600 元/晚；引用原文同步变化 | [07-doc-replacement/](../examples/cases/07-doc-replacement/README.md) |
| 08 资料边界 + 追问 | 3 轮讨论公司电脑下载网游/下班开黑；T1/T2 完成，T3 给出附条件结论 | [08-policy-gap-followup.md](../examples/cases/08-policy-gap-followup.md) |

### 2.1 两条以上执行路径（模型根据工具结果调整）

- **01**：两次检索 + 两次 `get_version_date` → 判断无冲突 → 作答；
- **02**：检索到 D001/D002 冲突 → 继续检索 → 列冲突并请用户确认；第二轮回合结合“二十来天”继续分析；
- **04**：改写问题 → 检索到三处期限 → 识别冲突；
- **06**：`query_the_docs` 失败 → 换查询词重试 → 成功 → 作答；
- **07**：同一问题、同一模型，只替换 D001 内容 → 答案从 350 变 600。

## 3. 验收点覆盖

| 验收点 | 覆盖 | 说明 |
|---|---|---|
| 明确答案 | ✅ | 01、05、06、07 |
| 文档冲突 | ✅ | 02、04、offline-conflict |
| 资料外问题 | ✅ | 03、08、offline-no-support |
| 改写后的相似问题 | ✅ | 04 |
| 上下文追问 | ✅ | 02、05、08 |
| 工具失败 | ✅ | 06、offline-tool-failure |
| 替换文档内容 | ✅ | 07（基线 vs 替换后） |
| 引用原文支持回答 | ✅ | 所有在线案例的 Guard 1 校验 + 07 对照 |
| 步数上限 | ✅ | `MAX_STEPS=100`；单测 `test_step_limit_returns_incomplete_status` |
| 至少两条执行路径 | ✅ | 见 §2.1 |

## 4. 实际失败与已知限制

| 问题 | 现象 | 影响 / 处理 |
|---|---|---|
| Guard 2 与 Model A 同源 | 可能偏宽松，复杂冲突仍可能漏判 | 已用独立 prompt + 证据包；后续可换独立模型/加规则 |
| finalize 后不复查 | 整理稿若改写结论/引用，不会被自动再校验 | 明确取舍；状态卡与 Trace 标注“未再次结构化校验” |
| 附条件结论的状态 | 08 的 T3 回答“不能认定违规（附两个边界）”，Model B 返回 `done=False` 且未置 `no_support`，状态显示“未完成 / 状态未知” | 状态枚举还不够细；后续可加 `conditional` / `needs_confirmation` |
| 纯闲聊问题 | “你好，请介绍一下你能做什么”会被 Guard 1 要求补引用，触发多轮反馈 | 项目范围是企业知识问答；可在 `init_turn` 增加闲聊分流 |
| 文档 ID 写法 | 模型可能写 “D6” 而不是 “D006” | 内部校验用整数 ID，不影响正确性；展示层可统一格式化 |
| 检索无 reranker | 语义相近但条款无关的片段可能进入上下文 | 已用单句/段落双层级 + 引用校验兜底；检索改进是下一步 |
| `read_full_document` 变动检测 | 用 mtime，不是内容 hash | 同秒修改可能漏检 |
| 工具失败为模拟 | `RAG_FAULT_TOOL` 注入，不是真实网络故障 | 用于可复现演示；真实故障路径同样走 ToolMessage 异常处理 |
| 离线 Mock 边界 | 脚本化模型不根据工具结果动态决策 | 只验证图/工具接线；真实效果看在线案例 |
| 中文路径写 FAISS | 旧代码 `faiss.write_index` 会失败 | 已改为 `serialize_index` + Python 写文件，并加测试/文档 |
| 浏览器历史 | 旧版只存消息摘要，未存 `parsed_answer` | 已升级；旧 3 份记录保留为升级前原始材料 |

## 5. 替换文档实验的验收

- 基线：D001 版本 2024-06-30，其他城市 350 元/晚 → 回答 350；
- 替换：版本 2025-01-01，其他城市 600 元/晚 → 回答 600；
- 两次工具调用结构相同（1 次检索 + finalize），说明变化来自文档内容而非模型随机性；
- 恢复后 D001 SHA256 与实验前一致，`solid_sentences.json` 中 350 出现 1 次、600 出现 0 次；
- 详见 [07-doc-replacement/README.md](../examples/cases/07-doc-replacement/README.md)。

## 6. 如何复核

```powershell
python run.py check
.venv\Scripts\python.exe -m pytest tests -q
python run.py serve          # 打开 8010，按 examples/README.md 的问题逐条试
python run.py demo --case normal
```

每个在线案例的 `.md` 都包含：

1. 执行路径表（节点、动作/观察、下一步）；
2. 结构化任务状态（Guard 1/2、引用、资料缺口、步数）；
3. 最终答复。
