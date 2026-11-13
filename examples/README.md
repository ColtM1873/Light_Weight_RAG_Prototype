# Examples · 运行记录与验收覆盖

本目录保存可复现的输入、实际输出和执行 Trace。每个案例包含：

- `.json`：机器可读的完整执行记录（问题、每步消息、工具参数/结果、Guard 结构化结果、最终状态）；
- `.md`：同一份记录的可读版本（执行路径表格 + 结构化任务状态 + 最终答复）。

> 在线案例使用 `deepseek-chat` 真实调用；`offline/` 使用脚本化模型，只验证图/工具/Guard 接线，不能证明真实模型决策效果。

## 1. 在线案例（真实模型）

| 编号 | 问题 | 覆盖验收点 | 记录 |
|---|---|---|---|
| 01 | 出差住宿标准是多少？ | 明确答案、引用原文 | [01-explicit-answer](cases/01-explicit-answer.md) |
| 02 | 我月初出差，住酒店一千块钱还没报呢，怎么报啊 → 到现在，也就二十来天吧 | 文档冲突、上下文追问 | [02-conflict-followup](cases/02-conflict-followup.md) |
| 03 | 公司年会在哪里举办？ | 资料外问题、资料缺口 | [03-no-support](cases/03-no-support.md) |
| 04 | 差旅费需要在几天内提交报销？ | 改写后的相似问题、文档冲突 | [04-paraphrase](cases/04-paraphrase.md) |
| 05 | 咱们工资一般是几号发啊？ → 那年终奖一般什么时候发呢？ | 上下文追问、明确答案 | [05-followup-salary](cases/05-followup-salary.md) |
| 06 | 出差住宿标准是多少？（首次检索注入超时） | 工具失败、模型调整、明确答案 | [06-tool-failure](cases/06-tool-failure.md) |
| 07 | 其他城市出差住宿每晚最多报销多少？ | 替换文档内容、答案随文档改变 | [07-doc-replacement](cases/07-doc-replacement/README.md) |
| 08 | 上班能用公司电脑下载网游吗？ → 下班下载总可以吧 → 下班叫上同事 Steam 开黑？ | 资料边界、上下文追问 | [08-policy-gap-followup](cases/08-policy-gap-followup.md) |

02 / 05 / 08 是从真实浏览器会话 `Chat-History/` 导出的多轮记录；其余由 `run.py ask` 生成。所有在线案例都经过 `finalize`，最终答复是整理稿。为保护隐私，08 中的人名已替换为“同事”。

### 工具失败（06）

`--fault-tool query_the_docs --fault-times 1` 让第一次检索返回“服务响应超时”。记录中可以看到：

1. 模型调用 `query_the_docs`；
2. Tool 返回失败信息（模拟故障 1/1）；
3. 模型换查询词再次调用 `query_the_docs`，成功；
4. 给出带引用的答复，随后 Guard 1/2 通过并 finalize。

### 替换文档内容（07）

`07-doc-replacement/` 内包含原始文档、替换后文档、diff、基线与替换后两条 Trace。只改 D001 的“其他城市 350 元 → 600 元”和版本日期，同一问题的答案从 350 变为 600；实验后已恢复原始文档并校验 SHA256。

## 2. 离线 Mock 案例（无需 API key）

| 编号 | 场景 | 运行命令 | 记录 |
|---|---|---|---|
| offline-normal | 明确答案 | `python run.py demo --case normal` | [normal](offline/normal.md) |
| offline-conflict | 文档冲突 | `python run.py demo --case conflict` | [conflict](offline/conflict.md) |
| offline-no-support | 资料外 | `python run.py demo --case no-support` | [no-support](offline/no-support.md) |
| offline-tool-failure | 工具失败（故障注入） | `python run.py demo --case tool-failure` | [tool-failure](offline/tool-failure.md) |

离线模式的主模型回复是预设脚本，**只能验证流程与工具接线**；真实决策效果以在线案例为准。

## 3. 验收点覆盖矩阵

| 验收点 | 覆盖案例 |
|---|---|
| 明确答案 | 01、05、06、07 |
| 文档冲突 | 02、04、offline-conflict |
| 资料外问题 | 03、08、offline-no-support |
| 改写后的相似问题 | 04 |
| 上下文追问 | 02、05、08 |
| 工具失败 | 06、offline-tool-failure |
| 替换文档内容 | 07（基线与替换后对照） |
| 引用原文支持回答 | 所有在线案例的 Guard 1 校验 + 07 对照 |
| 执行步数上限 | `MAX_STEPS=100`；单测 `test_step_limit_returns_incomplete_status` 验证达到上限时返回“本轮未完成” |
| 至少两条执行路径 | 01（检索→作答）、02（多轮 + 冲突）、04（冲突识别）、06（工具失败后重试）、07（同一问题两种文档内容） |

## 4. 重新生成命令

```powershell
# 单轮在线
python run.py ask -q "出差住宿标准是多少？" --case-id 01-explicit-answer `
  --title "明确答案：出差住宿标准" --acceptance "明确答案" `
  --trace examples/cases/01-explicit-answer.json

# 多轮在线（同一 thread）
python run.py ask -q "咱们工资一般是几号发啊？" -q "那年终奖一般什么时候发呢？" `
  --trace examples/cases/05-followup-salary.json

# 工具失败
python run.py ask -q "出差住宿标准是多少？" --fault-tool query_the_docs --fault-times 1 `
  --trace examples/cases/06-tool-failure.json

# 离线
python run.py demo --case normal --trace examples/offline/normal.json
```

## 5. Trace 字段

每个 JSON 顶层包含：

```json
{
  "case_id": "...",
  "mode": "online | online-web | offline-mock",
  "model": {"provider": "deepseek", "name": "deepseek-chat"},
  "acceptance_points": ["..."],
  "turns": [
    {
      "question": "...",
      "events": [
        {"kind": "user|ai|tool|guard|node", "node": "...", "content": "...",
         "tool_calls": [], "tool_args": {}, "guard": "parse|support"}
      ],
      "task_status": {
        "label": "已完成 | 文档冲突，待用户确认 | 资料不足，未给出确定答案 | 达到步数上限，本轮未完成",
        "guard1": "pass|feedback|not_run",
        "guard2": "support|unsupported|not_run",
        "done": true, "conflict": false, "no_support": false,
        "step_count": 8, "tool_calls": 1
      },
      "citations": [{"document_id": 1, "quoted_content": "..."}],
      "gaps": ["..."],
      "final_answer": "..."
    }
  ]
}
```

`task_status` 来自 Guard 1 的 `parsed_answer.status` 和 Guard 2 的 `support_verdict`；`finalize` 后的整理稿不会再次结构化校验（已在本轮状态中标注）。
