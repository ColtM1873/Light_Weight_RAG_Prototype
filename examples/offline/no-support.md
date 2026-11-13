# offline-no-support · 资料外问题（离线 Mock）

- 运行模式：`offline-mock`
- 模型：`scripted (offline)`
- 生成时间：2026-11-14T04:34:24+08:00
- 覆盖验收点：资料外问题

## Turn 1

**问题**：公司年会在哪里举办？

| # | 节点 | 类型 | 动作 / 观察 | 下一步 |
|---|---|---|---|---|
| 1 | input | user | 公司年会在哪里举办？ | 初始化完成；hash_changed=False；history_cleared=False；index_rebuild_error=None |
| 2 | init_turn | node | 初始化完成；hash_changed=False；history_cleared=False；index_rebuild_error=None | 模型继续调用工具：query_the_docs |
| 3 | call_main_llm | ai | 调用工具：query_the_docs {"query_str": "公司年会 举办地点", "number_of_retrieval": 4} | 执行工具 query_the_docs |
| 4 | tool_node | tool | [{"document_id": 3, "piece_content": "# 2025 年年终奖发放章程## 一、目的与适用范围第一条 为肯定员工年度贡献，激励优秀人才，规范年终奖发放管理，特制定本章程。第二条 本章程适用于截至 2025 年 12 月 31 日在职的全体正式员工。试用期员工、已提交离职申请的员工及劳务派遣人员不适用本章程，另有协议约定的从其约定。## 二、发放条件第三条 公司年度经营业绩达成率不低于 80% 时，发放年终奖。达成率低于 80% 的，由董事会决定是否缩减或取消发放。第四条 员工个人… | 模型生成答复 |
| 5 | call_main_llm | ai | 我在现有文档库里没有找到与“公司年会举办地点”相关的资料。 资料缺口：年会地点、时间及组织方。现有文档覆盖差旅报销、年终奖、云办公、招聘入职、薪酬福利、CRM 和信息安全，无法回答这个问题。 | Guard 1 通过：引用 0 条；任务状态 done=False, conflict=False, no_support=True |
| 6 | guard_parse | node | Guard 1 通过：引用 0 条；任务状态 done=False, conflict=False, no_support=True | Guard 2 判定：support（error_type=none）；理由：文档内容支持该回复。 |
| 7 | guard_support | node | Guard 2 判定：support（error_type=none）；理由：文档内容支持该回复。 | 进入 Guard 反馈（finalize） |
| 8 | finalize | guard | 现在请给出给用户的最终回复。 要求： 1. 保留结构化的内容：引用的文档 ID 与原文片段、资料缺口、任务状态； 2. 把结构化内容和流利、自然的文字说明放在一起，读起来像一段完整的答复； 3. 这是面向用户的最终版本，不要再提我们之间的核对或讨论过程，也不要用"更正说明"的口吻； 4. 直接给出结论、依据和必要的说明即可。 | 模型生成答复 |
| 9 | call_main_llm | ai | 我在现有文档库里没有找到与“公司年会举办地点”相关的资料。 资料缺口：年会地点、时间及组织方。现有文档覆盖差旅报销、年终奖、云办公、招聘入职、薪酬福利、CRM 和信息安全，无法回答这个问题。 如果你能提供年会相关文件，我可以再检索。 | 本轮结束 |

**结构化任务状态**：

- Guard 1：通过
- Guard 2：support
- 任务状态：资料不足，未给出确定答案
- done / conflict / no_support：False / False / True
- 资料缺口：
  - 年会地点、时间及组织方

**最终答复**：

> 我在现有文档库里没有找到与“公司年会举办地点”相关的资料。
> 资料缺口：年会地点、时间及组织方。现有文档覆盖差旅报销、年终奖、云办公、招聘入职、薪酬福利、CRM 和信息安全，无法回答这个问题。
>
> 如果你能提供年会相关文件，我可以再检索。

**本轮状态**：步数 8；工具调用 1 次；完成=True；Guard 反馈=无；Model C 辩论=无
