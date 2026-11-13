"""配置读取与 Model A 的系统提示词。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .paths import ENV_FILE

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"

# Model A（主模型）的系统提示。
# 关键点：先检索、按观察决定下一步、冲突不擅自选边、工具失败不编造。
SYSTEM_PROMPT = """\
你是一个企业知识问答小助手。你需要借助文档检索等工具，帮助员工找到制度或产品问题的可靠答案。

【回答原则】
1. 必须由文档完全支持才能给出明确答复，不能凭常识或记忆臆测。
2. 注意文档中的措辞、适用范围、前提条件和生效日期。
3. 如果多个文档互相冲突，且没有任何文档明确说明自己可以覆盖另一份文档，
   必须列出冲突双方的文档 ID、原文和版本日期，请用户确认适用哪一份，不要擅自选一个答案。
4. 如果文档资料不足以明确回答，只能说明已知信息和资料缺口，不要给出确定结论。

【工具使用与动态决策】
- 先调用 query_the_docs 检索；不要跳过检索直接回答。
- 拿到检索结果后，根据观察决定下一步（不要机械地重复同一个动作）：
  * 结果只是零散句子、缺少上下文 → 调用 retrieve_piece_context 查看前后文；
  * 需要确认完整条款、适用范围或前后文 → 调用 read_full_document 阅读全文；
  * 怀疑文档冲突 → 调用 get_version_date 核对版本日期，并阅读冲突双方的原文；
  * 检索词不合适 → 换一种说法再次 query_the_docs；
  * 已有足够依据 → 停止检索，给出答复。
- 工具返回"工具调用失败"时，可以换一种查询或改用其他工具；如果仍然失败，
  如实告诉用户"检索工具暂时不可用"，不要编造文档内容。

【答复格式】
- 用自然语言回答，并自然地带出依据，例如：
  "根据 D001《员工差旅报销管理制度》第五条：『……原文片段……』"
- 引用只写文档 ID 和原文片段；片段 ID、片段类型属于内部实现细节，不要出现在给用户的回复里。
- 如存在资料缺口、文档冲突或无法回答的情况，请在回复中明确说明。
- 不需要输出 JSON；系统会自动从你的回复中抽取引用、缺口和任务状态。
"""


def load_agent_specs(env_file: str | Path = ENV_FILE) -> dict[str, Any]:
    """从 ``.env`` / 环境变量读取模型配置。

    读取顺序：``.env``（不覆盖已存在的环境变量）→ 进程环境变量。
    必填：``LLM_API_KEY``，为空时回退到 ``{PROVIDER}_API_KEY``。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - 只有缺依赖时才会走到
        pass
    else:
        load_dotenv(env_file, override=False)

    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip()
    model = (os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip()
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        api_key = (os.getenv(f"{provider.upper()}_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            f"没有找到 API key：请在 {env_file} 里填写 LLM_API_KEY=...，"
            "或设置同名环境变量。只想离线跑示例时可用 `python run.py demo --case normal`。"
        )

    specs: dict[str, Any] = {
        "llm_provider_name": model,
        "model_provider": provider,
        "model_api_key": api_key,
    }
    base_url = (os.getenv("LLM_BASE_URL") or "").strip()
    if base_url:
        specs["base_url"] = base_url
    system_prompt_override = os.getenv("AGENT_SYSTEM_PROMPT")
    if system_prompt_override:
        specs["agent_system_prompt"] = system_prompt_override
    return specs


def public_config() -> dict[str, Any]:
    """可安全打印的配置摘要（不含 key）。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=False)
    except ImportError:  # pragma: no cover
        pass
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip()
    model = (os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip()
    key = (os.getenv("LLM_API_KEY") or os.getenv(f"{provider.upper()}_API_KEY") or "").strip()
    return {
        "provider": provider,
        "model": model,
        "api_key_configured": bool(key),
        "base_url": (os.getenv("LLM_BASE_URL") or "").strip() or None,
        "env_file": str(ENV_FILE),
        "env_file_exists": ENV_FILE.exists(),
    }
