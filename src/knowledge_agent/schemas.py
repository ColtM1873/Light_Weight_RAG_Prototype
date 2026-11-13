"""Guard 层使用的结构化输出 schema（Model B / Model C）。

字段的"是否缺失"由 ``validators.missing_required_fields`` 判断，
schema 本身不包含 missing_fields 之类的自报字段。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """一条文档引用：只要求文档 ID + 原文片段。"""

    document_id: int = Field(description="引用片段所属文档 ID")
    quoted_content: str = Field(description="引用到的原文片段，需与库中内容一致")


class AnswerStatus(BaseModel):
    """任务状态。"""

    done: bool = Field(description="任务是否完成")
    conflict: bool = Field(description="文档之间是否存在明确冲突")
    no_support: bool = Field(description="是否没有任何文档能支撑回答")


class StructuredAnswer(BaseModel):
    """Model B 从 Model A 回复中抽取出来的结构化内容。"""

    natural_language_reply: str = Field(description="给用户的自然语言回复全文（必填）")
    citations: list[Citation] = Field(default_factory=list, description="引用列表")
    gaps: list[str] = Field(default_factory=list, description="资料缺口")
    status: AnswerStatus


class SupportVerdict(BaseModel):
    """Model C 对"文档是否支持结论"的判定。"""

    support: Literal["support", "unsupported"]
    error_type: Literal["none", "overclaim", "underanswer"] = Field(
        description="none=无问题；overclaim=文档不足却给明确答复；underanswer=文档充足却回答不了"
    )
    reason: str = Field(description="判断理由，需引用具体文档内容")
