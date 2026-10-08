"""Typed contracts between the LLM and the rest of the pipeline."""
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["billing", "technical_issue", "account_access", "shipping",
                 "cancellation", "feature_request", "other"]
Urgency = Literal["low", "normal", "high", "critical"]


class Classification(BaseModel):
    intent: Intent
    urgency: Urgency = Field(description="critical = outage, security incident or legal threat")
    confidence: float = Field(ge=0.0, le=1.0,
                              description="probability that BOTH intent and urgency are correct")
    reason: str = Field(description="one short sentence")


class Draft(BaseModel):
    reply: str = Field(description="reply to the customer, at most 5 sentences")
    cited_case_ids: list[str] = Field(default_factory=list,
                                      description="ids of the resolved cases the reply is based on")
    needs_more_info: bool = Field(description="true if the ticket lacks details needed to resolve it")
