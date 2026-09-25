from typing import List, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str


class AssistantResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(BaseModel):
    query: str
    intent: Literal["policy_question", "general_question"] = "general_question"
    context: List[dict] = Field(default_factory=list)
    answer: str = ""
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0
