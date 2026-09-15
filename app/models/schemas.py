from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User question to answer using the knowledge base.",
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    latency_ms: float = Field(ge=0.0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)