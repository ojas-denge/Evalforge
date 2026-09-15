from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)

    expected_documents: list[str] = Field(
        min_length=1
    )

    expected_topics: list[str] = Field(
        default_factory=list
    )

    category: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)


class RetrievedEvidence(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    distance: float | None = Field(default=None, ge=0.0)
    text: str = Field(min_length=1)


class EvaluationResult(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    expected_documents: list[str] = Field(
        min_length=1
    )

    retrieved_evidence: list[RetrievedEvidence] = Field(
        default_factory=list
    )

    failure_type: str = Field(min_length=1)
    relevant_documents_found: bool
    first_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )
    missing_documents: list[str] = Field(default_factory=list)
    confounding_documents: list[str] = Field(default_factory=list)

    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    hit_at_5: float = Field(ge=0.0, le=1.0)

    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_3: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)

    mrr: float = Field(ge=0.0, le=1.0)

    retrieval_latency_ms: float = Field(ge=0.0)


class EvaluationRun(BaseModel):
    run_id: str = Field(min_length=1)
    dataset_size: int = Field(ge=0)

    results: list[EvaluationResult] = Field(
        default_factory=list
    )

    mean_hit_at_1: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_hit_at_3: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_hit_at_5: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_recall_at_1: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_recall_at_3: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_recall_at_5: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_mrr: float = Field(ge=0.0, le=1.0)
    mean_retrieval_latency_ms: float = Field(ge=0.0)
