from app.evaluation.runner import Evaluator
from app.generation.base import GenerationResult
from app.generation.usage import GenerationUsage
from app.models.evaluation import EvaluationCase, RetrievedEvidence


class FakeObservation:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeTracer:
    def __init__(self):
        self.calls = []

    def trace(self, name, *, input=None, metadata=None):
        tracer = self

        class TraceContext:
            def __enter__(self):
                self.observation = FakeObservation()
                tracer.calls.append(
                    {
                        "name": name,
                        "input": input,
                        "metadata": metadata,
                        "observation": self.observation,
                    }
                )
                return self.observation

            def __exit__(self, exc_type, exc, tb):
                return False

        return TraceContext()


class FakeRetriever:
    mode = "dense"
    candidate_k = None
    reranking_enabled = False
    reranker_candidate_k = 10
    hybrid_retrieval_enabled = False

    def retrieve(self, query, top_k):
        return type(
            "RetrievalResult",
            (),
            {
                "results": [
                    RetrievedEvidence(
                        rank=1,
                        document_id="doc-1",
                        chunk_id="chunk-1",
                        distance=0.05,
                        text=(
                            "EvalForge is an LLM evaluation and "
                            "observability platform."
                        ),
                    )
                ],
                "latency_ms": 12.5,
            },
        )()


class FakeGenerator:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)

        return GenerationResult(
            answer=(
                "EvalForge is an LLM evaluation and "
                "observability platform."
            ),
            model="test-model",
            provider="test-provider",
            usage=GenerationUsage(
                input_tokens=10,
                output_tokens=8,
            ),
            estimated_cost_usd=0.001,
            latency_ms=5.0,
            finish_reason="stop",
        )


def make_case(expected_topics=None):
    return EvaluationCase(
        case_id="test-001",
        question="What is EvalForge?",
        expected_answer=(
            "EvalForge is an LLM evaluation and "
            "observability platform."
        ),
        expected_documents=["doc-1"],
        expected_topics=expected_topics or [],
        category="general",
        difficulty="easy",
    )


def make_dataset(case):
    return type(
        "FakeDataset",
        (),
        {
            "cases": [case],
            "__len__": lambda self: 1,
        },
    )()


def test_evaluate_dataset_emits_evaluation_run_observation():
    tracer = FakeTracer()
    retriever = FakeRetriever()
    generator = FakeGenerator()

    evaluator = Evaluator(
        retriever=retriever,
        generator=generator,
        tracer=tracer,
    )

    run = evaluator.evaluate_dataset(
        make_dataset(make_case())
    )

    assert run.dataset_size == 1
    assert run.results[0].case_id == "test-001"

    evaluation_calls = [
        call
        for call in tracer.calls
        if call["name"] == "evaluation_run"
    ]

    assert len(evaluation_calls) == 1


def test_evaluator_generates_answer_from_retrieved_context():
    retriever = FakeRetriever()
    generator = FakeGenerator()

    evaluator = Evaluator(
        retriever=retriever,
        generator=generator,
    )

    evaluator.evaluate_case(
        make_case(
            expected_topics=[
                "LLM evaluation",
                "observability",
            ]
        )
    )

    assert len(generator.requests) == 1

    request = generator.requests[0]

    assert request.question == "What is EvalForge?"
    assert len(request.context) == 1
    assert request.context[0].document_id == "doc-1"


def test_evaluate_case_records_topic_coverage():
    evaluator = Evaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    result = evaluator.evaluate_case(
        make_case(
            expected_topics=[
                "LLM evaluation",
                "observability",
            ]
        )
    )

    assert result.generated_answer == (
        "EvalForge is an LLM evaluation and "
        "observability platform."
    )
    assert result.topic_coverage == 1.0


def test_evaluate_case_topic_coverage_is_none_without_topics():
    evaluator = Evaluator(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    result = evaluator.evaluate_case(
        make_case()
    )

    assert result.generated_answer == (
        "EvalForge is an LLM evaluation and "
        "observability platform."
    )
    assert result.topic_coverage is None
