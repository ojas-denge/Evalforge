from app.evaluation.runner import Evaluator
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

    def retrieve(self, query, top_k):
        return type(
            "RetrievalResult",
            (),
            {
                "results": [
                    RetrievedEvidence(
                        document_id="doc-1",
                        chunk_id="chunk-1",
                        score=0.95,
                    )
                ],
                "latency_ms": 12.5,
            },
        )()


def test_evaluate_dataset_emits_evaluation_run_observation():
    tracer = FakeTracer()
    retriever = FakeRetriever()
    evaluator = Evaluator(
        retriever=retriever,
        tracer=tracer,
    )

    case = EvaluationCase(
        case_id="test-001",
        question="What is EvalForge?",
        expected_answer="EvalForge is an LLM evaluation and observability platform.",
        expected_documents=["doc-1"],
        category="general",
        difficulty="easy",
    )

    dataset = type(
        "FakeDataset",
        (),
        {
            "cases": [case],
            "__len__": lambda self: 1,
        },
    )()