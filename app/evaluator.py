from uuid import uuid4

from app.evaluation import EvaluationDataset
from app.evaluation_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.models.evaluation import (
    EvaluationResult,
    EvaluationRun,
    RetrievedEvidence,
)
from app.retrieval.retriever import Retriever


class Evaluator:
    def __init__(
        self,
        retriever: Retriever | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()

    def evaluate_case(
        self,
        case,
    ) -> EvaluationResult:
        retrieval = self.retriever.retrieve(
            query=case.question,
            top_k=5,
        )

        retrieved_evidence = [
            RetrievedEvidence(
                rank=result.rank,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                distance=result.distance,
                text=result.text,
            )
            for result in retrieval.results
        ]

        retrieved_documents = [
            evidence.document_id
            for evidence in retrieved_evidence
        ]

        return EvaluationResult(
            case_id=case.case_id,
            question=case.question,
            expected_documents=case.expected_documents,
            retrieved_evidence=retrieved_evidence,
            hit_at_1=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                k=1,
            ),
            hit_at_3=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                k=3,
            ),
            hit_at_5=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                k=5,
            ),
            recall_at_1=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                k=1,
            ),
            recall_at_3=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                k=3,
            ),
            recall_at_5=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                k=5,
            ),
            mrr=reciprocal_rank(
                case.expected_documents,
                retrieved_documents,
            ),
            retrieval_latency_ms=retrieval.latency_ms,
        )

    def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
    ) -> EvaluationRun:
        results = [
            self.evaluate_case(case)
            for case in dataset.cases
        ]

        if results:
            mean_hit_at_1 = sum(
                result.hit_at_1
                for result in results
            ) / len(results)

            mean_hit_at_3 = sum(
                result.hit_at_3
                for result in results
            ) / len(results)

            mean_hit_at_5 = sum(
                result.hit_at_5
                for result in results
            ) / len(results)

            mean_recall_at_1 = sum(
                result.recall_at_1
                for result in results
            ) / len(results)

            mean_recall_at_3 = sum(
                result.recall_at_3
                for result in results
            ) / len(results)

            mean_recall_at_5 = sum(
                result.recall_at_5
                for result in results
            ) / len(results)

            mean_mrr = sum(
                result.mrr
                for result in results
            ) / len(results)

        else:
            mean_hit_at_1 = 0.0
            mean_hit_at_3 = 0.0
            mean_hit_at_5 = 0.0
            mean_recall_at_1 = 0.0
            mean_recall_at_3 = 0.0
            mean_recall_at_5 = 0.0
            mean_mrr = 0.0

        return EvaluationRun(
            run_id=str(uuid4()),
            dataset_size=len(results),
            results=results,
            mean_hit_at_1=mean_hit_at_1,
            mean_hit_at_3=mean_hit_at_3,
            mean_hit_at_5=mean_hit_at_5,
            mean_recall_at_1=mean_recall_at_1,
            mean_recall_at_3=mean_recall_at_3,
            mean_recall_at_5=mean_recall_at_5,
            mean_mrr=mean_mrr,
        )