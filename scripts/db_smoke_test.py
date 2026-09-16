import uuid
from datetime import datetime, timezone

from app.db.models import (
    EvaluationResultModel,
    EvaluationRunModel,
    RetrievedEvidenceModel,
)
from app.db.session import SessionLocal


def main() -> None:
    run_id = uuid.uuid4()

    session = SessionLocal()

    try:
        run = EvaluationRunModel(
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            dataset_size=1,
            retrieval_mode="dense",
            top_k=5,
            candidate_k=None,
            reranking_enabled=False,
            reranker_candidate_k=10,
            hybrid_retrieval_enabled=False,
            mean_hit_at_1=1.0,
            mean_hit_at_3=1.0,
            mean_hit_at_5=1.0,
            mean_recall_at_1=1.0,
            mean_recall_at_3=1.0,
            mean_recall_at_5=1.0,
            mean_mrr=1.0,
            mean_retrieval_latency_ms=20.0,
        )

        result = EvaluationResultModel(
            run_id=run_id,
            case_id="db_smoke_001",
            question="Does PostgreSQL persistence work?",
            expected_documents=["test_document"],
            failure_type="PASS",
            relevant_documents_found=True,
            first_relevant_rank=1,
            missing_documents=[],
            confounding_documents=[],
            hit_at_1=1.0,
            hit_at_3=1.0,
            hit_at_5=1.0,
            recall_at_1=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            mrr=1.0,
            retrieval_latency_ms=20.0,
        )

        evidence = RetrievedEvidenceModel(
            rank=1,
            chunk_id="test_chunk",
            document_id="test_document",
            distance=0.1,
            text="PostgreSQL persistence smoke test.",
        )

        result.evidence.append(evidence)
        run.results.append(result)

        session.add(run)
        session.commit()

        print(f"WRITE: {run_id}")

        stored_run = session.get(EvaluationRunModel, run_id)

        assert stored_run is not None
        assert len(stored_run.results) == 1
        assert len(stored_run.results[0].evidence) == 1

        print("READ: run found")
        print(f"READ: results={len(stored_run.results)}")
        print(f"READ: evidence={len(stored_run.results[0].evidence)}")

        session.delete(stored_run)
        session.commit()

        assert session.get(EvaluationRunModel, run_id) is None

        print("DELETE: run removed")
        print("DB SMOKE TEST PASSED")

    finally:
        session.close()


if __name__ == "__main__":
    main()