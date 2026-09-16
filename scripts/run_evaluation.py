from app.evaluation import EvaluationDataset
from app.evaluation.runner import Evaluator


def main() -> None:
    dataset = EvaluationDataset.load()
    evaluator = Evaluator()

    run = evaluator.evaluate_dataset(dataset)
    evaluator.tracer.flush()

    print("=" * 80)
    print("EVALFORGE RETRIEVAL EVALUATION")
    print("=" * 80)

    print(f"Run ID:  {run.run_id}")
    print(f"Dataset: {run.dataset_size} cases")
    print(f"Retriever: {evaluator.retriever.mode}")
    print()

    for result in run.results:
        print("-" * 80)
        print(f"Case: {result.case_id}")
        print(f"Question: {result.question}")

        print(
            "Expected:  "
            + ", ".join(result.expected_documents)
        )

        print("Retrieved:")

        for evidence in result.retrieved_evidence:
            print(
                f"  {evidence.rank}. "
                f"{evidence.document_id} "
                f"(distance={evidence.distance:.4f})"
                if evidence.distance is not None
                else "  (lexical candidate)"
            )

        print(f"Diagnostic: {result.failure_type}")

        print(
            "Relevant evidence found: "
            f"{'YES' if result.relevant_documents_found else 'NO'}"
        )

        print(
            "First relevant rank: "
            f"{result.first_relevant_rank or 'N/A'}"
        )

        print(
            "Missing documents: "
            + (", ".join(result.missing_documents) or "None")
        )

        print(
            "Confounding documents: "
            + (", ".join(result.confounding_documents) or "None")
        )

        print(
            f"Hit@1={result.hit_at_1:.1f} "
            f"Hit@3={result.hit_at_3:.1f} "
            f"Hit@5={result.hit_at_5:.1f}"
        )

        print(
            f"Recall@1={result.recall_at_1:.2f} "
            f"Recall@3={result.recall_at_3:.2f} "
            f"Recall@5={result.recall_at_5:.2f}"
        )

        print(f"MRR={result.mrr:.3f}")

        print(
            f"Latency={result.retrieval_latency_ms:.2f} ms"
        )

    print("=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)

    print(
        f"Mean Hit@1:    "
        f"{run.mean_hit_at_1:.3f}"
    )

    print(
        f"Mean Hit@3:    "
        f"{run.mean_hit_at_3:.3f}"
    )

    print(
        f"Mean Hit@5:    "
        f"{run.mean_hit_at_5:.3f}"
    )

    print(
        f"Mean Recall@1: "
        f"{run.mean_recall_at_1:.3f}"
    )

    print(
        f"Mean Recall@3: "
        f"{run.mean_recall_at_3:.3f}"
    )

    print(
        f"Mean Recall@5: "
        f"{run.mean_recall_at_5:.3f}"
    )

    print(
        f"Mean MRR:      "
        f"{run.mean_mrr:.3f}"
    )

    print(
        "Mean retrieval latency: "
        f"{run.mean_retrieval_latency_ms:.2f} ms"
    )

    print("=" * 80)
    print("RUN PROVENANCE")
    print("=" * 80)

    print(f"Run ID:       {run.run_id}")
    print(f"Created at:   {run.created_at}")
    print(f"Retriever:    {run.retrieval_config.mode}")
    print(f"Top K:        {run.retrieval_config.top_k}")
    print(f"Candidate K:  {run.retrieval_config.candidate_k}")
    print(
        f"Reranking:    "
        f"{run.retrieval_config.reranking_enabled}"
    )
    print(
        f"Reranker K:   "
        f"{run.retrieval_config.reranker_candidate_k}"
    )
    print(
        f"Hybrid:       "
        f"{run.retrieval_config.hybrid_retrieval_enabled}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()