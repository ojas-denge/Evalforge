from app.evaluation import EvaluationDataset


def test_golden_dataset_v2_loads_with_unique_case_ids() -> None:
    dataset = EvaluationDataset.load()

    assert len(dataset) == 75
    assert len({case.case_id for case in dataset.cases}) == 75
