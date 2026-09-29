from __future__ import annotations

from pathlib import Path

import numpy as np

from src.wafer_dataset import loadWaferDataset, selectWaferTestSubset


def _write_tsv(path: Path, labels: list[int], rows: list[list[float]]) -> None:
    matrix = np.column_stack((np.asarray(labels), np.asarray(rows, dtype=float)))
    np.savetxt(path, matrix, delimiter="\t", fmt="%.8g")


def test_load_wafer_tsv_and_keep_only_nominal_train(tmp_path: Path) -> None:
    wafer_dir = tmp_path / "Wafer"
    wafer_dir.mkdir()
    _write_tsv(
        wafer_dir / "Wafer_TRAIN.tsv",
        labels=[1, 1, 1, 1, -1],
        rows=[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [8, 9, 10]],
    )
    _write_tsv(
        wafer_dir / "Wafer_TEST.tsv",
        labels=[1, -1, 1, -1],
        rows=[[0, 1, 2], [8, 9, 10], [1, 2, 3], [7, 8, 9]],
    )

    dataset = loadWaferDataset(wafer_dir)

    assert dataset.normalSourceLabel == "1"
    assert dataset.anomalySourceLabel == "-1"
    assert dataset.trainSequences.shape == (4, 3)
    assert dataset.testSequences.shape == (4, 3)
    assert dataset.testLabels.tolist() == [0, 1, 0, 1]
    assert dataset.metadata["nTrainNormal"] == 4


def test_load_wafer_ts_format(tmp_path: Path) -> None:
    wafer_dir = tmp_path / "nested" / "Wafer"
    wafer_dir.mkdir(parents=True)
    header = """@problemName Wafer
@timestamps false
@univariate true
@classLabel true normal abnormal
@data
"""
    (wafer_dir / "Wafer_TRAIN.ts").write_text(
        header + "0,1,2:normal\n1,2,3:normal\n2,3,4:normal\n8,9,10:abnormal\n",
        encoding="utf-8",
    )
    (wafer_dir / "Wafer_TEST.ts").write_text(
        header + "0,1,2:normal\n8,9,10:abnormal\n",
        encoding="utf-8",
    )

    dataset = loadWaferDataset(wafer_dir)

    assert dataset.trainSequences.shape == (3, 3)
    assert dataset.testLabels.tolist() == [0, 1]
    assert dataset.normalSourceLabel == "normal"
    assert dataset.anomalySourceLabel == "abnormal"


def test_select_wafer_subset_is_deterministic(tmp_path: Path) -> None:
    wafer_dir = tmp_path / "Wafer"
    wafer_dir.mkdir()
    train_labels = [1] * 20 + [-1] * 4
    train_rows = [[float(i), float(i + 1)] for i in range(len(train_labels))]
    test_labels = [1] * 30 + [-1] * 10
    test_rows = [[float(i), float(i + 1)] for i in range(len(test_labels))]
    _write_tsv(wafer_dir / "Wafer_TRAIN.tsv", train_labels, train_rows)
    _write_tsv(wafer_dir / "Wafer_TEST.tsv", test_labels, test_rows)
    dataset = loadWaferDataset(wafer_dir)

    indices_one = selectWaferTestSubset(
        dataset, anomalyFraction=0.20, nNormal=20, seed=1405
    )
    indices_two = selectWaferTestSubset(
        dataset, anomalyFraction=0.20, nNormal=20, seed=1405
    )

    assert np.array_equal(indices_one, indices_two)
    labels = dataset.testLabels[indices_one]
    assert np.sum(labels == 0) == 20
    assert np.sum(labels == 1) == 5
