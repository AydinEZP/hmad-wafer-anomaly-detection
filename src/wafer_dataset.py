from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WaferDataset:
    """One-class view of the UCR/TSML Wafer dataset.

    ``trainSequences`` contains only nominal sequences. ``testSequences`` keeps
    the complete official test split. Project labels use 0 for nominal and 1
    for anomalous, independently of the original class encoding.
    """

    trainSequences: np.ndarray
    testSequences: np.ndarray
    testLabels: np.ndarray
    trainSourceLabels: np.ndarray
    testSourceLabels: np.ndarray
    normalSourceLabel: str
    anomalySourceLabel: str
    anomalyTypes: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        trainSequences = np.asarray(self.trainSequences, dtype=float)
        testSequences = np.asarray(self.testSequences, dtype=float)
        testLabels = np.asarray(self.testLabels, dtype=int).reshape(-1)
        trainSourceLabels = np.asarray(self.trainSourceLabels).reshape(-1)
        testSourceLabels = np.asarray(self.testSourceLabels).reshape(-1)
        anomalyTypes = np.asarray(self.anomalyTypes).reshape(-1)

        if trainSequences.ndim != 2 or testSequences.ndim != 2:
            raise ValueError("trainSequences and testSequences must be two-dimensional")
        if trainSequences.shape[1] != testSequences.shape[1]:
            raise ValueError("official Wafer train and test sequences must have equal length")
        if trainSequences.shape[0] != trainSourceLabels.size:
            raise ValueError("one source label is required for each retained train sequence")
        if testSequences.shape[0] != testLabels.size:
            raise ValueError("one project label is required for each test sequence")
        if testSequences.shape[0] != testSourceLabels.size:
            raise ValueError("one source label is required for each test sequence")
        if testSequences.shape[0] != anomalyTypes.size:
            raise ValueError("one anomaly type is required for each test sequence")
        if not np.all(np.isfinite(trainSequences)) or not np.all(np.isfinite(testSequences)):
            raise ValueError("Wafer contains missing or non-finite observations")
        if not np.all(np.isin(testLabels, [0, 1])):
            raise ValueError("testLabels must use 0 for nominal and 1 for anomaly")
        if not np.any(testLabels == 0) or not np.any(testLabels == 1):
            raise ValueError("the official test split must contain both classes")


def _normalize_label(value: object) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _find_split_file(datasetDirectory: Path, split: str) -> Path:
    splitUpper = split.upper()
    candidates: list[Path] = []
    for suffix in (".ts", ".tsv", ".txt", ".csv"):
        candidates.extend(datasetDirectory.rglob(f"*{splitUpper}*{suffix}"))
        candidates.extend(datasetDirectory.rglob(f"*{splitUpper.lower()}*{suffix}"))

    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not candidates:
        raise FileNotFoundError(
            f"Could not find the Wafer {splitUpper} file below {datasetDirectory}. "
            "Expected a file such as Wafer_TRAIN.ts or Wafer_TRAIN.tsv."
        )

    waferCandidates = [path for path in candidates if "wafer" in path.name.lower()]
    if waferCandidates:
        candidates = waferCandidates
    if len(candidates) > 1:
        exact = [
            path
            for path in candidates
            if path.stem.lower() in {f"wafer_{split.lower()}", f"wafer{split.lower()}"}
        ]
        if len(exact) == 1:
            return exact[0]
        formatted = "\n".join(f"- {path}" for path in candidates)
        raise ValueError(
            f"Multiple possible Wafer {splitUpper} files were found. "
            f"Set the path explicitly in the configuration.\n{formatted}"
        )
    return candidates[0]


def _load_tabular_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    frame = pd.read_csv(path, sep=delimiter, header=None)
    if frame.shape[1] < 2:
        # Some .txt exports are whitespace separated rather than tab separated.
        frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] < 2:
        raise ValueError(f"{path} does not contain a label column and sequence values")

    labels = frame.iloc[:, 0].map(_normalize_label).to_numpy(dtype=str)
    sequences = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    return sequences, labels


def _load_ts_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    inDataSection = False

    with path.open("r", encoding="utf-8-sig") as source:
        for lineNumber, rawLine in enumerate(source, start=1):
            line = rawLine.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                if line.lower().startswith("@data"):
                    inDataSection = True
                continue
            if not inDataSection:
                continue

            parts = line.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"Expected an equal-length univariate .ts row at {path}:{lineNumber}"
                )
            valueText, labelText = parts
            values = np.fromstring(valueText, sep=",", dtype=float)
            if values.size == 0:
                raise ValueError(f"No observations found at {path}:{lineNumber}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite observation found at {path}:{lineNumber}")
            rows.append(values)
            labels.append(_normalize_label(labelText))

    if not rows:
        raise ValueError(f"No @data rows were found in {path}")
    lengths = {row.size for row in rows}
    if len(lengths) != 1:
        raise ValueError("This HMAD implementation requires equal-length Wafer sequences")
    return np.vstack(rows), np.asarray(labels, dtype=str)


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".ts":
        return _load_ts_file(path)
    return _load_tabular_file(path)


def _infer_label_mapping(trainLabels: np.ndarray) -> tuple[str, str, dict[str, int]]:
    uniqueLabels, counts = np.unique(trainLabels.astype(str), return_counts=True)
    if uniqueLabels.size != 2:
        raise ValueError(
            f"Wafer must contain exactly two source classes; found {uniqueLabels.tolist()}"
        )
    countLookup = {str(label): int(count) for label, count in zip(uniqueLabels, counts)}
    anomalyLabel = str(uniqueLabels[int(np.argmin(counts))])
    normalLabel = str(uniqueLabels[int(np.argmax(counts))])
    if countLookup[anomalyLabel] == countLookup[normalLabel]:
        raise ValueError(
            "Cannot infer Wafer normal/anomaly mapping from equal class counts. "
            "Provide an explicit mapping instead."
        )
    return normalLabel, anomalyLabel, countLookup


def loadWaferDataset(
    datasetDirectory: str | Path,
    *,
    trainPath: str | Path | None = None,
    testPath: str | Path | None = None,
    normalSourceLabel: str | int | float | None = None,
    anomalySourceLabel: str | int | float | None = None,
) -> WaferDataset:
    """Load Wafer from extracted ``.ts`` or UCR ``.tsv`` files.

    By default the minority class in the official training split is interpreted
    as abnormal, matching the published Wafer class imbalance. Explicit source
    labels can be supplied through the configuration when desired.
    """

    directory = Path(datasetDirectory).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Wafer dataset directory does not exist: {directory}")

    resolvedTrainPath = (
        Path(trainPath).expanduser().resolve()
        if trainPath is not None
        else _find_split_file(directory, "TRAIN")
    )
    resolvedTestPath = (
        Path(testPath).expanduser().resolve()
        if testPath is not None
        else _find_split_file(directory, "TEST")
    )

    trainAll, trainLabels = _load_split(resolvedTrainPath)
    testSequences, testLabelsSource = _load_split(resolvedTestPath)
    if trainAll.shape[1] != testSequences.shape[1]:
        raise ValueError("Wafer TRAIN and TEST sequence lengths do not match")

    hasExplicitNormalLabel = normalSourceLabel is not None
    hasExplicitAnomalyLabel = anomalySourceLabel is not None

    if hasExplicitNormalLabel != hasExplicitAnomalyLabel:
        raise ValueError(
            "Provide both normalSourceLabel and anomalySourceLabel, "
            "or leave both unset."
        )

    mappingIsExplicit = (
        hasExplicitNormalLabel and hasExplicitAnomalyLabel
    )

    inferredNormal, inferredAnomaly, trainClassCounts = _infer_label_mapping(
        trainLabels
    )

    if mappingIsExplicit:
        normalLabel = _normalize_label(normalSourceLabel)
        anomalyLabel = _normalize_label(anomalySourceLabel)
    else:
        normalLabel = inferredNormal
        anomalyLabel = inferredAnomaly
        
    if normalLabel == anomalyLabel:
        raise ValueError("normalSourceLabel and anomalySourceLabel must differ")
    if normalLabel not in trainClassCounts or anomalyLabel not in trainClassCounts:
        raise ValueError(
            f"Configured labels are not present in Wafer TRAIN: {trainClassCounts}"
        )

    trainNormalMask = trainLabels == normalLabel
    trainSequences = trainAll[trainNormalMask]
    retainedTrainLabels = trainLabels[trainNormalMask]
    testProjectLabels = np.where(testLabelsSource == anomalyLabel, 1, 0).astype(int)

    unexpectedTestLabels = sorted(
        set(testLabelsSource.tolist()) - {normalLabel, anomalyLabel}
    )
    if unexpectedTestLabels:
        raise ValueError(f"Unexpected labels in Wafer TEST: {unexpectedTestLabels}")

    testUnique, testCounts = np.unique(testLabelsSource, return_counts=True)
    testClassCounts = {
        str(label): int(count) for label, count in zip(testUnique, testCounts)
    }
    anomalyTypes = np.where(testProjectLabels == 1, "abnormal", "nominal")

    metadata: dict[str, Any] = {
        "datasetName": "Wafer",
        "source": "UCR/TSML Time Series Classification Archive",
        "trainFile": str(resolvedTrainPath),
        "testFile": str(resolvedTestPath),
        "sequenceLength": int(trainSequences.shape[1]),
        "nOfficialTrain": int(trainAll.shape[0]),
        "nTrainNormal": int(trainSequences.shape[0]),
        "nOfficialTest": int(testSequences.shape[0]),
        "nTestNormal": int(np.sum(testProjectLabels == 0)),
        "nTestAnomaly": int(np.sum(testProjectLabels == 1)),
        "normalSourceLabel": normalLabel,
        "anomalySourceLabel": anomalyLabel,
        "trainSourceClassCounts": trainClassCounts,
        "testSourceClassCounts": testClassCounts,
        "labelMappingRule": (
            "explicit source labels from configuration"
            if mappingIsExplicit
            else "minority official training class inferred as abnormal"
        ),
        "labelMappingWasExplicit": mappingIsExplicit,
    }

    dataset = WaferDataset(
        trainSequences=trainSequences,
        testSequences=testSequences,
        testLabels=testProjectLabels,
        trainSourceLabels=retainedTrainLabels,
        testSourceLabels=testLabelsSource,
        normalSourceLabel=normalLabel,
        anomalySourceLabel=anomalyLabel,
        anomalyTypes=anomalyTypes,
        metadata=metadata,
    )
    dataset.validate()
    return dataset


def selectWaferTestSubset(
    dataset: WaferDataset,
    *,
    anomalyFraction: float,
    nNormal: int,
    seed: int,
) -> np.ndarray:
    """Return deterministic official-test indices for contamination sensitivity."""

    if not 0.0 < anomalyFraction < 1.0:
        raise ValueError("anomalyFraction must be in (0, 1)")
    normalIndices = np.flatnonzero(dataset.testLabels == 0)
    anomalyIndices = np.flatnonzero(dataset.testLabels == 1)
    if nNormal < 1 or nNormal > normalIndices.size:
        raise ValueError(
            f"nNormal must be between 1 and {normalIndices.size}; received {nNormal}"
        )
    nAnomaly = int(round(anomalyFraction / (1.0 - anomalyFraction) * nNormal))
    nAnomaly = max(1, nAnomaly)
    if nAnomaly > anomalyIndices.size:
        raise ValueError(
            f"Requested {nAnomaly} anomalies, but Wafer TEST contains only "
            f"{anomalyIndices.size}"
        )

    randomGenerator = np.random.default_rng(seed)
    selectedNormal = randomGenerator.choice(normalIndices, size=nNormal, replace=False)
    selectedAnomaly = randomGenerator.choice(anomalyIndices, size=nAnomaly, replace=False)
    combined = np.concatenate((selectedNormal, selectedAnomaly))
    return randomGenerator.permutation(combined)
