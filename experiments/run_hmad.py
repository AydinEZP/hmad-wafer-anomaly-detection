from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from src.hmad import SimplifiedHMAD
from src.joint_features import batchJointFeatures, computeJointFeatures
from src.plotting import (
    saveAucSensitivityPlot,
    saveConvergencePlot,
    saveDecodedPathPlot,
    saveJointFeatureHeatmap,
    saveRocPlot,
    saveSequencePairPlot,
    saveSequencePlot,
)
from src.evaluation import evaluateNormalityScores
from src.wafer_dataset import WaferDataset, loadWaferDataset, selectWaferTestSubset


def loadConfig(configPath: str | Path) -> dict:
    with open(configPath, "r", encoding="utf-8") as configFile:
        return yaml.safe_load(configFile)


def _resolveOptionalPath(value: object) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else projectRoot / path


def loadDataset(dataConfig: dict) -> WaferDataset:
    datasetDirectory = Path(dataConfig.get("datasetDirectory", "data/Wafer"))
    if not datasetDirectory.is_absolute():
        datasetDirectory = projectRoot / datasetDirectory
    return loadWaferDataset(
        datasetDirectory,
        trainPath=_resolveOptionalPath(dataConfig.get("trainPath")),
        testPath=_resolveOptionalPath(dataConfig.get("testPath")),
        normalSourceLabel=dataConfig.get("normalSourceLabel"),
        anomalySourceLabel=dataConfig.get("anomalySourceLabel"),
    )


def globalStatistics(sequence: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(sequence)),
        "variance": float(np.var(sequence)),
        "minimum": float(np.min(sequence)),
        "maximum": float(np.max(sequence)),
    }


def runToyExperiment(seed: int, figureDirectory: Path, tableDirectory: Path) -> dict:
    """Structural sanity check retained independently of the real dataset."""

    randomGenerator = np.random.default_rng(seed)
    lowValues = randomGenerator.normal(0.0, 1.0, size=50)
    highValues = randomGenerator.normal(4.0, 1.0, size=50)
    contiguousSequence = np.concatenate((lowValues, highValues))
    shuffledSequence = contiguousSequence.copy()
    randomGenerator.shuffle(shuffledSequence)

    contiguousStates = np.where(contiguousSequence < 2.0, 0, 1)
    shuffledStates = np.where(shuffledSequence < 2.0, 0, 1)
    contiguousFeatures = computeJointFeatures(contiguousSequence, contiguousStates, 2)
    shuffledFeatures = computeJointFeatures(shuffledSequence, shuffledStates, 2)

    pd.DataFrame(
        [
            {"sequence": "contiguous", **globalStatistics(contiguousSequence)},
            {"sequence": "shuffled", **globalStatistics(shuffledSequence)},
        ]
    ).to_csv(tableDirectory / "toy_summary_statistics.csv", index=False)

    featureNames = ["c11", "c12", "c21", "c22", "meanState1", "meanState2"]
    pd.DataFrame(
        [contiguousFeatures, shuffledFeatures],
        index=["contiguous", "shuffled"],
        columns=featureNames,
    ).to_csv(tableDirectory / "toy_joint_features.csv")

    saveSequencePlot(
        contiguousSequence,
        "Contiguous low- and high-mean blocks",
        figureDirectory / "toy_contiguous.pdf",
    )
    saveSequencePlot(
        shuffledSequence,
        "The same values in shuffled order",
        figureDirectory / "toy_shuffled.pdf",
    )

    return {
        "contiguousFeatures": contiguousFeatures.tolist(),
        "shuffledFeatures": shuffledFeatures.tolist(),
        "largestFeatureDifferences": np.argsort(
            np.abs(contiguousFeatures - shuffledFeatures)
        )[::-1][:3].tolist(),
    }


def writeDatasetDiagnostics(
    dataset: WaferDataset,
    decodedStates: np.ndarray,
    tableDirectory: Path,
) -> dict[str, float | int | str]:
    nominalMask = dataset.testLabels == 0
    anomalyMask = dataset.testLabels == 1
    nominalSequences = dataset.testSequences[nominalMask]
    anomalySequences = dataset.testSequences[anomalyMask]
    nominalStates = decodedStates[nominalMask]
    anomalyStates = decodedStates[anomalyMask]

    diagnostics: dict[str, float | int | str] = {
        "datasetName": "Wafer",
        "nOfficialTrain": int(dataset.metadata["nOfficialTrain"]),
        "nTrainNormal": int(dataset.trainSequences.shape[0]),
        "nTestNormal": int(nominalMask.sum()),
        "nTestAnomaly": int(anomalyMask.sum()),
        "sequenceLength": int(dataset.trainSequences.shape[1]),
        "totalTrainingTimeSteps": int(dataset.trainSequences.size),
        "normalSourceLabel": dataset.normalSourceLabel,
        "anomalySourceLabel": dataset.anomalySourceLabel,
        "nominalObservationMean": float(nominalSequences.mean()),
        "anomalyObservationMean": float(anomalySequences.mean()),
        "nominalObservationStd": float(nominalSequences.std()),
        "anomalyObservationStd": float(anomalySequences.std()),
        "nominalDecodedSwitchRate": float(
            np.mean(nominalStates[:, 1:] != nominalStates[:, :-1])
        ),
        "anomalyDecodedSwitchRate": float(
            np.mean(anomalyStates[:, 1:] != anomalyStates[:, :-1])
        ),
        "officialTestAnomalyFraction": float(np.mean(dataset.testLabels)),
    }
    (tableDirectory / "dataset_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    return diagnostics


def appendScoreRows(
    resultRows: list[dict],
    labels: np.ndarray,
    normalityScores: np.ndarray,
    *,
    modelName: str,
    representation: str,
    nu: float,
    evaluation: str,
    seed: int,
) -> None:
    predictions = (normalityScores < 0.0).astype(int)
    anomalyTypes = np.where(labels == 1, "abnormal", "nominal")
    for sampleIndex, (label, anomalyType, score, prediction) in enumerate(
        zip(labels, anomalyTypes, normalityScores, predictions)
    ):
        resultRows.append(
            {
                "sampleIndex": sampleIndex,
                "normalityScore": float(score),
                "anomalyScore": float(-score),
                "hardPrediction": int(prediction),
                "trueLabel": int(label),
                "anomalyType": str(anomalyType),
                "model": modelName,
                "representation": representation,
                "nu": float(nu),
                "rTest": float(np.mean(labels)),
                "evaluation": evaluation,
                "seed": seed,
            }
        )

def chooseRepresentativeSample(
    labels: np.ndarray,
    targetLabel: int,
    *,
    seed: int,
) -> int:
    """Select a deterministic example without using model scores."""

    labelArray = np.asarray(labels, dtype=int).reshape(-1)
    candidateIndices = np.flatnonzero(labelArray == targetLabel)

    if candidateIndices.size == 0:
        raise ValueError(f"No sample has target label {targetLabel}")

    randomGenerator = np.random.default_rng(seed)
    return int(randomGenerator.choice(candidateIndices))

def evaluateAbnormalClass(
    labels: np.ndarray,
    normalityScores: np.ndarray,
    modelName: str,
) -> dict:
    return {
        "model": modelName,
        "anomalyType": "abnormal",
        "nNormal": int(np.sum(labels == 0)),
        "nAnomaly": int(np.sum(labels == 1)),
        "rocAuc": float(roc_auc_score(labels, -normalityScores)),
    }


def _coerceBooleanSeries(series: pd.Series, columnName: str) -> pd.Series:
    """Convert a CSV boolean column to a strict pandas boolean series."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    unknownValues = sorted(set(normalized) - set(mapping))
    if unknownValues:
        raise ValueError(
            f"Column {columnName!r} contains invalid boolean values: "
            f"{unknownValues}"
        )

    return normalized.map(mapping).astype(bool)


def loadPrimaryOcsvmResults(
    tableDirectory: Path,
    dataset: WaferDataset,
    *,
    primaryRepresentation: str,
    primaryNu: float,
    expectedKernel: str,
    expectedGamma: object,
    expectedSeed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Load and validate the precomputed OC-SVM outputs.

    The OC-SVM must be executed first through ``experiments/run_ocsvm.py``.
    This function never refits the baseline. It verifies that the stored
    sample order, labels, configuration, and metrics match the current Wafer
    split before the scores are used in the integrated HMAD comparison.
    """

    metricPath = tableDirectory / "ocsvm_metrics.csv"
    scorePath = tableDirectory / "ocsvm_results.csv"

    missingPaths = [
        str(path)
        for path in (metricPath, scorePath)
        if not path.exists()
    ]
    if missingPaths:
        raise FileNotFoundError(
            "Required OC-SVM outputs are missing. Run "
            "`python experiments/run_ocsvm.py` before HMAD. Missing files:\n- "
            + "\n- ".join(missingPaths)
        )

    metricTable = pd.read_csv(metricPath)
    scoreTable = pd.read_csv(scorePath)

    requiredMetricColumns = {
        "model",
        "representation",
        "nu",
        "gamma",
        "evaluation",
        "rocAuc",
        "precision",
        "recall",
        "f1Score",
        "isPrimary",
    }
    requiredScoreColumns = {
        "sampleIndex",
        "trueLabel",
        "normalityScore",
        "representation",
        "nu",
        "gamma",
        "evaluation",
        "seed",
        "isPrimary",
    }

    missingMetricColumns = sorted(
        requiredMetricColumns - set(metricTable.columns)
    )
    missingScoreColumns = sorted(
        requiredScoreColumns - set(scoreTable.columns)
    )

    if missingMetricColumns:
        raise ValueError(
            "ocsvm_metrics.csv is missing columns: "
            + ", ".join(missingMetricColumns)
        )
    if missingScoreColumns:
        raise ValueError(
            "ocsvm_results.csv is missing columns: "
            + ", ".join(missingScoreColumns)
        )

    metricPrimaryFlags = _coerceBooleanSeries(
        metricTable["isPrimary"],
        "ocsvm_metrics.isPrimary",
    )
    scorePrimaryFlags = _coerceBooleanSeries(
        scoreTable["isPrimary"],
        "ocsvm_results.isPrimary",
    )

    primaryMetricMask = (
        metricPrimaryFlags
        & (metricTable["model"].astype(str) == "OC-SVM")
        & (
            metricTable["representation"].astype(str)
            == primaryRepresentation
        )
        & np.isclose(
            metricTable["nu"].to_numpy(dtype=float),
            primaryNu,
        )
        & (
            metricTable["evaluation"].astype(str)
            == "official_test"
        )
    )

    primaryScoreMask = (
        scorePrimaryFlags
        & (
            scoreTable["representation"].astype(str)
            == primaryRepresentation
        )
        & np.isclose(
            scoreTable["nu"].to_numpy(dtype=float),
            primaryNu,
        )
        & (
            scoreTable["evaluation"].astype(str)
            == "official_test"
        )
    )

    primaryMetricRows = metricTable.loc[primaryMetricMask].copy()
    primaryScoreRows = scoreTable.loc[primaryScoreMask].copy()

    if primaryMetricRows.shape[0] != 1:
        raise ValueError(
            "Expected exactly one primary OC-SVM metric row, found "
            f"{primaryMetricRows.shape[0]}"
        )

    expectedTestCount = int(dataset.testLabels.size)
    if primaryScoreRows.shape[0] != expectedTestCount:
        raise ValueError(
            "Primary OC-SVM score count does not match the Wafer test split: "
            f"{primaryScoreRows.shape[0]} != {expectedTestCount}"
        )

    primaryMetric = primaryMetricRows.iloc[0]

    if "kernel" in primaryMetricRows.columns:
        storedKernel = str(primaryMetric["kernel"]).lower()
        if storedKernel != str(expectedKernel).lower():
            raise ValueError(
                f"Stored OC-SVM kernel {storedKernel!r} does not match "
                f"configuration {expectedKernel!r}"
            )

    storedGamma = str(primaryMetric["gamma"])
    if storedGamma != str(expectedGamma):
        raise ValueError(
            f"Stored OC-SVM gamma {storedGamma!r} does not match "
            f"configuration {expectedGamma!r}"
        )

    primaryScoreRows = primaryScoreRows.sort_values(
        "sampleIndex"
    ).reset_index(drop=True)

    sampleIndices = primaryScoreRows["sampleIndex"].to_numpy(dtype=int)
    expectedIndices = np.arange(expectedTestCount, dtype=int)

    if not np.array_equal(sampleIndices, expectedIndices):
        raise ValueError(
            "Primary OC-SVM sampleIndex values must be the complete ordered "
            "range 0..n_test-1"
        )

    storedLabels = primaryScoreRows["trueLabel"].to_numpy(dtype=int)
    if not np.array_equal(storedLabels, dataset.testLabels.astype(int)):
        raise ValueError(
            "Primary OC-SVM labels do not match the current Wafer test split"
        )

    storedSeeds = primaryScoreRows["seed"].to_numpy(dtype=int)
    if not np.all(storedSeeds == expectedSeed):
        raise ValueError(
            "Primary OC-SVM scores were generated with a different seed"
        )

    normalityScores = primaryScoreRows[
        "normalityScore"
    ].to_numpy(dtype=float)

    if not np.all(np.isfinite(normalityScores)):
        raise ValueError(
            "Primary OC-SVM normality scores contain non-finite values"
        )

    recomputedMetrics = evaluateNormalityScores(
        dataset.testLabels,
        normalityScores,
    )
    for metricName in ("rocAuc", "precision", "recall", "f1Score"):
        storedValue = float(primaryMetric[metricName])
        recomputedValue = float(recomputedMetrics[metricName])
        if not np.isclose(
            storedValue,
            recomputedValue,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Stored primary OC-SVM {metricName} does not match scores: "
                f"{storedValue} != {recomputedValue}"
            )

    return normalityScores, metricTable.to_dict("records")


def writeReportNumbers(
    officialMetricTable: pd.DataFrame,
    hmadModel: SimplifiedHMAD,
    datasetDiagnostics: dict,
    primaryBaselineName: str,
    outputPath: Path,
) -> None:
    metricLookup = officialMetricTable.set_index("model").to_dict("index")
    meansText = ", ".join(f"{value:.4f}" for value in hmadModel.emissionMeans_)
    matrixRows = [
        " & ".join(f"{value:.4f}" for value in row)
        for row in hmadModel.transitionMatrix_
    ]

    def metric(model: str, key: str) -> float:
        return float(metricLookup.get(model, {}).get(key, float("nan")))

    content = "\n".join(
        [
            "\\newcommand{\\DatasetName}{Wafer}",
            f"\\newcommand{{\\PrimaryTestFraction}}{{{datasetDiagnostics['officialTestAnomalyFraction']:.4f}}}",
            f"\\newcommand{{\\DatasetTrainCount}}{{{datasetDiagnostics['nTrainNormal']}}}",
            f"\\newcommand{{\\DatasetTestNormalCount}}{{{datasetDiagnostics['nTestNormal']}}}",
            f"\\newcommand{{\\DatasetTestAnomalyCount}}{{{datasetDiagnostics['nTestAnomaly']}}}",
            f"\\newcommand{{\\DatasetSequenceLength}}{{{datasetDiagnostics['sequenceLength']}}}",
            f"\\newcommand{{\\NominalSwitchRate}}{{{datasetDiagnostics['nominalDecodedSwitchRate']:.4f}}}",
            f"\\newcommand{{\\AnomalySwitchRate}}{{{datasetDiagnostics['anomalyDecodedSwitchRate']:.4f}}}",
            f"\\newcommand{{\\HmadAucPrimary}}{{{metric('HMAD', 'rocAuc'):.4f}}}",
            f"\\newcommand{{\\HmadPrecisionPrimary}}{{{100.0 * metric('HMAD', 'precision'):.2f}}}",
            f"\\newcommand{{\\HmadRecallPrimary}}{{{100.0 * metric('HMAD', 'recall'):.2f}}}",
            f"\\newcommand{{\\HmadFOnePrimary}}{{{100.0 * metric('HMAD', 'f1Score'):.2f}}}",
            f"\\newcommand{{\\TrivialAucPrimary}}{{{metric('HMAD all-one ablation', 'rocAuc'):.4f}}}",
            f"\\newcommand{{\\StandaloneBaselineAucPrimary}}{{{metric(primaryBaselineName, 'rocAuc'):.4f}}}",
            f"\\newcommand{{\\WaferAbnormalHmadAuc}}{{{metric('HMAD', 'rocAuc'):.4f}}}",
            "\\newcommand{\\RapidSwitchHmadAuc}{\\text{N/A}}",
            "\\newcommand{\\BlockShuffleHmadAuc}{\\text{N/A}}",
            f"\\newcommand{{\\HmadIterations}}{{{hmadModel.nIterations_}}}",
            f"\\newcommand{{\\HmadEmissionMeans}}{{{meansText}}}",
            "\\newcommand{\\HmadTransitionMatrix}{%",
            "\\begin{bmatrix}",
            f"{matrixRows[0]} \\\\",
            matrixRows[1],
            "\\end{bmatrix}}",
        ]
    )
    outputPath.write_text(content + "\n", encoding="utf-8")


def runExperiment(configPath: str | Path, outputDirectory: str | Path) -> dict:
    config = loadConfig(configPath)
    outputPath = Path(outputDirectory)
    figureDirectory = outputPath / "figures"
    tableDirectory = outputPath / "tables"
    figureDirectory.mkdir(parents=True, exist_ok=True)
    tableDirectory.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    dataConfig = config["data"]
    modelConfig = config["hmad"]
    ocsvmConfig = config["ocsvm"]
    primaryNu = float(ocsvmConfig["primaryNu"])
    primaryRepresentation = str(ocsvmConfig["primaryRepresentation"])
    primaryBaselineName = f"OC-SVM {primaryRepresentation}"

    dataset = loadDataset(dataConfig)

    toyOutputDirectory = outputPath / "toy_validation"
    toyFigureDirectory = toyOutputDirectory / "figures"
    toyTableDirectory = toyOutputDirectory / "tables"
    toyFigureDirectory.mkdir(parents=True, exist_ok=True)
    toyTableDirectory.mkdir(parents=True, exist_ok=True)
    toyResult = runToyExperiment(
        seed,
        toyFigureDirectory,
        toyTableDirectory,
    )

    hmadModel = SimplifiedHMAD(
        nStates=int(modelConfig["nStates"]),
        nu=primaryNu,
        gamma=ocsvmConfig.get("gamma", "scale"),
        maxIterations=int(modelConfig["maxIterations"]),
        convergenceThreshold=float(modelConfig["convergenceThreshold"]),
        initialSelfTransition=float(modelConfig["initialSelfTransition"]),
        varianceFloor=float(modelConfig["varianceFloor"]),
        transitionSmoothing=float(modelConfig["transitionSmoothing"]),
        randomState=seed,
    ).fit(dataset.trainSequences)

    convergenceTable = pd.DataFrame(hmadModel.convergenceHistory_)
    convergenceTable.to_csv(tableDirectory / "hmad_convergence.csv", index=False)
    saveConvergencePlot(
        convergenceTable["iteration"].to_numpy(dtype=int),
        convergenceTable["changedFraction"].to_numpy(),
        convergenceTable["meanDecisionScore"].to_numpy(),
        figureDirectory / "hmad_convergence.pdf",
    )

    testDecodedStates = hmadModel.decodeSequences(dataset.testSequences)
    hmadScores = hmadModel.decisionFunction(dataset.testSequences)
    allTestJointFeatures = batchJointFeatures(
        dataset.testSequences,
        testDecodedStates,
        int(modelConfig["nStates"]),
    )

    expectedJointShape = (
        dataset.testSequences.shape[0],
        6,
    )

    if allTestJointFeatures.shape != expectedJointShape:
        raise AssertionError(
            "Unexpected official-test joint feature shape: "
            f"{allTestJointFeatures.shape}; "
            f"expected {expectedJointShape}"
        )

    if not np.all(
        np.isfinite(allTestJointFeatures)
    ):
        raise AssertionError(
            "Official-test joint features contain "
            "non-finite values"
        )

    transitionSums = np.sum(
        allTestJointFeatures[:, :4],
        axis=1,
    )

    if not np.allclose(
        transitionSums,
        1.0,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise AssertionError(
            "Official-test transition features "
            "do not sum to one"
        )

    allJointFeatureTable = pd.DataFrame(
        allTestJointFeatures,
        columns=[
            "c11",
            "c12",
            "c21",
            "c22",
            "meanState1",
            "meanState2",
        ],
    )

    allJointFeatureTable.insert(
        0,
        "trueLabel",
        dataset.testLabels.astype(int),
    )

    allJointFeatureTable.insert(
        0,
        "sampleIndex",
        np.arange(
            dataset.testSequences.shape[0],
            dtype=int,
        ),
    )

    allJointFeatureTable.to_csv(
        tableDirectory
        / "wafer_joint_features_all_test.csv",
        index=False,
    )
    trivialScores = hmadModel.decisionFunctionWithTrivialStates(dataset.testSequences)
    datasetDiagnostics = writeDatasetDiagnostics(
        dataset, testDecodedStates, tableDirectory
    )

    primaryBaselineScores, ocsvmMetricRows = loadPrimaryOcsvmResults(
        tableDirectory,
        dataset,
        primaryRepresentation=primaryRepresentation,
        primaryNu=primaryNu,
        expectedKernel=str(ocsvmConfig.get("kernel", "rbf")),
        expectedGamma=ocsvmConfig.get("gamma", "scale"),
        expectedSeed=seed,
    )

    # results.csv contains only the three primary models used in the
    # integrated comparison. The complete 12-configuration OC-SVM scores
    # remain in outputs/tables/ocsvm_results.csv.
    resultRows: list[dict] = []

    officialModelScores = {
        "HMAD": hmadScores,
        "HMAD all-one ablation": trivialScores,
        primaryBaselineName: primaryBaselineScores,
    }
    officialMetricRows: list[dict] = []
    for modelName, scores in officialModelScores.items():
        metrics = evaluateNormalityScores(dataset.testLabels, scores)
        officialMetricRows.append(
            {
                "model": modelName,
                "evaluation": "official_test",
                "nNormal": int(np.sum(dataset.testLabels == 0)),
                "nAnomaly": int(np.sum(dataset.testLabels == 1)),
                "rTest": float(np.mean(dataset.testLabels)),
                "seed": seed,
                "nu": primaryNu,
                **metrics,
            }
        )
        appendScoreRows(
            resultRows,
            dataset.testLabels,
            scores,
            modelName=modelName,
            representation=(
                "joint_features"
                if modelName.startswith("HMAD")
                else primaryRepresentation
            ),
            nu=primaryNu,
            evaluation="official_test",
            seed=seed,
        )

    officialMetricTable = pd.DataFrame(officialMetricRows)
    saveRocPlot(
        dataset.testLabels,
        officialModelScores,
        figureDirectory / "roc_curves_primary.pdf",
        title="ROC curves on the official Wafer test split",
    )

    # Sequence examples and Viterbi paths. Wafer has no interval-level labels.
    normalIndex = chooseRepresentativeSample(
        dataset.testLabels,
        0,
        seed=seed + 200,
    )

    anomalyIndex = chooseRepresentativeSample(
        dataset.testLabels,
        1,
        seed=seed + 201,
    )
    saveDecodedPathPlot(
        dataset.testSequences[normalIndex],
        testDecodedStates[normalIndex],
        "Decoded path for a nominal Wafer sequence",
        figureDirectory / "viterbi_normal.pdf",
    )
    saveDecodedPathPlot(
        dataset.testSequences[anomalyIndex],
        testDecodedStates[anomalyIndex],
        "Decoded path for an abnormal Wafer sequence",
        figureDirectory / "viterbi_anomalous.pdf",
    )
    saveSequencePairPlot(
        dataset.testSequences[normalIndex],
        dataset.testSequences[anomalyIndex],
        outputPath=figureDirectory / "sequence_examples.pdf",
        nominalTitle="Nominal Wafer sensor sequence",
        anomalousTitle="Abnormal Wafer sensor sequence",
    )
    pd.DataFrame(
        {
            "normalObservation": dataset.testSequences[normalIndex],
            "normalState": testDecodedStates[normalIndex] + 1,
            "anomalousObservation": dataset.testSequences[anomalyIndex],
            "anomalousState": testDecodedStates[anomalyIndex] + 1,
        }
    ).to_csv(tableDirectory / "decoded_path_examples.csv", index=False)

    # Heatmap uses model-decoded states, never unavailable ground-truth states.
    randomGenerator = np.random.default_rng(seed)
    normalIndices = np.flatnonzero(dataset.testLabels == 0)
    anomalyIndices = np.flatnonzero(dataset.testLabels == 1)
    nHeatmapNormal = min(int(dataConfig.get("heatmapNormalCount", 200)), normalIndices.size)
    nHeatmapAnomaly = min(int(dataConfig.get("heatmapAnomalyCount", 200)), anomalyIndices.size)
    heatmapIndices = np.concatenate(
        (
            randomGenerator.choice(normalIndices, nHeatmapNormal, replace=False),
            randomGenerator.choice(anomalyIndices, nHeatmapAnomaly, replace=False),
        )
    )
    heatmapFeatures = batchJointFeatures(
        dataset.testSequences[heatmapIndices],
        testDecodedStates[heatmapIndices],
        int(modelConfig["nStates"]),
    )
    saveJointFeatureHeatmap(
        heatmapFeatures,
        figureDirectory / "joint_feature_heatmap.pdf",
        nominalCount=nHeatmapNormal,
    )
    featureNames = ["c11", "c12", "c21", "c22", "meanState1", "meanState2"]
    heatmapTable = pd.DataFrame(heatmapFeatures, columns=featureNames)
    heatmapTable.insert(0, "label", dataset.testLabels[heatmapIndices])
    heatmapTable.insert(1, "sourceIndex", heatmapIndices)
    heatmapTable.to_csv(tableDirectory / "wafer_joint_features.csv", index=False)

    # Fixed-subset anomaly-fraction sensitivity reuses scores from the official test.
    sensitivityFractions = [float(value) for value in dataConfig["sensitivityFractions"]]
    sensitivityNormalCount = int(dataConfig["sensitivityNormalCount"])
    sensitivityRows: list[dict] = []
    aucByModel = {modelName: [] for modelName in officialModelScores}
    achievedFractions: list[float] = []
    for fractionIndex, targetFraction in enumerate(sensitivityFractions):
        subsetIndices = selectWaferTestSubset(
            dataset,
            anomalyFraction=targetFraction,
            nNormal=sensitivityNormalCount,
            seed=seed + 100 + fractionIndex,
        )
        labels = dataset.testLabels[subsetIndices]
        achievedFraction = float(np.mean(labels))
        achievedFractions.append(achievedFraction)
        subsetScoreLookup = {
            "HMAD": hmadScores[subsetIndices],
            "HMAD all-one ablation": trivialScores[subsetIndices],
            primaryBaselineName: primaryBaselineScores[subsetIndices],
        }
        for modelName, scores in subsetScoreLookup.items():
            metrics = evaluateNormalityScores(labels, scores)
            aucByModel[modelName].append(metrics["rocAuc"])
            sensitivityRows.append(
                {
                    "model": modelName,
                    "evaluation": "sensitivity_subset",
                    "targetRTest": targetFraction,
                    "nNormal": int(np.sum(labels == 0)),
                    "nAnomaly": int(np.sum(labels == 1)),
                    "rTest": achievedFraction,
                    "seed": seed + 100 + fractionIndex,
                    "nu": primaryNu,
                    **metrics,
                }
            )

    hmadMetricTable = pd.concat(
        [officialMetricTable, pd.DataFrame(sensitivityRows)], ignore_index=True
    )
    hmadMetricTable.to_csv(tableDirectory / "hmad_metrics.csv", index=False)
    saveAucSensitivityPlot(
        np.asarray(achievedFractions),
        {name: np.asarray(values) for name, values in aucByModel.items()},
        figureDirectory / "auc_vs_contamination.pdf",
    )

    typeMetricRows = [
        evaluateAbnormalClass(dataset.testLabels, scores, modelName)
        for modelName, scores in officialModelScores.items()
    ]
    pd.DataFrame(typeMetricRows).to_csv(
        tableDirectory / "metrics_by_anomaly_type.csv", index=False
    )
    pd.DataFrame(resultRows).to_csv(outputPath / "results.csv", index=False)

    learnedParameters = {
        "emissionMeans": hmadModel.emissionMeans_.tolist(),
        "emissionVariances": hmadModel.emissionVariances_.tolist(),
        "initialProbabilities": hmadModel.initialProbabilities_.tolist(),
        "transitionMatrix": hmadModel.transitionMatrix_.tolist(),
        "nIterations": hmadModel.nIterations_,
        "converged": bool(hmadModel.converged_),
    }
    (tableDirectory / "learned_parameters.json").write_text(
        json.dumps(learnedParameters, indent=2) + "\n", encoding="utf-8"
    )

    writeReportNumbers(
        officialMetricTable,
        hmadModel,
        datasetDiagnostics,
        primaryBaselineName,
        projectRoot / "generated_numbers.tex",
    )

    summary = {
        "dataset": dataset.metadata,
        "datasetDiagnostics": datasetDiagnostics,
        "toyExperiment": toyResult,
        "learnedParameters": learnedParameters,
        "officialMetrics": officialMetricRows,
        "ocsvmMetrics": ocsvmMetricRows,
        "ocsvmArtifacts": {
            "metrics": str(tableDirectory / "ocsvm_metrics.csv"),
            "scores": str(tableDirectory / "ocsvm_results.csv"),
        },
        "sensitivityMetrics": sensitivityRows,
        "metricsByAnomalyType": typeMetricRows,
        "primaryBaseline": {
            "representation": primaryRepresentation,
            "nu": primaryNu,
            "modelName": primaryBaselineName,
        },
    }
    (outputPath / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(projectRoot / "configs" / "default.yaml")
    )
    parser.add_argument("--output", default=str(projectRoot / "outputs"))
    arguments = parser.parse_args()
    summary = runExperiment(arguments.config, arguments.output)
    print(json.dumps(summary["learnedParameters"], indent=2))


if __name__ == "__main__":
    main()
