from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


projectRoot = Path(__file__).resolve().parents[1]

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))


from src.baseline_features import buildFeatureRepresentation
from src.evaluation import evaluateNormalityScores
from src.gaussian_required_data import (
    BASE_SEED,
    Q9_SEED,
    SENSITIVITY_POOL_SEED,
    SUPPLEMENTARY_SEED,
    makeExactFractionSubsetFromPool,
    makeGaussianTestSet,
    makeGaussianTrainingSet,
    makeQuestion9Dataset,
    makeSensitivityMasterPool,
    saveGaussianTestSet,
)
from src.hmad import SimplifiedHMAD
from src.joint_features import batchJointFeatures
from src.ocsvm_baseline import OcsvmBaseline, fitOcsvmBaseline
from src.plotting import (
    saveAucSensitivityPlot,
    saveConvergencePlot,
    saveDecodedPathPlot,
    saveRocPlot,
)

FEATURE_NAMES = [
    "c11",
    "c12",
    "c21",
    "c22",
    "meanState1",
    "meanState2",
]


def saveFigureBothFormats(
    figure: plt.Figure,
    outputBasePath: Path,
) -> None:
    """Save one figure as both PDF and PNG."""

    outputBasePath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdfPath = outputBasePath.with_suffix(
        ".pdf"
    )

    pngPath = outputBasePath.with_suffix(
        ".png"
    )

    figure.savefig(
        pdfPath,
        bbox_inches="tight",
    )

    figure.savefig(
        pngPath,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def assignThresholdStates(
    sequences: np.ndarray,
    *,
    threshold: float = 2.0,
) -> np.ndarray:
    """Apply the hard state rule required in Question 9.

    Project rule:
        state 1 if x_t < 2
        state 2 if x_t >= 2

    Internal code convention:
        0 = state 1
        1 = state 2
    """

    sequenceArray = np.asarray(
        sequences,
        dtype=float,
    )

    if sequenceArray.ndim != 2:
        raise ValueError(
            "sequences must have shape "
            "(nSequences, sequenceLength)"
        )

    if not np.all(
        np.isfinite(sequenceArray)
    ):
        raise ValueError(
            "sequences contain non-finite values"
        )

    return (
        sequenceArray >= threshold
    ).astype(int)


def validateQuestion9Features(
    featureMatrix: np.ndarray,
    *,
    expectedRows: int = 250,
) -> None:
    """Validate the required 250 by 6 feature matrix."""

    featureArray = np.asarray(
        featureMatrix,
        dtype=float,
    )

    if featureArray.shape != (
        expectedRows,
        6,
    ):
        raise AssertionError(
            "Question 9 feature matrix has "
            f"shape {featureArray.shape}; "
            f"expected {(expectedRows, 6)}"
        )

    if not np.all(
        np.isfinite(featureArray)
    ):
        raise AssertionError(
            "Question 9 features contain "
            "non-finite values"
        )

    transitionSums = np.sum(
        featureArray[:, :4],
        axis=1,
    )

    if not np.allclose(
        transitionSums,
        1.0,
        rtol=1e-8,
        atol=1e-8,
    ):
        maximumError = float(
            np.max(
                np.abs(
                    transitionSums - 1.0
                )
            )
        )

        raise AssertionError(
            "Normalized transition features "
            "do not sum to one. "
            f"Maximum error: {maximumError}"
        )


def saveQuestion9Heatmap(
    featureMatrix: np.ndarray,
    outputBasePath: Path,
    *,
    nNormal: int,
) -> None:
    """Plot the grouped 250 by 6 joint-feature heatmap."""

    figure, axis = plt.subplots(
        figsize=(9.2, 7.0)
    )

    image = axis.imshow(
        featureMatrix,
        aspect="auto",
        interpolation="nearest",
    )

    axis.axhline(
        nNormal - 0.5,
        linestyle="--",
        linewidth=1.3,
    )

    axis.set_xticks(
        np.arange(
            len(FEATURE_NAMES)
        )
    )

    axis.set_xticklabels(
        FEATURE_NAMES,
        rotation=35,
        ha="right",
    )

    axis.set_xlabel(
        "Joint-feature component"
    )

    axis.set_ylabel(
        "Sequence index"
    )

    axis.set_title(
        "Question 9: Gaussian joint-feature matrix"
    )

    axis.text(
        5.45,
        nNormal / 2,
        "Normal",
        rotation=90,
        va="center",
    )

    axis.text(
        5.45,
        nNormal
        + (
            featureMatrix.shape[0]
            - nNormal
        )
        / 2,
        "Anomaly",
        rotation=90,
        va="center",
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Feature value",
    )

    figure.tight_layout()

    saveFigureBothFormats(
        figure,
        outputBasePath,
    )


def saveQuestion9SequenceExamples(
    sequences: np.ndarray,
    labels: np.ndarray,
    anomalyStarts: np.ndarray,
    anomalyEnds: np.ndarray,
    outputBasePath: Path,
) -> dict[str, int]:
    """Plot one fixed normal and one fixed anomalous sequence."""

    normalIndices = np.flatnonzero(
        labels == 0
    )

    anomalyIndices = np.flatnonzero(
        labels == 1
    )

    if (
        normalIndices.size == 0
        or anomalyIndices.size == 0
    ):
        raise ValueError(
            "Both normal and anomalous sequences "
            "are required"
        )

    # Q9 is unshuffled, so this deterministic
    # choice does not depend on model scores.
    normalIndex = int(
        normalIndices[0]
    )

    anomalyIndex = int(
        anomalyIndices[0]
    )

    anomalyStart = int(
        anomalyStarts[
            anomalyIndex,
            0,
        ]
    )

    anomalyEnd = int(
        anomalyEnds[
            anomalyIndex,
            0,
        ]
    )

    timeSteps = np.arange(
        sequences.shape[1]
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.2),
        sharex=True,
    )

    axes[0].plot(
        timeSteps,
        sequences[normalIndex],
        linewidth=1.3,
    )

    axes[0].set_title(
        "Nominal Gaussian sequence"
    )

    axes[0].set_ylabel(
        "Observation"
    )

    axes[0].grid(
        alpha=0.25
    )

    axes[1].plot(
        timeSteps,
        sequences[anomalyIndex],
        linewidth=1.3,
    )

    axes[1].axvspan(
        anomalyStart,
        anomalyEnd - 1,
        alpha=0.2,
        label="Inserted anomaly block",
    )

    axes[1].set_title(
        "Anomalous Gaussian sequence"
    )

    axes[1].set_xlabel(
        "Time step"
    )

    axes[1].set_ylabel(
        "Observation"
    )

    axes[1].grid(
        alpha=0.25
    )

    axes[1].legend()

    figure.tight_layout()

    saveFigureBothFormats(
        figure,
        outputBasePath,
    )

    return {
        "normalIndex": normalIndex,
        "anomalyIndex": anomalyIndex,
        "anomalyStart": anomalyStart,
        "anomalyEnd": anomalyEnd,
    }


def writeTrainingMetadata(
    trainSequences: np.ndarray,
    dataDirectory: Path,
) -> dict[str, object]:
    """Serialize the common clean Gaussian training pool."""

    dataDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        dataDirectory
        / "gaussian_train_sequences.npy",
        trainSequences,
    )

    metadata = {
        "dataset": "Required Gaussian training pool",
        "seed": BASE_SEED,
        "nTrainNormal": int(
            trainSequences.shape[0]
        ),
        "sequenceLength": int(
            trainSequences.shape[1]
        ),
        "totalTrainingTimeSteps": int(
            trainSequences.size
        ),
        "trainingContainsOnlyNormalSequences": True,
        "nominalDistribution": {
            "mean": 0.0,
            "standardDeviation": 1.0,
        },
    }

    (
        dataDirectory
        / "gaussian_train_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return metadata

def loadSavedGaussianTrainingSet(
    outputDirectory: str | Path,
) -> np.ndarray:
    """Load the exact training pool generated by Question 9."""

    trainingPath = (
        Path(outputDirectory)
        / "gaussian_required"
        / "data"
        / "gaussian_train_sequences.npy"
    )

    if not trainingPath.exists():
        raise FileNotFoundError(
            "The common Gaussian training pool was not found. "
            "Run Question 9 before Question 10."
        )

    trainSequences = np.load(
        trainingPath
    )

    if trainSequences.shape != (
        200,
        100,
    ):
        raise ValueError(
            "Unexpected Gaussian training shape: "
            f"{trainSequences.shape}"
        )

    if not np.all(
        np.isfinite(trainSequences)
    ):
        raise ValueError(
            "Gaussian training data contain "
            "non-finite values"
        )

    return trainSequences

def runQuestion9(
    outputDirectory: str | Path,
) -> dict[str, object]:
    """Run the complete required Question 9 experiment."""

    outputPath = Path(
        outputDirectory
    )

    gaussianOutputPath = (
        outputPath
        / "gaussian_required"
    )

    dataDirectory = (
        gaussianOutputPath
        / "data"
    )

    tableDirectory = (
        gaussianOutputPath
        / "tables"
    )

    figureDirectory = (
        gaussianOutputPath
        / "figures"
    )

    dataDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tableDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figureDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Common clean training data:
    # generated once and reused later by
    # both OC-SVM and HMAD.
    trainSequences = (
        makeGaussianTrainingSet()
    )

    trainingMetadata = (
        writeTrainingMetadata(
            trainSequences,
            dataDirectory,
        )
    )

    # Exact Question 9 data:
    # 200 normal + 50 anomalous,
    # one block of length 20.
    question9Dataset = (
        makeQuestion9Dataset()
    )

    question9Metadata = (
        saveGaussianTestSet(
            question9Dataset,
            dataDirectory,
            prefix="question9",
        )
    )

    # Required hard threshold states:
    # z_t = 1 when x_t < 2,
    # z_t = 2 otherwise.
    thresholdStates = (
        assignThresholdStates(
            question9Dataset.sequences,
            threshold=2.0,
        )
    )

    featureMatrix = (
        batchJointFeatures(
            question9Dataset.sequences,
            thresholdStates,
            2,
        )
    )

    validateQuestion9Features(
        featureMatrix
    )

    featureTable = pd.DataFrame(
        featureMatrix,
        columns=FEATURE_NAMES,
    )

    featureTable.insert(
        0,
        "trueLabel",
        question9Dataset.labels,
    )

    featureTable.insert(
        0,
        "sampleIndex",
        np.arange(
            question9Dataset.labels.size,
            dtype=int,
        ),
    )

    featureTable.insert(
        2,
        "anomalyStart",
        question9Dataset.anomalyStart,
    )

    featureTable.insert(
        3,
        "anomalyEnd",
        question9Dataset.anomalyEnd,
    )

    featureTable.to_csv(
        tableDirectory
        / "question9_joint_features.csv",
        index=False,
    )

    stateTable = pd.DataFrame(
        thresholdStates,
        columns=[
            f"t{timeIndex}"
            for timeIndex in range(
                thresholdStates.shape[1]
            )
        ],
    )

    stateTable.insert(
        0,
        "trueLabel",
        question9Dataset.labels,
    )

    stateTable.insert(
        0,
        "sampleIndex",
        np.arange(
            question9Dataset.labels.size,
            dtype=int,
        ),
    )

    stateTable.to_csv(
        tableDirectory
        / "question9_threshold_states.csv",
        index=False,
    )

    saveQuestion9Heatmap(
        featureMatrix,
        figureDirectory
        / "question9_joint_feature_heatmap",
        nNormal=200,
    )

    exampleMetadata = (
        saveQuestion9SequenceExamples(
            question9Dataset.sequences,
            question9Dataset.labels,
            question9Dataset.anomalyStarts,
            question9Dataset.anomalyEnds,
            figureDirectory
            / "question9_sequence_examples",
        )
    )

    normalFeatures = featureMatrix[
        question9Dataset.labels == 0
    ]

    anomalyFeatures = featureMatrix[
        question9Dataset.labels == 1
    ]

    featureSummaryRows = []

    for featureIndex, featureName in enumerate(
        FEATURE_NAMES
    ):
        featureSummaryRows.append(
            {
                "feature": featureName,
                "normalMean": float(
                    np.mean(
                        normalFeatures[
                            :,
                            featureIndex,
                        ]
                    )
                ),
                "anomalyMean": float(
                    np.mean(
                        anomalyFeatures[
                            :,
                            featureIndex,
                        ]
                    )
                ),
                "absoluteMeanDifference": float(
                    abs(
                        np.mean(
                            anomalyFeatures[
                                :,
                                featureIndex,
                            ]
                        )
                        - np.mean(
                            normalFeatures[
                                :,
                                featureIndex,
                            ]
                        )
                    )
                ),
            }
        )

    featureSummaryTable = pd.DataFrame(
        featureSummaryRows
    ).sort_values(
        "absoluteMeanDifference",
        ascending=False,
    )

    featureSummaryTable.to_csv(
        tableDirectory
        / "question9_feature_group_summary.csv",
        index=False,
    )

    summary = {
        "experiment": "Simulation Question 9",
        "training": trainingMetadata,
        "question9Dataset": question9Metadata,
        "stateAssignment": {
            "rule": (
                "internal state 0 if x < 2; "
                "internal state 1 otherwise"
            ),
            "reportConvention": (
                "state 1 if x < 2; "
                "state 2 otherwise"
            ),
            "threshold": 2.0,
        },
        "jointFeatureShape": list(
            featureMatrix.shape
        ),
        "transitionFeaturesSumToOne": bool(
            np.allclose(
                featureMatrix[:, :4].sum(
                    axis=1
                ),
                1.0,
            )
        ),
        "allFeaturesFinite": bool(
            np.isfinite(
                featureMatrix
            ).all()
        ),
        "exampleSequences": exampleMetadata,
        "largestMeanDifferenceFeature": str(
            featureSummaryTable.iloc[0][
                "feature"
            ]
        ),
        "seedPolicyConfirmedByTA": True,
        "seedPolicy": (
            "The clean training pool uses seed 1405. "
            "Independent deterministic derived seeds are used "
            "for Q9, held-out test pools, and subset shuffling."
        ),
        "currentSeeds": {
            "training": BASE_SEED,
            "question9": Q9_SEED,
        },
    }

    (
        gaussianOutputPath
        / "question9_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Question 9 feature matrix:",
        featureMatrix.shape,
    )

    print(
        "Normal sequences:",
        int(
            np.sum(
                question9Dataset.labels == 0
            )
        ),
    )

    print(
        "Anomalous sequences:",
        int(
            np.sum(
                question9Dataset.labels == 1
            )
        ),
    )

    print()

    print(
        featureSummaryTable.to_string(
            index=False
        )
    )

    print()

    print(
        "Wrote Question 9 outputs to:",
        gaussianOutputPath,
    )

    return summary

def runQuestion10(
    outputDirectory: str | Path,
) -> tuple[dict[str, object], SimplifiedHMAD]:
    """Train simplified HMAD for Simulation Question 10."""

    outputPath = Path(
        outputDirectory
    )

    gaussianOutputPath = (
        outputPath
        / "gaussian_required"
    )

    tableDirectory = (
        gaussianOutputPath
        / "tables"
    )

    figureDirectory = (
        gaussianOutputPath
        / "figures"
    )

    tableDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figureDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainSequences = (
        loadSavedGaussianTrainingSet(
            outputDirectory
        )
    )

    hmadModel = SimplifiedHMAD(
        nStates=2,
        nu=0.10,
        gamma="scale",
        maxIterations=10,
        convergenceThreshold=0.01,
        initialSelfTransition=0.90,
        varianceFloor=1.0e-6,
        transitionSmoothing=1.0e-3,
        randomState=BASE_SEED,
    ).fit(
        trainSequences
    )

    convergenceTable = pd.DataFrame(
        hmadModel.convergenceHistory_
    )

    requiredColumns = {
        "iteration",
        "changedFraction",
        "meanDecisionScore",
    }

    if not requiredColumns.issubset(
        convergenceTable.columns
    ):
        raise AssertionError(
            "HMAD convergence history does not "
            "contain the required columns"
        )

    if convergenceTable.empty:
        raise AssertionError(
            "HMAD convergence history is empty"
        )

    if convergenceTable.shape[0] > 10:
        raise AssertionError(
            "HMAD exceeded the maximum of "
            "10 training iterations"
        )

    convergenceTable.to_csv(
        tableDirectory
        / "question10_hmad_convergence.csv",
        index=False,
    )

    saveConvergencePlot(
        convergenceTable[
            "iteration"
        ].to_numpy(dtype=int),
        convergenceTable[
            "changedFraction"
        ].to_numpy(dtype=float),
        convergenceTable[
            "meanDecisionScore"
        ].to_numpy(dtype=float),
        figureDirectory
        / "question10_hmad_convergence.pdf",
    )

    trainingStates = np.asarray(
        hmadModel.trainingStates_,
        dtype=int,
    )

    stateCounts = np.bincount(
        trainingStates.reshape(-1),
        minlength=2,
    )

    stateFractions = (
        stateCounts
        / trainingStates.size
    )

    stateCollapseObserved = bool(
        np.count_nonzero(stateCounts) < 2
    )

    trainingFeatures = np.asarray(
        hmadModel.trainingFeatures_,
        dtype=float,
    )

    if trainingStates.shape != (
        200,
        100,
    ):
        raise AssertionError(
            "Unexpected HMAD training-state shape: "
            f"{trainingStates.shape}"
        )

    if trainingFeatures.shape != (
        200,
        6,
    ):
        raise AssertionError(
            "Unexpected HMAD joint-feature shape: "
            f"{trainingFeatures.shape}"
        )

    if not np.all(
        np.isfinite(trainingFeatures)
    ):
        raise AssertionError(
            "HMAD training features contain "
            "non-finite values"
        )

    transitionSums = np.sum(
        trainingFeatures[:, :4],
        axis=1,
    )

    if not np.allclose(
        transitionSums,
        1.0,
        rtol=1.0e-8,
        atol=1.0e-8,
    ):
        raise AssertionError(
            "Question 10 transition features "
            "do not sum to one"
        )

    stateTable = pd.DataFrame(
        trainingStates + 1,
        columns=[
            f"t{timeIndex}"
            for timeIndex in range(
                trainingStates.shape[1]
            )
        ],
    )

    stateTable.insert(
        0,
        "sampleIndex",
        np.arange(
            trainingStates.shape[0],
            dtype=int,
        ),
    )

    stateTable.to_csv(
        tableDirectory
        / "question10_training_states.csv",
        index=False,
    )

    featureTable = pd.DataFrame(
        trainingFeatures,
        columns=FEATURE_NAMES,
    )

    featureTable.insert(
        0,
        "sampleIndex",
        np.arange(
            trainingFeatures.shape[0],
            dtype=int,
        ),
    )

    featureTable.to_csv(
        tableDirectory
        / "question10_training_joint_features.csv",
        index=False,
    )

    learnedParameters = {
        "nStates": 2,
        "nu": 0.10,
        "gamma": "scale",
        "maxIterations": 10,
        "convergenceThreshold": 0.01,
        "initialSelfTransition": 0.90,
        "varianceFloor": 1.0e-6,
        "transitionSmoothing": 1.0e-3,
        "seed": BASE_SEED,
        "emissionMeans": (
            hmadModel.emissionMeans_.tolist()
        ),
        "emissionVariances": (
            hmadModel.emissionVariances_.tolist()
        ),
        "initialProbabilities": (
            hmadModel.initialProbabilities_.tolist()
        ),
        "transitionMatrix": (
            hmadModel.transitionMatrix_.tolist()
        ),
        "nIterations": int(
            hmadModel.nIterations_
        ),
        "converged": bool(
            hmadModel.converged_
        ),
        "finalChangedFraction": float(
            convergenceTable.iloc[-1][
                "changedFraction"
            ]
        ),
        "finalMeanDecisionScore": float(
            convergenceTable.iloc[-1][
                "meanDecisionScore"
            ]
        ),
        "stateCounts": {
            "state1": int(stateCounts[0]),
            "state2": int(stateCounts[1]),
        },
        "stateFractions": {
            "state1": float(stateFractions[0]),
            "state2": float(stateFractions[1]),
        },
        "stateCollapseObserved": (
            stateCollapseObserved
        ),
        "stateCollapseAcceptedByTA": True,
        "stateCollapseInterpretation": (
            "The clean N(0,1) training data contain one nominal "
            "regime, so complete collapse to one active state is "
            "an accepted empirical outcome."
        ),
        "seedPolicyConfirmedByTA": True,
    }

    parameterPath = (
        tableDirectory
        / "question10_learned_parameters.json"
    )

    parameterPath.write_text(
        json.dumps(
            learnedParameters,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "experiment": (
            "Simulation Question 10"
        ),
        "trainingShape": list(
            trainSequences.shape
        ),
        "jointFeatureShape": list(
            trainingFeatures.shape
        ),
        "parameters": learnedParameters,
        "convergenceArtifact": str(
            tableDirectory
            / "question10_hmad_convergence.csv"
        ),
        "convergenceFigure": str(
            figureDirectory
            / "question10_hmad_convergence.pdf"
        ),
        "reusedCommonTrainingPool": True,
    }

    (
        gaussianOutputPath
        / "question10_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Question 10 HMAD training"
    )

    print(
        convergenceTable.to_string(
            index=False
        )
    )

    print()

    print(
        "Emission means:",
        hmadModel.emissionMeans_,
    )

    print(
        "Emission variances:",
        hmadModel.emissionVariances_,
    )

    print(
        "Transition matrix:"
    )

    print(
        hmadModel.transitionMatrix_
    )

    print(
        "Iterations:",
        hmadModel.nIterations_,
    )

    print(
        "Converged:",
        hmadModel.converged_,
    )

    return (
        summary,
        hmadModel,
    )


GAUSSIAN_OCSVM_REPRESENTATIONS = [
    "flattened",
    "global_summary",
    "window_summary",
]

GAUSSIAN_PRIMARY_REPRESENTATION = "global_summary"
GAUSSIAN_PRIMARY_NU = 0.10
GAUSSIAN_N_WINDOWS = 5
GAUSSIAN_SENSITIVITY_FRACTIONS = [
    0.01,
    0.05,
    0.10,
    0.20,
]


def _fractionTag(
    fraction: float,
) -> str:
    return (
        f"{fraction:.2f}"
        .replace(".", "p")
    )


def _validateQuestion11TestSet(
    sequences: np.ndarray,
    labels: np.ndarray,
    *,
    expectedFraction: float,
) -> None:
    sequenceArray = np.asarray(
        sequences,
        dtype=float,
    )

    labelArray = np.asarray(
        labels,
        dtype=int,
    ).reshape(-1)

    if sequenceArray.shape != (
        1000,
        100,
    ):
        raise AssertionError(
            "Question 11 sensitivity test set has "
            f"shape {sequenceArray.shape}; "
            "expected (1000, 100)"
        )

    if labelArray.shape != (
        1000,
    ):
        raise AssertionError(
            "Question 11 labels have an "
            f"unexpected shape: {labelArray.shape}"
        )

    if not np.all(
        np.isfinite(sequenceArray)
    ):
        raise AssertionError(
            "Question 11 sequences contain "
            "non-finite values"
        )

    if not np.all(
        np.isin(labelArray, [0, 1])
    ):
        raise AssertionError(
            "Question 11 labels must use "
            "0 = normal and 1 = anomaly"
        )

    achievedFraction = float(
        np.mean(labelArray)
    )

    if not np.isclose(
        achievedFraction,
        expectedFraction,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError(
            "Question 11 anomaly fraction is "
            f"{achievedFraction}; "
            f"expected {expectedFraction}"
        )


def _fitGaussianOcsvmBaselines(
    trainSequences: np.ndarray,
) -> dict[str, OcsvmBaseline]:
    """Fit all required OC-SVM representations once on clean train data."""

    baselineModels: dict[
        str,
        OcsvmBaseline,
    ] = {}

    for representation in (
        GAUSSIAN_OCSVM_REPRESENTATIONS
    ):
        trainFeatures = (
            buildFeatureRepresentation(
                trainSequences,
                representation,
                nWindows=(
                    GAUSSIAN_N_WINDOWS
                ),
            )
        )

        baselineModels[
            representation
        ] = fitOcsvmBaseline(
            trainFeatures,
            representation=representation,
            nu=GAUSSIAN_PRIMARY_NU,
            gamma="scale",
            nWindows=(
                GAUSSIAN_N_WINDOWS
            ),
        )

    return baselineModels


def _scoreGaussianOcsvmBaselines(
    baselineModels: dict[
        str,
        OcsvmBaseline,
    ],
    sequences: np.ndarray,
) -> dict[str, np.ndarray]:
    scoreLookup: dict[
        str,
        np.ndarray,
    ] = {}

    for (
        representation,
        baselineModel,
    ) in baselineModels.items():
        testFeatures = (
            buildFeatureRepresentation(
                sequences,
                representation,
                nWindows=(
                    GAUSSIAN_N_WINDOWS
                ),
            )
        )

        scoreLookup[
            representation
        ] = baselineModel.decisionFunction(
            testFeatures
        )

    return scoreLookup


def _appendQuestion11ScoreRows(
    rows: list[dict[str, object]],
    *,
    labels: np.ndarray,
    normalityScores: np.ndarray,
    modelName: str,
    representation: str,
    anomalyFraction: float,
    evaluation: str,
    seed: int,
) -> None:
    labelArray = np.asarray(
        labels,
        dtype=int,
    ).reshape(-1)

    scoreArray = np.asarray(
        normalityScores,
        dtype=float,
    ).reshape(-1)

    if labelArray.shape != scoreArray.shape:
        raise ValueError(
            "Question 11 labels and scores "
            "must have equal shapes"
        )

    if not np.all(
        np.isfinite(scoreArray)
    ):
        raise ValueError(
            f"{modelName} scores contain "
            "non-finite values"
        )

    hardPredictions = (
        scoreArray < 0.0
    ).astype(int)

    for sampleIndex in range(
        labelArray.size
    ):
        rows.append(
            {
                "evaluation": evaluation,
                "targetRTest": (
                    anomalyFraction
                ),
                "achievedRTest": float(
                    np.mean(labelArray)
                ),
                "sampleIndex": (
                    sampleIndex
                ),
                "trueLabel": int(
                    labelArray[
                        sampleIndex
                    ]
                ),
                "model": modelName,
                "representation": (
                    representation
                ),
                "nu": (
                    GAUSSIAN_PRIMARY_NU
                ),
                "normalityScore": float(
                    scoreArray[
                        sampleIndex
                    ]
                ),
                "anomalyScore": float(
                    -scoreArray[
                        sampleIndex
                    ]
                ),
                "hardPrediction": int(
                    hardPredictions[
                        sampleIndex
                    ]
                ),
                "seed": seed,
            }
        )


def _chooseCorrectQuestion11Sample(
    labels: np.ndarray,
    normalityScores: np.ndarray,
    *,
    targetLabel: int,
) -> tuple[int, bool]:
    """Choose the first correctly classified sample deterministically.

    If no correctly classified sample exists, return the first sample
    of the requested class and mark the selection as a fallback.
    """

    labelArray = np.asarray(
        labels,
        dtype=int,
    ).reshape(-1)

    scoreArray = np.asarray(
        normalityScores,
        dtype=float,
    ).reshape(-1)

    predictions = (
        scoreArray < 0.0
    ).astype(int)

    correctIndices = np.flatnonzero(
        (labelArray == targetLabel)
        & (predictions == targetLabel)
    )

    if correctIndices.size > 0:
        return (
            int(correctIndices[0]),
            True,
        )

    classIndices = np.flatnonzero(
        labelArray == targetLabel
    )

    if classIndices.size == 0:
        raise ValueError(
            "No Question 11 sample has "
            f"target label {targetLabel}"
        )

    return (
        int(classIndices[0]),
        False,
    )


def _saveQuestion11DecodedPair(
    *,
    normalSequence: np.ndarray,
    normalStates: np.ndarray,
    anomalousSequence: np.ndarray,
    anomalousStates: np.ndarray,
    anomalyStart: int,
    anomalyEnd: int,
    outputBasePath: Path,
) -> None:
    """Plot normal and anomalous Viterbi paths side by side."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 4.6),
        sharex=True,
    )

    examples = [
        (
            axes[0],
            normalSequence,
            normalStates,
            "Correctly classified normal",
            -1,
            -1,
        ),
        (
            axes[1],
            anomalousSequence,
            anomalousStates,
            "Correctly classified anomaly",
            anomalyStart,
            anomalyEnd,
        ),
    ]

    for (
        axis,
        sequence,
        states,
        title,
        blockStart,
        blockEnd,
    ) in examples:
        timeSteps = np.arange(
            sequence.size
        )

        axis.plot(
            timeSteps,
            sequence,
            linewidth=1.25,
            label="Observation",
        )

        if (
            blockStart >= 0
            and blockEnd > blockStart
        ):
            axis.axvspan(
                blockStart,
                blockEnd - 1,
                alpha=0.2,
                label=(
                    "True anomaly block"
                ),
            )

        stateAxis = axis.twinx()

        stateAxis.step(
            timeSteps,
            states + 1,
            where="mid",
            linestyle="--",
            linewidth=1.2,
            label="Viterbi state",
        )

        axis.set_title(title)
        axis.set_xlabel("Time step")
        axis.set_ylabel("Observation")
        stateAxis.set_ylabel(
            "Decoded state"
        )
        stateAxis.set_yticks([1, 2])
        axis.grid(alpha=0.25)

        handles1, labels1 = (
            axis.get_legend_handles_labels()
        )

        handles2, labels2 = (
            stateAxis.get_legend_handles_labels()
        )

        axis.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper left",
        )

    figure.suptitle(
        "Question 11: Viterbi-decoded paths"
    )

    figure.tight_layout()

    saveFigureBothFormats(
        figure,
        outputBasePath,
    )


def _saveSensitivityMasterPool(
    pool: object,
    dataDirectory: Path,
) -> dict[str, object]:
    """Serialize the common class-conditional sensitivity pool."""

    dataDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        dataDirectory
        / "sensitivity_pool_normal.npy",
        pool.normalSequences,
    )

    np.save(
        dataDirectory
        / "sensitivity_pool_anomaly.npy",
        pool.anomalousSequences,
    )

    np.save(
        dataDirectory
        / "sensitivity_pool_anomaly_starts.npy",
        pool.anomalyStarts,
    )

    np.save(
        dataDirectory
        / "sensitivity_pool_anomaly_ends.npy",
        pool.anomalyEnds,
    )

    np.save(
        dataDirectory
        / "sensitivity_pool_n_blocks.npy",
        pool.nAnomalyBlocks,
    )

    metadata = {
        "dataset": (
            "Required Gaussian sensitivity "
            "master pool"
        ),
        "seed": (
            SENSITIVITY_POOL_SEED
        ),
        "nNormalPool": int(
            pool.normalSequences.shape[0]
        ),
        "nAnomalyPool": int(
            pool.anomalousSequences.shape[0]
        ),
        "sequenceLength": int(
            pool.normalSequences.shape[1]
        ),
        "anomalyBlockLength": 20,
        "nBlocksPerAnomalousSequence": 1,
        "nominalDistribution": {
            "mean": 0.0,
            "standardDeviation": 1.0,
        },
        "anomalyBlockDistribution": {
            "mean": 4.0,
            "standardDeviation": 1.0,
        },
        "sharedAcrossModels": True,
        "sharedAcrossFractions": True,
    }

    (
        dataDirectory
        / "sensitivity_pool_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return metadata


def evaluateAggregateViterbiAlignment(
    decodedStates: np.ndarray,
    labels: np.ndarray,
    anomalyStarts: np.ndarray,
    anomalyEnds: np.ndarray,
    *,
    highStateIndex: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate high-emission-state alignment for every anomaly."""

    stateArray = np.asarray(
        decodedStates,
        dtype=int,
    )

    labelArray = np.asarray(
        labels,
        dtype=int,
    ).reshape(-1)

    startArray = np.asarray(
        anomalyStarts,
        dtype=int,
    )

    endArray = np.asarray(
        anomalyEnds,
        dtype=int,
    )

    if stateArray.ndim != 2:
        raise ValueError(
            "decodedStates must be a matrix"
        )

    if labelArray.shape[0] != stateArray.shape[0]:
        raise ValueError(
            "labels and decodedStates have different row counts"
        )

    if startArray.shape != endArray.shape:
        raise ValueError(
            "anomalyStarts and anomalyEnds must have equal shapes"
        )

    if startArray.ndim != 2:
        raise ValueError(
            "Anomaly interval arrays must be two-dimensional"
        )

    if startArray.shape[0] != stateArray.shape[0]:
        raise ValueError(
            "Anomaly intervals and decodedStates have different row counts"
        )

    anomalyIndices = np.flatnonzero(
        labelArray == 1
    )

    if anomalyIndices.size == 0:
        raise ValueError(
            "No anomalous sequences are available"
        )

    rows: list[dict[str, object]] = []

    totalTruePositive = 0
    totalFalsePositive = 0
    totalFalseNegative = 0
    totalTrueNegative = 0

    for sampleIndex in anomalyIndices:
        trueBlockMask = np.zeros(
            stateArray.shape[1],
            dtype=bool,
        )

        for blockStart, blockEnd in zip(
            startArray[sampleIndex],
            endArray[sampleIndex],
            strict=True,
        ):
            if blockStart < 0:
                continue

            if not (
                0 <= blockStart < blockEnd <= stateArray.shape[1]
            ):
                raise ValueError(
                    "An anomaly interval is outside the sequence range"
                )

            trueBlockMask[
                blockStart:blockEnd
            ] = True

        if not np.any(trueBlockMask):
            raise ValueError(
                "An anomalous sequence has no valid anomaly interval"
            )

        predictedBlockMask = (
            stateArray[sampleIndex]
            == highStateIndex
        )

        truePositive = int(
            np.sum(
                predictedBlockMask
                & trueBlockMask
            )
        )

        falsePositive = int(
            np.sum(
                predictedBlockMask
                & ~trueBlockMask
            )
        )

        falseNegative = int(
            np.sum(
                ~predictedBlockMask
                & trueBlockMask
            )
        )

        trueNegative = int(
            np.sum(
                ~predictedBlockMask
                & ~trueBlockMask
            )
        )

        precision = (
            truePositive
            / (
                truePositive
                + falsePositive
            )
            if (
                truePositive
                + falsePositive
            )
            > 0
            else 0.0
        )

        recall = (
            truePositive
            / (
                truePositive
                + falseNegative
            )
            if (
                truePositive
                + falseNegative
            )
            > 0
            else 0.0
        )

        f1Score = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            )
            > 0
            else 0.0
        )

        insideFraction = float(
            np.mean(
                predictedBlockMask[
                    trueBlockMask
                ]
            )
        )

        outsideFraction = float(
            np.mean(
                predictedBlockMask[
                    ~trueBlockMask
                ]
            )
        )

        rows.append(
            {
                "sampleIndex": int(sampleIndex),
                "truePositiveTimeSteps": truePositive,
                "falsePositiveTimeSteps": falsePositive,
                "falseNegativeTimeSteps": falseNegative,
                "trueNegativeTimeSteps": trueNegative,
                "localizationPrecision": precision,
                "localizationRecall": recall,
                "localizationF1": f1Score,
                "highStateFractionInsideBlock": insideFraction,
                "highStateFractionOutsideBlock": outsideFraction,
                "exactPathBlockMatch": bool(
                    np.array_equal(
                        predictedBlockMask,
                        trueBlockMask,
                    )
                ),
            }
        )

        totalTruePositive += truePositive
        totalFalsePositive += falsePositive
        totalFalseNegative += falseNegative
        totalTrueNegative += trueNegative

    resultTable = pd.DataFrame(rows)

    microPrecision = (
        totalTruePositive
        / (
            totalTruePositive
            + totalFalsePositive
        )
        if (
            totalTruePositive
            + totalFalsePositive
        )
        > 0
        else 0.0
    )

    microRecall = (
        totalTruePositive
        / (
            totalTruePositive
            + totalFalseNegative
        )
        if (
            totalTruePositive
            + totalFalseNegative
        )
        > 0
        else 0.0
    )

    microF1 = (
        2.0
        * microPrecision
        * microRecall
        / (
            microPrecision
            + microRecall
        )
        if (
            microPrecision
            + microRecall
        )
        > 0
        else 0.0
    )

    summary = {
        "nAnomalousSequences": int(
            anomalyIndices.size
        ),
        "higherMeanStateIndexReportConvention": int(
            highStateIndex + 1
        ),
        "meanHighStateFractionInsideBlock": float(
            resultTable[
                "highStateFractionInsideBlock"
            ].mean()
        ),
        "meanHighStateFractionOutsideBlock": float(
            resultTable[
                "highStateFractionOutsideBlock"
            ].mean()
        ),
        "microLocalizationPrecision": float(
            microPrecision
        ),
        "microLocalizationRecall": float(
            microRecall
        ),
        "microLocalizationF1": float(
            microF1
        ),
        "exactPathBlockMatchCount": int(
            resultTable[
                "exactPathBlockMatch"
            ].sum()
        ),
        "exactPathBlockMatchFraction": float(
            resultTable[
                "exactPathBlockMatch"
            ].mean()
        ),
        "timeStepConfusionMatrix": {
            "truePositive": int(
                totalTruePositive
            ),
            "falsePositive": int(
                totalFalsePositive
            ),
            "falseNegative": int(
                totalFalseNegative
            ),
            "trueNegative": int(
                totalTrueNegative
            ),
        },
    }

    return resultTable, summary


def runQuestion11(
    outputDirectory: str | Path,
    hmadModel: SimplifiedHMAD,
) -> dict[str, object]:
    """Run prediction, ablation, ROC, and contamination sensitivity."""

    outputPath = Path(
        outputDirectory
    )

    gaussianOutputPath = (
        outputPath
        / "gaussian_required"
    )

    dataDirectory = (
        gaussianOutputPath
        / "data"
    )

    tableDirectory = (
        gaussianOutputPath
        / "tables"
    )

    figureDirectory = (
        gaussianOutputPath
        / "figures"
    )

    for directory in [
        dataDirectory,
        tableDirectory,
        figureDirectory,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    trainSequences = (
        loadSavedGaussianTrainingSet(
            outputDirectory
        )
    )

    baselineModels = (
        _fitGaussianOcsvmBaselines(
            trainSequences
        )
    )

    baselineConfigurationRows = []

    for (
        representation,
        baselineModel,
    ) in baselineModels.items():
        featureDimension = int(
            buildFeatureRepresentation(
                trainSequences[:1],
                representation,
                nWindows=(
                    GAUSSIAN_N_WINDOWS
                ),
            ).shape[1]
        )

        baselineConfigurationRows.append(
            {
                "representation": (
                    representation
                ),
                "featureDimension": (
                    featureDimension
                ),
                "kernel": "rbf",
                "nu": float(
                    baselineModel.nu
                ),
                "gamma": str(
                    baselineModel.gamma
                ),
                "nWindows": int(
                    baselineModel.nWindows
                ),
                "isPrimaryGaussianBaseline": bool(
                    representation
                    == (
                        GAUSSIAN_PRIMARY_REPRESENTATION
                    )
                ),
            }
        )

    pd.DataFrame(
        baselineConfigurationRows
    ).to_csv(
        tableDirectory
        / "question11_ocsvm_configurations.csv",
        index=False,
    )

    masterPool = (
        makeSensitivityMasterPool(
            nNormalPool=990,
            nAnomalyPool=200,
            sequenceLength=100,
            anomalyBlockLength=20,
            nBlocks=1,
            minimumGap=0,
            seed=(
                SENSITIVITY_POOL_SEED
            ),
        )
    )

    masterPoolMetadata = (
        _saveSensitivityMasterPool(
            masterPool,
            dataDirectory,
        )
    )

    sensitivityMetricRows: list[
        dict[str, object]
    ] = []

    sensitivityScoreRows: list[
        dict[str, object]
    ] = []

    aucLookup: dict[
        str,
        list[float],
    ] = {
        "HMAD": [],
        "HMAD all-one ablation": [],
        (
            "OC-SVM "
            + GAUSSIAN_PRIMARY_REPRESENTATION
        ): [],
    }

    allModelAucLookup: dict[
        str,
        list[float],
    ] = {
        "HMAD": [],
        "HMAD all-one ablation": [],
    }

    for representation in (
        GAUSSIAN_OCSVM_REPRESENTATIONS
    ):
        allModelAucLookup[
            f"OC-SVM {representation}"
        ] = []

    exactTenPercentArtifacts: dict[
        str,
        object,
    ] | None = None

    aggregateAlignmentSummary: dict[
        str,
        object,
    ] | None = None

    for fractionIndex, anomalyFraction in enumerate(
        GAUSSIAN_SENSITIVITY_FRACTIONS
    ):
        subsetSeed = (
            1500
            + fractionIndex
        )

        testSet = (
            makeExactFractionSubsetFromPool(
                masterPool,
                anomalyFraction,
                nTotal=1000,
                shuffleSeed=subsetSeed,
            )
        )

        _validateQuestion11TestSet(
            testSet.sequences,
            testSet.labels,
            expectedFraction=(
                anomalyFraction
            ),
        )

        fractionTag = _fractionTag(
            anomalyFraction
        )

        saveGaussianTestSet(
            testSet,
            dataDirectory,
            prefix=(
                "question11_rtest_"
                + fractionTag
            ),
        )

        decodedStates = (
            hmadModel.decodeSequences(
                testSet.sequences
            )
        )

        if decodedStates.shape != (
            1000,
            100,
        ):
            raise AssertionError(
                "Question 11 decoded-state "
                f"shape is {decodedStates.shape}; "
                "expected (1000, 100)"
            )

        hmadJointFeatures = (
            batchJointFeatures(
                testSet.sequences,
                decodedStates,
                2,
            )
        )

        if hmadJointFeatures.shape != (
            1000,
            6,
        ):
            raise AssertionError(
                "Question 11 HMAD feature "
                f"shape is {hmadJointFeatures.shape}; "
                "expected (1000, 6)"
            )

        if not np.allclose(
            hmadJointFeatures[:, :4].sum(
                axis=1
            ),
            1.0,
            rtol=1.0e-8,
            atol=1.0e-8,
        ):
            raise AssertionError(
                "Question 11 transition "
                "features do not sum to one"
            )

        hmadScores = (
            hmadModel.decisionFunction(
                testSet.sequences
            )
        )

        trivialScores = (
            hmadModel
            .decisionFunctionWithTrivialStates(
                testSet.sequences
            )
        )

        baselineScoreLookup = (
            _scoreGaussianOcsvmBaselines(
                baselineModels,
                testSet.sequences,
            )
        )

        modelScoreLookup: dict[
            str,
            tuple[np.ndarray, str],
        ] = {
            "HMAD": (
                hmadScores,
                "joint_features_viterbi",
            ),
            "HMAD all-one ablation": (
                trivialScores,
                "joint_features_all_one",
            ),
        }

        for (
            representation,
            scores,
        ) in baselineScoreLookup.items():
            modelScoreLookup[
                f"OC-SVM {representation}"
            ] = (
                scores,
                representation,
            )

        for (
            modelName,
            (
                normalityScores,
                representation,
            ),
        ) in modelScoreLookup.items():
            metrics = (
                evaluateNormalityScores(
                    testSet.labels,
                    normalityScores,
                )
            )

            metricRow = {
                "model": modelName,
                "representation": (
                    representation
                ),
                "evaluation": (
                    "gaussian_sensitivity"
                ),
                "targetRTest": float(
                    anomalyFraction
                ),
                "achievedRTest": float(
                    np.mean(
                        testSet.labels
                    )
                ),
                "nNormal": int(
                    np.sum(
                        testSet.labels == 0
                    )
                ),
                "nAnomaly": int(
                    np.sum(
                        testSet.labels == 1
                    )
                ),
                "nu": (
                    GAUSSIAN_PRIMARY_NU
                ),
                "seed": subsetSeed,
                **metrics,
            }

            sensitivityMetricRows.append(
                metricRow
            )

            allModelAucLookup[
                modelName
            ].append(
                metrics["rocAuc"]
            )

            if modelName in aucLookup:
                aucLookup[
                    modelName
                ].append(
                    metrics["rocAuc"]
                )

            _appendQuestion11ScoreRows(
                sensitivityScoreRows,
                labels=testSet.labels,
                normalityScores=(
                    normalityScores
                ),
                modelName=modelName,
                representation=(
                    representation
                ),
                anomalyFraction=(
                    anomalyFraction
                ),
                evaluation=(
                    "gaussian_sensitivity"
                ),
                seed=subsetSeed,
            )

        if np.isclose(
            anomalyFraction,
            0.10,
            rtol=0.0,
            atol=1.0e-12,
        ):
            primaryBaselineName = (
                "OC-SVM "
                + (
                    GAUSSIAN_PRIMARY_REPRESENTATION
                )
            )

            primaryBaselineScores = (
                baselineScoreLookup[
                    GAUSSIAN_PRIMARY_REPRESENTATION
                ]
            )

            saveRocPlot(
                testSet.labels,
                {
                    "HMAD": (
                        hmadScores
                    ),
                    (
                        "OC-SVM "
                        + (
                            GAUSSIAN_PRIMARY_REPRESENTATION
                        )
                    ): (
                        primaryBaselineScores
                    ),
                    "HMAD all-one ablation": (
                        trivialScores
                    ),
                },
                figureDirectory
                / "question11_roc_rtest_0p10.pdf",
                title=(
                    "Question 11 ROC curves "
                    "at exact r_test = 0.10"
                ),
            )

            normalIndex, normalWasCorrect = (
                _chooseCorrectQuestion11Sample(
                    testSet.labels,
                    hmadScores,
                    targetLabel=0,
                )
            )

            anomalyIndex, anomalyWasCorrect = (
                _chooseCorrectQuestion11Sample(
                    testSet.labels,
                    hmadScores,
                    targetLabel=1,
                )
            )

            anomalyStart = int(
                testSet.anomalyStart[
                    anomalyIndex
                ]
            )

            anomalyEnd = int(
                testSet.anomalyEnd[
                    anomalyIndex
                ]
            )

            saveDecodedPathPlot(
                testSet.sequences[
                    normalIndex
                ],
                decodedStates[
                    normalIndex
                ],
                (
                    "Question 11 decoded path: "
                    "normal sequence"
                ),
                figureDirectory
                / "question11_viterbi_normal.pdf",
            )

            saveDecodedPathPlot(
                testSet.sequences[
                    anomalyIndex
                ],
                decodedStates[
                    anomalyIndex
                ],
                (
                    "Question 11 decoded path: "
                    "anomalous sequence"
                ),
                figureDirectory
                / "question11_viterbi_anomaly.pdf",
                anomalyStart=(
                    anomalyStart
                ),
                anomalyEnd=(
                    anomalyEnd
                ),
            )

            _saveQuestion11DecodedPair(
                normalSequence=(
                    testSet.sequences[
                        normalIndex
                    ]
                ),
                normalStates=(
                    decodedStates[
                        normalIndex
                    ]
                ),
                anomalousSequence=(
                    testSet.sequences[
                        anomalyIndex
                    ]
                ),
                anomalousStates=(
                    decodedStates[
                        anomalyIndex
                    ]
                ),
                anomalyStart=(
                    anomalyStart
                ),
                anomalyEnd=(
                    anomalyEnd
                ),
                outputBasePath=(
                    figureDirectory
                    / (
                        "question11_viterbi_"
                        "examples_side_by_side"
                    )
                ),
            )

            highStateIndex = int(
                np.argmax(
                    hmadModel.emissionMeans_
                )
            )

            (
                aggregateAlignmentTable,
                aggregateAlignmentSummary,
            ) = evaluateAggregateViterbiAlignment(
                decodedStates,
                testSet.labels,
                testSet.anomalyStarts,
                testSet.anomalyEnds,
                highStateIndex=highStateIndex,
            )

            aggregateAlignmentTable.to_csv(
                tableDirectory
                / (
                    "question11_viterbi_alignment_"
                    "all_anomalies.csv"
                ),
                index=False,
            )

            anomalyPath = (
                decodedStates[
                    anomalyIndex
                ]
            )

            insideMask = np.zeros(
                anomalyPath.size,
                dtype=bool,
            )

            insideMask[
                anomalyStart:anomalyEnd
            ] = True

            highStateInsideFraction = float(
                np.mean(
                    anomalyPath[
                        insideMask
                    ]
                    == highStateIndex
                )
            )

            highStateOutsideFraction = float(
                np.mean(
                    anomalyPath[
                        ~insideMask
                    ]
                    == highStateIndex
                )
            )

            decodedExampleTable = (
                pd.DataFrame(
                    {
                        "timeStep": np.arange(
                            testSet.sequences.shape[
                                1
                            ],
                            dtype=int,
                        ),
                        "normalObservation": (
                            testSet.sequences[
                                normalIndex
                            ]
                        ),
                        "normalDecodedState": (
                            decodedStates[
                                normalIndex
                            ]
                            + 1
                        ),
                        "anomalyObservation": (
                            testSet.sequences[
                                anomalyIndex
                            ]
                        ),
                        "anomalyDecodedState": (
                            decodedStates[
                                anomalyIndex
                            ]
                            + 1
                        ),
                        "isInsideTrueAnomalyBlock": (
                            insideMask.astype(int)
                        ),
                    }
                )
            )

            decodedExampleTable.to_csv(
                tableDirectory
                / (
                    "question11_decoded_"
                    "path_examples.csv"
                ),
                index=False,
            )

            exactTenPercentRows = [
                row
                for row in (
                    sensitivityMetricRows
                )
                if np.isclose(
                    float(
                        row[
                            "targetRTest"
                        ]
                    ),
                    0.10,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ]

            pd.DataFrame(
                exactTenPercentRows
            ).to_csv(
                tableDirectory
                / (
                    "question11_metrics_"
                    "rtest_0p10.csv"
                ),
                index=False,
            )

            exactTenPercentArtifacts = {
                "nNormal": int(
                    np.sum(
                        testSet.labels == 0
                    )
                ),
                "nAnomaly": int(
                    np.sum(
                        testSet.labels == 1
                    )
                ),
                "rTest": float(
                    np.mean(
                        testSet.labels
                    )
                ),
                "primaryBaseline": (
                    primaryBaselineName
                ),
                "normalExampleIndex": (
                    normalIndex
                ),
                "normalExampleCorrect": (
                    normalWasCorrect
                ),
                "anomalyExampleIndex": (
                    anomalyIndex
                ),
                "anomalyExampleCorrect": (
                    anomalyWasCorrect
                ),
                "anomalyStart": (
                    anomalyStart
                ),
                "anomalyEnd": (
                    anomalyEnd
                ),
                "higherMeanStateIndexReportConvention": (
                    highStateIndex + 1
                ),
                "higherMeanStateFractionInsideBlock": (
                    highStateInsideFraction
                ),
                "higherMeanStateFractionOutsideBlock": (
                    highStateOutsideFraction
                ),
            }

    sensitivityMetricTable = (
        pd.DataFrame(
            sensitivityMetricRows
        )
    )

    sensitivityMetricTable.to_csv(
        tableDirectory
        / "question11_sensitivity_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        sensitivityScoreRows
    ).to_csv(
        tableDirectory
        / "question11_sensitivity_scores.csv",
        index=False,
    )

    fractionArray = np.asarray(
        GAUSSIAN_SENSITIVITY_FRACTIONS,
        dtype=float,
    )

    saveAucSensitivityPlot(
        fractionArray,
        {
            modelName: np.asarray(
                aucValues,
                dtype=float,
            )
            for (
                modelName,
                aucValues,
            ) in aucLookup.items()
        },
        figureDirectory
        / (
            "question11_auc_vs_"
            "contamination.pdf"
        ),
    )

    saveAucSensitivityPlot(
        fractionArray,
        {
            modelName: np.asarray(
                aucValues,
                dtype=float,
            )
            for (
                modelName,
                aucValues,
            ) in (
                allModelAucLookup.items()
            )
        },
        figureDirectory
        / (
            "question11_auc_vs_"
            "contamination_all_models.pdf"
        ),
    )

    if exactTenPercentArtifacts is None:
        raise AssertionError(
            "The exact r_test = 0.10 "
            "experiment was not produced"
        )

    if aggregateAlignmentSummary is None:
        raise AssertionError(
            "Aggregate Viterbi alignment was not computed"
        )

    modelAtTenPercent = (
        sensitivityMetricTable[
            np.isclose(
                sensitivityMetricTable[
                    "targetRTest"
                ].to_numpy(dtype=float),
                0.10,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]
        .sort_values(
            "rocAuc",
            ascending=False,
        )
        .to_dict(
            orient="records"
        )
    )

    summary = {
        "experiment": (
            "Simulation Question 11"
        ),
        "reusedQuestion10Model": True,
        "reusedCommonTrainingPool": True,
        "masterPool": (
            masterPoolMetadata
        ),
        "requiredFractions": (
            GAUSSIAN_SENSITIVITY_FRACTIONS
        ),
        "exactFractionsAchieved": True,
        "primaryGaussianBaseline": {
            "representation": (
                GAUSSIAN_PRIMARY_REPRESENTATION
            ),
            "selectionRationale": (
                "Pre-specified before evaluation; "
                "global summary is translation-"
                "invariant to the random anomaly-"
                "block position."
            ),
            "nu": (
                GAUSSIAN_PRIMARY_NU
            ),
            "gamma": "scale",
        },
        "exactTenPercent": (
            exactTenPercentArtifacts
        ),
        "metricsAtTenPercent": (
            modelAtTenPercent
        ),
        "scoreConvention": {
            "largerNormalityScoreMeans": (
                "more normal"
            ),
            "anomalyScore": (
                "-normalityScore"
            ),
            "hardAnomalyPrediction": (
                "normalityScore < 0"
            ),
        },
        "viterbiAlignmentInterpretation": {
            "higherMeanEmissionState": (
                int(
                    np.argmax(
                        hmadModel.emissionMeans_
                    )
                )
                + 1
            ),
            "insideBlockFraction": (
                exactTenPercentArtifacts[
                    (
                        "higherMeanState"
                        "FractionInsideBlock"
                    )
                ]
            ),
            "outsideBlockFraction": (
                exactTenPercentArtifacts[
                    (
                        "higherMeanState"
                        "FractionOutsideBlock"
                    )
                ]
            ),
        },
        "aggregateViterbiAlignmentAtTenPercent": (
            aggregateAlignmentSummary
        ),
        "seedPolicyConfirmedByTA": True,
        "seedPolicy": (
            "The clean training pool uses seed 1405. "
            "Independent deterministic derived seeds are used "
            "for Q9, held-out test pools, and subset shuffling."
        ),
        "currentSeeds": {
            "training": BASE_SEED,
            "masterSensitivityPool": (
                SENSITIVITY_POOL_SEED
            ),
            "subsetShuffleSeeds": [
                1500
                + fractionIndex
                for fractionIndex in range(
                    len(
                        GAUSSIAN_SENSITIVITY_FRACTIONS
                    )
                )
            ],
        },
    }

    (
        gaussianOutputPath
        / "question11_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Question 11 metrics at "
        "exact r_test = 0.10"
    )

    print(
        pd.DataFrame(
            modelAtTenPercent
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "Viterbi alignment:"
    )

    print(
        json.dumps(
            summary[
                "viterbiAlignmentInterpretation"
            ],
            indent=2,
        )
    )

    print()
    print(
        "Aggregate Viterbi alignment "
        "across all anomalies:"
    )
    print(
        json.dumps(
            summary[
                "aggregateViterbiAlignmentAtTenPercent"
            ],
            indent=2,
        )
    )

    print()

    print(
        "Wrote Question 11 outputs to:",
        gaussianOutputPath,
    )

    return summary


GAUSSIAN_BLOCK_LENGTH_SWEEP = [
    5,
    10,
    20,
    30,
    40,
]

GAUSSIAN_BLOCK_COUNT_SWEEP = [
    1,
    2,
    3,
]

GAUSSIAN_SUPPLEMENTARY_R_TEST = 0.10
GAUSSIAN_SUPPLEMENTARY_N_NORMAL = 900
GAUSSIAN_SUPPLEMENTARY_N_ANOMALY = 100
GAUSSIAN_BLOCK_COUNT_SWEEP_LENGTH = 10
GAUSSIAN_BLOCK_COUNT_MINIMUM_GAP = 2


def _appendSupplementaryScoreRows(
    rows: list[dict[str, object]],
    *,
    labels: np.ndarray,
    normalityScores: np.ndarray,
    modelName: str,
    representation: str,
    sweepType: str,
    parameterName: str,
    parameterValue: int,
    anomalyBlockLength: int,
    nBlocks: int,
    seed: int,
) -> None:
    """Store continuous scores and zero-threshold predictions."""

    labelArray = np.asarray(
        labels,
        dtype=int,
    ).reshape(-1)

    scoreArray = np.asarray(
        normalityScores,
        dtype=float,
    ).reshape(-1)

    if labelArray.shape != scoreArray.shape:
        raise ValueError(
            "Supplementary labels and scores "
            "must have equal shapes"
        )

    if not np.all(
        np.isfinite(scoreArray)
    ):
        raise ValueError(
            f"{modelName} supplementary scores "
            "contain non-finite values"
        )

    hardPredictions = (
        scoreArray < 0.0
    ).astype(int)

    achievedRTest = float(
        np.mean(labelArray)
    )

    for sampleIndex in range(
        labelArray.size
    ):
        rows.append(
            {
                "sweepType": sweepType,
                "parameterName": parameterName,
                "parameterValue": int(
                    parameterValue
                ),
                "anomalyBlockLength": int(
                    anomalyBlockLength
                ),
                "nBlocks": int(nBlocks),
                "totalAnomalousSteps": int(
                    anomalyBlockLength
                    * nBlocks
                ),
                "targetRTest": (
                    GAUSSIAN_SUPPLEMENTARY_R_TEST
                ),
                "achievedRTest": (
                    achievedRTest
                ),
                "sampleIndex": int(
                    sampleIndex
                ),
                "trueLabel": int(
                    labelArray[
                        sampleIndex
                    ]
                ),
                "model": modelName,
                "representation": (
                    representation
                ),
                "nu": (
                    GAUSSIAN_PRIMARY_NU
                ),
                "normalityScore": float(
                    scoreArray[
                        sampleIndex
                    ]
                ),
                "anomalyScore": float(
                    -scoreArray[
                        sampleIndex
                    ]
                ),
                "hardPrediction": int(
                    hardPredictions[
                        sampleIndex
                    ]
                ),
                "seed": int(seed),
            }
        )


def _saveSupplementaryMetricPlot(
    metricTable: pd.DataFrame,
    *,
    sweepType: str,
    xColumn: str,
    xLabel: str,
    metricColumn: str,
    yLabel: str,
    title: str,
    outputBasePath: Path,
) -> None:
    """Plot one sequence-level metric for every evaluated model."""

    sweepTable = metricTable[
        metricTable["sweepType"]
        == sweepType
    ].copy()

    if sweepTable.empty:
        raise ValueError(
            f"No rows are available for sweep {sweepType}"
        )

    figure, axis = plt.subplots(
        figsize=(8.8, 5.4)
    )

    for modelName in (
        sweepTable["model"]
        .drop_duplicates()
        .tolist()
    ):
        modelTable = sweepTable[
            sweepTable["model"]
            == modelName
        ].sort_values(xColumn)

        axis.plot(
            modelTable[xColumn].to_numpy(
                dtype=float
            ),
            modelTable[
                metricColumn
            ].to_numpy(dtype=float),
            marker="o",
            linewidth=1.4,
            label=modelName,
        )

    axis.set_xlabel(xLabel)
    axis.set_ylabel(yLabel)
    axis.set_title(title)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(
        loc="best",
        fontsize=8,
    )

    figure.tight_layout()

    saveFigureBothFormats(
        figure,
        outputBasePath,
    )


def _saveSupplementaryLocalizationPlot(
    alignmentTable: pd.DataFrame,
    *,
    sweepType: str,
    xColumn: str,
    xLabel: str,
    title: str,
    outputBasePath: Path,
) -> None:
    """Plot aggregate time-step localization metrics for Viterbi."""

    sweepTable = alignmentTable[
        alignmentTable["sweepType"]
        == sweepType
    ].sort_values(xColumn)

    if sweepTable.empty:
        raise ValueError(
            f"No alignment rows are available for {sweepType}"
        )

    figure, axis = plt.subplots(
        figsize=(8.2, 5.1)
    )

    metricLookup = {
        "Precision": (
            "microLocalizationPrecision"
        ),
        "Recall": (
            "microLocalizationRecall"
        ),
        "F1": (
            "microLocalizationF1"
        ),
        "Exact path match": (
            "exactPathBlockMatchFraction"
        ),
    }

    for label, columnName in (
        metricLookup.items()
    ):
        axis.plot(
            sweepTable[xColumn].to_numpy(
                dtype=float
            ),
            sweepTable[columnName].to_numpy(
                dtype=float
            ),
            marker="o",
            linewidth=1.4,
            label=label,
        )

    axis.set_xlabel(xLabel)
    axis.set_ylabel("Localization metric")
    axis.set_title(title)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    figure.tight_layout()

    saveFigureBothFormats(
        figure,
        outputBasePath,
    )


def _evaluateSupplementaryCondition(
    *,
    testSet: object,
    hmadModel: SimplifiedHMAD,
    baselineModels: dict[
        str,
        OcsvmBaseline,
    ],
    sweepType: str,
    parameterName: str,
    parameterValue: int,
    anomalyBlockLength: int,
    nBlocks: int,
    seed: int,
    metricRows: list[dict[str, object]],
    scoreRows: list[dict[str, object]],
    alignmentDetailTables: list[pd.DataFrame],
    alignmentSummaryRows: list[
        dict[str, object]
    ],
) -> None:
    """Evaluate all models and Viterbi localization for one condition."""

    _validateQuestion11TestSet(
        testSet.sequences,
        testSet.labels,
        expectedFraction=(
            GAUSSIAN_SUPPLEMENTARY_R_TEST
        ),
    )

    decodedStates = (
        hmadModel.decodeSequences(
            testSet.sequences
        )
    )

    if decodedStates.shape != (
        1000,
        100,
    ):
        raise AssertionError(
            "Supplementary decoded-state shape is "
            f"{decodedStates.shape}; expected (1000, 100)"
        )

    hmadScores = (
        hmadModel.decisionFunction(
            testSet.sequences
        )
    )

    trivialScores = (
        hmadModel
        .decisionFunctionWithTrivialStates(
            testSet.sequences
        )
    )

    baselineScoreLookup = (
        _scoreGaussianOcsvmBaselines(
            baselineModels,
            testSet.sequences,
        )
    )

    modelScoreLookup: dict[
        str,
        tuple[np.ndarray, str],
    ] = {
        "HMAD": (
            hmadScores,
            "joint_features_viterbi",
        ),
        "HMAD all-one ablation": (
            trivialScores,
            "joint_features_all_one",
        ),
    }

    for representation, scores in (
        baselineScoreLookup.items()
    ):
        modelScoreLookup[
            f"OC-SVM {representation}"
        ] = (
            scores,
            representation,
        )

    for modelName, (
        normalityScores,
        representation,
    ) in modelScoreLookup.items():
        metrics = evaluateNormalityScores(
            testSet.labels,
            normalityScores,
        )

        metricRows.append(
            {
                "sweepType": sweepType,
                "parameterName": parameterName,
                "parameterValue": int(
                    parameterValue
                ),
                "anomalyBlockLength": int(
                    anomalyBlockLength
                ),
                "nBlocks": int(nBlocks),
                "totalAnomalousSteps": int(
                    anomalyBlockLength
                    * nBlocks
                ),
                "model": modelName,
                "representation": (
                    representation
                ),
                "evaluation": (
                    "gaussian_supplementary"
                ),
                "targetRTest": (
                    GAUSSIAN_SUPPLEMENTARY_R_TEST
                ),
                "achievedRTest": float(
                    np.mean(testSet.labels)
                ),
                "nNormal": int(
                    np.sum(
                        testSet.labels == 0
                    )
                ),
                "nAnomaly": int(
                    np.sum(
                        testSet.labels == 1
                    )
                ),
                "nu": (
                    GAUSSIAN_PRIMARY_NU
                ),
                "seed": int(seed),
                **metrics,
            }
        )

        _appendSupplementaryScoreRows(
            scoreRows,
            labels=testSet.labels,
            normalityScores=(
                normalityScores
            ),
            modelName=modelName,
            representation=(
                representation
            ),
            sweepType=sweepType,
            parameterName=parameterName,
            parameterValue=parameterValue,
            anomalyBlockLength=(
                anomalyBlockLength
            ),
            nBlocks=nBlocks,
            seed=seed,
        )

    highStateIndex = int(
        np.argmax(
            hmadModel.emissionMeans_
        )
    )

    (
        alignmentDetails,
        alignmentSummary,
    ) = evaluateAggregateViterbiAlignment(
        decodedStates,
        testSet.labels,
        testSet.anomalyStarts,
        testSet.anomalyEnds,
        highStateIndex=highStateIndex,
    )

    alignmentDetails.insert(
        0,
        "parameterValue",
        int(parameterValue),
    )
    alignmentDetails.insert(
        0,
        "parameterName",
        parameterName,
    )
    alignmentDetails.insert(
        0,
        "sweepType",
        sweepType,
    )
    alignmentDetails["anomalyBlockLength"] = int(
        anomalyBlockLength
    )
    alignmentDetails["nBlocks"] = int(
        nBlocks
    )
    alignmentDetails["totalAnomalousSteps"] = int(
        anomalyBlockLength
        * nBlocks
    )
    alignmentDetails["seed"] = int(seed)

    alignmentDetailTables.append(
        alignmentDetails
    )

    alignmentSummaryRows.append(
        {
            "sweepType": sweepType,
            "parameterName": parameterName,
            "parameterValue": int(
                parameterValue
            ),
            "anomalyBlockLength": int(
                anomalyBlockLength
            ),
            "nBlocks": int(nBlocks),
            "totalAnomalousSteps": int(
                anomalyBlockLength
                * nBlocks
            ),
            "seed": int(seed),
            **alignmentSummary,
        }
    )


def runSupplementaryGaussianSweeps(
    outputDirectory: str | Path,
    hmadModel: SimplifiedHMAD,
) -> dict[str, object]:
    """Run required variations of block length and block count.

    Question 9--11 remain fixed at one 20-step block. These sweeps are
    supplementary robustness experiments required by the dataset section.
    """

    outputPath = Path(outputDirectory)

    gaussianOutputPath = (
        outputPath
        / "gaussian_required"
    )

    dataDirectory = (
        gaussianOutputPath
        / "data"
        / "supplementary_sweeps"
    )

    tableDirectory = (
        gaussianOutputPath
        / "tables"
    )

    figureDirectory = (
        gaussianOutputPath
        / "figures"
    )

    for directory in [
        dataDirectory,
        tableDirectory,
        figureDirectory,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    trainSequences = (
        loadSavedGaussianTrainingSet(
            outputDirectory
        )
    )

    baselineModels = (
        _fitGaussianOcsvmBaselines(
            trainSequences
        )
    )

    metricRows: list[
        dict[str, object]
    ] = []
    scoreRows: list[
        dict[str, object]
    ] = []
    alignmentDetailTables: list[
        pd.DataFrame
    ] = []
    alignmentSummaryRows: list[
        dict[str, object]
    ] = []

    # The same deterministic seed and unshuffled ordering are used across
    # conditions so the nominal sequences and pre-insertion anomaly
    # baselines remain directly comparable.
    for anomalyBlockLength in (
        GAUSSIAN_BLOCK_LENGTH_SWEEP
    ):
        testSet = makeGaussianTestSet(
            nNormal=(
                GAUSSIAN_SUPPLEMENTARY_N_NORMAL
            ),
            nAnomaly=(
                GAUSSIAN_SUPPLEMENTARY_N_ANOMALY
            ),
            sequenceLength=100,
            anomalyBlockLength=(
                anomalyBlockLength
            ),
            anomalyMean=4.0,
            anomalyStd=1.0,
            nBlocks=1,
            minimumGap=0,
            seed=SUPPLEMENTARY_SEED,
            shuffle=False,
        )

        saveGaussianTestSet(
            testSet,
            dataDirectory,
            prefix=(
                "block_length_"
                f"{anomalyBlockLength:03d}"
            ),
        )

        _evaluateSupplementaryCondition(
            testSet=testSet,
            hmadModel=hmadModel,
            baselineModels=baselineModels,
            sweepType="block_length",
            parameterName=(
                "anomalyBlockLength"
            ),
            parameterValue=(
                anomalyBlockLength
            ),
            anomalyBlockLength=(
                anomalyBlockLength
            ),
            nBlocks=1,
            seed=SUPPLEMENTARY_SEED,
            metricRows=metricRows,
            scoreRows=scoreRows,
            alignmentDetailTables=(
                alignmentDetailTables
            ),
            alignmentSummaryRows=(
                alignmentSummaryRows
            ),
        )

    for nBlocks in (
        GAUSSIAN_BLOCK_COUNT_SWEEP
    ):
        testSet = makeGaussianTestSet(
            nNormal=(
                GAUSSIAN_SUPPLEMENTARY_N_NORMAL
            ),
            nAnomaly=(
                GAUSSIAN_SUPPLEMENTARY_N_ANOMALY
            ),
            sequenceLength=100,
            anomalyBlockLength=(
                GAUSSIAN_BLOCK_COUNT_SWEEP_LENGTH
            ),
            anomalyMean=4.0,
            anomalyStd=1.0,
            nBlocks=nBlocks,
            minimumGap=(
                GAUSSIAN_BLOCK_COUNT_MINIMUM_GAP
            ),
            seed=SUPPLEMENTARY_SEED,
            shuffle=False,
        )

        saveGaussianTestSet(
            testSet,
            dataDirectory,
            prefix=(
                "block_count_"
                f"{nBlocks:02d}"
            ),
        )

        _evaluateSupplementaryCondition(
            testSet=testSet,
            hmadModel=hmadModel,
            baselineModels=baselineModels,
            sweepType="block_count",
            parameterName="nBlocks",
            parameterValue=nBlocks,
            anomalyBlockLength=(
                GAUSSIAN_BLOCK_COUNT_SWEEP_LENGTH
            ),
            nBlocks=nBlocks,
            seed=SUPPLEMENTARY_SEED,
            metricRows=metricRows,
            scoreRows=scoreRows,
            alignmentDetailTables=(
                alignmentDetailTables
            ),
            alignmentSummaryRows=(
                alignmentSummaryRows
            ),
        )

    metricTable = pd.DataFrame(
        metricRows
    )
    scoreTable = pd.DataFrame(
        scoreRows
    )
    alignmentSummaryTable = pd.DataFrame(
        alignmentSummaryRows
    )
    alignmentDetailTable = pd.concat(
        alignmentDetailTables,
        ignore_index=True,
    )

    metricTable.to_csv(
        tableDirectory
        / "supplementary_sweep_metrics.csv",
        index=False,
    )
    scoreTable.to_csv(
        tableDirectory
        / "supplementary_sweep_scores.csv",
        index=False,
    )
    alignmentSummaryTable.to_csv(
        tableDirectory
        / (
            "supplementary_viterbi_"
            "alignment_summary.csv"
        ),
        index=False,
    )
    alignmentDetailTable.to_csv(
        tableDirectory
        / (
            "supplementary_viterbi_"
            "alignment_details.csv"
        ),
        index=False,
    )

    for metricColumn, yLabel in [
        ("rocAuc", "ROC-AUC"),
        ("f1Score", "F1-score at zero threshold"),
    ]:
        metricTag = (
            "auc"
            if metricColumn == "rocAuc"
            else "f1"
        )

        _saveSupplementaryMetricPlot(
            metricTable,
            sweepType="block_length",
            xColumn="anomalyBlockLength",
            xLabel="Anomaly block length",
            metricColumn=metricColumn,
            yLabel=yLabel,
            title=(
                "Gaussian robustness versus "
                "anomaly block length"
            ),
            outputBasePath=(
                figureDirectory
                / (
                    "supplementary_block_"
                    f"length_{metricTag}"
                )
            ),
        )

        _saveSupplementaryMetricPlot(
            metricTable,
            sweepType="block_count",
            xColumn="nBlocks",
            xLabel=(
                "Number of anomaly blocks"
            ),
            metricColumn=metricColumn,
            yLabel=yLabel,
            title=(
                "Gaussian robustness versus "
                "number of anomaly blocks"
            ),
            outputBasePath=(
                figureDirectory
                / (
                    "supplementary_block_"
                    f"count_{metricTag}"
                )
            ),
        )

    _saveSupplementaryLocalizationPlot(
        alignmentSummaryTable,
        sweepType="block_length",
        xColumn="anomalyBlockLength",
        xLabel="Anomaly block length",
        title=(
            "Viterbi localization versus "
            "anomaly block length"
        ),
        outputBasePath=(
            figureDirectory
            / (
                "supplementary_block_length_"
                "viterbi_localization"
            )
        ),
    )

    _saveSupplementaryLocalizationPlot(
        alignmentSummaryTable,
        sweepType="block_count",
        xColumn="nBlocks",
        xLabel="Number of anomaly blocks",
        title=(
            "Viterbi localization versus "
            "number of anomaly blocks"
        ),
        outputBasePath=(
            figureDirectory
            / (
                "supplementary_block_count_"
                "viterbi_localization"
            )
        ),
    )

    summary = {
        "experiment": (
            "Supplementary Gaussian robustness sweeps"
        ),
        "status": "supplementary",
        "questions9To11RemainUnchanged": True,
        "fixedSettings": {
            "nTotal": 1000,
            "nNormal": (
                GAUSSIAN_SUPPLEMENTARY_N_NORMAL
            ),
            "nAnomaly": (
                GAUSSIAN_SUPPLEMENTARY_N_ANOMALY
            ),
            "rTest": (
                GAUSSIAN_SUPPLEMENTARY_R_TEST
            ),
            "nominalDistribution": {
                "mean": 0.0,
                "standardDeviation": 1.0,
            },
            "anomalyDistribution": {
                "mean": 4.0,
                "standardDeviation": 1.0,
            },
            "seed": SUPPLEMENTARY_SEED,
            "shuffle": False,
            "commonNominalAndAnomalyBaselinesAcrossConditions": True,
            "seedPolicyConfirmedByTA": True,
        },
        "blockLengthSweep": {
            "values": (
                GAUSSIAN_BLOCK_LENGTH_SWEEP
            ),
            "nBlocks": 1,
            "metrics": (
                metricTable[
                    metricTable["sweepType"]
                    == "block_length"
                ].to_dict(orient="records")
            ),
            "viterbiAlignment": (
                alignmentSummaryTable[
                    alignmentSummaryTable[
                        "sweepType"
                    ]
                    == "block_length"
                ].to_dict(orient="records")
            ),
        },
        "blockCountSweep": {
            "values": (
                GAUSSIAN_BLOCK_COUNT_SWEEP
            ),
            "anomalyBlockLength": (
                GAUSSIAN_BLOCK_COUNT_SWEEP_LENGTH
            ),
            "minimumGap": (
                GAUSSIAN_BLOCK_COUNT_MINIMUM_GAP
            ),
            "metrics": (
                metricTable[
                    metricTable["sweepType"]
                    == "block_count"
                ].to_dict(orient="records")
            ),
            "viterbiAlignment": (
                alignmentSummaryTable[
                    alignmentSummaryTable[
                        "sweepType"
                    ]
                    == "block_count"
                ].to_dict(orient="records")
            ),
        },
        "scoreConvention": {
            "largerNormalityScoreMeans": (
                "more normal"
            ),
            "anomalyScore": (
                "-normalityScore"
            ),
            "hardAnomalyPrediction": (
                "normalityScore < 0"
            ),
        },
    }

    (
        gaussianOutputPath
        / "supplementary_sweeps_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Supplementary block-length sweep metrics"
    )
    print(
        metricTable[
            metricTable["sweepType"]
            == "block_length"
        ].to_string(index=False)
    )

    print()
    print(
        "Supplementary block-count sweep metrics"
    )
    print(
        metricTable[
            metricTable["sweepType"]
            == "block_count"
        ].to_string(index=False)
    )

    print()
    print(
        "Wrote supplementary sweep outputs to:",
        gaussianOutputPath,
    )

    return summary


def runGaussianRequiredExperiment(
    outputDirectory: str | Path,
) -> dict[str, object]:
    """Run all required and supplementary Gaussian experiments."""

    question9Summary = runQuestion9(
        outputDirectory
    )

    question10Summary, hmadModel = runQuestion10(
        outputDirectory
    )

    question11Summary = runQuestion11(
        outputDirectory,
        hmadModel,
    )

    supplementarySummary = (
        runSupplementaryGaussianSweeps(
            outputDirectory,
            hmadModel,
        )
    )

    combinedSummary = {
        "question9": question9Summary,
        "question10": question10Summary,
        "question11": question11Summary,
        "supplementarySweeps": (
            supplementarySummary
        ),
    }

    gaussianOutputPath = (
        Path(outputDirectory)
        / "gaussian_required"
    )
    gaussianOutputPath.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        gaussianOutputPath
        / "summary.json"
    ).write_text(
        json.dumps(
            combinedSummary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return combinedSummary

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=str(
            projectRoot
            / "outputs"
        ),
    )

    arguments = parser.parse_args()

    summary = runGaussianRequiredExperiment(
        arguments.output
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

if __name__ == "__main__":
    main()