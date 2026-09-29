"""Run the reproducible OC-SVM baseline on Wafer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

projectRoot = Path(__file__).resolve().parents[1]

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from src.baseline_features import buildFeatureRepresentation
from src.evaluation import evaluateNormalityScores
from src.ocsvm_baseline import fitOcsvmBaseline
from src.plotting import saveFigure
from src.wafer_dataset import WaferDataset, loadWaferDataset


def loadConfig(configPath: str | Path) -> dict:
    with Path(configPath).open(
        "r",
        encoding="utf-8",
    ) as configFile:
        return yaml.safe_load(configFile)


def resolveProjectPath(
    pathValue: object,
) -> Path | None:
    """Resolve a configuration path relative to the project root."""

    if pathValue is None:
        return None

    pathText = str(pathValue).strip()

    if not pathText:
        return None

    path = Path(pathText)

    if not path.is_absolute():
        path = projectRoot / path

    return path.resolve()


def loadDataset(
    dataConfig: dict,
) -> WaferDataset:
    datasetDirectory = resolveProjectPath(
        dataConfig.get(
            "datasetDirectory",
            "data/Wafer",
        )
    )

    if datasetDirectory is None:
        raise ValueError(
            "data.datasetDirectory must be configured"
        )

    return loadWaferDataset(
        datasetDirectory,
        trainPath=resolveProjectPath(
            dataConfig.get("trainPath")
        ),
        testPath=resolveProjectPath(
            dataConfig.get("testPath")
        ),
        normalSourceLabel=dataConfig.get(
            "normalSourceLabel"
        ),
        anomalySourceLabel=dataConfig.get(
            "anomalySourceLabel"
        ),
    )


def saveAucVsNuPlot(
    metricTable: pd.DataFrame,
    outputPath: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(7.2, 4.6)
    )

    for representation, group in metricTable.groupby(
        "representation",
        sort=False,
    ):
        orderedGroup = group.sort_values("nu")

        axis.plot(
            orderedGroup["nu"],
            orderedGroup["rocAuc"],
            marker="o",
            linewidth=1.6,
            label=representation,
        )

    axis.set_xlabel(r"$\nu$")
    axis.set_ylabel("ROC-AUC")
    axis.set_title(
        "OC-SVM ROC-AUC sensitivity to nu"
    )
    axis.set_xticks(
        sorted(metricTable["nu"].unique())
    )
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    saveFigure(
        figure,
        outputPath,
    )


def saveRepresentationComparisonPlot(
    metricTable: pd.DataFrame,
    primaryNu: float,
    outputPath: Path,
) -> None:
    selectedRows = metricTable[
        np.isclose(
            metricTable["nu"].to_numpy(dtype=float),
            primaryNu,
        )
    ].copy()

    if selectedRows.empty:
        raise ValueError(
            f"No rows were found for primary nu={primaryNu}"
        )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.6)
    )

    axis.bar(
        selectedRows["representation"],
        selectedRows["rocAuc"],
    )

    axis.set_xlabel("Feature representation")
    axis.set_ylabel("ROC-AUC")
    axis.set_title(
        f"OC-SVM representations at nu={primaryNu:.2f}"
    )
    axis.set_ylim(0.0, 1.05)
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for index, value in enumerate(
        selectedRows["rocAuc"]
    ):
        axis.text(
            index,
            float(value) + 0.02,
            f"{float(value):.4f}",
            ha="center",
        )

    figure.tight_layout()

    saveFigure(
        figure,
        outputPath,
    )


def savePrimaryScoreDistributionPlot(
    scoreTable: pd.DataFrame,
    primaryRepresentation: str,
    primaryNu: float,
    outputPath: Path,
) -> None:
    primaryRows = scoreTable[
        (scoreTable["representation"] == primaryRepresentation)
        & np.isclose(
            scoreTable["nu"].to_numpy(dtype=float),
            primaryNu,
        )
    ]

    if primaryRows.empty:
        raise ValueError(
            "Primary OC-SVM score rows were not found"
        )

    normalScores = primaryRows.loc[
        primaryRows["trueLabel"] == 0,
        "anomalyScore",
    ].to_numpy(dtype=float)

    anomalyScores = primaryRows.loc[
        primaryRows["trueLabel"] == 1,
        "anomalyScore",
    ].to_numpy(dtype=float)

    figure, axis = plt.subplots(
        figsize=(7.2, 4.6)
    )

    axis.hist(
        normalScores,
        bins=45,
        density=True,
        alpha=0.55,
        label="Normal",
    )

    axis.hist(
        anomalyScores,
        bins=45,
        density=True,
        alpha=0.55,
        label="Anomaly",
    )

    axis.axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Zero decision boundary",
    )

    axis.set_xlabel("Anomaly score")
    axis.set_ylabel("Density")
    axis.set_title(
        "Primary OC-SVM anomaly-score distribution"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    saveFigure(
        figure,
        outputPath,
    )


def runOcsvmExperiment(
    configPath: str | Path,
    outputDirectory: str | Path,
) -> dict:
    config = loadConfig(configPath)

    seed = int(config["seed"])
    dataConfig = config["data"]
    ocsvmConfig = config["ocsvm"]

    outputPath = Path(outputDirectory)
    tableDirectory = outputPath / "tables"
    figureDirectory = outputPath / "figures"

    tableDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )
    figureDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = loadDataset(dataConfig)

    representations = [
        str(value)
        for value in ocsvmConfig["representations"]
    ]

    nuValues = [
        float(value)
        for value in ocsvmConfig["nuValues"]
    ]

    primaryRepresentation = str(
        ocsvmConfig["primaryRepresentation"]
    )

    primaryNu = float(
        ocsvmConfig["primaryNu"]
    )

    gamma = ocsvmConfig.get(
        "gamma",
        "scale",
    )

    kernel = str(
        ocsvmConfig.get(
            "kernel",
            "rbf",
        )
    ).lower()

    nWindows = int(
        ocsvmConfig.get(
            "windowCount",
            5,
        )
    )

    if kernel != "rbf":
        raise ValueError(
            "The current OC-SVM baseline implementation "
            "supports the RBF kernel only"
        )

    if primaryRepresentation not in representations:
        raise ValueError(
            "primaryRepresentation must be included "
            "in the representation grid"
        )

    if not any(
        np.isclose(primaryNu, nu)
        for nu in nuValues
    ):
        raise ValueError(
            "primaryNu must be included in the nu grid"
        )

    metricRows: list[dict] = []
    scoreFrames: list[pd.DataFrame] = []
    featureDimensions: dict[str, int] = {}

    for representation in representations:
        trainFeatures = buildFeatureRepresentation(
            dataset.trainSequences,
            representation,
            nWindows=nWindows,
        )

        testFeatures = buildFeatureRepresentation(
            dataset.testSequences,
            representation,
            nWindows=nWindows,
        )

        featureDimensions[representation] = int(
            trainFeatures.shape[1]
        )

        if trainFeatures.shape[1] != testFeatures.shape[1]:
            raise ValueError(
                f"Train/test feature dimensions differ "
                f"for {representation}"
            )

        for nu in nuValues:
            baseline = fitOcsvmBaseline(
                trainFeatures,
                representation=representation,
                nu=nu,
                gamma=gamma,
                nWindows=nWindows,
            )

            normalityScores = baseline.decisionFunction(
                testFeatures
            )

            metrics = evaluateNormalityScores(
                dataset.testLabels,
                normalityScores,
            )

            isPrimary = (
                representation == primaryRepresentation
                and np.isclose(nu, primaryNu)
            )

            metricRows.append(
                {
                    "model": "OC-SVM",
                    "representation": representation,
                    "featureDimension": int(
                        trainFeatures.shape[1]
                    ),
                    "kernel": kernel,
                    "nu": float(nu),
                    "gamma": gamma,
                    "evaluation": "official_test",
                    "seed": seed,
                    "nTrainNormal": int(
                        dataset.trainSequences.shape[0]
                    ),
                    "nNormal": int(
                        np.sum(dataset.testLabels == 0)
                    ),
                    "nAnomaly": int(
                        np.sum(dataset.testLabels == 1)
                    ),
                    "rTest": float(
                        np.mean(dataset.testLabels)
                    ),
                    "isPrimary": bool(isPrimary),
                    **metrics,
                }
            )

            scoreFrames.append(
                pd.DataFrame(
                    {
                        "sampleIndex": np.arange(
                            dataset.testLabels.size,
                            dtype=int,
                        ),
                        "trueLabel": dataset.testLabels.astype(
                            int
                        ),
                        "normalityScore": normalityScores,
                        "anomalyScore": -normalityScores,
                        "hardPrediction": (
                            normalityScores < 0.0
                        ).astype(int),
                        "model": "OC-SVM",
                        "representation": representation,
                        "featureDimension": int(
                            trainFeatures.shape[1]
                        ),
                        "kernel": kernel,
                        "nu": float(nu),
                        "gamma": gamma,
                        "isPrimary": bool(isPrimary),
                        "evaluation": "official_test",
                        "seed": seed,
                    }
                )
            )

    metricTable = pd.DataFrame(metricRows)
    scoreTable = pd.concat(
        scoreFrames,
        ignore_index=True,
    )

    expectedMetricRows = (
        len(representations) * len(nuValues)
    )

    expectedScoreRows = (
        expectedMetricRows
        * dataset.testLabels.size
    )

    if metricTable.shape[0] != expectedMetricRows:
        raise AssertionError(
            "Unexpected number of metric rows"
        )

    if scoreTable.shape[0] != expectedScoreRows:
        raise AssertionError(
            "Unexpected number of score rows"
        )

    if not metricTable["rocAuc"].between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "ROC-AUC values must lie in [0, 1]"
        )

    metricOutputPath = (
        tableDirectory / "ocsvm_metrics.csv"
    )

    scoreOutputPath = (
        tableDirectory / "ocsvm_results.csv"
    )

    metricTable.to_csv(
        metricOutputPath,
        index=False,
    )

    scoreTable.to_csv(
        scoreOutputPath,
        index=False,
    )

    saveAucVsNuPlot(
        metricTable,
        figureDirectory / "ocsvm_auc_vs_nu.pdf",
    )

    saveRepresentationComparisonPlot(
        metricTable,
        primaryNu,
        figureDirectory
        / "ocsvm_representation_comparison.pdf",
    )

    savePrimaryScoreDistributionPlot(
        scoreTable,
        primaryRepresentation,
        primaryNu,
        figureDirectory
        / "ocsvm_score_distribution.pdf",
    )

    primaryRow = metricTable.loc[
        metricTable["isPrimary"]
    ]

    if primaryRow.shape[0] != 1:
        raise AssertionError(
            "Exactly one primary configuration is required"
        )

    primaryRecord = primaryRow.iloc[0]

    bestAucRecord = metricTable.loc[
        metricTable["rocAuc"].idxmax()
    ]

    summary = {
        "dataset": "Wafer",
        "seed": seed,
        "nTrainNormal": int(
            dataset.trainSequences.shape[0]
        ),
        "nTest": int(
            dataset.testSequences.shape[0]
        ),
        "nTestNormal": int(
            np.sum(dataset.testLabels == 0)
        ),
        "nTestAnomaly": int(
            np.sum(dataset.testLabels == 1)
        ),
        "sequenceLength": int(
            dataset.testSequences.shape[1]
        ),
        "featureDimensions": featureDimensions,
        "primaryConfiguration": {
            "representation": primaryRepresentation,
            "nu": primaryNu,
            "kernel": kernel,
            "gamma": gamma,
        },
        "primaryMetrics": {
            "rocAuc": float(
                primaryRecord["rocAuc"]
            ),
            "precision": float(
                primaryRecord["precision"]
            ),
            "recall": float(
                primaryRecord["recall"]
            ),
            "f1Score": float(
                primaryRecord["f1Score"]
            ),
        },
        "highestObservedTestAuc": {
            "representation": str(
                bestAucRecord["representation"]
            ),
            "nu": float(
                bestAucRecord["nu"]
            ),
            "rocAuc": float(
                bestAucRecord["rocAuc"]
            ),
            "note": (
                "Reported as an observed sensitivity result; "
                "it was not used to redefine the "
                "pre-specified primary configuration."
            ),
        },
    }

    (
        tableDirectory
        / "ocsvm_run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        metricTable[
            [
                "representation",
                "featureDimension",
                "nu",
                "rocAuc",
                "precision",
                "recall",
                "f1Score",
                "isPrimary",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"Wrote {metricTable.shape[0]} metric rows to "
        f"{metricOutputPath}"
    )
    print(
        f"Wrote {scoreTable.shape[0]} score rows to "
        f"{scoreOutputPath}"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(
            projectRoot
            / "configs"
            / "default.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            projectRoot
            / "outputs"
        ),
    )

    arguments = parser.parse_args()

    runOcsvmExperiment(
        arguments.config,
        arguments.output,
    )


if __name__ == "__main__":
    main()
