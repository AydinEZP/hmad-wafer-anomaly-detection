from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


projectRoot = Path(__file__).resolve().parents[1]

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))


def requireColumns(
    table: pd.DataFrame,
    requiredColumns: set[str],
    tableName: str,
) -> None:
    missingColumns = requiredColumns.difference(table.columns)

    if missingColumns:
        missingText = ", ".join(sorted(missingColumns))
        raise ValueError(
            f"{tableName} is missing required columns: {missingText}"
        )


def auditModelScores(
    modelTable: pd.DataFrame,
    modelName: str,
    expectedSampleCount: int,
) -> dict[str, object]:
    orderedTable = modelTable.sort_values(
        "sampleIndex"
    ).reset_index(drop=True)

    if orderedTable.shape[0] != expectedSampleCount:
        raise ValueError(
            f"{modelName}: expected {expectedSampleCount} rows, "
            f"found {orderedTable.shape[0]}"
        )

    sampleIndices = orderedTable[
        "sampleIndex"
    ].to_numpy(dtype=int)

    expectedIndices = np.arange(
        expectedSampleCount,
        dtype=int,
    )

    if not np.array_equal(
        sampleIndices,
        expectedIndices,
    ):
        raise ValueError(
            f"{modelName}: sampleIndex must contain each official "
            "test index exactly once"
        )

    labels = orderedTable[
        "trueLabel"
    ].to_numpy(dtype=int)

    normalityScores = orderedTable[
        "normalityScore"
    ].to_numpy(dtype=float)

    anomalyScores = orderedTable[
        "anomalyScore"
    ].to_numpy(dtype=float)

    storedPredictions = orderedTable[
        "hardPrediction"
    ].to_numpy(dtype=int)

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(
            f"{modelName}: labels must use 0 = normal and 1 = anomaly"
        )

    if np.unique(labels).size != 2:
        raise ValueError(
            f"{modelName}: both normal and anomaly labels are required"
        )

    if not np.all(np.isfinite(normalityScores)):
        raise ValueError(
            f"{modelName}: normality scores contain non-finite values"
        )

    if not np.all(np.isfinite(anomalyScores)):
        raise ValueError(
            f"{modelName}: anomaly scores contain non-finite values"
        )

    if not np.allclose(
        anomalyScores,
        -normalityScores,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            f"{modelName}: anomalyScore is not equal to -normalityScore"
        )

    expectedPredictions = (
        normalityScores < 0.0
    ).astype(int)

    if not np.array_equal(
        storedPredictions,
        expectedPredictions,
    ):
        raise ValueError(
            f"{modelName}: stored hard predictions do not use "
            "the zero normality-score threshold"
        )

    normalMask = labels == 0
    anomalyMask = labels == 1

    aucFromAnomalyScore = float(
        roc_auc_score(
            labels,
            anomalyScores,
        )
    )

    aucFromNormalityScore = float(
        roc_auc_score(
            labels,
            normalityScores,
        )
    )

    precision = float(
        precision_score(
            labels,
            expectedPredictions,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            labels,
            expectedPredictions,
            zero_division=0,
        )
    )

    f1Score = float(
        f1_score(
            labels,
            expectedPredictions,
            zero_division=0,
        )
    )

    trueNegative, falsePositive, falseNegative, truePositive = (
        confusion_matrix(
            labels,
            expectedPredictions,
            labels=[0, 1],
        ).ravel()
    )

    normalMean = float(
        np.mean(normalityScores[normalMask])
    )

    anomalyMean = float(
        np.mean(normalityScores[anomalyMask])
    )

    normalMedian = float(
        np.median(normalityScores[normalMask])
    )

    anomalyMedian = float(
        np.median(normalityScores[anomalyMask])
    )

    return {
        "model": modelName,
        "nSamples": int(labels.size),
        "nNormal": int(np.sum(normalMask)),
        "nAnomaly": int(np.sum(anomalyMask)),
        "meanNormalityScoreNormal": normalMean,
        "meanNormalityScoreAnomaly": anomalyMean,
        "medianNormalityScoreNormal": normalMedian,
        "medianNormalityScoreAnomaly": anomalyMedian,
        "normalSamplesHaveLargerMeanNormalityScore": bool(
            normalMean > anomalyMean
        ),
        "aucUsingNegativeNormalityScore": aucFromAnomalyScore,
        "aucUsingPositiveNormalityScore": aucFromNormalityScore,
        "precisionAtZero": precision,
        "recallAtZero": recall,
        "f1AtZero": f1Score,
        "trueNegative": int(trueNegative),
        "falsePositive": int(falsePositive),
        "falseNegative": int(falseNegative),
        "truePositive": int(truePositive),
    }


def auditJointFeatures(
    jointFeaturePath: Path,
) -> dict[str, object]:
    jointFeatureTable = pd.read_csv(
        jointFeaturePath
    )

    featureColumns = [
        "c11",
        "c12",
        "c21",
        "c22",
        "meanState1",
        "meanState2",
    ]

    requireColumns(
        jointFeatureTable,
        set(featureColumns),
        "wafer_joint_features.csv",
    )

    featureMatrix = jointFeatureTable[
        featureColumns
    ].to_numpy(dtype=float)

    if featureMatrix.shape[1] != 6:
        raise ValueError(
            "The HMAD joint feature representation must have "
            "exactly six columns"
        )

    if not np.all(np.isfinite(featureMatrix)):
        raise ValueError(
            "The HMAD joint feature matrix contains non-finite values"
        )

    transitionSums = np.sum(
        featureMatrix[:, :4],
        axis=1,
    )

    maximumTransitionSumError = float(
        np.max(
            np.abs(
                transitionSums - 1.0
            )
        )
    )

    if not np.allclose(
        transitionSums,
        1.0,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError(
            "The four normalized transition features do not "
            "sum to one"
        )

    return {
        "nAuditedSequences": int(
            featureMatrix.shape[0]
        ),
        "featureDimension": int(
            featureMatrix.shape[1]
        ),
        "allFeaturesFinite": True,
        "transitionBlockSumsToOne": True,
        "maximumTransitionSumError": (
            maximumTransitionSumError
        ),
    }


def compareWithMetricTable(
    auditTable: pd.DataFrame,
    metricTable: pd.DataFrame,
) -> None:
    officialMetrics = metricTable[
        metricTable["evaluation"] == "official_test"
    ].copy()

    for auditRow in auditTable.to_dict(
        orient="records"
    ):
        modelName = str(
            auditRow["model"]
        )

        matchingRows = officialMetrics[
            officialMetrics["model"] == modelName
        ]

        if matchingRows.shape[0] != 1:
            raise ValueError(
                f"{modelName}: expected exactly one official metric row, "
                f"found {matchingRows.shape[0]}"
            )

        storedRow = matchingRows.iloc[0]

        comparisons = {
            "rocAuc": (
                auditRow[
                    "aucUsingNegativeNormalityScore"
                ]
            ),
            "precision": auditRow[
                "precisionAtZero"
            ],
            "recall": auditRow[
                "recallAtZero"
            ],
            "f1Score": auditRow[
                "f1AtZero"
            ],
        }

        for metricName, recalculatedValue in comparisons.items():
            storedValue = float(
                storedRow[metricName]
            )

            absoluteTolerance = (
                1e-5
                if metricName == "rocAuc"
                else 1e-10
            )

            if not np.isclose(
                storedValue,
                float(recalculatedValue),
                rtol=1e-9,
                atol=absoluteTolerance,
            ):
                raise ValueError(
                    f"{modelName}: stored {metricName}={storedValue} "
                    f"does not match recalculated value "
                    f"{recalculatedValue}"
                )


def main() -> None:
    tableDirectory = (
        projectRoot
        / "outputs"
        / "tables"
    )

    resultsPath = (
        projectRoot
        / "outputs"
        / "results.csv"
    )

    metricPath = (
        tableDirectory
        / "hmad_metrics.csv"
    )

    jointFeaturePath = (
        tableDirectory
        / "wafer_joint_features.csv"
    )

    for requiredPath in [
        resultsPath,
        metricPath,
        jointFeaturePath,
    ]:
        if not requiredPath.exists():
            raise FileNotFoundError(
                f"Required output file was not found: {requiredPath}"
            )

    resultTable = pd.read_csv(
        resultsPath
    )

    metricTable = pd.read_csv(
        metricPath
    )

    requireColumns(
        resultTable,
        {
            "sampleIndex",
            "normalityScore",
            "anomalyScore",
            "hardPrediction",
            "trueLabel",
            "model",
            "evaluation",
        },
        "results.csv",
    )

    officialResultTable = resultTable[
        resultTable["evaluation"] == "official_test"
    ].copy()

    if officialResultTable.empty:
        raise ValueError(
            "results.csv contains no official_test rows"
        )

    expectedSampleCount = int(
        officialResultTable[
            "sampleIndex"
        ].nunique()
    )

    modelNames = list(
        officialResultTable[
            "model"
        ].drop_duplicates()
    )

    auditRows = []

    for modelName in modelNames:
        modelTable = officialResultTable[
            officialResultTable["model"] == modelName
        ]

        auditRows.append(
            auditModelScores(
                modelTable,
                str(modelName),
                expectedSampleCount,
            )
        )

    auditTable = pd.DataFrame(
        auditRows
    )

    compareWithMetricTable(
        auditTable,
        metricTable,
    )

    jointFeatureAudit = auditJointFeatures(
        jointFeaturePath
    )

    auditCsvPath = (
        tableDirectory
        / "wafer_score_audit.csv"
    )

    auditJsonPath = (
        tableDirectory
        / "wafer_score_audit.json"
    )

    auditTable.to_csv(
        auditCsvPath,
        index=False,
    )

    auditPayload = {
        "dataset": "Wafer",
        "evaluation": "official_test",
        "scoreConvention": {
            "largerNormalityScoreMeans": "more normal",
            "anomalyScore": "-normalityScore",
            "hardAnomalyDecision": "normalityScore < 0",
        },
        "models": auditRows,
        "jointFeatureAudit": jointFeatureAudit,
        "storedMetricsMatchRecalculation": True,
    }

    auditJsonPath.write_text(
        json.dumps(
            auditPayload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    displayColumns = [
        "model",
        "meanNormalityScoreNormal",
        "meanNormalityScoreAnomaly",
        "aucUsingNegativeNormalityScore",
        "aucUsingPositiveNormalityScore",
        "precisionAtZero",
        "recallAtZero",
        "f1AtZero",
    ]

    print(
        auditTable[
            displayColumns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        json.dumps(
            jointFeatureAudit,
            indent=2,
        )
    )

    print()
    print(
        f"Wrote score audit to: {auditCsvPath}"
    )
    print(
        f"Wrote audit metadata to: {auditJsonPath}"
    )


if __name__ == "__main__":
    main()