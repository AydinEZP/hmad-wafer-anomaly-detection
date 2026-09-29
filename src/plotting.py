from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def saveFigure(figure: plt.Figure, outputPath: str | Path) -> None:
    outputFile = Path(outputPath)
    figure.savefig(outputFile, bbox_inches="tight")
    if outputFile.suffix.lower() == ".pdf":
        figure.savefig(outputFile.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def saveSequencePlot(
    sequence: np.ndarray,
    title: str,
    outputPath: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 3.4))
    axis.plot(np.arange(len(sequence)), sequence, linewidth=1.4)
    axis.set_xlabel("Time step")
    axis.set_ylabel("Observation")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    saveFigure(figure, outputPath)



def saveSequencePairPlot(
    nominalSequence: np.ndarray,
    anomalousSequence: np.ndarray,
    anomalyStart: int = -1,
    anomalyEnd: int = -1,
    outputPath: str | Path = "sequence_examples.pdf",
    nominalTitle: str = "Nominal sequence",
    anomalousTitle: str = "Anomalous sequence",
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    timeSteps = np.arange(len(nominalSequence))
    axes[0].plot(timeSteps, nominalSequence, linewidth=1.35)
    axes[0].set_title(nominalTitle)
    axes[0].set_ylabel("Observation")
    axes[0].grid(alpha=0.25)

    axes[1].plot(timeSteps, anomalousSequence, linewidth=1.35)
    if anomalyStart >= 0 and anomalyEnd > anomalyStart:
        axes[1].axvspan(
            anomalyStart, anomalyEnd - 1, alpha=0.18, label="Known anomaly interval"
        )
        axes[1].legend(loc="upper right")
    axes[1].set_title(anomalousTitle)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Observation")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    saveFigure(figure, outputPath)

def saveJointFeatureHeatmap(
    featureMatrix: np.ndarray,
    outputPath: str | Path,
    nominalCount: int | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    image = axis.imshow(featureMatrix, aspect="auto")
    if nominalCount is not None and 0 < nominalCount < featureMatrix.shape[0]:
        axis.axhline(nominalCount - 0.5, linewidth=1.2, linestyle="--")
    axis.set_xlabel("Joint feature")
    axis.set_ylabel("Sequence index")
    axis.set_xticks(np.arange(6))
    axis.set_xticklabels(
        ["c11", "c12", "c21", "c22", "mean state 1", "mean state 2"],
        rotation=25,
        ha="right",
    )
    axis.set_title("Joint features for nominal and anomalous sequences")
    figure.colorbar(image, ax=axis, label="Feature value")
    figure.tight_layout()
    saveFigure(figure, outputPath)


def saveConvergencePlot(
    iterations: np.ndarray,
    changedFractions: np.ndarray,
    meanDecisionScores: np.ndarray,
    outputPath: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.plot(iterations, changedFractions, marker="o", label="Changed-state fraction")
    axis.plot(iterations, meanDecisionScores, marker="s", label="Mean decision score")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Recorded value")
    axis.set_title("Simplified HMAD convergence diagnostics")
    axis.set_xticks(iterations)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    saveFigure(figure, outputPath)


def saveRocPlot(
    labels: np.ndarray,
    modelScores: dict[str, np.ndarray],
    outputPath: str | Path,
    title: str = "ROC curves on the fixed test set",
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    for modelName, normalityScores in modelScores.items():
        falsePositiveRate, truePositiveRate, _ = roc_curve(labels, -normalityScores)
        axis.plot(falsePositiveRate, truePositiveRate, linewidth=1.6, label=modelName)
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    saveFigure(figure, outputPath)


def saveAucSensitivityPlot(
    fractions: np.ndarray,
    modelAucValues: dict[str, np.ndarray],
    outputPath: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    for modelName, aucValues in modelAucValues.items():
        axis.plot(fractions, aucValues, marker="o", linewidth=1.6, label=modelName)
    axis.set_xlabel("Test anomaly fraction")
    axis.set_ylabel("ROC-AUC")
    axis.set_title("ROC-AUC across test anomaly fractions")
    axis.set_xticks(fractions)
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    saveFigure(figure, outputPath)


def saveDecodedPathPlot(
    sequence: np.ndarray,
    states: np.ndarray,
    title: str,
    outputPath: str | Path,
    anomalyStart: int = -1,
    anomalyEnd: int = -1,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 3.8))
    timeSteps = np.arange(len(sequence))
    axis.plot(timeSteps, sequence, linewidth=1.4, label="Observation")
    if anomalyStart >= 0 and anomalyEnd > anomalyStart:
        axis.axvspan(anomalyStart, anomalyEnd - 1, alpha=0.18, label="Inserted block")
    stateAxis = axis.twinx()
    stateAxis.step(timeSteps, states + 1, where="mid", linestyle="--", label="Decoded state")
    axis.set_xlabel("Time step")
    axis.set_ylabel("Observation")
    stateAxis.set_ylabel("Decoded state")
    stateAxis.set_yticks(np.arange(1, int(np.max(states)) + 2))
    axis.set_title(title)
    axis.grid(alpha=0.25)
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = stateAxis.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    figure.tight_layout()
    saveFigure(figure, outputPath)
