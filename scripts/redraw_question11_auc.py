from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


projectRoot = Path(__file__).resolve().parents[1]

metricPath = (
    projectRoot
    / "outputs"
    / "gaussian_required"
    / "tables"
    / "question11_sensitivity_metrics.csv"
)

figureDirectory = (
    projectRoot
    / "outputs"
    / "gaussian_required"
    / "figures"
)


def drawAucPlot(
    metricTable: pd.DataFrame,
    modelNames: list[str],
    outputName: str,
) -> None:
    selectedTable = metricTable[
        metricTable["model"].isin(modelNames)
    ].copy()

    if selectedTable.empty:
        raise ValueError(
            "No matching Question 11 metrics were found."
        )

    fractions = np.array(
        sorted(
            selectedTable["targetRTest"].unique()
        ),
        dtype=float,
    )

    nModels = len(modelNames)

    if nModels == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -0.0018,
            0.0018,
            nModels,
        )

    lineStyles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
    ]

    markerStyles = [
        "o",
        "s",
        "^",
        "D",
        "X",
    ]

    figure, axis = plt.subplots(
        figsize=(9.0, 5.8)
    )

    for modelIndex, modelName in enumerate(
        modelNames
    ):
        modelTable = selectedTable[
            selectedTable["model"] == modelName
        ].sort_values(
            "targetRTest"
        )

        if modelTable.shape[0] != fractions.size:
            raise ValueError(
                f"Incomplete sensitivity results for {modelName}"
            )

        displayFractions = (
            modelTable[
                "targetRTest"
            ].to_numpy(dtype=float)
            + offsets[modelIndex]
        )

        aucValues = modelTable[
            "rocAuc"
        ].to_numpy(dtype=float)

        axis.plot(
            displayFractions,
            aucValues,
            marker=markerStyles[
                modelIndex
                % len(markerStyles)
            ],
            linestyle=lineStyles[
                modelIndex
                % len(lineStyles)
            ],
            linewidth=1.8,
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.7,
            label=modelName,
        )

    axis.set_xticks(
        fractions
    )

    axis.set_xticklabels(
        [
            f"{fraction:.2f}"
            for fraction in fractions
        ]
    )

    axis.set_xlabel(
        "Test anomaly fraction"
    )

    axis.set_ylabel(
        "ROC-AUC"
    )

    axis.set_title(
        "ROC-AUC across test anomaly fractions"
    )

    # A narrow interval makes the exactly overlapping
    # AUC=1 results easier to inspect.
    minimumAuc = float(
        selectedTable["rocAuc"].min()
    )

    if minimumAuc >= 0.98:
        axis.set_ylim(
            0.975,
            1.005,
        )
    else:
        axis.set_ylim(
            0.0,
            1.05,
        )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        loc="lower left"
    )

    figure.text(
        0.5,
        0.01,
        (
            "Small horizontal offsets are used only "
            "to reveal exactly overlapping curves; "
            "the original anomaly fractions are unchanged."
        ),
        ha="center",
        fontsize=8.5,
    )

    figure.tight_layout(
        rect=(0.0, 0.05, 1.0, 1.0)
    )

    figureDirectory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        figureDirectory
        / f"{outputName}.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        figureDirectory
        / f"{outputName}.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    if not metricPath.exists():
        raise FileNotFoundError(
            f"Metric file was not found: {metricPath}"
        )

    metricTable = pd.read_csv(
        metricPath
    )

    requiredColumns = {
        "model",
        "targetRTest",
        "rocAuc",
    }

    missingColumns = (
        requiredColumns
        - set(metricTable.columns)
    )

    if missingColumns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(
                sorted(missingColumns)
            )
        )

    drawAucPlot(
        metricTable,
        [
            "HMAD",
            "HMAD all-one ablation",
            "OC-SVM global_summary",
        ],
        "question11_auc_vs_contamination",
    )

    drawAucPlot(
        metricTable,
        [
            "HMAD",
            "HMAD all-one ablation",
            "OC-SVM flattened",
            "OC-SVM global_summary",
            "OC-SVM window_summary",
        ],
        "question11_auc_vs_contamination_all_models",
    )

    print(
        "Redrew Question 11 AUC figures without "
        "rerunning Q9, Q10, or Q11."
    )


if __name__ == "__main__":
    main()