"""Run validation, Wafer experiments, and required Gaussian experiments."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from experiments.run_hmad import runExperiment
from experiments.run_ocsvm import runOcsvmExperiment


def _copy_pdf_tree(sourceDirectory: Path, targetDirectory: Path) -> None:
    """Copy every PDF below sourceDirectory while preserving subdirectories."""

    if not sourceDirectory.exists():
        return

    for sourcePath in sourceDirectory.rglob("*.pdf"):
        relativePath = sourcePath.relative_to(sourceDirectory)
        destinationPath = targetDirectory / relativePath
        destinationPath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sourcePath, destinationPath)


def sync_report_artifacts() -> None:
    """Copy validated Wafer and Gaussian figures into report directories."""

    overleafDirectory = projectRoot / "overleaf_source"

    targetRoots = [projectRoot / "figures"]
    if overleafDirectory.exists():
        targetRoots.append(overleafDirectory / "figures")

    waferFigureDirectory = projectRoot / "outputs" / "figures"
    gaussianFigureDirectory = (
        projectRoot
        / "outputs"
        / "gaussian_required"
        / "figures"
    )

    for targetRoot in targetRoots:
        targetRoot.mkdir(parents=True, exist_ok=True)

        # Keep existing Wafer figure names unchanged.
        _copy_pdf_tree(
            waferFigureDirectory,
            targetRoot,
        )

        # Keep Gaussian artifacts separate to avoid name collisions.
        _copy_pdf_tree(
            gaussianFigureDirectory,
            targetRoot / "gaussian_required",
        )

    generatedNumbers = projectRoot / "generated_numbers.tex"
    if generatedNumbers.exists() and overleafDirectory.exists():
        shutil.copy2(
            generatedNumbers,
            overleafDirectory / generatedNumbers.name,
        )


def _load_json_if_present(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as inputFile:
        payload = json.load(inputFile)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, found {type(payload).__name__}"
        )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(projectRoot / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(projectRoot / "outputs"),
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
    )
    arguments = parser.parse_args()

    outputDirectory = Path(arguments.output)

    if not arguments.skip_tests:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(projectRoot / "tests"),
                "-q",
            ],
            check=True,
            cwd=projectRoot,
        )

    subprocess.run(
        [
            sys.executable,
            str(projectRoot / "experiments" / "run_hmm_tests.py"),
        ],
        check=True,
        cwd=projectRoot,
    )

    # Main real-world Wafer baseline experiment.
    ocsvmSummary = runOcsvmExperiment(
        arguments.config,
        arguments.output,
    )

    # Main real-world Wafer HMAD experiment.
    hmadSummary = runExperiment(
        arguments.config,
        arguments.output,
    )

    # Required Gaussian Questions 9--11 plus supplementary robustness sweeps.
    subprocess.run(
        [
            sys.executable,
            str(
                projectRoot
                / "experiments"
                / "run_gaussian_required.py"
            ),
            "--output",
            str(outputDirectory),
        ],
        check=True,
        cwd=projectRoot,
    )

    sync_report_artifacts()

    gaussianSummary = _load_json_if_present(
        outputDirectory
        / "gaussian_required"
        / "summary.json"
    )

    finalSummary = {
        "wafer": {
            "ocsvmPrimaryConfiguration": ocsvmSummary[
                "primaryConfiguration"
            ],
            "ocsvmPrimaryMetrics": ocsvmSummary[
                "primaryMetrics"
            ],
            "hmadLearnedParameters": hmadSummary[
                "learnedParameters"
            ],
        },
        "gaussianRequired": {
            "completed": gaussianSummary is not None,
            "summaryPath": str(
                outputDirectory
                / "gaussian_required"
                / "summary.json"
            ),
            "questions9To11Present": bool(
                gaussianSummary
                and all(
                    key in gaussianSummary
                    for key in (
                        "question9",
                        "question10",
                        "question11",
                    )
                )
            ),
            "supplementarySweepsPresent": bool(
                gaussianSummary
                and "supplementarySweeps" in gaussianSummary
            ),
        },
    }

    integratedSummaryPath = outputDirectory / "integrated_summary.json"
    integratedSummaryPath.parent.mkdir(parents=True, exist_ok=True)
    integratedSummaryPath.write_text(
        json.dumps(finalSummary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(finalSummary, indent=2))
    print(f"Wrote integrated summary to: {integratedSummaryPath}")


if __name__ == "__main__":
    main()
