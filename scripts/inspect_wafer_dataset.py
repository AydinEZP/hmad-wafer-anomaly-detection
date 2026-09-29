from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

projectRoot = Path(__file__).resolve().parents[1]

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from src.wafer_dataset import loadWaferDataset


def resolveProjectPath(pathValue: str | None) -> Path | None:
    """Resolve a configuration path relative to the project root."""

    if pathValue is None or str(pathValue).strip() == "":
        return None

    path = Path(str(pathValue))

    if not path.is_absolute():
        path = projectRoot / path

    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(projectRoot / "configs" / "default.yaml"),
        help="Path to the YAML configuration file.",
    )

    arguments = parser.parse_args()

    configPath = resolveProjectPath(arguments.config)

    if configPath is None or not configPath.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {configPath}"
        )

    with configPath.open("r", encoding="utf-8") as configFile:
        config = yaml.safe_load(configFile)

    dataConfig = config["data"]

    datasetDirectory = resolveProjectPath(
        dataConfig["datasetDirectory"]
    )
    trainPath = resolveProjectPath(
        dataConfig.get("trainPath")
    )
    testPath = resolveProjectPath(
        dataConfig.get("testPath")
    )

    dataset = loadWaferDataset(
        datasetDirectory,
        trainPath=trainPath,
        testPath=testPath,
        normalSourceLabel=dataConfig.get("normalSourceLabel"),
        anomalySourceLabel=dataConfig.get("anomalySourceLabel"),
    )

    print(json.dumps(dataset.metadata, indent=2))
    print("trainSequences:", dataset.trainSequences.shape)
    print("testSequences:", dataset.testSequences.shape)
    print("testLabels:", dataset.testLabels.shape)


if __name__ == "__main__":
    main()