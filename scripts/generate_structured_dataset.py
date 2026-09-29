"""Generate the fixed structured sequence dataset used by the project."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from src.structured_dataset import (
    createStructuredSequenceDataset,
    saveStructuredSequenceDataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(projectRoot / "data" / "legacy_synthetic"))
    parser.add_argument("--seed", type=int, default=1405)
    arguments = parser.parse_args()

    dataset = createStructuredSequenceDataset(seed=arguments.seed)
    datasetPath, metadataPath, manifestPath = saveStructuredSequenceDataset(
        dataset, arguments.output
    )
    print(f"dataset: {datasetPath}")
    print(f"metadata: {metadataPath}")
    print(f"manifest: {manifestPath}")
    print(
        f"train={dataset.trainSequences.shape}, test={dataset.testSequences.shape}, "
        f"normal_test={(dataset.testLabels == 0).sum()}, "
        f"anomaly_test={(dataset.testLabels == 1).sum()}"
    )


if __name__ == "__main__":
    main()
