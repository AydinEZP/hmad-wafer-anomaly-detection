from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class OcsvmBaseline:
    scaler: StandardScaler
    model: OneClassSVM
    representation: str
    nu: float
    gamma: str | float
    nWindows: int

    def decisionFunction(self, features: np.ndarray) -> np.ndarray:
        featureArray = np.asarray(features, dtype=float)
        return self.model.decision_function(
            self.scaler.transform(featureArray)
        ).reshape(-1)


def fitOcsvmBaseline(
    trainFeatures: np.ndarray,
    *,
    representation: str,
    nu: float,
    gamma: str | float = "scale",
    nWindows: int = 5,
) -> OcsvmBaseline:
    featureArray = np.asarray(trainFeatures, dtype=float)
    if featureArray.ndim != 2 or featureArray.shape[0] == 0:
        raise ValueError("trainFeatures must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(featureArray)):
        raise ValueError("trainFeatures contain non-finite values")

    scaler = StandardScaler().fit(featureArray)
    model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
    model.fit(scaler.transform(featureArray))
    return OcsvmBaseline(
        scaler=scaler,
        model=model,
        representation=representation,
        nu=float(nu),
        gamma=gamma,
        nWindows=int(nWindows),
    )
