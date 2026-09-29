import numpy as np

from src.hmad import SimplifiedHMAD
from src.standalone_support import makeTrainSet


def testFitProducesValidParametersAndScores() -> None:
    trainSequences = makeTrainSet(40, 30, 1405)
    model = SimplifiedHMAD(maxIterations=4, randomState=1405).fit(trainSequences)
    scores = model.decisionFunction(trainSequences[:5])
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert np.allclose(model.transitionMatrix_.sum(axis=1), 1.0)
    assert model.trainingFeatures_.shape == (40, 6)
    assert 1 <= model.nIterations_ <= 4


def testTrivialStateAblationReturnsOneScorePerSequence() -> None:
    trainSequences = makeTrainSet(30, 25, 1405)
    model = SimplifiedHMAD(maxIterations=3, randomState=1405).fit(trainSequences)
    scores = model.decisionFunctionWithTrivialStates(trainSequences[:7])
    assert scores.shape == (7,)
    assert np.all(np.isfinite(scores))


def testInjectedOneBasedDecoderIsAccepted() -> None:
    def oneBasedDecoder(sequence, initialProbabilities, transitionMatrix, means, variances):
        return np.where(sequence < np.mean(means), 1, 2)

    trainSequences = makeTrainSet(20, 20, 1405)
    model = SimplifiedHMAD(
        maxIterations=2,
        randomState=1405,
        viterbiDecoder=oneBasedDecoder,
    ).fit(trainSequences)
    states = model.decodeSequence(trainSequences[0])
    assert states.min() >= 0
    assert states.max() < 2


def testPredictionUsesNormalityScoreBoundary() -> None:
    trainSequences = makeTrainSet(30, 25, 1405)
    model = SimplifiedHMAD(maxIterations=3, randomState=1405).fit(trainSequences)
    scores = model.decisionFunction(trainSequences[:6])
    predictions = model.predict(trainSequences[:6])
    assert np.array_equal(predictions, np.where(scores >= 0.0, 1, -1))
