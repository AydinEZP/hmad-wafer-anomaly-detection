# Wafer Dataset README

## Role in this project

Wafer is the final real-world dataset used for sequence-level anomaly detection and for the main comparison between fixed-vector OC-SVM baselines and simplified HMAD.

## Source and domain

- Dataset name: **Wafer**
- Source archive: UCR/TSML Time Series Classification Archive
- Data type: univariate process-control sensor sequence
- Domain: semiconductor wafer fabrication monitoring
- Label availability: one class label per complete sequence
- Interval-level anomaly annotation: not available

## Files used by the final configuration

```text
data/Wafer/Wafer_TRAIN.ts
data/Wafer/Wafer_TEST.ts
```

The source archive also provides ARFF copies, but `configs/default.yaml` explicitly selects the `.ts` files. This repository does not redistribute either format. The TRAIN and TEST partitions must not be combined.

## Source-label and project-label mapping

| Meaning | Source label | Internal label |
|---|---:|---:|
| Normal | `1` | `0` |
| Anomaly | `-1` | `1` |

The explicit mapping is stored in `configs/default.yaml`.

## Final split after one-class conversion

| Quantity | Value |
|---|---:|
| Official TRAIN sequences | 1,000 |
| Normal sequences retained for training | 903 |
| Abnormal official-TRAIN sequences excluded from fitting | 97 |
| Official TEST sequences | 6,164 |
| Normal TEST sequences | 5,499 |
| Anomalous TEST sequences | 665 |
| Sequence length | 152 |
| Dimensions per time step | 1 |
| Normal training time steps | 137,256 |
| Official TEST anomaly fraction | approximately 0.107884 |

Only normal sequences from the official TRAIN split are used to fit the one-class models. The complete official TEST split is retained for the primary evaluation.

## Why Wafer satisfies the dataset requirement

- Each example is an independent ordered sequence, not an unordered collection of points.
- The training set contains substantially more than 100 normal sequences and more than 5,000 total training time steps.
- The test set contains hundreds of anomalous sequences and thousands of normal sequences.
- Labels are available at sequence level.
- The official split is preserved and no difficult test samples are removed after viewing results.
- The anomaly is not defined by a manually selected extreme-value threshold.

## Loading and validation

Run from the repository root:

```bat
python scripts\inspect_wafer_dataset.py
```

Expected shapes:

```text
trainSequences: (903, 152)
testSequences: (6164, 152)
testLabels: (6164,)
```

The loader validates finite values, equal sequence lengths, both TEST classes, and the explicit source-label mapping.

## Preprocessing

Three OC-SVM sequence representations are constructed:

1. `flattened`: all 152 ordered values;
2. `global_summary`: mean, standard deviation, minimum, maximum, and median;
3. `window_summary`: five relative windows, each with the same five statistics, for 25 features.

For every model configuration, `StandardScaler` is fitted only on normal training features and is then applied unchanged to test features.

HMAD uses two Gaussian hidden states. Each sequence is converted to six joint features:

```text
c11, c12, c21, c22, meanState1, meanState2
```

The state paths are Viterbi-decoded because ground-truth hidden states are not provided.

## Primary evaluation protocol

- Primary OC-SVM: `window_summary`, RBF kernel, `nu=0.10`, `gamma="scale"`.
- Main test: complete official TEST split.
- ROC-AUC score: `-normalityScore` because label 1 means anomaly.
- Hard anomaly prediction: `normalityScore < 0`.
- No anomaly interval is highlighted in Wafer figures.

## Limitations

- Wafer was originally distributed as a time-series classification dataset, not as a purpose-built one-class benchmark.
- The test labels identify whether a sequence is abnormal but not where the fault occurs inside the sequence.
- Exact localization performance cannot be measured on Wafer.
- A two-state scalar Gaussian HMAD representation may not match the physical mechanism that separates Wafer classes.
