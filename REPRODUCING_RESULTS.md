# Reproducing the Results

## 1. Create an isolated environment

Python 3.10 or newer is required; Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment for your operating system and install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

## 2. Run the automated tests

```bash
python -m pytest -q
```

The tests exercise feature extraction, evaluation, HMM utilities, Viterbi decoding, HMAD integration, synthetic-data generation, and Wafer parsing with temporary fixtures.

## 3. Run the self-contained Gaussian experiments

```bash
python experiments/run_gaussian_required.py
```

The Gaussian arrays are generated deterministically. Results, tables, and figures are written below `outputs/gaussian_required/`.

## 4. Obtain the Wafer dataset

Download the official Wafer TRAIN and TEST files from the UCR/TSML Time Series Classification Archive and place exactly these files in `data/Wafer/`:

```text
Wafer_TRAIN.ts
Wafer_TEST.ts
```

Do not merge the official partitions. Validate the files before running the real-data experiments:

```bash
python scripts/inspect_wafer_dataset.py
```

Expected shapes after one-class conversion:

```text
trainSequences: (903, 152)
testSequences:  (6164, 152)
testLabels:     (6164,)
```

## 5. Run the integrated pipeline

```bash
python experiments/run_all.py
```

To skip the test step after it has already passed:

```bash
python experiments/run_all.py --skip-tests
```

The integrated pipeline runs Markov/HMM/Viterbi validation, Wafer OC-SVM baselines, Wafer HMAD, required Gaussian experiments, and supplementary sensitivity sweeps. It writes generated artifacts under `outputs/`; these files are intentionally ignored by Git.

## 6. Interpret the archived tables

The CSV files under `results/` are selected outputs from the archived project run. They are included so that the README metrics can be audited without bundling raw datasets or large generated arrays. Reproduced floating-point results may vary slightly across compatible dependency versions because the requirements specify minimum versions rather than a locked environment.
