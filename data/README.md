# Dataset Index

The final project uses two distinct datasets for different purposes.

## 1. Wafer real dataset

Location:

```text
data/Wafer/
```

Wafer is the main real-world sequence-level anomaly-detection benchmark. The raw TRAIN and TEST files are not redistributed in this repository. Obtain them from the UCR/TSML Time Series Classification Archive and place the two `.ts` files in this directory. See:

```text
data/Wafer/README.md
```

## 2. Required Gaussian synthetic dataset

Definition and generator documentation:

```text
data/Gaussian/README.md
src/gaussian_required_data.py
```

Gaussian arrays are generated deterministically during execution and serialized under:

```text
outputs/gaussian_required/data/
```

The `data/Gaussian/` directory contains documentation only; it is not a manually downloaded dataset folder.

## Legacy data

An earlier structured synthetic dataset and generated binary arrays were intentionally omitted from this portfolio edition. They are not selected by `configs/default.yaml` and are not required by the automated tests.
