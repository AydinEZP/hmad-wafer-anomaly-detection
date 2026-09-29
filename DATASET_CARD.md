# Final Dataset Card

The final project uses one real dataset and one required controlled synthetic dataset. They serve different experimental purposes and must be reported separately.

## Wafer

- Source: UCR/TSML Time Series Classification Archive.
- Unit of observation: one univariate sequence of length 152.
- Official TRAIN: 1,000 sequences; 903 normal sequences are retained for one-class fitting.
- Official TEST: 6,164 sequences, containing 5,499 normal and 665 anomalous examples.
- Source label mapping: `1 = normal`, `-1 = anomaly`.
- Project label mapping: `0 = normal`, `1 = anomaly`.
- Main evaluation: complete official TEST split.
- Ground-truth anomaly interval: unavailable.
- Main limitation: sequence-level detection can be evaluated, but exact fault localization cannot.

Full details: `data/Wafer/README.md`.

## Required Gaussian data

- Generated programmatically; no external download.
- Normal distribution: `N(0,1)`.
- Anomaly-block distribution: `N(4,1)`.
- Clean train: 200 normal sequences of length 100.
- Question 9: 200 normal and 50 anomalous sequences, one block of length 20.
- Question 11: exact 1,000-sequence test conditions at anomaly fractions 0.01, 0.05, 0.10, and 0.20.
- Required Questions 9--11 use one anomaly block of length 20.
- Block intervals are stored, enabling time-step localization evaluation.
- Supplementary experiments vary block length and block count.

Full details: `data/Gaussian/README.md`.

## Shared integrity controls

- Training contains only normal sequences.
- Test data do not participate in feature-scaler fitting.
- All compared models use the same test condition.
- Continuous anomaly scores equal the negative of the model normality scores.
- The hard anomaly boundary is fixed at normality score zero.
- Seeds are deterministic and documented in the dataset README files.
