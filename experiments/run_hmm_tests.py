"""Run Markov, HMM, and Viterbi simulations and save machine-readable outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.markov import markov_log_probability
from src.hmm import sample_gaussian_hmm
from src.viterbi import brute_force_decode, state_accuracy, viterbi_decode

SEED = 1405
PI = np.array([2.0 / 3.0, 1.0 / 3.0])
A = np.array([[0.95, 0.05], [0.10, 0.90]])
MEANS = np.array([0.0, 4.0])
VARIANCES = np.array([1.0, 1.0])


def run_markov_example() -> dict:
    z = np.array([0, 0, 0, 1, 1, 0, 0], dtype=int)
    log_prob = markov_log_probability(z, PI, A)
    return {
        "state_sequence_code_labels": z.tolist(),
        "state_sequence_report_labels": (z + 1).tolist(),
        "log_probability": log_prob,
        "raw_probability": float(np.exp(log_prob)),
    }


def run_generator_example(rng: np.random.Generator) -> dict:
    z, x = sample_gaussian_hmm(PI, A, MEANS, VARIANCES, 100, rng=rng)
    return {
        "length": int(x.size),
        "state_counts": np.bincount(z, minlength=2).tolist(),
        "observation_mean": float(x.mean()),
        "observation_std": float(x.std(ddof=1)),
        "first_10_states_report_labels": (z[:10] + 1).tolist(),
        "first_10_observations": x[:10].round(6).tolist(),
    }


def run_bruteforce_validation(
    rng: np.random.Generator, n_sequences: int = 20, length: int = 8
) -> dict:
    agreements = 0
    score_max_abs_difference = 0.0
    for _ in range(n_sequences):
        _, x = sample_gaussian_hmm(PI, A, MEANS, VARIANCES, length, rng=rng)
        vit = viterbi_decode(x, PI, A, MEANS, VARIANCES)
        brute = brute_force_decode(x, PI, A, MEANS, VARIANCES)
        agreements += int(np.array_equal(vit.path, brute.path))
        score_max_abs_difference = max(
            score_max_abs_difference, abs(vit.log_score - brute.log_score)
        )
    return {
        "n_sequences": n_sequences,
        "length": length,
        "exact_path_agreements": agreements,
        "agreement_rate": agreements / n_sequences,
        "max_abs_log_score_difference": score_max_abs_difference,
    }


def run_state_recovery(
    rng: np.random.Generator, n_sequences: int = 200, length: int = 100
) -> dict:
    accuracies = []
    finite_failures = 0
    total_correct = 0
    total_steps = 0
    for _ in range(n_sequences):
        z, x = sample_gaussian_hmm(PI, A, MEANS, VARIANCES, length, rng=rng)
        decoded = viterbi_decode(x, PI, A, MEANS, VARIANCES)
        acc = state_accuracy(z, decoded.path)
        accuracies.append(acc)
        total_correct += int(np.sum(z == decoded.path))
        total_steps += length
        finite_failures += int(not np.isfinite(decoded.log_score))
    arr = np.asarray(accuracies)
    return {
        "n_sequences": n_sequences,
        "length": length,
        "micro_state_accuracy": total_correct / total_steps,
        "mean_sequence_accuracy": float(arr.mean()),
        "std_sequence_accuracy": float(arr.std(ddof=1)),
        "min_sequence_accuracy": float(arr.min()),
        "max_sequence_accuracy": float(arr.max()),
        "finite_log_score_failures": finite_failures,
    }


def run_underflow_demo(
    rng: np.random.Generator, length: int = 5000
) -> tuple[dict, np.ndarray, np.ndarray]:
    z, _ = sample_gaussian_hmm(PI, A, MEANS, VARIANCES, length, rng=rng)
    factors = np.empty(length, dtype=float)
    factors[0] = PI[z[0]]
    factors[1:] = A[z[:-1], z[1:]]
    raw_prefix = np.cumprod(factors)
    log_prefix = np.cumsum(np.log(factors))
    zero_idx = np.flatnonzero(raw_prefix == 0.0)
    first_underflow_length = int(zero_idx[0] + 1) if zero_idx.size else None
    result = {
        "length": length,
        "first_zero_raw_product_prefix_length": first_underflow_length,
        "final_raw_probability": float(raw_prefix[-1]),
        "final_log_probability": float(log_prefix[-1]),
        "all_log_prefix_values_finite": bool(np.all(np.isfinite(log_prefix))),
    }
    return result, raw_prefix, log_prefix


def save_underflow_plot(raw_prefix: np.ndarray, log_prefix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, raw_prefix.size + 1)
    tiny = np.nextafter(0.0, 1.0)
    raw_log = np.log(np.maximum(raw_prefix, tiny))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(steps, log_prefix, label="Log-space cumulative probability")
    ax.plot(steps, raw_log, linestyle="--", label="Log of raw cumulative product")
    ax.set_xlabel("Prefix length")
    ax.set_ylabel("Log probability")
    ax.set_title("Numerical underflow in a length-5000 Markov path")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(results: dict, path: Path) -> None:
    brute = results["brute_force_validation"]
    recovery = results["state_recovery"]
    text = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Validation of the log-space Viterbi implementation.}}
\\label{{tab:viterbi-validation}}
\\begin{{tabular}}{{lllll}}
\\toprule
Validation check & Sequences & Length & Metric & Result \\\\
\\midrule
Brute-force optimum & {brute['n_sequences']} & $T={brute['length']}$ & Exact path agreement & {brute['exact_path_agreements']}/{brute['n_sequences']} \\\\
Generated-HMM recovery & {recovery['n_sequences']} & $T={recovery['length']}$ & State accuracy & {100*recovery['micro_state_accuracy']:.2f}\\% \\\\
Finite-score check & {recovery['n_sequences']} & $T={recovery['length']}$ & Failure count & {recovery['finite_log_score_failures']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = {
        "configuration": {
            "seed": SEED,
            "pi": PI.tolist(),
            "A": A.tolist(),
            "means": MEANS.tolist(),
            "variances": VARIANCES.tolist(),
        },
        "sq4_markov_log_probability": run_markov_example(),
        "sq5_generator_example": run_generator_example(rng),
        "brute_force_validation": run_bruteforce_validation(rng),
        "state_recovery": run_state_recovery(rng),
    }
    underflow, raw_prefix, log_prefix = run_underflow_demo(rng)
    results["sq7_underflow_demo"] = underflow

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "hmm_viterbi_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    save_underflow_plot(
        raw_prefix, log_prefix, out_dir / "figures" / "markov_underflow.pdf"
    )
    write_latex_table(results, out_dir / "tables" / "viterbi_validation_table.tex")

    print(json.dumps(results, indent=2))
    print(f"Saved results to {out_dir / 'hmm_viterbi_results.json'}")


if __name__ == "__main__":
    main()
