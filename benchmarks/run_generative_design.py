"""Train a conditional VAE and sample high-performing designs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.airfoil import load_airfoil
from datasets.concrete import load_concrete
from models.cvae import CVAEConfig, ConditionalVAE
from models.mlp import MLPRegressorTorch
from inverse_design import flatten_dataset

TASKS = {
    "airfoil": {
        "loader": load_airfoil,
        "objective": "minimize",
        "objective_label": "Sound Pressure Level (dB)",
        "default_percentile": 0.1,
    },
    "concrete": {
        "loader": load_concrete,
        "objective": "maximize",
        "objective_label": "Compressive Strength (MPa)",
        "default_percentile": 0.9,
    },
}


def parse_hidden_dims(values: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(v) for v in values)
    if not dims:
        raise argparse.ArgumentTypeError("At least one hidden dimension is required.")
    if any(v <= 0 for v in dims):
        raise argparse.ArgumentTypeError("Hidden dimensions must be positive integers.")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS.keys()), required=True)
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimensionality.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=(128, 128),
        help="Hidden layer sizes for encoder/decoder (space-separated).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs.")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight on KL divergence term (beta-VAE coefficient).",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Desired performance value. Overrides percentile-based target when set.",
    )
    parser.add_argument(
        "--target-percentile",
        type=float,
        default=None,
        help="Percentile used to derive target when --target is not provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="How many designs to sample from the trained decoder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy/torch and dataset split.",
    )
    parser.add_argument(
        "--eval-surrogate",
        action="store_true",
        help="Also train a fast MLP surrogate to score generated designs.",
    )
    parser.add_argument(
        "--surrogate-epochs",
        type=int,
        default=400,
        help="Epochs for the auxiliary surrogate scorer when --eval-surrogate is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for JSON output.",
    )
    parser.add_argument("--device", type=str, default=None, help="Force computation device.")
    args = parser.parse_args()
    args.hidden_dims = parse_hidden_dims(args.hidden_dims)
    if args.latent_dim <= 0:
        parser.error("--latent-dim must be positive.")
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--batch-size and --epochs must be positive.")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive.")
    if args.beta <= 0:
        parser.error("--beta must be positive.")
    return args


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEF_VALIDITY_MARGIN = 0.05


def load_dataset_with_metadata(loader, seed: int) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Load dataset splits alongside optional metadata."""
    output = loader(random_state=seed, return_metadata=True)
    if len(output) == 7:
        *splits, metadata = output
    else:
        splits = output
        metadata = {}
    return tuple(splits), dict(metadata)


def inverse_transform(data: np.ndarray, scaler: Any) -> np.ndarray:
    """Invert standardization using a fitted scaler."""
    return data * scaler.scale_ + scaler.mean_


def evaluate_validity(
    generated: np.ndarray,
    metadata: dict[str, Any],
    margin: float = DEF_VALIDITY_MARGIN,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return validity mask, unscaled designs, and violation counts per feature."""
    scaler = metadata.get("scaler")
    feature_names = metadata.get("feature_names")
    train_min = metadata.get("train_min")
    train_max = metadata.get("train_max")
    if scaler is None or feature_names is None or train_min is None or train_max is None:
        raise ValueError("Dataset metadata missing fields for validity checking.")

    unscaled = inverse_transform(generated, scaler)
    span = np.maximum(train_max - train_min, 1e-12)
    lower = (train_min - margin * span)[None, :]
    upper = (train_max + margin * span)[None, :]

    below = unscaled < lower
    above = unscaled > upper
    violations = below | above
    valid_mask = ~np.any(violations, axis=1)

    violation_counts = {
        str(feature_names[idx]): int(violations[:, idx].sum())
        for idx in range(len(feature_names))
    }
    return valid_mask, unscaled, violation_counts


def compute_diversity(designs: np.ndarray) -> float | None:
    """Compute mean pairwise Euclidean distance for a set of designs."""
    n = designs.shape[0]
    if n < 2:
        return None
    diff = designs[:, None, :] - designs[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n, 1)
    return float(dist[iu].mean())


def train_surrogate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
) -> MLPRegressorTorch:
    """Train the surrogate model used for scoring generated designs."""
    surrogate = MLPRegressorTorch(in_dim=X_train.shape[1], epochs=epochs)
    surrogate.fit(X_train, y_train)
    return surrogate


def score_surrogate(surrogate: MLPRegressorTorch, X: np.ndarray) -> np.ndarray:
    """Predict objective scores using the fitted surrogate."""
    return surrogate.predict(X)


def load_inverse_design_results(task: str, output_dir: Path) -> dict[str, Any] | None:
    """Load inverse design benchmark results if they exist."""
    path = output_dir / f"{task}_inverse_design.json"
    if not path.exists():
        return None
    with path.open("r") as fh:
        return json.load(fh)


def select_best_surrogate_entry(
    results: dict[str, Any], objective: str
) -> tuple[str, str, dict[str, Any]] | None:
    """Return (surrogate_name, fraction_key, fraction_payload) for the top performer."""
    surrogates = results.get("surrogates", {})
    best: tuple[str, str, dict[str, Any]] | None = None
    best_score: float | None = None
    for name, surrogate_entry in surrogates.items():
        for frac_key, frac_payload in surrogate_entry.get("fractions", {}).items():
            summary = frac_payload.get("summary", {})
            final_best = summary.get("final_best_mean")
            if final_best is None:
                continue
            if best_score is None:
                best_score = final_best
                best = (name, frac_key, frac_payload)
                continue
            if objective == "maximize":
                if final_best > best_score:
                    best_score = final_best
                    best = (name, frac_key, frac_payload)
            else:
                if final_best < best_score:
                    best_score = final_best
                    best = (name, frac_key, frac_payload)
    return best


def select_best_run(runs: list[dict[str, Any]], objective: str) -> dict[str, Any] | None:
    """Pick the single best run from the inverse design benchmark."""
    if not runs:
        return None
    if objective == "maximize":
        return max(runs, key=lambda run: run.get("best_value", float("-inf")))
    return min(runs, key=lambda run: run.get("best_value", float("inf")))


def find_best_candidate_index(run: dict[str, Any], objective: str) -> tuple[int, float]:
    """Reconstruct which candidate achieved the reported best objective value."""
    tol = 1e-8
    observed: dict[int, float] = {}
    for idx, value in zip(run.get("initial_indices", []), run.get("initial_objectives", [])):
        observed[int(idx)] = float(value)
    if not observed:
        raise ValueError("Run is missing initial observations.")
    if objective == "maximize":
        best_idx = max(observed, key=observed.get)
    else:
        best_idx = min(observed, key=observed.get)
    best_value = observed[best_idx]

    for trace in run.get("traces", []):
        idx = int(trace["candidate_index"])
        value = float(trace["objective_value"])
        observed[idx] = value
        if objective == "maximize":
            if value > best_value + tol:
                best_idx = idx
                best_value = value
        else:
            if value < best_value - tol:
                best_idx = idx
                best_value = value

    reported_best = float(run.get("best_value", best_value))
    if abs(best_value - reported_best) > tol:
        for idx, value in observed.items():
            if abs(value - reported_best) <= tol:
                best_idx = idx
                best_value = value
                break
    return best_idx, best_value



def plot_performance_histogram(
    scores: np.ndarray,
    objective_label: str,
    destination: Path,
) -> None:
    """Persist a histogram of surrogate-evaluated performance."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=30, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.set_xlabel(f"Predicted {objective_label}")
    ax.set_ylabel("Count")
    ax.set_title("CVAE Design Performance")
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)



def is_better_than_baseline(
    scores: np.ndarray,
    baseline: float,
    objective: str,
) -> np.ndarray:
    """Return a boolean mask indicating which scores beat the baseline."""
    if objective == "maximize":
        return scores > baseline
    return scores < baseline


def resolve_target(
    y: np.ndarray,
    objective: str,
    explicit_target: float | None,
    percentile: float | None,
    default_percentile: float,
) -> tuple[float, float | None]:
    if explicit_target is not None:
        return float(explicit_target), None
    pct = default_percentile if percentile is None else float(percentile)
    if not 0 < pct < 1:
        raise ValueError("Percentile must lie in (0, 1).")
    target = float(np.quantile(y, pct))
    return target, pct


def train_cvae(
    args: argparse.Namespace,
    splits: tuple[np.ndarray, ...],
) -> tuple[ConditionalVAE, dict]:
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    config = CVAEConfig(
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        beta=args.beta,
        device=args.device,
    )

    model = ConditionalVAE(x_dim=X_fit.shape[1], cond_dim=1, config=config)
    model.fit(X_fit, y_fit)

    recon_test = model.reconstruct(X_test, y_test)
    recon_mse = float(np.mean((recon_test - X_test) ** 2))

    metrics = {
        "train_loss_last": model.training_history_[-1] if model.training_history_ else None,
        "train_loss_first": model.training_history_[0] if model.training_history_ else None,
        "reconstruction_mse_test": recon_mse,
    }
    return model, metrics


def evaluate_with_surrogate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    generated: np.ndarray,
    epochs: int,
) -> tuple[MLPRegressorTorch, np.ndarray]:
    surrogate = train_surrogate_model(X_train, y_train, epochs=epochs)
    preds = score_surrogate(surrogate, generated)
    return surrogate, preds





def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    task_cfg = TASKS[args.task]
    loader = task_cfg["loader"]

    splits, metadata = load_dataset_with_metadata(loader, args.seed)
    model, metrics = train_cvae(args, splits)

    X_train, y_train, X_val, y_val, _, _ = splits
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    target_value, target_percentile = resolve_target(
        y_fit,
        objective=task_cfg["objective"],
        explicit_target=args.target,
        percentile=args.target_percentile,
        default_percentile=task_cfg["default_percentile"],
    )

    targets = np.full(shape=(args.num_samples,), fill_value=target_value, dtype=np.float32)
    generated = model.generate(targets, num_samples=args.num_samples)

    valid_mask = np.ones(args.num_samples, dtype=bool)
    violation_counts: dict[str, int] = {}
    generated_unscaled: np.ndarray | None = None
    if metadata:
        try:
            valid_mask, generated_unscaled, violation_counts = evaluate_validity(generated, metadata)
        except ValueError as exc:
            print(f"Warning: {exc}. Treating all generated samples as valid.")
            generated_unscaled = None
            violation_counts = {}

    valid_count = int(valid_mask.sum())
    validity_metrics = {
        "valid_count": valid_count,
        "invalid_count": int(args.num_samples - valid_count),
        "valid_percentage": float(100.0 * valid_count / args.num_samples),
        "violation_counts": violation_counts,
    }
    valid_indices = [int(idx) for idx in np.flatnonzero(valid_mask)]

    diversity_value = None
    if generated_unscaled is not None and valid_count >= 2:
        diversity_value = compute_diversity(generated_unscaled[valid_mask])
    diversity_metrics = {
        "average_pairwise_distance": diversity_value,
        "space": "original" if generated_unscaled is not None else None,
    }

    surrogate_model: MLPRegressorTorch | None = None
    surrogate_scores: np.ndarray | None = None
    valid_scores: np.ndarray | None = None
    performance_stats: dict[str, float] | None = None
    performance_hist_path: Path | None = None

    if args.eval_surrogate:
        surrogate_model, surrogate_scores = evaluate_with_surrogate(
            X_train=X_fit,
            y_train=y_fit,
            generated=generated,
            epochs=args.surrogate_epochs,
        )
        if valid_count > 0:
            valid_scores = surrogate_scores[valid_mask]
            if valid_scores.size > 0:
                performance_stats = {
                    "mean": float(np.mean(valid_scores)),
                    "std": float(np.std(valid_scores)),
                    "min": float(np.min(valid_scores)),
                    "max": float(np.max(valid_scores)),
                }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if valid_scores is not None and valid_scores.size > 0:
        performance_hist_path = args.output_dir / f"{args.task}_generative_performance_hist.png"
        plot_performance_histogram(
            valid_scores,
            task_cfg["objective_label"],
            performance_hist_path,
        )

    paper1_comparison: dict[str, Any] | None = None
    inverse_results = load_inverse_design_results(args.task, args.output_dir)
    if inverse_results is not None:
        results_seed = inverse_results.get("random_state")
        if results_seed is not None and int(results_seed) != int(args.seed):
            print(
                "Warning: inverse design results use random_state",
                results_seed,
                "which differs from current seed; skipping comparison.",
            )
        else:
            selection = select_best_surrogate_entry(inverse_results, task_cfg["objective"])
            if selection is not None:
                surrogate_name, frac_key, frac_payload = selection
                best_run = select_best_run(frac_payload.get("runs", []), task_cfg["objective"])
                if best_run is not None:
                    try:
                        best_idx, _ = find_best_candidate_index(best_run, task_cfg["objective"])
                        X_flat, y_flat = flatten_dataset(splits)
                        best_design_scaled = X_flat[best_idx]
                        best_objective = float(y_flat[best_idx])
                        best_design_unscaled = None
                        if metadata and metadata.get("scaler") is not None:
                            best_design_unscaled = inverse_transform(
                                best_design_scaled[None, :], metadata["scaler"]
                            )[0]
                        surrogate_prediction = None
                        if surrogate_model is not None:
                            surrogate_prediction = float(
                                score_surrogate(surrogate_model, best_design_scaled[None, :])[0]
                            )
                        better_count = None
                        better_fraction = None
                        if (
                            valid_scores is not None
                            and valid_scores.size > 0
                            and surrogate_prediction is not None
                        ):
                            better_mask = is_better_than_baseline(
                                valid_scores,
                                surrogate_prediction,
                                task_cfg["objective"],
                            )
                            better_count = int(np.sum(better_mask))
                            better_fraction = float(better_count / valid_scores.size)

                        paper1_comparison = {
                            "surrogate_name": surrogate_name,
                            "initial_fraction": frac_key,
                            "run_id": int(best_run.get("run_id", -1)),
                            "candidate_index": int(best_idx),
                            "objective_value": best_objective,
                            "surrogate_prediction": surrogate_prediction,
                            "generative_valid_better_count": better_count,
                            "generative_valid_better_fraction": better_fraction,
                            "comparison_basis": "surrogate_prediction"
                            if surrogate_prediction is not None
                            else "objective_value",
                        }
                        paper1_comparison["design_scaled"] = best_design_scaled.tolist()
                        if best_design_unscaled is not None:
                            paper1_comparison["design_unscaled"] = best_design_unscaled.tolist()
                    except (ValueError, IndexError) as err:
                        print(f"Warning: unable to compute Paper 1 comparison ({err}).")
            else:
                print("Warning: inverse design results missing surrogate entries; skipping comparison.")

    out_path = args.output_dir / f"{args.task}_generative_cvae.json"
    payload = {
        "task": args.task,
        "objective_label": task_cfg["objective_label"],
        "config": {
            "latent_dim": args.latent_dim,
            "hidden_dims": list(args.hidden_dims),
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "beta": args.beta,
            "device": str(model.device),
        },
        "seed": int(args.seed),
        "target_value": target_value,
        "target_percentile": target_percentile,
        "num_samples": int(args.num_samples),
        "metrics": metrics,
        "generated_designs": generated.tolist(),
        "validity": validity_metrics,
        "valid_design_indices": valid_indices,
        "diversity": diversity_metrics,
        "surrogate_scores": surrogate_scores.tolist() if surrogate_scores is not None else None,
        "performance": {
            "valid_scores_summary": performance_stats,
            "histogram_path": str(performance_hist_path) if performance_hist_path else None,
        },
        "paper1_comparison": paper1_comparison,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved generative designs to {out_path}")
    if metrics["train_loss_last"] is not None:
        print(
            "Final train loss:",
            f"{metrics['train_loss_last']:.4f}",
            "(first epoch:",
            f"{metrics['train_loss_first']:.4f})",
        )
    print(f"Target value used: {target_value:.3f}")
    print(
        "Validity:",
        f"{valid_count}/{args.num_samples}",
        f"({validity_metrics['valid_percentage']:.1f}% valid)",
    )
    if diversity_value is not None:
        print(f"Diversity (avg pairwise distance): {diversity_value:.3f}")
    if valid_scores is not None and valid_scores.size > 0 and performance_stats is not None:
        print(
            "Surrogate (valid) score stats -- mean:",
            f"{performance_stats['mean']:.3f}",
            "std:",
            f"{performance_stats['std']:.3f}",
        )
    elif args.eval_surrogate:
        print("No valid surrogate-scored designs to summarize.")
    if performance_hist_path:
        print(f"Saved performance histogram to {performance_hist_path}")
    if paper1_comparison is not None:
        print(
            "Best surrogate design:",
            f"{paper1_comparison['surrogate_name']} (fraction {paper1_comparison['initial_fraction']})",
            f"objective={paper1_comparison['objective_value']:.3f}",
        )
        if paper1_comparison["generative_valid_better_fraction"] is not None:
            print(
                "Generative designs outperform best surrogate design in",
                f"{paper1_comparison['generative_valid_better_fraction'] * 100:.1f}%",
                "of valid samples (surrogate scores).",
            )
        else:
            print("Paper 1 best design comparison recorded (surrogate scores unavailable).")




if __name__ == "__main__":
    main()
