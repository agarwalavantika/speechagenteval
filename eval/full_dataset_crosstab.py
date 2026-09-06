"""
full_dataset_crosstab.py — SpeechAgentEval, full-dataset suppressed-signal analysis

train_probe.py's headline "suppressed signal" number comes from a single
held-out test split (~15% of data), which for a class as rare as B1 can be
a very small n (e.g. 5 cases) — exciting but statistically fragile.

This script computes the same cross-tabulation across ALL labeled rows
instead, using speaker-grouped K-fold cross-validation so every row still
gets an out-of-fold (i.e. the model never saw that row during training)
probability estimate — this avoids the optimistic bias of just re-scoring
the training data, while using far more of your B1/B3 cases than a single
test split would.

Usage:
    from full_dataset_crosstab import run_full_dataset_crosstab
    result = run_full_dataset_crosstab(
        manifest_path="judge_scores_full_500_fixed_paths.csv",
        layer="layer_12",   # use whatever train_probe.py found as the best layer
        probe_type="logreg",
        n_folds=5,
    )
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from train_probe import load_manifest, load_activations, build_probe, reliability_plot


def cross_validated_predictions(
    df: pd.DataFrame,
    layer: str,
    probe_type: str = "logreg",
    group_col: str = "speaker_id",
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns an array of out-of-fold predicted probabilities, one per row of
    df, in the same order as df. Each row's prediction comes from a probe
    that was NOT trained on that row's speaker (grouped k-fold), so this is
    a fair, leakage-free estimate for every sample, not just a held-out
    slice.
    """
    X = load_activations(df, layer)
    y = df["is_ambiguous"].values
    groups = df[group_col].values

    n_unique_groups = len(np.unique(groups))
    effective_folds = min(n_folds, n_unique_groups)
    if effective_folds < n_folds:
        print(f"WARNING: only {n_unique_groups} unique speakers, reducing folds "
              f"from {n_folds} to {effective_folds}")

    gkf = GroupKFold(n_splits=effective_folds)
    oof_proba = np.zeros(len(df))

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        probe = build_probe(probe_type)
        probe.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = probe.predict_proba(X[test_idx])[:, 1]
        fold_auroc = roc_auc_score(y[test_idx], oof_proba[test_idx]) if len(set(y[test_idx])) > 1 else float("nan")
        print(f"  Fold {fold_i + 1}/{effective_folds}: {len(test_idx)} held-out samples, AUROC = {fold_auroc:.3f}")

    return oof_proba


def full_cross_tabulate(df: pd.DataFrame, oof_proba: np.ndarray, threshold: float = 0.5) -> tuple:
    """Same logic as train_probe.py's cross_tabulate, but using out-of-fold
    predictions across the full dataset instead of a single test split."""
    df = df.copy()
    df["probe_says_ambiguous"] = oof_proba >= threshold
    df["output_was_confident"] = df["b_label"] == "B1"

    clean_df = df[df["b_label"].isin(["B1", "B3"])]

    table = pd.crosstab(
        clean_df["probe_says_ambiguous"].map({True: "probe: ambiguous", False: "probe: clear"}),
        clean_df["output_was_confident"].map({True: "output: confident (B1)", False: "output: hedged (B3)"}),
    )

    suppressed = clean_df[(clean_df["probe_says_ambiguous"]) & (clean_df["output_was_confident"])]
    n_b1 = clean_df["output_was_confident"].sum()

    print(f"\n=== Full-dataset cross-tabulation (N={len(clean_df)} B1/B3 rows, "
          f"out-of-fold predictions) ===")
    print(table)
    print(f"\nSuppressed-signal cases: {len(suppressed)}/{len(clean_df)} of B1/B3 rows "
          f"({len(suppressed) / max(len(clean_df), 1):.1%})")
    if n_b1:
        print(f"As a fraction of ALL B1 (silent propagation) cases in the dataset: "
              f"{len(suppressed)}/{n_b1} ({len(suppressed) / n_b1:.1%})")
        print(f"(Compare to your single-test-split estimate — this uses "
              f"{n_b1} B1 cases instead of whatever small number the test split had, "
              f"a much more stable estimate.)")

    return table, suppressed


def run_full_dataset_crosstab(
    manifest_path: str,
    layer: str,
    probe_type: str = "logreg",
    n_folds: int = 5,
    out_dir: str = "full_probe_results",
) -> dict:
    import os
    os.makedirs(out_dir, exist_ok=True)

    df = load_manifest(manifest_path)
    print(f"Loaded {len(df)} samples, {df['is_ambiguous'].mean():.1%} labeled ambiguous")

    print(f"\nRunning {n_folds}-fold speaker-grouped cross-validation on layer {layer}...")
    oof_proba = cross_validated_predictions(df, layer, probe_type=probe_type, n_folds=n_folds)

    y = df["is_ambiguous"].values
    overall_auroc = roc_auc_score(y, oof_proba)
    overall_acc = accuracy_score(y, oof_proba >= 0.5)
    overall_brier = brier_score_loss(y, oof_proba)
    print(f"\nOverall out-of-fold performance: AUROC={overall_auroc:.3f}, "
          f"Accuracy={overall_acc:.3f}, Brier={overall_brier:.3f}")

    reliability_path = f"{out_dir}/full_reliability_diagram.png"
    reliability_plot(y, oof_proba, reliability_path)
    print(f"Reliability diagram saved to {reliability_path}")

    table, suppressed = full_cross_tabulate(df, oof_proba)
    table.to_csv(f"{out_dir}/full_cross_tabulation.csv")
    suppressed.to_csv(f"{out_dir}/full_suppressed_signal_cases.csv", index=False)

    print(f"\nSuppressed-signal case sample_ids (up to 20 shown):")
    print(suppressed["sample_id"].tolist()[:20])

    return {
        "overall_auroc": overall_auroc,
        "overall_accuracy": overall_acc,
        "overall_brier": overall_brier,
        "n_suppressed": len(suppressed),
        "n_b1_total": int((df["b_label"] == "B1").sum()),
        "table": table,
        "suppressed_df": suppressed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--layer", required=True, help="Which layer to use, e.g. 'layer_12' (from train_probe.py's best layer)")
    parser.add_argument("--probe_type", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--out_dir", default="full_probe_results")
    args = parser.parse_args()

    run_full_dataset_crosstab(
        manifest_path=args.manifest,
        layer=args.layer,
        probe_type=args.probe_type,
        n_folds=args.n_folds,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
