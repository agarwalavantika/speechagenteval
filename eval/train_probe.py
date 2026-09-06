"""
train_probe.py — SpeechAgentEval Phase 2.5: Internal Calibration Probing

Trains a probe on LLM hidden-state activations to predict whether an input
was *actually* ambiguous (ground truth, derived from the ASR-error taxonomy),
then cross-tabulates the probe's finding against the model's *expressed*
confidence in its output (from your taxonomy's C-axis labels).

Expected inputs
----------------
1. A manifest CSV with columns:
     sample_id, speaker_id, activation_path, layer_available,
     b_label, c_label
   - activation_path: path to a .npz file containing one array per layer,
     keys like "layer_0", "layer_1", ... "layer_N" (produced by the
     extraction hook — see extract_activations.py)
   - b_label: one of {B1, B2, B3, not_applicable} (from judge/manual labels)
   - c_label: one of {C1, C2, C3, C4}

2. Ground-truth ambiguity target is derived, not manually labeled:
     is_ambiguous = 1 if b_label in {B1, B3} else 0
   (B1/B3 both mean the ASR error changed meaning; B2/not_applicable don't.)

Usage
-----
python train_probe.py --manifest manifest.csv --activations_root /path/to/acts \
    --split_by speaker_id --probe_type logreg

This script:
1. Loads the manifest + activations
2. Splits train/val/test by speaker (never split by sample — avoids leakage
   from the same speaker appearing in both train and test)
3. Sweeps candidate layers, picks the best by validation AUROC
4. Trains the final probe on the chosen layer
5. Reports test accuracy, AUROC, and a calibration (reliability) plot
6. Cross-tabulates probe predictions against expressed confidence (C-axis)
   to surface the "suppressed signal" cell — cases where the probe detects
   internal ambiguity but the model's output was confident anyway
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt


AMBIGUOUS_B_LABELS = {"B1", "B3"}
CONFIDENT_C_LABELS = {"C1", "C4"}   # output looked confident / behaved normally
HEDGED_C_LABELS = {"C2", "C3"}      # output hedged or asked for clarification


@dataclass
class ProbeResult:
    layer: str
    val_auroc: float
    test_auroc: float
    test_accuracy: float
    test_brier: float


def load_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    required = {"sample_id", "speaker_id", "activation_path", "b_label", "c_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    df["is_ambiguous"] = df["b_label"].isin(AMBIGUOUS_B_LABELS).astype(int)
    return df


def load_activations(df: pd.DataFrame, layer_key: str) -> np.ndarray:
    """Load one layer's activation vector per row, in manifest order."""
    vectors = []
    for path in df["activation_path"]:
        with np.load(path) as npz:
            if layer_key not in npz:
                raise KeyError(f"{layer_key} not found in {path}. Available: {list(npz.keys())}")
            vectors.append(npz[layer_key])
    return np.stack(vectors)


def speaker_grouped_split(df: pd.DataFrame, group_col: str, test_size=0.2, val_size=0.2, seed=42):
    """Split so no speaker appears in more than one of train/val/test."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df[group_col]))
    trainval_df = df.iloc[trainval_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df[group_col]))

    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    return train_df, val_df, test_df


def build_probe(probe_type: str):
    if probe_type == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    elif probe_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(48,), max_iter=2000, random_state=42)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")


def evaluate_probe(probe, X, y) -> dict:
    proba = probe.predict_proba(X)[:, 1]
    preds = probe.predict(X)
    return {
        "auroc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
        "accuracy": accuracy_score(y, preds),
        "brier": brier_score_loss(y, proba),
        "proba": proba,
    }


def layer_sweep(train_df, val_df, layer_keys, probe_type):
    """Try each layer, return results sorted by val AUROC (best first)."""
    results = []
    y_train = train_df["is_ambiguous"].values
    y_val = val_df["is_ambiguous"].values

    for layer_key in layer_keys:
        X_train = load_activations(train_df, layer_key)
        X_val = load_activations(val_df, layer_key)

        probe = build_probe(probe_type)
        probe.fit(X_train, y_train)
        val_metrics = evaluate_probe(probe, X_val, y_val)

        results.append({"layer": layer_key, "val_auroc": val_metrics["auroc"], "probe": probe})
        print(f"  {layer_key}: val AUROC = {val_metrics['auroc']:.3f}")

    results.sort(key=lambda r: (r["val_auroc"] if not np.isnan(r["val_auroc"]) else -1), reverse=True)
    return results


def reliability_plot(y_true, y_proba, out_path, n_bins=10):
    """Save a calibration/reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_acc, bin_conf, bin_count = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_acc.append(y_true[mask].mean())
        bin_conf.append(y_proba[mask].mean())
        bin_count.append(mask.sum())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(bin_conf, bin_acc, marker="o", label="Probe")
    plt.xlabel("Predicted probability of ambiguity")
    plt.ylabel("Observed fraction ambiguous")
    plt.title("Probe calibration (reliability diagram)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def cross_tabulate(df: pd.DataFrame, probe_proba: np.ndarray, threshold=0.5) -> pd.DataFrame:
    """
    The key output: does the probe's internal signal agree with the model's
    expressed behavior? Surfaces the 'suppressed signal' cell.

    Uses b_label (not c_label) to define "confident vs hedged" output:
    B1 (silent_propagation) = the agent committed to the wrong content
    without flagging uncertainty — this is the real "confident" case.
    B3 (correct_rejection) = the agent asked for clarification — the real
    "hedged" case. c_label's C4 ("appropriate") is NOT a reliable proxy for
    this distinction, since C4 covers both confidently-correct AND
    appropriately-hedging responses — using it here would conflate the two
    and make the cross-tab meaningless (verified: on a dataset where the
    judge rarely predicts C2/C3, nearly everything falls into C4/C1 and the
    cross-tab collapses to one cell). B2 (graceful_degradation) and
    not_applicable rows are excluded from this specific cross-tab since
    they don't represent a clean confident-vs-hedged contrast — restrict to
    only the B1/B3 rows in the input df before calling this if you want the
    cleanest version of the headline number.
    """
    df = df.copy()
    df["probe_says_ambiguous"] = (probe_proba >= threshold)
    df["output_was_confident"] = df["b_label"] == "B1"
    df["output_was_hedged"] = df["b_label"] == "B3"

    # Restrict the headline cross-tab to the clean B1-vs-B3 contrast rows
    clean_df = df[df["b_label"].isin(["B1", "B3"])]
    if len(clean_df) == 0:
        print("WARNING: no B1/B3 rows in this split — cannot compute the clean cross-tab. "
              "Falling back to full df (B1 vs everything-else), interpret with caution.")
        clean_df = df

    table = pd.crosstab(
        clean_df["probe_says_ambiguous"].map({True: "probe: ambiguous", False: "probe: clear"}),
        clean_df["output_was_confident"].map({True: "output: confident (B1)", False: "output: hedged (B3)"}),
    )

    suppressed = clean_df[(clean_df["probe_says_ambiguous"]) & (clean_df["output_was_confident"])]
    n_b1 = (clean_df["output_was_confident"]).sum()
    print("\n=== Cross-tabulation: probe signal vs. expressed confidence (B1 vs B3 rows only) ===")
    print(table)
    print(f"\nSuppressed-signal cases (probe detected ambiguity, agent silently committed anyway — B1): "
          f"{len(suppressed)}/{len(clean_df)} of B1/B3 rows ({len(suppressed) / max(len(clean_df),1):.1%})")
    if n_b1:
        print(f"As a fraction of B1 (silent propagation) cases specifically: "
              f"{len(suppressed)}/{n_b1} ({len(suppressed) / n_b1:.1%}) — "
              f"this is your strongest 'model knew and said it anyway' number.")
    print("Sample IDs:", suppressed["sample_id"].tolist()[:15], "..." if len(suppressed) > 15 else "")

    return table, suppressed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--layers", nargs="+", default=[f"layer_{i}" for i in (8, 12, 16, 20, 24, 28)],
                         help="Layer keys to sweep, matching keys in the .npz activation files")
    parser.add_argument("--split_by", default="speaker_id")
    parser.add_argument("--probe_type", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--out_dir", default="probe_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_manifest(args.manifest)
    print(f"Loaded {len(df)} samples, {df['is_ambiguous'].mean():.1%} labeled ambiguous")

    train_df, val_df, test_df = speaker_grouped_split(df, args.split_by)
    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    print("\n=== Layer sweep (selecting by validation AUROC) ===")
    sweep_results = layer_sweep(train_df, val_df, args.layers, args.probe_type)
    best = sweep_results[0]
    best_layer = best["layer"]
    print(f"\nBest layer: {best_layer} (val AUROC = {best['val_auroc']:.3f})")

    # Retrain on train+val for the final probe, evaluate on held-out test
    final_train_df = pd.concat([train_df, val_df])
    X_final_train = load_activations(final_train_df, best_layer)
    y_final_train = final_train_df["is_ambiguous"].values
    X_test = load_activations(test_df, best_layer)
    y_test = test_df["is_ambiguous"].values

    final_probe = build_probe(args.probe_type)
    final_probe.fit(X_final_train, y_final_train)
    test_metrics = evaluate_probe(final_probe, X_test, y_test)

    result = ProbeResult(
        layer=best_layer,
        val_auroc=best["val_auroc"],
        test_auroc=test_metrics["auroc"],
        test_accuracy=test_metrics["accuracy"],
        test_brier=test_metrics["brier"],
    )
    print(f"\n=== Final test performance (layer={best_layer}) ===")
    print(f"Accuracy: {result.test_accuracy:.3f} | AUROC: {result.test_auroc:.3f} | Brier: {result.test_brier:.3f}")
    print("Brier score interpretation: lower is better-calibrated; compare against a "
          "coin-flip baseline of ~0.25 for balanced classes.")

    reliability_path = os.path.join(args.out_dir, "reliability_diagram.png")
    reliability_plot(y_test, test_metrics["proba"], reliability_path)
    print(f"Reliability diagram saved to {reliability_path}")

    table, suppressed = cross_tabulate(test_df, test_metrics["proba"])
    table.to_csv(os.path.join(args.out_dir, "cross_tabulation.csv"))
    suppressed.to_csv(os.path.join(args.out_dir, "suppressed_signal_cases.csv"), index=False)

    summary = {
        "best_layer": best_layer,
        "val_auroc": result.val_auroc,
        "test_auroc": result.test_auroc,
        "test_accuracy": result.test_accuracy,
        "test_brier": result.test_brier,
        "n_train": len(final_train_df),
        "n_test": len(test_df),
        "suppressed_signal_rate": len(suppressed) / len(test_df),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {os.path.join(args.out_dir, 'summary.json')}")
    print("\nCAVEAT for your writeup: a decodable signal in activations shows the "
          "information is linearly present, not that it causally drives the output. "
          "Do not claim causation from this probe alone.")


if __name__ == "__main__":
    main()
