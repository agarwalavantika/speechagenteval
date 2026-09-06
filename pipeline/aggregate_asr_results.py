"""
aggregate_asr_results.py — SpeechAgentEval Stage 1 analysis

Merges multiple ASR result CSVs (one per speaker/severity run from
run_asr_batch.py, or your existing asr_test.csv / asr_test_control.csv
files) with severity metadata, and reproduces the core Stage 1 findings
automatically:

  - Mean/median WER and CER per severity bucket
  - "Hallucination rate": % of samples where Whisper transcribed 2+ words
    for what should have been a single word / short command — a specific,
    citable failure mode distinct from plain word-substitution errors
  - Correlation between Whisper's own confidence (avg logprob) and actual
    WER, per severity bucket — tests whether ASR confidence is a reliable
    self-flagging signal (spoiler from initial data: it's weak, and weaker
    for dysarthric speech, which matters for your taxonomy design)

Usage (as importable functions, recommended in a notebook):
    from aggregate_asr_results import aggregate_results

    summary_df, merged_df = aggregate_results(
        result_csvs={
            "M16": "/content/drive/.../asr_test.csv",
            "CF02": "/content/drive/.../asr_test_control.csv",
            # add more as you run more speakers
        },
        severity_csv="/content/drive/.../speaker_severity.csv",
    )
    summary_df  # one row per speaker, with severity + all metrics

Usage (as a script, if you've saved a mapping file):
    python aggregate_asr_results.py \
        --result_csvs_json result_csv_paths.json \
        --severity_csv speaker_severity.csv \
        --out_summary summary.csv
"""

import argparse
import json
import os

import pandas as pd


def load_severity_lookup(severity_csv: str) -> pd.DataFrame:
    df = pd.read_csv(severity_csv)
    df["speaker_id"] = df["speaker_id"].str.upper()
    return df


def extract_speaker_id(sample_id: str) -> str:
    """sample_id format: {SpeakerID}_{Block}_{WordCode}_{Mic} -> SpeakerID"""
    return sample_id.split("_")[0].upper()


def compute_hallucination_flag(row: pd.Series) -> bool:
    """True if Whisper transcribed 2+ words where CER/WER context suggests
    a short target — approximated here via the normalized transcript's word
    count. This is a heuristic proxy, not a ground-truth label; validate
    spot-checks manually if you rely on this for a headline claim."""
    n_words = len(str(row.get("asr_transcript_normalized", "")).split())
    return n_words >= 2


def aggregate_results(result_csvs: dict, severity_csv: str) -> tuple:
    """
    result_csvs: dict mapping a label (e.g. speaker_id) -> path to that
                 speaker's ASR result CSV.
    Returns (summary_df, merged_df):
        summary_df: one row per speaker/label with aggregate metrics
        merged_df: all rows concatenated, with severity + speaker_id columns
                   added, for finer-grained analysis if needed
    """
    severity_df = load_severity_lookup(severity_csv)
    severity_map = dict(zip(severity_df["speaker_id"], severity_df["severity"]))
    intel_map = (dict(zip(severity_df["speaker_id"], severity_df["intelligibility_pct"]))
                 if "intelligibility_pct" in severity_df.columns else {})

    all_rows = []
    summary_rows = []

    for label, path in result_csvs.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {label}")
            continue

        df = pd.read_csv(path)
        df["speaker_id"] = df["sample_id"].apply(extract_speaker_id)

        # Prefer the severity found via the manifest lookup; fall back to
        # 'control' if the label itself suggests it and no match is found.
        inferred_speaker = df["speaker_id"].iloc[0] if len(df) else label.upper()
        severity = severity_map.get(inferred_speaker)
        if severity is None:
            severity = "control" if "control" in label.lower() or "cf" in inferred_speaker.lower() or "cm" in inferred_speaker.lower() else "unknown"
        intelligibility = intel_map.get(inferred_speaker)

        df["severity"] = severity
        df["intelligibility_pct"] = intelligibility
        df["source_label"] = label
        df["is_hallucination"] = df.apply(compute_hallucination_flag, axis=1)

        all_rows.append(df)

        conf_wer_corr = df["asr_confidence"].corr(df["wer"]) if df["wer"].notna().sum() > 1 else float("nan")

        summary_rows.append({
            "label": label,
            "speaker_id": inferred_speaker,
            "severity": severity,
            "intelligibility_pct": intelligibility,
            "n_samples": len(df),
            "mean_wer": df["wer"].mean(),
            "median_wer": df["wer"].median(),
            "mean_cer": df["cer"].mean(),
            "pct_any_error": (df["wer"] > 0).mean(),
            "hallucination_rate": df["is_hallucination"].mean(),
            "mean_asr_confidence": df["asr_confidence"].mean(),
            "confidence_wer_correlation": conf_wer_corr,
        })

    merged_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    # Order by intelligibility so the severity gradient reads naturally
    if "intelligibility_pct" in summary_df.columns and summary_df["intelligibility_pct"].notna().any():
        summary_df = summary_df.sort_values("intelligibility_pct", na_position="first")

    return summary_df, merged_df


def print_report(summary_df: pd.DataFrame):
    print("\n=== Stage 1 ASR Summary — severity gradient ===\n")
    display_cols = ["label", "severity", "intelligibility_pct", "n_samples",
                     "mean_wer", "hallucination_rate", "confidence_wer_correlation"]
    print(summary_df[display_cols].to_string(index=False))

    if len(summary_df) >= 2:
        control_row = summary_df[summary_df["severity"] == "control"]
        dysarthric_rows = summary_df[summary_df["severity"] != "control"]
        if len(control_row) and len(dysarthric_rows):
            control_halluc = control_row["hallucination_rate"].mean()
            dysarthric_halluc = dysarthric_rows["hallucination_rate"].mean()
            if control_halluc > 0:
                ratio = dysarthric_halluc / control_halluc
                print(f"\nHallucination rate: control={control_halluc:.1%}, "
                      f"dysarthric (avg)={dysarthric_halluc:.1%} "
                      f"({ratio:.1f}x higher)")
            print("\nCAVEAT: with only a few speakers so far, treat this as a "
                  "directional finding, not a generalized claim — expand across "
                  "more speakers per severity bucket before stating this broadly.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_csvs_json", required=True,
                         help="Path to a JSON file mapping label -> result CSV path, "
                              'e.g. {"M16": "asr_test.csv", "CF02": "asr_test_control.csv"}')
    parser.add_argument("--severity_csv", required=True)
    parser.add_argument("--out_summary", required=True)
    args = parser.parse_args()

    with open(args.result_csvs_json) as f:
        result_csvs = json.load(f)

    summary_df, merged_df = aggregate_results(result_csvs, args.severity_csv)

    out_dir = os.path.dirname(args.out_summary)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)

    print_report(summary_df)
    print(f"\nSummary written to {args.out_summary}")


if __name__ == "__main__":
    main()
