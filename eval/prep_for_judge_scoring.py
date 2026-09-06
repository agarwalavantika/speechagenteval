"""
prep_for_judge_scoring.py — SpeechAgentEval

Merges agent output (from run_agent_batch.py) with the manifest to attach
ground_truth_text, producing the exact schema run_local_judge() expects
(sample_id, ground_truth, asr_transcript, agent_response), plus severity/
word_category kept as reference columns for later taxonomy breakdowns.

Usage:
    python prep_for_judge_scoring.py \
        --agent_csv agent_stratified.csv \
        --manifest ua_speech_manifest.csv \
        --out_csv agent_stratified_for_judge.csv
"""

import argparse
import os

import pandas as pd


def prep_for_judge(agent_csv: str, manifest_csv: str) -> pd.DataFrame:
    agent = pd.read_csv(agent_csv)
    manifest = pd.read_csv(manifest_csv)

    merged = agent.merge(
        manifest[["sample_id", "ground_truth_text", "severity", "word_category", "speaker_id"]],
        on="sample_id", how="left",
    )
    merged = merged.rename(columns={"ground_truth_text": "ground_truth"})

    missing = merged["ground_truth"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows have no matching ground_truth after merge — "
              f"these sample_ids weren't found in the manifest. Check for ID mismatches.")

    required = {"sample_id", "ground_truth", "asr_transcript", "agent_response"}
    missing_cols = required - set(merged.columns)
    if missing_cols:
        raise ValueError(f"Output is missing required columns for the judge: {missing_cols}")

    print(f"Prepared {len(merged)} rows for judge scoring")
    print("Severity distribution:")
    print(merged["severity"].value_counts())

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    merged = prep_for_judge(args.agent_csv, args.manifest)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Written to {args.out_csv}")


if __name__ == "__main__":
    main()
