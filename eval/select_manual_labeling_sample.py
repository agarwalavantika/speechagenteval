"""
select_manual_labeling_sample.py — SpeechAgentEval, prep for judge validation

Selects a stratified subset of Stage 2 (agent) output for you to manually
label before validating the LLM-judge (see validate_judge.ipynb).

KNOWN CONFOUND (see project notes): most UA-Speech items are isolated words
without command context, so agents reasonably ask for clarification ~65-70%
of the time regardless of ASR correctness — even on clean/control speech.
The 'computer_command' word category is the only one where a plausible
unambiguous correct action exists (e.g. "Delete", "Control"), making it the
cleanest signal for genuine ASR-error-driven confusion vs. structural
single-word ambiguity. This sampler deliberately over-represents
computer_command relative to its natural frequency, so your manual labels
(and therefore judge validation) aren't dominated by the confounded
categories. Report this sampling choice explicitly in your methodology
write-up — it's a deliberate stratification decision, not neutral random
sampling.

Usage:
    python select_manual_labeling_sample.py \
        --agent_csv /content/drive/.../results/agent_stratified.csv \
        --manifest /content/drive/.../ua_speech_manifest.csv \
        --out_csv /content/drive/.../eval/manual_labels_todo.csv \
        --n_total 60
"""

import argparse
import os

import pandas as pd


# Target composition: weight toward computer_command (cleanest signal) while
# still covering other categories for the confound-caveat analysis.
CATEGORY_WEIGHTS = {
    "computer_command": 0.35,
    "uncommon_word": 0.20,
    "common_word": 0.20,
    "digit": 0.15,
    "radio_alphabet_letter": 0.10,
}


def select_sample(
    agent_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    n_total: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    df = agent_df.merge(
        manifest_df[["sample_id", "speaker_id", "severity", "ground_truth_text", "word_category"]],
        on="sample_id", how="left",
    )
    df = df[df["agent_response"].notna() & (df["agent_response"].str.strip() != "")]

    parts = []
    for category, weight in CATEGORY_WEIGHTS.items():
        n_category = max(1, round(n_total * weight))
        cat_df = df[df["word_category"] == category]
        if len(cat_df) == 0:
            print(f"WARNING: no samples found for category '{category}', skipping")
            continue

        # Spread across severity within the category too, not just randomly
        severities = cat_df["severity"].unique()
        per_severity = max(1, n_category // len(severities))
        severity_samples = []
        for sev in severities:
            sev_df = cat_df[cat_df["severity"] == sev]
            n = min(per_severity, len(sev_df))
            severity_samples.append(sev_df.sample(n=n, random_state=seed))
        cat_sample = pd.concat(severity_samples) if severity_samples else cat_df.sample(
            n=min(n_category, len(cat_df)), random_state=seed
        )
        parts.append(cat_sample)

    result = pd.concat(parts).drop_duplicates(subset="sample_id").reset_index(drop=True)

    print("Selected sample composition:")
    print("\nBy word_category:")
    print(result["word_category"].value_counts())
    print("\nBy severity:")
    print(result["severity"].value_counts())
    print(f"\nTotal selected: {len(result)}")

    return result


def format_for_labeling(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Match the schema validate_judge.ipynb expects, with blank label columns to fill in."""
    out = sample_df[["sample_id", "ground_truth_text", "asr_transcript", "agent_response"]].copy()
    out = out.rename(columns={"ground_truth_text": "ground_truth"})
    out["manual_b_label"] = ""
    out["manual_c_label"] = ""
    out["notes"] = ""
    # keep severity/word_category as reference columns for your own
    # sanity-checking while labeling, without breaking the notebook's
    # expected core columns
    out["severity_ref"] = sample_df["severity"].values
    out["word_category_ref"] = sample_df["word_category"].values
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--n_total", type=int, default=60)
    args = parser.parse_args()

    agent_df = pd.read_csv(args.agent_csv)
    manifest_df = pd.read_csv(args.manifest)

    sample_df = select_sample(agent_df, manifest_df, n_total=args.n_total)
    out_df = format_for_labeling(sample_df)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nWritten to {args.out_csv}")
    print("\nNext: open this CSV and fill in manual_b_label / manual_c_label for each row,")
    print("using the definitions in judge_prompt.py, before running validate_judge.ipynb.")


if __name__ == "__main__":
    main()
