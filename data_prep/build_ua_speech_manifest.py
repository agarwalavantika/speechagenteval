"""
build_ua_speech_manifest.py — SpeechAgentEval data prep (v2)

Builds the master UA-Speech manifest using the REAL speaker severity table
and word-code-to-ground-truth-text lookup extracted from the dataset's own
documentation (speaker_wordlist.xls), rather than guessed values.

Folder layout assumed (confirmed from actual dataset):
    audio/{SPEAKER_ID}/{SPEAKER_ID}_{BLOCK}_{WORD_CODE}_{MIC}.wav
    audio/control/{CONTROL_SPEAKER_ID}/...   (control speakers grouped separately)

Filename format: {SpeakerID}_{Block}_{WordCode}_{Mic}.wav
    e.g. F02_B1_C10_M2.wav   -> speaker F02, block B1, word code C10, mic M2
    e.g. M16_B3_UW9_M3.wav   -> speaker M16, block B3, word code UW9, mic M3

IMPORTANT ground-truth lookup quirk (verified against the actual wordlist):
    - For D (digit), C (computer command), CW (common word), and L (radio
      alphabet) codes, the code alone maps to a word — e.g. "D5" -> "Five",
      "C8" -> "Control", "LA" -> "Alpha". Block is NOT part of the lookup key.
    - For UW (uncommon word) codes, the BLOCK IS part of the lookup key —
      e.g. filename word code "UW9" in block B3 must be looked up as
      "B3_UW9" -> "hoist". Looking up "UW9" alone will not match.
    This script handles this automatically; see `build_lookup_key()`.

Required companion files (included alongside this script):
    - word_code_lookup.csv   (columns: word, file_code)      <- from Word_filename sheet
    - speaker_severity.csv   (columns: speaker_id, severity, intelligibility_pct)  <- from Speaker sheet

Known dataset notes (from the documentation, worth keeping in mind):
    - M02, M03, F01 are not included in this release (recorded under a
      different protocol).
    - M06 exists in the audio but the speaker did not approve redistribution
      of results — consider excluding M06 from any published findings/report
      even if the audio is present, out of respect for that restriction.
    - M13's intelligibility score was "not obtained" in the source docs —
      treat as severity='unknown', don't fabricate a number.

Colab/Kaggle usage:
    from google.colab import drive
    drive.mount('/content/drive')

Usage:
    python build_ua_speech_manifest.py \
        --audio_dir /content/drive/MyDrive/UASpeech/audio \
        --word_lookup /content/drive/MyDrive/SpeechAgentEval/data_prep/word_code_lookup.csv \
        --severity_csv /content/drive/MyDrive/SpeechAgentEval/data_prep/speaker_severity.csv \
        --out_manifest /content/drive/MyDrive/SpeechAgentEval/manifests/ua_speech_manifest.csv
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<speaker_id>[A-Za-z]{1,2}\d{2})_(?P<block>B\d)_(?P<word_code>[A-Za-z]+\d*)_(?P<mic>M\d)\.wav$",
    re.IGNORECASE,
)

# Speakers explicitly noted in the dataset docs as excluded from redistribution
# of results, even though audio may be present locally.
EXCLUDE_FROM_RESULTS = {"M06"}

WORD_CODE_CATEGORY = {
    "D": "digit",
    "CW": "common_word",
    "UW": "uncommon_word",
    "C": "computer_command",
    "L": "radio_alphabet_letter",
}


def classify_word_code(word_code: str) -> str:
    """
    Radio-alphabet codes (LA, LB, ... LZ) are exactly 'L' + one letter, with
    no digit suffix, so a generic 'grab the leading letters' regex swallows
    the whole code and misses the L->radio_alphabet_letter mapping. Check
    for that pattern explicitly before falling back to the general case.
    """
    code = word_code.upper()
    if re.match(r"^L[A-Z]$", code):
        return WORD_CODE_CATEGORY["L"]
    prefix_match = re.match(r"^([A-Za-z]+)", code)
    if not prefix_match:
        return "unknown"
    return WORD_CODE_CATEGORY.get(prefix_match.group(1), "unknown")


def build_lookup_key(block: str, word_code: str) -> str:
    """
    UW codes need the block prefix to find the correct ground-truth word
    (block-specific uncommon words); all other code types look up directly.
    """
    category = classify_word_code(word_code)
    if category == "uncommon_word":
        return f"{block.upper()}_{word_code.upper()}"
    return word_code.upper()


def load_word_lookup(word_lookup_csv: str) -> dict:
    df = pd.read_csv(word_lookup_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"word", "file_code"}.issubset(df.columns):
        raise ValueError(f"word_lookup CSV must have columns [word, file_code], got {list(df.columns)}")
    return dict(zip(df["file_code"].str.upper(), df["word"]))


def load_severity_map(severity_csv: str) -> pd.DataFrame:
    df = pd.read_csv(severity_csv)
    required = {"speaker_id", "severity"}
    if not required.issubset(df.columns):
        raise ValueError(f"severity_csv must have columns {required}, got {list(df.columns)}")
    df["speaker_id"] = df["speaker_id"].str.upper()
    return df


def build_manifest(audio_dir: str, word_lookup: dict, severity_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unmatched_files = []
    unknown_words = set()
    unknown_speakers = set()

    severity_map = dict(zip(severity_df["speaker_id"], severity_df["severity"]))
    intel_map = (dict(zip(severity_df["speaker_id"], severity_df["intelligibility_pct"]))
                 if "intelligibility_pct" in severity_df.columns else {})

    audio_paths = list(Path(audio_dir).rglob("*.wav"))
    print(f"Found {len(audio_paths)} .wav files under {audio_dir}")

    for path in audio_paths:
        m = FILENAME_PATTERN.match(path.name)
        if not m:
            unmatched_files.append(path.name)
            continue

        speaker_id = m.group("speaker_id").upper()
        block = m.group("block").upper()
        word_code = m.group("word_code").upper()
        mic = m.group("mic").upper()

        # Control speakers: infer from folder path (audio/control/...) since
        # control speaker IDs (CF02, CM01, etc.) may not always be reflected
        # cleanly in severity_csv — folder location is the ground truth here.
        is_control = "control" in [p.lower() for p in path.parts]

        if is_control:
            severity = "control"
            intelligibility = None
        else:
            severity = severity_map.get(speaker_id, "unknown")
            intelligibility = intel_map.get(speaker_id)
            if speaker_id not in severity_map:
                unknown_speakers.add(speaker_id)

        lookup_key = build_lookup_key(block, word_code)
        ground_truth_text = word_lookup.get(lookup_key)
        if ground_truth_text is None:
            unknown_words.add(lookup_key)

        rows.append({
            "sample_id": path.stem,
            "audio_path": str(path),
            "speaker_id": speaker_id,
            "block": block,
            "word_code": word_code,
            "word_category": classify_word_code(word_code),
            "mic": mic,
            "ground_truth_text": ground_truth_text if ground_truth_text is not None else "",
            "severity": severity,
            "intelligibility_pct": intelligibility,
            "is_control": is_control,
            "exclude_from_results": speaker_id in EXCLUDE_FROM_RESULTS,
        })

    if unmatched_files:
        print(f"\nWARNING: {len(unmatched_files)} files didn't match the expected naming "
              f"pattern and were skipped. First few: {unmatched_files[:5]}")

    if unknown_speakers:
        print(f"\nWARNING: {len(unknown_speakers)} speaker(s) not found in severity_csv: "
              f"{sorted(unknown_speakers)}. Labeled severity='unknown'.")

    if unknown_words:
        print(f"\nWARNING: {len(unknown_words)} word code(s) not found in word_lookup: "
              f"{sorted(unknown_words)[:10]}{'...' if len(unknown_words) > 10 else ''}. "
              f"ground_truth_text left blank for these rows — check word_code_lookup.csv coverage.")

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    print("\n=== Manifest summary ===")
    print(f"Total samples: {len(df)}")
    print("\nBy severity:")
    print(df["severity"].value_counts())
    print("\nBy word category:")
    print(df["word_category"].value_counts())
    print(f"\nUnique speakers: {df['speaker_id'].nunique()}")
    n_excluded = df["exclude_from_results"].sum()
    if n_excluded:
        print(f"\nNOTE: {n_excluded} rows belong to speaker(s) excluded from "
              f"result redistribution per dataset docs ({EXCLUDE_FROM_RESULTS}). "
              f"Filter these out before publishing findings, even though they're "
              f"kept in the manifest for completeness.")
    n_missing_gt = (df["ground_truth_text"] == "").sum()
    if n_missing_gt:
        print(f"\nNOTE: {n_missing_gt} rows have no ground_truth_text match — "
              f"these will need manual resolution before WER can be computed for them.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--word_lookup", required=True, help="Path to word_code_lookup.csv")
    parser.add_argument("--severity_csv", required=True, help="Path to speaker_severity.csv")
    parser.add_argument("--out_manifest", required=True)
    args = parser.parse_args()

    word_lookup = load_word_lookup(args.word_lookup)
    severity_df = load_severity_map(args.severity_csv)
    df = build_manifest(args.audio_dir, word_lookup, severity_df)

    out_dir = os.path.dirname(args.out_manifest)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_manifest, index=False)
    print(f"\nManifest written to {args.out_manifest}")

    print_summary(df)


if __name__ == "__main__":
    main()
