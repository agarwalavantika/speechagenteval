"""
run_asr_batch.py — SpeechAgentEval Stage 1, scaled across speakers/severities

Runs Whisper over any manifest (or a filtered subset — one speaker, one
severity bucket, control-only, etc.) and produces a results CSV with the
same schema as your existing asr_test.csv / asr_test_control.csv files:

    sample_id, asr_transcript, asr_transcript_normalized, asr_confidence,
    asr_seconds, wer, cer

Designed for Colab/Kaggle: checkpoints periodically to the output CSV, and
skips samples already present in an existing output file, so a disconnect
just means re-running the same command to resume.

Usage (as a script):
    python run_asr_batch.py \
        --manifest /content/drive/MyDrive/SpeechAgentEval/manifests/ua_speech_manifest.csv \
        --out_csv /content/drive/MyDrive/SpeechAgentEval/results/asr_M05.csv \
        --filter_speaker M05 \
        --model_size small

Usage (as importable functions, recommended in a notebook cell):
    from run_asr_batch import run_asr_batch
    df = run_asr_batch(
        manifest_path=".../ua_speech_manifest.csv",
        out_csv=".../results/asr_M05.csv",
        filter_speaker="M05",
        model_size="small",
    )
"""

import argparse
import os
import re
import time

import pandas as pd
import jiwer

try:
    import whisper
except ImportError:
    whisper = None


_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

_CER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    if not reference or not reference.strip():
        return float("nan"), float("nan")
    wer = jiwer.wer(reference, hypothesis, reference_transform=_WER_TRANSFORM, hypothesis_transform=_WER_TRANSFORM)
    cer = jiwer.cer(reference, hypothesis, reference_transform=_CER_TRANSFORM, hypothesis_transform=_CER_TRANSFORM)
    return wer, cer


def load_manifest_subset(
    manifest_path: str,
    filter_speaker: str | None = None,
    filter_severity: str | None = None,
    control_only: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    if filter_speaker:
        df = df[df["speaker_id"].str.upper() == filter_speaker.upper()]
    if filter_severity:
        df = df[df["severity"].str.lower() == filter_severity.lower()]
    if control_only:
        df = df[df["is_control"] == True]

    if "ground_truth_text" in df.columns:
        before = len(df)
        df = df[df["ground_truth_text"].notna() & (df["ground_truth_text"] != "")]
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with no ground_truth_text (can't compute WER for these)")

    return df


def run_asr_batch(
    manifest_path: str,
    out_csv: str,
    filter_speaker: str | None = None,
    filter_severity: str | None = None,
    control_only: bool = False,
    model_size: str = "small",
    checkpoint_every: int = 20,
    resume: bool = True,
) -> pd.DataFrame:
    if whisper is None:
        raise ImportError("openai-whisper is not installed. Run: pip install openai-whisper --break-system-packages")

    df = load_manifest_subset(manifest_path, filter_speaker, filter_severity, control_only)
    print(f"Loaded {len(df)} samples to transcribe")

    results = []
    done_ids = set()
    if resume and os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        results = existing.to_dict("records")
        done_ids = set(existing["sample_id"])
        print(f"Resuming: {len(done_ids)} samples already done, skipping those")

    df = df[~df["sample_id"].isin(done_ids)]
    if len(df) == 0:
        print("Nothing left to transcribe.")
        return pd.DataFrame(results)

    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows()):
        start = time.time()
        try:
            result = model.transcribe(row["audio_path"])
            transcript = result["text"].strip()
            avg_logprob = (
                sum(seg.get("avg_logprob", 0.0) for seg in result["segments"]) / len(result["segments"])
                if result["segments"] else None
            )
        except Exception as e:
            print(f"  ERROR on {row['sample_id']}: {e}")
            transcript = ""
            avg_logprob = None

        elapsed = time.time() - start
        normalized = normalize_text(transcript)
        ground_truth = str(row.get("ground_truth_text", "")).strip()
        wer, cer = compute_wer_cer(ground_truth, normalized)

        results.append({
            "sample_id": row["sample_id"],
            "asr_transcript": transcript,
            "asr_transcript_normalized": normalized,
            "asr_confidence": avg_logprob,
            "asr_seconds": elapsed,
            "wer": wer,
            "cer": cer,
        })

        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(out_csv, index=False)
            print(f"  Checkpoint: {i + 1}/{len(df)} done this run "
                  f"({len(results)} total in output file)")

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nDone. {len(results)} total samples in {out_csv}")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--filter_speaker", default=None)
    parser.add_argument("--filter_severity", default=None,
                         help="e.g. 'very low', 'low', 'mid', 'high', 'control'")
    parser.add_argument("--control_only", action="store_true")
    parser.add_argument("--model_size", default="small", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    run_asr_batch(
        manifest_path=args.manifest,
        out_csv=args.out_csv,
        filter_speaker=args.filter_speaker,
        filter_severity=args.filter_severity,
        control_only=args.control_only,
        model_size=args.model_size,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
