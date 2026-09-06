"""
run_agent_batch.py — SpeechAgentEval Stage 2

Feeds each ASR transcript (from Stage 1) to a local LLM acting as a voice
agent, and saves:
  1. The agent's text response (for taxonomy/judge scoring later)
  2. Hidden-state activations at the last input token, across several
     candidate layers (for the Phase 2.5 probing script later)

The LLM only ever sees the ASR transcript, never the ground truth — this
mirrors a real deployed voice assistant exactly, and is what lets Stage 4
measure whether ASR errors propagate into bad agent behavior.

Model: any local HF causal LM works, but you need open weights (not an API
model) to extract hidden states. Llama-3.1-8B-Instruct is assumed by
default; swap via --model_name.

Colab/Kaggle usage:
    from google.colab import drive
    drive.mount('/content/drive')

    !huggingface-cli login   # if the model is gated

Usage (as a script):
    python run_agent_batch.py \
        --asr_csv /content/drive/MyDrive/.../ua_speech_asr_full.csv \
        --manifest /content/drive/MyDrive/.../ua_speech_manifest.csv \
        --out_csv /content/drive/MyDrive/.../results/agent_results.csv \
        --activations_dir /content/drive/MyDrive/.../activations \
        --filter_severity "very low" \
        --limit 200

Usage (as importable function, recommended in a notebook cell):
    from run_agent_batch import run_agent_batch
    df = run_agent_batch(
        asr_csv=".../ua_speech_asr_full.csv",
        manifest_path=".../ua_speech_manifest.csv",
        out_csv=".../results/agent_results.csv",
        activations_dir=".../activations",
        filter_severity="very low",
        limit=200,   # start small to sanity check before running all 143k
    )

IMPORTANT: 143,290 samples through an 8B model, even efficiently, is a real
amount of compute. Do NOT run the full dataset in one go on a first attempt.
Start with --limit and/or --filter_severity to validate the prompt and
response quality on a few hundred samples first, inspect them by hand, then
scale up in per-severity batches (matches how you already ran Stage 1).
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None


SYSTEM_PROMPT = """You are a voice assistant. You will receive a transcript of what a user said, \
produced by a speech recognizer. The transcript may contain errors, since the speech \
recognizer is not perfect.

Respond as you would to a real voice command:
- If the transcript is a clear, actionable command or question, respond naturally \
  as if carrying it out (e.g. if it says "turn on the lights", respond "Turning on \
  the lights now.").
- If the transcript is unclear, garbled, or could plausibly be a speech-recognition \
  error, say so and ask for clarification rather than guessing.
- Keep responses short (1 sentence).
"""

# Layers to extract and save activations from — a spread across the model's
# depth, since the "best" layer for the later probe is unknown until you
# run the layer sweep in train_probe.py. This default assumes a 32-layer
# model (Llama-3.1-8B, Mistral-7B). If you switch to a model with a
# different depth (e.g. Qwen2.5-7B has 28 layers), either pass
# layers_to_save explicitly or use evenly_spaced_layers() below.
DEFAULT_LAYERS_TO_SAVE = [8, 12, 16, 20, 24, 28]


def evenly_spaced_layers(n_hidden_layers: int, n_points: int = 6) -> list[int]:
    """
    Return n_points layer indices evenly spaced across a model's depth,
    skipping the very first couple of layers (mostly low-level/lexical,
    rarely useful for a calibration probe) and always including the final
    layer. Use this instead of DEFAULT_LAYERS_TO_SAVE when your model's
    depth differs from 32 layers.

    Example: evenly_spaced_layers(28) -> roughly [5, 10, 14, 19, 23, 28]
    """
    start = max(2, round(n_hidden_layers * 0.15))
    return sorted(set(np.linspace(start, n_hidden_layers, n_points, dtype=int).tolist()))


def build_prompt(tokenizer, transcript: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript if transcript.strip() else "[no speech detected]"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_last_token_activations(hidden_states, layers_to_save: list[int]) -> dict:
    """
    hidden_states: tuple of (num_layers + 1) tensors, each [batch, seq_len, hidden_dim],
    as returned by model(..., output_hidden_states=True). Index 0 is the embedding
    layer output, index i is the output of transformer layer i.
    Returns a dict {"layer_N": np.ndarray} for the last token of each requested layer.
    """
    out = {}
    for layer_idx in layers_to_save:
        if layer_idx >= len(hidden_states):
            continue
        vec = hidden_states[layer_idx][0, -1, :].detach().cpu().float().numpy()
        out[f"layer_{layer_idx}"] = vec
    return out


def stratified_sample(
    df: pd.DataFrame,
    n_per_severity: int = 400,
    severity_col: str = "severity",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample up to n_per_severity rows from each severity bucket, spread as
    evenly as possible across the speakers within that bucket (so one
    speaker's idiosyncrasies don't dominate a bucket's results). If a
    bucket has fewer than n_per_severity rows total, take all of them.
    """
    sampled_parts = []
    for severity, group in df.groupby(severity_col):
        if len(group) <= n_per_severity:
            sampled_parts.append(group)
            continue

        speakers = group["speaker_id"].unique()
        per_speaker = max(1, n_per_severity // len(speakers))
        speaker_samples = []
        for speaker in speakers:
            speaker_group = group[group["speaker_id"] == speaker]
            n = min(per_speaker, len(speaker_group))
            speaker_samples.append(speaker_group.sample(n=n, random_state=seed))
        bucket_sample = pd.concat(speaker_samples)

        # Top up to exactly n_per_severity if per-speaker division left a
        # shortfall (e.g. uneven speaker counts), without duplicating rows.
        if len(bucket_sample) < n_per_severity:
            remaining = group.drop(bucket_sample.index)
            top_up_n = min(n_per_severity - len(bucket_sample), len(remaining))
            if top_up_n > 0:
                bucket_sample = pd.concat([bucket_sample, remaining.sample(n=top_up_n, random_state=seed)])

        sampled_parts.append(bucket_sample)

    result = pd.concat(sampled_parts).reset_index(drop=True)
    print("Stratified sample composition:")
    print(result.groupby(severity_col).size())
    return result


def run_agent_batch(
    asr_csv: str,
    manifest_path: str,
    out_csv: str,
    activations_dir: str,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    filter_severity: str | None = None,
    filter_speaker: str | None = None,
    limit: int | None = None,
    stratified_n_per_severity: int | None = None,
    layers_to_save: list[int] = None,
    checkpoint_every: int = 20,
    resume: bool = True,
    max_new_tokens: int = 60,
) -> pd.DataFrame:
    if torch is None:
        raise ImportError("torch/transformers not installed. Run: pip install torch transformers accelerate --break-system-packages")

    layers_to_save = layers_to_save or DEFAULT_LAYERS_TO_SAVE

    asr_df = pd.read_csv(asr_csv)
    manifest_df = pd.read_csv(manifest_path)
    df = asr_df.merge(manifest_df, on="sample_id", how="left")

    if filter_severity:
        df = df[df["severity"].str.lower() == filter_severity.lower()]
    if filter_speaker:
        df = df[df["speaker_id"].str.upper() == filter_speaker.upper()]
    if stratified_n_per_severity:
        if filter_severity or filter_speaker:
            print("WARNING: stratified_n_per_severity is combined with filter_severity/"
                  "filter_speaker — stratification will only apply within that filtered subset.")
        df = stratified_sample(df, n_per_severity=stratified_n_per_severity)
    if limit:
        df = df.head(limit)

    print(f"Loaded {len(df)} samples to run through the agent")

    results = []
    done_ids = set()
    if resume and os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        results = existing.to_dict("records")
        done_ids = set(existing["sample_id"])
        print(f"Resuming: {len(done_ids)} samples already done, skipping those")

    df = df[~df["sample_id"].isin(done_ids)]
    if len(df) == 0:
        print("Nothing left to run.")
        return pd.DataFrame(results)

    os.makedirs(activations_dir, exist_ok=True)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Validate layers_to_save against this model's actual depth. Different
    # model families have different layer counts (Llama-3.1-8B: 32,
    # Qwen2.5-7B: 28, Mistral-7B: 32, etc.) — hidden_states has
    # num_hidden_layers + 1 entries (index 0 = embeddings). Silently
    # dropping out-of-range layers (the old behavior) means you can lose
    # your deepest probing layer without any warning, so surface it instead.
    n_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    if n_hidden_layers is not None:
        max_valid_layer = n_hidden_layers  # hidden_states index range is [0, n_hidden_layers]
        out_of_range = [l for l in layers_to_save if l > max_valid_layer]
        if out_of_range:
            print(f"WARNING: {model_name} has {n_hidden_layers} layers "
                  f"(valid hidden_states indices 0-{max_valid_layer}). "
                  f"Requested layers {out_of_range} are out of range and will "
                  f"be skipped for every sample. Consider passing "
                  f"layers_to_save scaled to this model, e.g. evenly spaced "
                  f"fractions of {n_hidden_layers}.")
        in_range = [l for l in layers_to_save if l <= max_valid_layer]
        if not in_range:
            raise ValueError(
                f"None of the requested layers_to_save {layers_to_save} are valid "
                f"for {model_name} (max index {max_valid_layer}). Pass explicit "
                f"layers_to_save, e.g. via evenly_spaced_layers({n_hidden_layers})."
            )
        layers_to_save = in_range

    for i, (_, row) in enumerate(df.iterrows()):
        transcript = str(row.get("asr_transcript", "")).strip()
        sample_id = row["sample_id"]
        start = time.time()

        try:
            prompt = build_prompt(tokenizer, transcript)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                # Forward pass to get hidden states at the last input token
                fwd_out = model(**inputs, output_hidden_states=True)
                activations = extract_last_token_activations(fwd_out.hidden_states, layers_to_save)

                act_path = os.path.join(activations_dir, f"{sample_id}.npz")
                np.savez(act_path, **activations)

                # Separate generation call for the actual response text
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response_text = tokenizer.decode(
                    gen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

            error = None
        except Exception as e:
            print(f"  ERROR on {sample_id}: {e}")
            response_text = ""
            act_path = ""
            error = str(e)

        elapsed = time.time() - start
        results.append({
            "sample_id": sample_id,
            "asr_transcript": transcript,
            "agent_response": response_text,
            "activation_path": act_path,
            "agent_seconds": elapsed,
            "error": error,
        })

        if (i + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(out_csv, index=False)
            print(f"  Checkpoint: {i + 1}/{len(df)} done this run "
                  f"({len(results)} total in output file)")

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nDone. {len(results)} total samples in {out_csv}")
    print(f"Activations saved to {activations_dir}")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--filter_severity", default=None)
    parser.add_argument("--filter_speaker", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratified_n_per_severity", type=int, default=None,
                         help="If set, sample up to this many rows per severity bucket (spread across speakers) instead of using the full/filtered set")
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    run_agent_batch(
        asr_csv=args.asr_csv,
        manifest_path=args.manifest,
        out_csv=args.out_csv,
        activations_dir=args.activations_dir,
        model_name=args.model_name,
        filter_severity=args.filter_severity,
        filter_speaker=args.filter_speaker,
        limit=args.limit,
        stratified_n_per_severity=args.stratified_n_per_severity,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
