"""
local_judge.py — SpeechAgentEval, LLM-judge without any API key

Uses a local open-weight model (same kind you already run for Stage 2 —
Qwen2.5-7B-Instruct, Llama, Mistral, etc.) as the judge instead of calling
the Anthropic API. No key, no billing, no network dependency beyond the
one-time model download.

Trade-off to be upfront about: a 7-8B local model is a weaker judge than
Claude/GPT-4-class models — expect somewhat noisier classifications,
especially on subtle cases (like the "commits to wrong content as fact"
pattern you calibrated on manually). This is exactly why the validation
step (comparing against your manual labels) matters MORE here, not less —
run it, check kappa, and if it's too weak on a specific category, that's
useful information (either fix the prompt with more explicit few-shot
examples, or fall back to manual labels only for the final analysis, which
is a legitimate methodology choice to state plainly in your report).

Usage (as importable functions, in a notebook cell):
    from local_judge import run_local_judge, validate_against_manual

    labeled_df = run_local_judge(
        manual_csv=".../manual_labels_todo.csv",   # or your full agent output
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    validate_against_manual(labeled_df)   # prints kappa, confusion matrices
"""

import json
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None

from judge_prompt import SYSTEM_PROMPT, build_user_prompt


_model = None
_tokenizer = None


def load_judge_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", load_in_4bit: bool = True):
    """
    Loads once and caches — call run_local_judge() repeatedly without reloading.

    load_in_4bit=True (default) uses bitsandbytes 4-bit quantization, cutting
    memory footprint roughly 3-4x (e.g. ~14GB -> ~4-5GB for a 7B model). This
    is necessary on single ~15GB GPUs (Kaggle's typical T4) where a 7B model
    at float16 alone consumes nearly the entire GPU, leaving no room for
    activations/KV cache during generation. Quality impact is typically small
    for a classification task like this judge; if you have more GPU memory
    available (e.g. an A100), pass load_in_4bit=False for full precision.
    """
    global _model, _tokenizer
    if torch is None:
        raise ImportError("torch/transformers not installed. Run: pip install torch transformers accelerate --break-system-packages")
    if _model is None:
        print(f"Loading judge model: {model_name} (4-bit: {load_in_4bit})")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "bitsandbytes not installed. Run: pip install bitsandbytes --break-system-packages"
                )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto"
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
        _model.eval()
    return _model, _tokenizer


def parse_judge_json(text: str) -> dict:
    """
    Local models are less reliable than Claude/GPT-4 at strict JSON-only
    output — they sometimes add preamble, markdown fences, or trailing
    commentary. This extracts the first {...} block rather than assuming
    the whole response is clean JSON.
    """
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"b_label": "ERROR", "c_label": "ERROR", "confidence": "low",
                 "justification": f"No JSON found in output: {text[:200]}"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"b_label": "ERROR", "c_label": "ERROR", "confidence": "low",
                 "justification": f"JSON parse error: {e}. Raw: {text[:200]}"}


def call_local_judge(ground_truth: str, asr_transcript: str, agent_response: str,
                      model_name: str = "Qwen/Qwen2.5-7B-Instruct", max_new_tokens: int = 150,
                      load_in_4bit: bool = True) -> dict:
    model, tokenizer = load_judge_model(model_name, load_in_4bit=load_in_4bit)
    user_prompt = build_user_prompt(ground_truth, asr_transcript, agent_response)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_judge_json(response_text)


def run_local_judge(
    manual_csv: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    checkpoint_every: int = 10,
    out_csv: str | None = None,
    load_in_4bit: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(manual_csv)
    required = {"sample_id", "ground_truth", "asr_transcript", "agent_response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual_csv missing required columns: {missing}")

    results = []
    for i, row in df.iterrows():
        label = call_local_judge(
            str(row["ground_truth"]), str(row["asr_transcript"]), str(row["agent_response"]),
            model_name=model_name, load_in_4bit=load_in_4bit,
        )
        label["sample_id"] = row["sample_id"]
        results.append(label)

        if (i + 1) % checkpoint_every == 0:
            print(f"  {i + 1}/{len(df)} judged...")
            if out_csv:
                pd.DataFrame(results).to_csv(out_csv, index=False)

    judge_df = pd.DataFrame(results)
    merged = df.merge(judge_df, on="sample_id", suffixes=("", "_judge"))

    if out_csv:
        merged.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")

    n_errors = (judge_df["b_label"] == "ERROR").sum()
    if n_errors:
        print(f"WARNING: {n_errors}/{len(judge_df)} judge calls failed to parse. "
              f"These local-model failures are usually fixable by lowering "
              f"max_new_tokens issues or tightening the prompt — inspect the "
              f"'justification' field on ERROR rows for the raw output.")

    return merged


def validate_against_manual(merged_df: pd.DataFrame) -> None:
    """Same reporting as validate_judge.ipynb — kappa, accuracy, confusion matrices."""
    if "manual_b_label" not in merged_df.columns or "manual_c_label" not in merged_df.columns:
        print("No manual_b_label/manual_c_label columns found — pass a df that includes your manual labels.")
        return

    valid = merged_df[(merged_df["b_label"] != "ERROR") & (merged_df["c_label"] != "ERROR")]
    print(f"{len(valid)}/{len(merged_df)} judge calls usable for validation")

    b_acc = accuracy_score(valid["manual_b_label"], valid["b_label"])
    c_acc = accuracy_score(valid["manual_c_label"], valid["c_label"])
    b_kappa = cohen_kappa_score(valid["manual_b_label"], valid["b_label"])
    c_kappa = cohen_kappa_score(valid["manual_c_label"], valid["c_label"])

    print(f"\nAxis B (propagation) — accuracy: {b_acc:.2%}, Cohen's kappa: {b_kappa:.3f}")
    print(f"Axis C (agent behavior) — accuracy: {c_acc:.2%}, Cohen's kappa: {c_kappa:.3f}")

    print("\nInterpretation: >0.6 substantial, 0.4-0.6 moderate (refine prompt/add "
          "few-shot examples and re-run), <0.4 weak (don't trust yet).")

    for axis, manual_col, judge_col in [("B", "manual_b_label", "b_label"), ("C", "manual_c_label", "c_label")]:
        labels = sorted(set(valid[manual_col]) | set(valid[judge_col]))
        print(f"\nAxis {axis} confusion matrix:")
        print(pd.DataFrame(
            confusion_matrix(valid[manual_col], valid[judge_col], labels=labels),
            index=labels, columns=labels,
        ))

    disagreements = valid[
        (valid["manual_b_label"] != valid["b_label"]) | (valid["manual_c_label"] != valid["c_label"])
    ]
    if len(disagreements):
        print(f"\n{len(disagreements)} disagreement rows — inspect these to decide whether to "
              f"refine the prompt (add these as few-shot examples) or trust manual labels only:")
        print(disagreements[["sample_id", "ground_truth", "asr_transcript", "agent_response",
                              "manual_b_label", "b_label", "manual_c_label", "c_label"]].to_string())
