# SpeechAgentEval

[![Live Demo](https://img.shields.io/badge/Live%20Demo-speechagenteval.streamlit.app-blue)](https://speechagenteval.streamlit.app)

**An evaluation harness for voice-agent pipelines (ASR → LLM), stress-tested against real dysarthric speech.**

Most evaluations of voice assistants stop at Word Error Rate. This project asks the more important question: **when speech recognition gets something wrong, does the error get caught, get harmlessly absorbed, or silently propagate into a confidently wrong response?** And can we tell, from the model's own internal activations, whether it "knew" something was off even when its output gave no sign of it?

Built on [UA-Speech](http://www.isle.illinois.edu/sst/data/UASpeech/), a real corpus of dysarthric speech spanning a range of severities, with a full pipeline from raw audio through ASR, an LLM agent, an automated failure taxonomy, and an interpretability probe on the agent's hidden states.

---

## Headline findings

**1. Propagation risk is non-monotonic with speech severity — it peaks in the middle, not at the extreme.**

| Severity | ASR error rate | Silent propagation (B1) | False confidence (C1) |
|---|---|---|---|
| control | 35% | 7% | 7% |
| high | 26% | 4% | 4% |
| **mid** | 66% | **22%** | **22%** |
| **low** | 78% | **29%** | **29%** |
| very low | 91% | 12% | 12% |

At moderate degradation, ASR errors are frequent *and* still plausible-sounding enough that both Whisper and the downstream LLM agent get fooled into treating fabricated content as real. At the most extreme degradation, failure becomes obvious enough that the pipeline more often catches itself (correct-rejection rate jumps to 66%).

**2. Whisper doesn't just mis-transcribe dysarthric speech — it hallucinates fluent sentences from single words**, especially at moderate severity, and its own confidence score becomes an unreliable (sometimes actively misleading) signal exactly where it's needed most.

**3. A linear probe on the LLM agent's activations detects a decodable "ambiguity" signal in 81% of silent-propagation cases (60/74, 5-fold cross-validated, AUROC 0.805)** — even though the model's *output* showed no sign of uncertainty in those cases. This suggests the relevant information is often present internally but not expressed, a distinct and more concerning failure mode than simple ignorance. *(Correlational — see Limitations.)*

---

## Pipeline overview

```
UA-Speech audio ──▶ Whisper (ASR) ──▶ Local LLM agent ──▶ Response + activations
                         │                    │
                     WER/CER              hidden states
                         │                    │
                         └──────┬─────────────┘
                                ▼
                    LLM-judge (taxonomy labels)
                     validated against manual labels
                                │
                                ▼
                  Failure taxonomy + probe analysis
                                │
                                ▼
                         Streamlit dashboard
```

### Failure taxonomy

| Axis | Label | Meaning |
|---|---|---|
| B (propagation) | B1 | ASR error changed meaning; agent acted on it confidently, no hedge |
| | B2 | ASR error present, but response harmless anyway |
| | B3 | Agent noticed something was off, asked for clarification |
| | not_applicable | ASR was essentially correct |
| C (behavior) | C1 | False confidence — confident answer to genuinely ambiguous input |
| | C2 | No repair mechanism at all |
| | C3 | Overcorrection — asked for clarification when input was clear |
| | C4 | Appropriate behavior |

---

## Methodology notes (read before trusting the numbers)

- **Judge validation**: taxonomy labels are assigned by a locally-run open-weight LLM (Qwen2.5-7B, 4-bit quantized — no API costs), validated against 50 hand-labeled examples. **Axis B: Cohen's κ = 0.541 (moderate). Axis C: κ = 0.707 (substantial).** The judge underwent 4 rounds of targeted prompt refinement; results oscillated between rounds before converging, and the final version was locked in deliberately rather than over-fit further to the validation set.
- **Isolated-word confound**: most UA-Speech test items are single words without command context, which structurally elevates clarification-seeking behavior *independent of ASR correctness* — even clean control speech gets a ~65-70% clarification rate. This is reported as a methodology finding, not hidden; taxonomy results should be read with this in mind rather than assumed to reflect ASR-error detection alone.
- **Probe caveat**: the probe shows that ambiguity-relevant information is *linearly decodable* from activations. It does **not** show this information causally drives the model's output. Establishing causation would require an intervention study (e.g. activation patching) — a natural extension beyond this project's current scope.
- **Sample scale**: Stage 1 (ASR) runs on the full 143,290-sample UA-Speech corpus. Stages 2 onward (LLM agent, taxonomy, probe) run on a stratified 500-sample subset (100 per severity bucket, balanced across speakers) for compute tractability — this is a deliberate design choice, not a limitation of convenience.

---

## Repo structure

```
data_prep/       UA-Speech manifest construction (real speaker severity + word-code ground truth)
pipeline/        Stage 1 (Whisper ASR) and Stage 2 (LLM agent + activation extraction)
eval/            Failure taxonomy, LLM-judge (+ validation), probe training
dashboard/       Streamlit app (deployed version)
results/         Final CSVs/plots backing the headline findings
```

## Running it

Each stage is a standalone, resumable script — see docstrings in each file for full usage. Rough order:

```bash
# 1. Build the manifest from raw UA-Speech audio + provided metadata
python data_prep/build_ua_speech_manifest.py --audio_dir <path> --word_lookup data_prep/word_code_lookup.csv --severity_csv data_prep/speaker_severity.csv --out_manifest manifest.csv

# 2. Run ASR
python pipeline/run_asr_batch.py --manifest manifest.csv --out_csv asr_results.csv

# 3. Run the LLM agent (stratified sample) + save activations
python pipeline/run_agent_batch.py --asr_csv asr_results.csv --manifest manifest.csv --out_csv agent_results.csv --activations_dir activations/ --stratified_n_per_severity 100

# 4. Score with the validated local judge
python eval/prep_for_judge_scoring.py --agent_csv agent_results.csv --manifest manifest.csv --out_csv for_judge.csv
python -c "from eval.local_judge import run_local_judge; run_local_judge('for_judge.csv', out_csv='judged.csv')"

# 5. Train the probe + get the suppressed-signal cross-tabulation
python eval/train_probe.py --manifest judged.csv --out_dir probe_results
python eval/full_dataset_crosstab.py --manifest judged.csv --layer layer_12 --out_dir full_probe_results

# 6. Explore results
cd dashboard && pip install -r requirements.txt && streamlit run app.py
```

## Stack

Whisper (ASR) · Qwen2.5-7B-Instruct (agent + local judge, 4-bit quantized) · scikit-learn (probe) · Streamlit + Plotly (dashboard) · pandas/numpy throughout

## Limitations & next steps

- Probe shows correlation, not causation — activation patching is the natural next step
- Judge Axis B agreement is moderate, not substantial — worth further refinement or a larger manually-labeled set
- Only a subset of UA-Speech speakers per severity bucket (3-13 depending on bucket) — more speakers would tighten confidence intervals
- The isolated-word confound could be resolved by extending to conversational (not single-word) test inputs

---

*Built by Avantika Agarwal. Part of a broader focus on building AI systems that can be evaluated and trusted, not just measured on accuracy.*
