"""
app.py — SpeechAgentEval Dashboard

Streamlit dashboard presenting the project's findings:
1. Overview / headline numbers
2. Severity-gradient ASR results (Stage 1: WER, hallucination rate)
3. Taxonomy breakdown by severity (Stage 4: propagation, false confidence)
4. Probe / suppressed-signal results (Phase 2.5: interpretability)
5. Browsable individual pipeline traces, filterable

Expected data files (place alongside this app.py, or adjust DATA_DIR below):
    - judge_scores_full_500.csv       (main taxonomy-labeled dataset)
    - full_cross_tabulation.csv       (probe cross-tab summary)
    - full_suppressed_signal_cases.csv (probe results merged with taxonomy)
    - full_reliability_diagram.png    (probe calibration plot)

Run locally:
    streamlit run app.py

Deploy: push this file + requirements.txt + the data CSVs/PNG to a Hugging
Face Space (Streamlit SDK) — no code changes needed for that.
"""

import os

import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = os.environ.get("SPEECHAGENTEVAL_DATA_DIR", ".")

st.set_page_config(page_title="SpeechAgentEval", layout="wide", page_icon="🎙️")

SEVERITY_ORDER = ["control", "high", "mid", "low", "very low"]


@st.cache_data
def load_data():
    paths = {
        "main": os.path.join(DATA_DIR, "judge_scores_full_500.csv"),
        "crosstab": os.path.join(DATA_DIR, "full_cross_tabulation.csv"),
        "suppressed": os.path.join(DATA_DIR, "full_suppressed_signal_cases.csv"),
    }
    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, missing

    main_df = pd.read_csv(paths["main"])
    crosstab_df = pd.read_csv(paths["crosstab"])
    suppressed_df = pd.read_csv(paths["suppressed"])
    return {"main": main_df, "crosstab": crosstab_df, "suppressed": suppressed_df}, []


data, missing_files = load_data()

if data is None:
    st.error(
        f"Missing data files: {missing_files}. Place these CSVs in the same "
        f"directory as app.py (or set SPEECHAGENTEVAL_DATA_DIR)."
    )
    st.stop()

main_df = data["main"]
crosstab_df = data["crosstab"]
suppressed_df = data["suppressed"]

# ---------------------------------------------------------------------------
st.title("🎙️ SpeechAgentEval")
st.caption(
    "An evaluation harness for voice-agent pipelines (ASR → LLM), stress-tested "
    "against dysarthric speech (UA-Speech). Measures not just transcription "
    "accuracy, but where and how ASR errors propagate into agent behavior."
)

tab_overview, tab_severity, tab_taxonomy, tab_probe, tab_browse = st.tabs(
    ["Overview", "Severity Gradient", "Taxonomy Breakdown", "Probe / Interpretability", "Browse Traces"]
)

# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Headline findings")

    n_total = len(main_df)
    n_b1 = (main_df["b_label"] == "B1").sum()
    n_b3 = (main_df["b_label"] == "B3").sum()
    n_c1 = (main_df["c_label"] == "C1").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples evaluated", n_total)
    col2.metric("Silent propagation (B1)", f"{n_b1} ({n_b1/n_total:.0%})")
    col3.metric("Correct rejection (B3)", f"{n_b3} ({n_b3/n_total:.0%})")
    col4.metric("False confidence (C1)", f"{n_c1} ({n_c1/n_total:.0%})")

    st.markdown("---")
    st.markdown(
        """
**Key finding — propagation risk is non-monotonic with severity.**
The most dangerous zone isn't the most severely degraded speech — it's *moderate*
degradation (mid/low severity), where ASR errors are frequent but still
plausible-looking enough that both Whisper and the downstream LLM agent are
fooled into treating fabricated content as real. At the most extreme severity,
failure becomes obvious enough that the pipeline more often catches itself.

**Suppressed-signal finding.** Using a linear probe on the LLM's internal
activations, **81% of silent-propagation (B1) cases** show a decodable signal
that the input was ambiguous — even though the model's *output* gave no
indication of uncertainty. This suggests the model's internal state often
"knows" more than it says, a distinct and more concerning failure mode than
simple ignorance. *(Correlational, not causal — see Probe tab for caveats.)*
        """
    )

# ---------------------------------------------------------------------------
with tab_severity:
    st.subheader("ASR severity gradient (Stage 1)")

    if "severity" in main_df.columns and "b_label" in main_df.columns:
        sev_summary = (
            main_df.groupby("severity")
            .agg(
                n=("sample_id", "count"),
                pct_B1=("b_label", lambda s: (s == "B1").mean()),
                pct_B3=("b_label", lambda s: (s == "B3").mean()),
                pct_C1=("c_label", lambda s: (s == "C1").mean()),
            )
            .reindex(SEVERITY_ORDER)
            .reset_index()
        )

        fig = px.bar(
            sev_summary,
            x="severity",
            y=["pct_B1", "pct_C1"],
            barmode="group",
            labels={"value": "Rate", "severity": "Severity", "variable": "Metric"},
            title="Silent propagation (B1) and false confidence (C1) rate by severity",
            category_orders={"severity": SEVERITY_ORDER},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sev_summary.style.format({"pct_B1": "{:.1%}", "pct_B3": "{:.1%}", "pct_C1": "{:.1%}"}))

        st.caption(
            "Note: many test items are isolated single words without command context, "
            "which structurally elevates clarification-seeking behavior independent of "
            "ASR correctness — see methodology notes for details."
        )
    else:
        st.warning("severity or b_label column not found in dataset.")

# ---------------------------------------------------------------------------
with tab_taxonomy:
    st.subheader("Failure taxonomy breakdown")

    col1, col2 = st.columns(2)
    with col1:
        b_counts = main_df["b_label"].value_counts().reset_index()
        b_counts.columns = ["b_label", "count"]
        fig_b = px.pie(b_counts, names="b_label", values="count", title="Axis B — ASR propagation outcome")
        st.plotly_chart(fig_b, use_container_width=True)

    with col2:
        c_counts = main_df["c_label"].value_counts().reset_index()
        c_counts.columns = ["c_label", "count"]
        fig_c = px.pie(c_counts, names="c_label", values="count", title="Axis C — Agent behavior")
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
**Taxonomy definitions**
- **B1 — Silent propagation**: ASR error changed meaning; agent acted on it confidently, no hedge.
- **B2 — Graceful degradation**: ASR error present, but response was harmless anyway.
- **B3 — Correct rejection**: agent noticed something was off and asked for clarification.
- **C1 — False confidence**: agent answered confidently despite genuinely ambiguous input.
- **C4 — Appropriate**: agent's behavior matched the situation.

Labels were assigned by a locally-run LLM judge (Qwen2.5-7B), validated against
50 hand-labeled examples: **Axis B kappa = 0.541 (moderate)**, **Axis C kappa =
0.707 (substantial)** agreement with human labels.
        """
    )

# ---------------------------------------------------------------------------
with tab_probe:
    st.subheader("Interpretability: does the model's activations \"know\" more than it says?")

    diagram_path = os.path.join(DATA_DIR, "full_reliability_diagram.png")
    col1, col2 = st.columns([1, 1])
    with col1:
        if os.path.exists(diagram_path):
            st.image(diagram_path, caption="Probe calibration (reliability diagram)")
        else:
            st.info("Reliability diagram image not found in data directory.")

    with col2:
        st.markdown("**Cross-tabulation: probe signal vs. expressed confidence**")
        st.dataframe(crosstab_df)

        n_b1_total = (main_df["b_label"] == "B1").sum()
        n_suppressed = len(suppressed_df[suppressed_df["output_was_confident"] == True]) if "output_was_confident" in suppressed_df.columns else len(suppressed_df)
        if n_b1_total:
            st.metric(
                "Suppressed-signal rate (of B1 cases)",
                f"{n_suppressed}/{n_b1_total} ({n_suppressed/n_b1_total:.1%})",
            )

    st.markdown(
        """
**What this shows**: a logistic-regression probe was trained on the LLM's hidden-state
activations to predict whether an input was genuinely ambiguous (ground truth from the
taxonomy labels), evaluated with 5-fold speaker-grouped cross-validation (AUROC ≈ 0.80).

For samples where the agent's output showed **silent propagation (B1)** — i.e. it
committed confidently to a wrong interpretation — the probe still detected an internal
ambiguity signal in the large majority of cases. This means the relevant information
was often linearly present in the model's activations even when absent from its output.

**Important caveat**: this shows the signal is *decodable*, not that it *causally
drives* the model's behavior. A model could represent uncertainty internally without
that representation influencing what it ultimately says. Establishing causation would
require an intervention study (e.g. activation patching) — a natural next step beyond
this project's current scope.
        """
    )

# ---------------------------------------------------------------------------
with tab_browse:
    st.subheader("Browse individual pipeline traces")

    col1, col2, col3 = st.columns(3)
    with col1:
        severity_filter = st.multiselect(
            "Severity", options=sorted(main_df["severity"].dropna().unique()), default=None
        )
    with col2:
        b_filter = st.multiselect(
            "B label (ASR propagation)", options=sorted(main_df["b_label"].dropna().unique()), default=None
        )
    with col3:
        c_filter = st.multiselect(
            "C label (agent behavior)", options=sorted(main_df["c_label"].dropna().unique()), default=None
        )

    filtered = main_df.copy()
    if severity_filter:
        filtered = filtered[filtered["severity"].isin(severity_filter)]
    if b_filter:
        filtered = filtered[filtered["b_label"].isin(b_filter)]
    if c_filter:
        filtered = filtered[filtered["c_label"].isin(c_filter)]

    st.caption(f"Showing {len(filtered)} of {len(main_df)} samples")

    display_cols = [c for c in ["sample_id", "severity", "ground_truth", "asr_transcript",
                                 "agent_response", "b_label", "c_label", "justification"]
                     if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, height=500)
