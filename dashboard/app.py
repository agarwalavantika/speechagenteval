"""
app.py — SpeechAgentEval Dashboard
===================================
Streamlit dashboard presenting findings from SpeechAgentEval:
    1. Overview / headline numbers
    2. Severity-gradient ASR results (WER, hallucination, propagation)
    3. Taxonomy breakdown by severity
    4. Probe / suppressed-signal results (mechanistic interpretability)
    5. Browsable individual pipeline traces, filterable

Expected data files (place alongside this app.py, or set SPEECHAGENTEVAL_DATA_DIR):
    - judge_scores_full_500.csv        (main taxonomy-labeled dataset)
    - full_cross_tabulation.csv        (probe cross-tab summary)
    - full_suppressed_signal_cases.csv (probe results merged with taxonomy)
    - full_reliability_diagram.png     (probe calibration plot)

Deploy: push this file + requirements.txt + data CSVs/PNG to a HuggingFace Space
(Streamlit SDK) — no code changes needed.
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Config ───────────────────────────────────────────────────────────────────
# Use the directory where app.py lives — works locally AND on Streamlit Cloud
DATA_DIR = os.environ.get(
    "SPEECHAGENTEVAL_DATA_DIR",
    os.path.dirname(os.path.abspath(__file__))
)
SEVERITY_ORDER = ["control", "high", "mid", "low", "very low"]

COLORS = {
    "B1": "#E53935",   # red — silent propagation (bad)
    "B2": "#43A047",   # green — graceful degradation (ok)
    "B3": "#1E88E5",   # blue — correct rejection (good)
    "not_applicable": "#90A4AE",
    "C1": "#E53935",
    "C2": "#FB8C00",
    "C3": "#FDD835",
    "C4": "#43A047",
}

st.set_page_config(
    page_title="SpeechAgentEval",
    layout="wide",
    page_icon="🎙️",
    menu_items={
        "Get Help": "https://github.com/agarwalavantika/speechagenteval",
        "Report a bug": "https://github.com/agarwalavantika/speechagenteval/issues",
        "About": "SpeechAgentEval — evaluation harness for voice-agent pipelines under dysarthric speech."
    }
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.safety-banner {
    background: linear-gradient(90deg, #1B3A5C 0%, #2563A8 100%);
    padding: 14px 20px;
    border-radius: 8px;
    color: white;
    margin-bottom: 18px;
    font-size: 0.92rem;
    line-height: 1.5;
}
.safety-banner b { color: #90CAF9; }
.finding-box {
    background: #F0F4FF;
    border-left: 4px solid #2563A8;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 10px 0;
    font-size: 0.9rem;
}
.caveat-box {
    background: #FFF8E1;
    border-left: 4px solid #FFA000;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 10px 0;
    font-size: 0.88rem;
}
.metric-label { font-size: 0.78rem; color: #555; font-weight: 600; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = {
        "main":       os.path.join(DATA_DIR, "judge_scores_full_500.csv"),
        "crosstab":   os.path.join(DATA_DIR, "full_cross_tabulation.csv"),
        "suppressed": os.path.join(DATA_DIR, "full_suppressed_signal_cases.csv"),
    }
    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, missing
    return {
        "main":       pd.read_csv(paths["main"]),
        "crosstab":   pd.read_csv(paths["crosstab"]),
        "suppressed": pd.read_csv(paths["suppressed"]),
    }, []

data, missing_files = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=40)
    st.markdown("## 🎙️ SpeechAgentEval")
    st.markdown(
        "Evaluation harness stress-testing voice-agent pipelines "
        "against **real dysarthric speech** (UA-Speech corpus)."
    )
    st.markdown("---")
    st.markdown("### Pipeline")
    st.code("UA-Speech audio\n  → Whisper (ASR)\n  → Qwen2.5-7B agent\n  → LLM-judge taxonomy\n  → Probe (activations)", language=None)
    st.markdown("---")
    st.markdown("### Key numbers")
    st.markdown("- **143,290** total audio samples (Stage 1)")
    st.markdown("- **500** stratified samples (Stages 2–4)")
    st.markdown("- **AUROC 0.805** — probe on agent activations")
    st.markdown("- **κ = 0.707** — judge validation (Axis C)")
    st.markdown("---")
    st.markdown("### Links")
    st.markdown("📁 [GitHub repo](https://github.com/agarwalavantika/speechagenteval)")
    st.markdown("👤 [Avantika Agarwal](https://agarwalavantika.github.io)")
    st.markdown("📄 [Portfolio](https://agarwalavantika.github.io)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="safety-banner">
<b>SpeechAgentEval</b> &nbsp;·&nbsp; An evaluation harness for voice-agent pipelines (ASR → LLM),
stress-tested against real dysarthric speech (UA-Speech). Asks not just <i>what</i> goes wrong, 
but <i>whether the model internally knew something was wrong</i> — even when its output gave no sign of it.
&nbsp;&nbsp;<a href="https://github.com/agarwalavantika/speechagenteval" style="color:#90CAF9;">
GitHub ↗</a>
</div>
""", unsafe_allow_html=True)

# ── Data check ────────────────────────────────────────────────────────────────
if data is None:
    st.error(f"⚠️ Missing data files: **{missing_files}**")
    st.info(
        "Place the following files in the same directory as `app.py` "
        "(or set the `SPEECHAGENTEVAL_DATA_DIR` environment variable):\n\n"
        "- `judge_scores_full_500.csv`\n"
        "- `full_cross_tabulation.csv`\n"
        "- `full_suppressed_signal_cases.csv`\n"
        "- `full_reliability_diagram.png`\n\n"
        "See [the repository](https://github.com/agarwalavantika/speechagenteval) "
        "for instructions on generating these files."
    )
    st.stop()

main_df       = data["main"]
crosstab_df   = data["crosstab"]
suppressed_df = data["suppressed"]

# Ensure severity is categorical in correct order
if "severity" in main_df.columns:
    main_df["severity"] = pd.Categorical(
        main_df["severity"], categories=SEVERITY_ORDER, ordered=True
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_severity, tab_taxonomy, tab_probe, tab_browse = st.tabs([
    "📊 Overview",
    "📈 Severity Gradient",
    "🗂️ Taxonomy Breakdown",
    "🔬 Probe / Interpretability",
    "🔍 Browse Traces",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Headline findings")

    n_total = len(main_df)
    n_b1    = (main_df["b_label"] == "B1").sum()
    n_b3    = (main_df["b_label"] == "B3").sum()
    n_c1    = (main_df["c_label"] == "C1").sum()
    n_c4    = (main_df["c_label"] == "C4").sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Samples evaluated", f"{n_total:,}")
    col2.metric("Silent propagation (B1)", f"{n_b1} ({n_b1/n_total:.0%})",
                delta=f"Risk peaks at mid/low severity", delta_color="inverse")
    col3.metric("Correct rejection (B3)", f"{n_b3} ({n_b3/n_total:.0%})",
                delta="Self-correction", delta_color="normal")
    col4.metric("False confidence (C1)", f"{n_c1} ({n_c1/n_total:.0%})",
                delta_color="inverse")
    col5.metric("Appropriate behavior (C4)", f"{n_c4} ({n_c4/n_total:.0%})",
                delta_color="normal")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
<div class="finding-box">
<b>Finding 1 — Non-monotonic propagation risk</b><br>
The most dangerous failure zone isn't the most severely degraded speech — it's
<b>moderate degradation (mid/low severity)</b>, where ASR errors are frequent
but still plausible-sounding enough to fool both Whisper and the downstream agent.
At extreme severity, failure becomes obvious and the pipeline more often catches itself.
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="finding-box">
<b>Finding 2 — Suppressed uncertainty signal</b><br>
A linear probe on the LLM agent's internal activations detects a decodable 
"ambiguity" signal in <b>81% of silent-propagation (B1) cases</b> (AUROC 0.805),
even when the model's output expressed no uncertainty.
The information is <i>present internally but not expressed</i> — a distinct failure
mode from simple ignorance, directly relevant to oversight and deceptive confidence.
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="finding-box">
<b>Finding 3 — Evaluation-metric failure</b><br>
Whisper's confidence scores are <b>actively misleading at moderate degradation</b>:
highest false-confidence rate exactly where reliable uncertainty signalling is
most needed.
</div>
""", unsafe_allow_html=True)

    with col_r:
        # Propagation table
        table_data = {
            "Severity":              ["control", "high", "mid", "low", "very low"],
            "ASR error rate":        ["35%", "26%", "66%", "78%", "91%"],
            "Silent propagation (B1)": ["7%", "4%", "22%", "29%", "12%"],
            "False confidence (C1)": ["7%", "4%", "22%", "29%", "12%"],
        }
        df_table = pd.DataFrame(table_data)
        st.markdown("**Propagation risk by severity bucket**")
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        st.markdown("""
<div class="caveat-box">
<b>Methodology note:</b> Many UA-Speech test items are isolated single words
without command context, which structurally elevates clarification-seeking
behavior (B3) independent of ASR correctness. Results should be read with
this in mind. See the repository for full methodology notes.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEVERITY GRADIENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_severity:
    st.subheader("ASR severity gradient — how failure modes shift across severity")
    st.caption(
        "Each severity bucket contains 100 stratified samples from UA-Speech. "
        "Severity labels (control / high / mid / low / very low) reflect intelligibility "
        "ratings from the UA-Speech corpus annotations."
    )

    if "severity" in main_df.columns and "b_label" in main_df.columns:
        sev_summary = (
            main_df.groupby("severity", observed=True)
            .agg(
                n          = ("sample_id", "count"),
                pct_B1     = ("b_label", lambda s: (s == "B1").mean()),
                pct_B2     = ("b_label", lambda s: (s == "B2").mean()),
                pct_B3     = ("b_label", lambda s: (s == "B3").mean()),
                pct_C1     = ("c_label", lambda s: (s == "C1").mean()),
                pct_C4     = ("c_label", lambda s: (s == "C4").mean()),
            )
            .reindex(SEVERITY_ORDER)
            .reset_index()
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_b = px.bar(
                sev_summary,
                x="severity",
                y=["pct_B1", "pct_B2", "pct_B3"],
                barmode="group",
                labels={"value": "Rate", "severity": "Severity bucket", "variable": "Outcome"},
                title="Axis B — ASR propagation outcomes by severity",
                category_orders={"severity": SEVERITY_ORDER},
                color_discrete_map={
                    "pct_B1": COLORS["B1"],
                    "pct_B2": COLORS["B2"],
                    "pct_B3": COLORS["B3"],
                },
            )
            fig_b.update_layout(yaxis_tickformat=".0%", legend_title_text="Outcome")
            fig_b.for_each_trace(lambda t: t.update(name=t.name.replace("pct_", "")))
            st.plotly_chart(fig_b, use_container_width=True)

        with col2:
            fig_c = px.bar(
                sev_summary,
                x="severity",
                y=["pct_C1", "pct_C4"],
                barmode="group",
                labels={"value": "Rate", "severity": "Severity bucket", "variable": "Behavior"},
                title="Axis C — Agent behavior by severity",
                category_orders={"severity": SEVERITY_ORDER},
                color_discrete_map={
                    "pct_C1": COLORS["C1"],
                    "pct_C4": COLORS["C4"],
                },
            )
            fig_c.update_layout(yaxis_tickformat=".0%", legend_title_text="Behavior")
            fig_c.for_each_trace(lambda t: t.update(name=t.name.replace("pct_", "")))
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("**Full severity summary table**")
        display_summary = sev_summary.copy()
        for col in ["pct_B1","pct_B2","pct_B3","pct_C1","pct_C4"]:
            display_summary[col] = display_summary[col].map("{:.1%}".format)
        display_summary.columns = ["Severity","N","B1 (silent)","B2 (graceful)","B3 (rejection)","C1 (false conf.)","C4 (appropriate)"]
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

    else:
        st.warning("`severity` or `b_label` column not found in dataset.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_taxonomy:
    st.subheader("Failure taxonomy — full distribution")

    col1, col2 = st.columns(2)

    with col1:
        b_counts = main_df["b_label"].value_counts().reset_index()
        b_counts.columns = ["label", "count"]
        b_counts["color"] = b_counts["label"].map(COLORS)
        fig_b = px.pie(
            b_counts, names="label", values="count",
            title="Axis B — ASR propagation outcome",
            color="label",
            color_discrete_map=COLORS,
            hole=0.35,
        )
        fig_b.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_b, use_container_width=True)

    with col2:
        c_counts = main_df["c_label"].value_counts().reset_index()
        c_counts.columns = ["label", "count"]
        fig_c = px.pie(
            c_counts, names="label", values="count",
            title="Axis C — Agent behavior",
            color="label",
            color_discrete_map=COLORS,
            hole=0.35,
        )
        fig_c.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")
    st.markdown("### Taxonomy definitions")

    col_b, col_c = st.columns(2)
    with col_b:
        st.markdown("""
**Axis B — ASR error propagation**

| Label | Meaning |
|-------|---------|
| **B1** | Silent propagation — agent acted confidently on misrecognised input |
| **B2** | Graceful degradation — error present but response harmless |
| **B3** | Correct rejection — agent flagged something was off |
| **not_applicable** | ASR was essentially correct |

**Judge validation:** Cohen's κ = 0.541 (moderate agreement) against 50 hand-labelled examples.
""")
    with col_c:
        st.markdown("""
**Axis C — Agent behavior**

| Label | Meaning |
|-------|---------|
| **C1** | False confidence — committed to answer despite genuinely ambiguous input |
| **C2** | No repair mechanism at all |
| **C3** | Overcorrection — asked for clarification on clear input |
| **C4** | Appropriate behavior |

**Judge validation:** Cohen's κ = 0.707 (substantial agreement) against 50 hand-labelled examples.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROBE / INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_probe:
    st.subheader("Mechanistic interpretability — does the model internally 'know' more than it says?")

    st.markdown("""
<div class="finding-box">
<b>Core question:</b> When the agent silently propagates an ASR error (B1), 
is there <i>any</i> signal in its internal activations that something was wrong — 
even though its output gave no indication?
<br><br>
<b>Answer:</b> Yes, in 81% of B1 cases. A linear probe on layer-12 hidden states 
achieves AUROC 0.805 (5-fold speaker-grouped CV), detecting a decodable ambiguity 
signal that is absent from the model's output. This is the <b>suppressed-signal</b> pattern.
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        diagram_path = os.path.join(DATA_DIR, "full_reliability_diagram.png")
        if os.path.exists(diagram_path):
            st.image(diagram_path, caption="Probe reliability diagram (calibration)")
        else:
            st.info("📊 Reliability diagram not found. Add `full_reliability_diagram.png` to the data directory.")

    with col2:
        st.markdown("**Cross-tabulation: probe signal vs. expressed output confidence**")
        st.dataframe(crosstab_df, use_container_width=True)

        n_b1_total    = (main_df["b_label"] == "B1").sum()
        n_suppressed  = len(suppressed_df)

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("B1 (silent propagation) cases", n_b1_total)
        col_m2.metric(
            "With suppressed ambiguity signal",
            f"{n_suppressed} ({n_suppressed/n_b1_total:.0%})" if n_b1_total else "N/A"
        )

    st.markdown("---")
    st.markdown("### What this means (and what it doesn't)")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**What the probe shows:**
- A logistic regression trained on the LLM's layer-12 hidden states predicts 
  whether an input was genuinely ambiguous (ground truth from taxonomy labels)
- Evaluated with 5-fold **speaker-grouped** cross-validation (no speaker leakage)
- AUROC ≈ 0.80 — well above chance across all folds
- In B1 cases (silent propagation), the ambiguity signal is present in 81% of cases
  despite being absent from the model's expressed output

This means: **the information was there internally, but not used**.
""")
    with col_r:
        st.markdown("""
<div class="caveat-box">
<b>Important caveat — correlation, not causation:</b><br>
The probe shows the ambiguity signal is <i>linearly decodable</i> from activations. 
It does not establish that this signal <i>causally drives</i> (or fails to drive) 
the model's output. A model could represent uncertainty internally without 
that representation influencing generation.<br><br>
Establishing causation requires an <b>intervention study</b> (e.g. activation patching, 
causal scrubbing) — a natural and planned extension of this work.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BROWSE TRACES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_browse:
    st.subheader("Browse individual pipeline traces")
    st.caption(
        "Filter by severity, propagation outcome (Axis B), or agent behavior (Axis C) "
        "to inspect individual samples. Each row is one audio sample through the full pipeline."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        severity_opts = sorted(main_df["severity"].dropna().unique().tolist()) \
            if "severity" in main_df.columns else []
        severity_filter = st.multiselect("Severity", options=severity_opts)
    with col2:
        b_opts = sorted(main_df["b_label"].dropna().unique().tolist()) \
            if "b_label" in main_df.columns else []
        b_filter = st.multiselect("Axis B — propagation outcome", options=b_opts)
    with col3:
        c_opts = sorted(main_df["c_label"].dropna().unique().tolist()) \
            if "c_label" in main_df.columns else []
        c_filter = st.multiselect("Axis C — agent behavior", options=c_opts)

    filtered = main_df.copy()
    if severity_filter:
        filtered = filtered[filtered["severity"].isin(severity_filter)]
    if b_filter:
        filtered = filtered[filtered["b_label"].isin(b_filter)]
    if c_filter:
        filtered = filtered[filtered["c_label"].isin(c_filter)]

    n_filtered = len(filtered)
    col_info, col_highlight = st.columns([2, 1])
    with col_info:
        st.caption(f"Showing **{n_filtered}** of **{len(main_df)}** samples")
    with col_highlight:
        if b_filter and "B1" in b_filter:
            n_b1_filtered = (filtered["b_label"] == "B1").sum()
            st.caption(f"🔴 {n_b1_filtered} silent-propagation cases in this view")

    display_cols = [c for c in [
        "sample_id", "severity",
        "ground_truth", "asr_transcript",
        "agent_response",
        "b_label", "c_label",
        "justification",
    ] if c in filtered.columns]

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        height=500,
    )

    if n_filtered > 0 and "asr_transcript" in filtered.columns and "ground_truth" in filtered.columns:
        st.markdown("---")
        st.markdown("**Spot-check a sample**")
        sample_ids = filtered["sample_id"].tolist() if "sample_id" in filtered.columns else list(range(n_filtered))
        chosen = st.selectbox("Select sample ID", sample_ids)
        row = filtered[filtered["sample_id"] == chosen].iloc[0] \
            if "sample_id" in filtered.columns else filtered.iloc[chosen]

        col_a, col_b_col = st.columns(2)
        with col_a:
            st.markdown("**Ground truth**")
            st.info(row.get("ground_truth", "N/A"))
            st.markdown("**ASR transcript**")
            st.warning(row.get("asr_transcript", "N/A"))
        with col_b_col:
            st.markdown("**Agent response**")
            st.text_area("", value=str(row.get("agent_response", "N/A")), height=100, disabled=True, label_visibility="collapsed")
            st.markdown(f"**B label:** `{row.get('b_label','?')}`  ·  **C label:** `{row.get('c_label','?')}`")
            if "justification" in row:
                st.markdown("**Judge justification**")
                st.caption(str(row["justification"]))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.82rem;'>"
    "SpeechAgentEval · Built by "
    "<a href='https://agarwalavantika.github.io' style='color:#2563A8;'>Avantika Agarwal</a>"
    " · "
    "<a href='https://github.com/agarwalavantika/speechagenteval' style='color:#2563A8;'>GitHub</a>"
    " · Part of a broader focus on AI systems that can be evaluated and trusted, not just measured on accuracy."
    "</div>",
    unsafe_allow_html=True,
)
