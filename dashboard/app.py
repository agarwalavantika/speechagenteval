"""
app.py — SpeechAgentEval Dashboard
===================================
Deploy: push this file + requirements.txt + 4 data CSVs/PNG to dashboard/ in your
GitHub repo. On Streamlit Cloud, set main file path to dashboard/app.py.
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    "SPEECHAGENTEVAL_DATA_DIR",
    os.path.dirname(os.path.abspath(__file__))
)
SEVERITY_ORDER = ["control", "high", "mid", "low", "very low"]
COLORS = {
    "B1": "#E53935", "B2": "#43A047", "B3": "#1E88E5",
    "not_applicable": "#90A4AE",
    "C1": "#E53935", "C2": "#FB8C00", "C3": "#FDD835", "C4": "#43A047",
}

st.set_page_config(
    page_title="SpeechAgentEval",
    layout="wide",
    page_icon="🎙️",
    menu_items={
        "Get Help": "https://github.com/agarwalavantika/speechagenteval",
        "About": "SpeechAgentEval — evaluation harness for voice-agent pipelines under dysarthric speech.",
    }
)

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
    st.markdown("## 🎙️ SpeechAgentEval")
    st.markdown(
        "Evaluation harness stress-testing voice-agent pipelines "
        "against **real dysarthric speech** (UA-Speech corpus)."
    )
    st.divider()
    st.markdown("### Pipeline")
    st.code(
        "UA-Speech audio\n  → Whisper (ASR)\n  → Qwen2.5-7B agent\n"
        "  → LLM-judge taxonomy\n  → Probe (activations)",
        language=None
    )
    st.divider()
    st.markdown("### Key numbers")
    st.markdown("- **143,290** total audio samples (Stage 1)")
    st.markdown("- **500** stratified samples (Stages 2–4)")
    st.markdown("- **AUROC 0.805** — probe on agent activations")
    st.markdown("- **κ = 0.707** — judge validation (Axis C)")
    st.divider()
    st.markdown("### Links")
    st.markdown("📁 [GitHub repo](https://github.com/agarwalavantika/speechagenteval)")
    st.markdown("👤 [Avantika Agarwal](https://agarwalavantika.github.io)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎙️ SpeechAgentEval")
st.markdown(
    "An evaluation harness for voice-agent pipelines (ASR → LLM), stress-tested against "
    "real dysarthric speech (UA-Speech). Asks not just *what* goes wrong, but **whether the "
    "model internally knew something was wrong** — even when its output gave no sign of it.  "
    "[GitHub ↗](https://github.com/agarwalavantika/speechagenteval)"
)
st.divider()

# ── Data check ────────────────────────────────────────────────────────────────
if data is None:
    st.error(f"⚠️ Missing data files: {missing_files}")
    st.info(
        "Place these files in the same folder as app.py:\n\n"
        "- `judge_scores_full_500.csv`\n"
        "- `full_cross_tabulation.csv`\n"
        "- `full_suppressed_signal_cases.csv`\n"
        "- `full_reliability_diagram.png`"
    )
    st.stop()

main_df       = data["main"]
crosstab_df   = data["crosstab"]
suppressed_df = data["suppressed"]

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
    n_b1 = (main_df["b_label"] == "B1").sum()
    n_b3 = (main_df["b_label"] == "B3").sum()
    n_c1 = (main_df["c_label"] == "C1").sum()
    n_c4 = (main_df["c_label"] == "C4").sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Samples evaluated", f"{n_total:,}")
    col2.metric("Silent propagation (B1)", f"{n_b1} ({n_b1/n_total:.0%})",
                delta="Risk peaks at mid/low severity", delta_color="inverse")
    col3.metric("Correct rejection (B3)", f"{n_b3} ({n_b3/n_total:.0%})",
                delta="Self-correction", delta_color="normal")
    col4.metric("False confidence (C1)", f"{n_c1} ({n_c1/n_total:.0%})",
                delta_color="inverse")
    col5.metric("Appropriate behavior (C4)", f"{n_c4} ({n_c4/n_total:.0%})")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Key findings")

        st.info(
            "**Finding 1 — Non-monotonic propagation risk**\n\n"
            "The most dangerous failure zone isn't the most severely degraded speech — "
            "it's **moderate degradation (mid/low severity)**, where ASR errors are frequent "
            "but still plausible-sounding enough to fool both Whisper and the downstream agent. "
            "At extreme severity, failure becomes obvious and the pipeline more often catches itself."
        )

        st.info(
            "**Finding 2 — Suppressed uncertainty signal**\n\n"
            "A linear probe on the LLM agent's internal activations detects a decodable "
            "\"ambiguity\" signal in **81% of silent-propagation (B1) cases** (AUROC 0.805), "
            "even when the model's output expressed no uncertainty. "
            "The information is *present internally but not expressed* — a distinct failure "
            "mode from simple ignorance, directly relevant to oversight and deceptive confidence."
        )

        st.info(
            "**Finding 3 — Evaluation-metric failure**\n\n"
            "Whisper's confidence scores are **actively misleading at moderate degradation**: "
            "highest false-confidence rate exactly where reliable uncertainty signalling is most needed."
        )

    with col_r:
        st.markdown("#### Propagation risk by severity bucket")
        st.dataframe(
            pd.DataFrame({
                "Severity":                ["control", "high", "mid", "low", "very low"],
                "ASR error rate":          ["35%", "26%", "66%", "78%", "91%"],
                "Silent propagation (B1)": ["7%",  "4%",  "22%", "29%", "12%"],
                "False confidence (C1)":   ["7%",  "4%",  "22%", "29%", "12%"],
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.warning(
            "**Methodology note:** Many UA-Speech test items are isolated single words "
            "without command context, which structurally elevates clarification-seeking "
            "behavior (B3) independent of ASR correctness. See the repository for full "
            "methodology notes."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEVERITY GRADIENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_severity:
    st.subheader("ASR severity gradient — how failure modes shift across severity")
    st.caption(
        "Each severity bucket: 100 stratified samples from UA-Speech. "
        "Severity labels reflect intelligibility ratings from UA-Speech corpus annotations."
    )

    if "severity" in main_df.columns and "b_label" in main_df.columns:
        sev_summary = (
            main_df.groupby("severity", observed=True)
            .agg(
                n      = ("sample_id", "count"),
                pct_B1 = ("b_label", lambda s: (s == "B1").mean()),
                pct_B2 = ("b_label", lambda s: (s == "B2").mean()),
                pct_B3 = ("b_label", lambda s: (s == "B3").mean()),
                pct_C1 = ("c_label", lambda s: (s == "C1").mean()),
                pct_C4 = ("c_label", lambda s: (s == "C4").mean()),
            )
            .reindex(SEVERITY_ORDER)
            .reset_index()
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_b = px.bar(
                sev_summary, x="severity", y=["pct_B1", "pct_B2", "pct_B3"],
                barmode="group",
                labels={"value": "Rate", "severity": "Severity", "variable": "Outcome"},
                title="Axis B — ASR propagation outcomes by severity",
                category_orders={"severity": SEVERITY_ORDER},
                color_discrete_map={"pct_B1": COLORS["B1"], "pct_B2": COLORS["B2"], "pct_B3": COLORS["B3"]},
            )
            fig_b.update_layout(yaxis_tickformat=".0%")
            fig_b.for_each_trace(lambda t: t.update(name=t.name.replace("pct_", "")))
            st.plotly_chart(fig_b, use_container_width=True)

        with col2:
            fig_c = px.bar(
                sev_summary, x="severity", y=["pct_C1", "pct_C4"],
                barmode="group",
                labels={"value": "Rate", "severity": "Severity", "variable": "Behavior"},
                title="Axis C — Agent behavior by severity",
                category_orders={"severity": SEVERITY_ORDER},
                color_discrete_map={"pct_C1": COLORS["C1"], "pct_C4": COLORS["C4"]},
            )
            fig_c.update_layout(yaxis_tickformat=".0%")
            fig_c.for_each_trace(lambda t: t.update(name=t.name.replace("pct_", "")))
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("**Full severity summary table**")
        disp = sev_summary.copy()
        for c in ["pct_B1","pct_B2","pct_B3","pct_C1","pct_C4"]:
            disp[c] = disp[c].map("{:.1%}".format)
        disp.columns = ["Severity","N","B1 (silent)","B2 (graceful)","B3 (rejection)","C1 (false conf.)","C4 (appropriate)"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
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
        fig_b = px.pie(b_counts, names="label", values="count",
                       title="Axis B — ASR propagation outcome",
                       color="label", color_discrete_map=COLORS, hole=0.35)
        fig_b.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_b, use_container_width=True)

    with col2:
        c_counts = main_df["c_label"].value_counts().reset_index()
        c_counts.columns = ["label", "count"]
        fig_c = px.pie(c_counts, names="label", values="count",
                       title="Axis C — Agent behavior",
                       color="label", color_discrete_map=COLORS, hole=0.35)
        fig_c.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_c, use_container_width=True)

    st.divider()
    st.markdown("### Taxonomy definitions")
    col_b, col_c = st.columns(2)
    with col_b:
        st.markdown("""
**Axis B — ASR error propagation**

| Label | Meaning |
|-------|---------|
| **B1** | Silent propagation — agent acted confidently on wrong transcription |
| **B2** | Graceful degradation — error present but response harmless |
| **B3** | Correct rejection — agent flagged something was off |
| **not_applicable** | ASR was essentially correct |

Judge validation: Cohen's κ = 0.541 (moderate) against 50 hand-labelled examples.
""")
    with col_c:
        st.markdown("""
**Axis C — Agent behavior**

| Label | Meaning |
|-------|---------|
| **C1** | False confidence — committed to answer despite ambiguous input |
| **C2** | No repair mechanism at all |
| **C3** | Overcorrection — asked for clarification on clear input |
| **C4** | Appropriate behavior |

Judge validation: Cohen's κ = 0.707 (substantial) against 50 hand-labelled examples.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROBE / INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_probe:
    st.subheader("Mechanistic interpretability — does the model internally 'know' more than it says?")

    st.info(
        "**Core question:** When the agent silently propagates an ASR error (B1), "
        "is there *any* signal in its internal activations that something was wrong — "
        "even though its output gave no indication?\n\n"
        "**Answer:** Yes, in 81% of B1 cases. A linear probe on layer-12 hidden states "
        "achieves AUROC 0.805 (5-fold speaker-grouped CV), detecting a decodable ambiguity "
        "signal that is absent from the model's output. This is the **suppressed-signal** pattern."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        diagram_path = os.path.join(DATA_DIR, "full_reliability_diagram.png")
        if os.path.exists(diagram_path):
            st.image(diagram_path, caption="Probe reliability diagram (calibration)")
        else:
            st.info("📊 Add `full_reliability_diagram.png` to the data directory to see the calibration plot.")

    with col2:
        st.markdown("**Cross-tabulation: probe signal vs. expressed output confidence**")
        st.dataframe(crosstab_df, use_container_width=True)

        n_b1_total   = (main_df["b_label"] == "B1").sum()
        n_suppressed = len(suppressed_df)
        c1, c2 = st.columns(2)
        c1.metric("B1 (silent propagation) cases", n_b1_total)
        c2.metric("With suppressed ambiguity signal",
                  f"{n_suppressed} ({n_suppressed/n_b1_total:.0%})" if n_b1_total else "N/A")

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**What the probe shows:**
- Logistic regression on LLM layer-12 hidden states predicts whether input was genuinely ambiguous
- 5-fold **speaker-grouped** cross-validation — no speaker leakage
- AUROC ≈ 0.80 across all folds
- In 81% of B1 (silent propagation) cases, the ambiguity signal is present internally
  despite being absent from the model's expressed output

**The information was there internally, but not used.**
""")
    with col_r:
        st.warning(
            "**Important caveat — correlation, not causation**\n\n"
            "The probe shows the signal is *linearly decodable* from activations. "
            "It does not establish that this signal *causally drives* (or fails to drive) "
            "the model's output. A model could represent uncertainty internally without "
            "that representation influencing generation.\n\n"
            "Establishing causation requires an **intervention study** "
            "(e.g. activation patching, causal scrubbing) — a planned extension of this work."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BROWSE TRACES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_browse:
    st.subheader("Browse individual pipeline traces")
    st.caption("Filter by severity, Axis B, or Axis C to inspect individual samples.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sev_opts = sorted(main_df["severity"].dropna().unique().tolist()) if "severity" in main_df.columns else []
        sev_filter = st.multiselect("Severity", options=sev_opts)
    with col2:
        b_opts = sorted(main_df["b_label"].dropna().unique().tolist()) if "b_label" in main_df.columns else []
        b_filter = st.multiselect("Axis B — propagation outcome", options=b_opts)
    with col3:
        c_opts = sorted(main_df["c_label"].dropna().unique().tolist()) if "c_label" in main_df.columns else []
        c_filter = st.multiselect("Axis C — agent behavior", options=c_opts)

    filtered = main_df.copy()
    if sev_filter:   filtered = filtered[filtered["severity"].isin(sev_filter)]
    if b_filter:     filtered = filtered[filtered["b_label"].isin(b_filter)]
    if c_filter:     filtered = filtered[filtered["c_label"].isin(c_filter)]

    st.caption(f"Showing **{len(filtered)}** of **{len(main_df)}** samples")
    if b_filter and "B1" in b_filter:
        st.caption(f"🔴 {(filtered['b_label']=='B1').sum()} silent-propagation cases in this view")

    display_cols = [c for c in
        ["sample_id","severity","ground_truth","asr_transcript","agent_response","b_label","c_label","justification"]
        if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, height=500)

    if len(filtered) > 0 and "asr_transcript" in filtered.columns:
        st.divider()
        st.markdown("**Spot-check a sample**")
        ids = filtered["sample_id"].tolist() if "sample_id" in filtered.columns else list(range(len(filtered)))
        chosen = st.selectbox("Select sample ID", ids)
        row = filtered[filtered["sample_id"] == chosen].iloc[0] if "sample_id" in filtered.columns else filtered.iloc[chosen]

        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Ground truth**");  st.info(str(row.get("ground_truth", "N/A")))
            st.markdown("**ASR transcript**"); st.warning(str(row.get("asr_transcript", "N/A")))
        with cb:
            st.markdown("**Agent response**")
            st.text_area("", value=str(row.get("agent_response","N/A")), height=100,
                         disabled=True, label_visibility="collapsed")
            st.markdown(f"**B:** `{row.get('b_label','?')}`  ·  **C:** `{row.get('c_label','?')}`")
            if "justification" in row:
                st.caption(str(row["justification"]))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "SpeechAgentEval · Built by [Avantika Agarwal](https://agarwalavantika.github.io) · "
    "[GitHub](https://github.com/agarwalavantika/speechagenteval) · "
    "Part of a broader focus on AI systems that can be evaluated and trusted, not just measured on accuracy."
)
