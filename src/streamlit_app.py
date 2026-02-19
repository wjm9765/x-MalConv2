"""
Binary-Hunter Streamlit App
MalConvGCT Explainable AI — Integrated Gradients with Adversarial Attack Analysis
"""

import sys
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Path Setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "MalConv2-main"))

from src.compute_DeepShap import MalConvGCTExplainable
from src.adversarial_malware import generate_adversarial_example
from src.utils import load_config

CKPT_PATH = os.path.join(
    PROJECT_ROOT, "models", "MalConv2-main", "malconvGCT_nocat.checkpoint"
)


# ── Model ───────────────────────────────────────────────────────────

@st.cache_resource
def _load_checkpoint():
    return torch.load(CKPT_PATH, map_location="cpu", weights_only=False)


def create_model():
    """Create a fresh MalConvGCTExplainable with pretrained weights."""
    config = load_config()
    mdl = config.get("model", {}).get("malconv", {})

    model = MalConvGCTExplainable(
        out_size=mdl.get("num_classes", 2),
        channels=mdl.get("channels", 256),
        window_size=mdl.get("window_size", 256),
        stride=mdl.get("stride", 64),
        embd_size=mdl.get("embd_size", 8),
    )
    ckpt = _load_checkpoint()
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ── Charting ────────────────────────────────────────────────────────

def make_chart(data, title, orig_len=None, total_len=None, threshold=1e-5):
    """Plotly bar chart for SHAP attribution. Highlights padding region."""
    if data.ndim > 1:
        data = data.flatten()

    mask = np.abs(data) > threshold
    indices = np.where(mask)[0]
    values = data[indices]
    display_len = total_len or len(data)

    fig = go.Figure()

    # Padding region background
    if orig_len is not None and total_len is not None and orig_len < total_len:
        fig.add_vrect(
            x0=orig_len, x1=total_len,
            fillcolor="rgba(255, 165, 0, 0.10)",
            line=dict(width=1, color="rgba(255, 165, 0, 0.4)", dash="dot"),
            annotation_text="⬅ Adversarial Padding",
            annotation_position="top right",
            annotation_font=dict(size=11, color="darkorange"),
        )

    if len(indices) == 0:
        fig.add_annotation(
            text="No significant contributions",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray"),
        )
    else:
        pos = values > 0
        neg = ~pos
        bar_w = max(display_len * 0.00285, 28)
        if pos.any():
            fig.add_trace(go.Bar(
                x=indices[pos], y=values[pos],
                marker_color="rgba(220, 40, 40, 0.9)",
                name="↑ Malware", width=bar_w,
            ))
        if neg.any():
            fig.add_trace(go.Bar(
                x=indices[neg], y=values[neg],
                marker_color="rgba(30, 80, 220, 0.9)",
                name="↓ Benign", width=bar_w,
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Byte Index", range=[0, display_len]),
        yaxis=dict(title="Attribution"),
        height=380,
        margin=dict(l=60, r=20, t=50, b=45),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=12),
        ),
        bargap=0,
        plot_bgcolor="#f8f9fb",
        paper_bgcolor="#f8f9fb",
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.07)",
        showline=True, linewidth=1, linecolor="#ccc", mirror=True,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.1)",
        zeroline=True, zerolinecolor="black", zerolinewidth=1,
        showline=True, linewidth=1, linecolor="#ccc", mirror=True,
    )
    return fig


# ── Additivity Verification ────────────────────────────────────────

def verify_additivity(model, shap_ctx, shap_feat, input_len, logit_val, desc):
    """Verify sum(attr) + base_value ≈ model_output."""
    device = next(model.parameters()).device
    embd_size = model.embd_size
    baseline = torch.zeros((1, input_len, 2 * embd_size)).to(device)

    model._is_explaining = True
    if hasattr(model, "context_net"):
        model.context_net._is_explaining = True

    try:
        with torch.no_grad():
            base_logits = model(baseline)
            base_value = base_logits[0, 1].item()
    finally:
        model._is_explaining = False
        if hasattr(model, "context_net"):
            model.context_net._is_explaining = False

    shap_sum = float(np.sum(shap_ctx) + np.sum(shap_feat))
    reconstructed = shap_sum + base_value
    diff = abs(reconstructed - logit_val)

    return {
        "desc": desc,
        "shap_sum": shap_sum,
        "base_value": base_value,
        "reconstructed": reconstructed,
        "model_output": logit_val,
        "diff": diff,
        "passed": diff < 1.0,
    }


# ── Analysis Pipeline ──────────────────────────────────────────────

def run_analysis(file_path, status_text=None):
    """Full pipeline: Original → Attack → Adversarial → Verify."""
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        raw = f.read(4_000_000)
    orig_bytes = np.frombuffer(raw, dtype=np.uint8)
    orig_len = len(orig_bytes)

    # Phase 1: Original
    if status_text:
        status_text.text(f"[{filename}] Phase 1/3: Original analysis (IG)...")
    model = create_model()
    orig_t = torch.tensor(
        orig_bytes.astype(np.int32) + 1, dtype=torch.long
    ).unsqueeze(0)

    out_orig = model(orig_t)
    logits_orig = out_orig[0]
    prob_orig = F.softmax(logits_orig, dim=1)[0, 1].item()
    logit_orig = logits_orig[0, 1].item()
    ctx_orig, feat_orig = out_orig[3], out_orig[4]

    v_orig = verify_additivity(
        model, ctx_orig, feat_orig, orig_len, logit_orig, "Original"
    )

    # Phase 2: Adversarial attack
    if status_text:
        status_text.text(f"[{filename}] Phase 2/3: Adversarial attack...")
    adv_bytes = generate_adversarial_example(model, orig_bytes, target_class=0)
    del model

    # Phase 3: Adversarial analysis
    if status_text:
        status_text.text(f"[{filename}] Phase 3/3: Adversarial analysis (IG)...")
    model_adv = create_model()
    adv_t = torch.tensor(
        adv_bytes.astype(np.int32) + 1, dtype=torch.long
    ).unsqueeze(0)

    out_adv = model_adv(adv_t)
    logits_adv = out_adv[0]
    prob_adv = F.softmax(logits_adv, dim=1)[0, 1].item()
    logit_adv = logits_adv[0, 1].item()
    ctx_adv, feat_adv = out_adv[3], out_adv[4]
    total_len = len(adv_bytes)

    v_adv = verify_additivity(
        model_adv, ctx_adv, feat_adv, total_len, logit_adv, "Adversarial"
    )
    del model_adv

    return {
        "filename": filename,
        "orig_len": orig_len,
        "total_len": total_len,
        "prob_orig": prob_orig,
        "prob_adv": prob_adv,
        "ctx_orig": ctx_orig,
        "feat_orig": feat_orig,
        "ctx_adv": ctx_adv,
        "feat_adv": feat_adv,
        "v_orig": v_orig,
        "v_adv": v_adv,
    }


# ── Display ─────────────────────────────────────────────────────────

def display_result(r):
    """Render a single file's analysis results."""
    st.markdown("---")
    st.subheader(f"📄 {r['filename']}")

    # ── Before Attack ──
    st.markdown(
        '<div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin-bottom:16px; background:#fafafa;">'
        f'<p style="margin:0 0 8px 0; font-size:15px; font-weight:600; color:#333;">'
        f'Before Attack &nbsp;<span style="color:#1a73e8; font-weight:700;">{r["prob_orig"]*100:.1f}%</span> malware</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_chart(r["ctx_orig"], "Context Attribution", total_len=r["orig_len"]),
        use_container_width=True, key=f"{r['filename']}_ctx_orig",
    )
    st.plotly_chart(
        make_chart(r["feat_orig"], "Feature Attribution", total_len=r["orig_len"]),
        use_container_width=True, key=f"{r['filename']}_feat_orig",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── After Attack ──
    st.markdown(
        '<div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin-bottom:16px; background:#fafafa;">'
        f'<p style="margin:0 0 8px 0; font-size:15px; font-weight:600; color:#333;">'
        f'After Attack &nbsp;<span style="color:#d93025; font-weight:700;">{r["prob_adv"]*100:.1f}%</span> malware</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_chart(
            r["ctx_adv"], "Context Attribution (Adversarial)",
            orig_len=r["orig_len"], total_len=r["total_len"],
        ),
        use_container_width=True, key=f"{r['filename']}_ctx_adv",
    )
    st.plotly_chart(
        make_chart(
            r["feat_adv"], "Feature Attribution (Adversarial)",
            orig_len=r["orig_len"], total_len=r["total_len"],
        ),
        use_container_width=True, key=f"{r['filename']}_feat_adv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Additivity Verification ──
    with st.expander("🔬 Additivity Verification"):
        for v in [r["v_orig"], r["v_adv"]]:
            icon = "✅" if v["passed"] else "❌"
            st.code(
                f"[{v['desc']}] {icon} {'PASS' if v['passed'] else 'FAIL'}\n"
                f"  SHAP Sum  : {v['shap_sum']:.4f}\n"
                f"  Base Value: {v['base_value']:.4f}\n"
                f"  Sum+Base  : {v['reconstructed']:.4f}\n"
                f"  Model Out : {v['model_output']:.4f}\n"
                f"  Diff      : {v['diff']:.6f}",
                language=None,
            )


# ── Main ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(layout="wide", page_title="Binary-Hunter XAI", page_icon="🔍")
    st.title("🔍 Binary-Hunter")
    st.caption(
        "MalConvGCT Explainable AI — "
        "Integrated Gradients Attribution with Adversarial Attack Analysis"
    )

    # CLI args (passed via: streamlit run app.py -- --file <name>)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    args, _ = parser.parse_known_args()

    # Determine target files
    if args.file:
        target_path = os.path.join(PROJECT_ROOT, args.file)
        if not os.path.exists(target_path):
            st.error(f"File not found: `{target_path}`")
            return
        file_list = [target_path]
        st.info(f"🎯 Single target: **{args.file}**")
    else:
        data_dir = os.path.join(PROJECT_ROOT, "data")
        if not os.path.isdir(data_dir):
            st.error(
                "`data/` directory not found. "
                "Run `./scripts/run_xMalconv` to extract data.zip first."
            )
            return
        file_list = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if not f.startswith(".")
        )
        st.info(f"📂 Processing **{len(file_list)}** files from `data/`")

    # Process
    progress = st.progress(0, text="Starting analysis...")
    status = st.empty()
    results = []

    for i, fp in enumerate(file_list):
        progress.progress(
            i / len(file_list),
            text=f"Processing {os.path.basename(fp)} ({i + 1}/{len(file_list)})...",
        )
        result = run_analysis(fp, status_text=status)
        results.append(result)

    progress.progress(1.0, text="✅ Analysis complete!")
    status.empty()

    # Render
    for r in results:
        display_result(r)


if __name__ == "__main__":
    main()
