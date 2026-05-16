"""
AFC Loss Simulation Suite — Streamlit App
Run with: streamlit run afc_streamlit.py
Install:  pip install streamlit plotly numpy
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AFC Loss Simulations", layout="wide", page_icon="🔬")

st.markdown("""
<style>
    .main { background-color: #020817; }
    .stApp { background-color: #020817; color: #e2e8f0; }
    h1, h2, h3 { color: #e2e8f0; }
    .insight-box {
        border-left: 3px solid #334155;
        padding: 10px 16px;
        background: #0f172a;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 12px;
        border-radius: 0 4px 4px 0;
    }
    .insight-box strong { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    'conflict':    '#ef4444',
    'duplicate':   '#3b82f6',
    'independent': '#22c55e',
    'gamma':       ['#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#22c55e'],
    'accent':      '#94a3b8',
    'bg':          '#020817',
    'grid':        '#1e293b',
}

PLOT_LAYOUT = dict(
    paper_bgcolor='#020817',
    plot_bgcolor='#0f172a',
    font=dict(color='#e2e8f0', family='monospace'),
    xaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155'),
    yaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155'),
    legend=dict(bgcolor='#0f172a', bordercolor='#334155', borderwidth=1),
    margin=dict(l=40, r=40, t=40, b=40),
)


st.title("AFC Loss — Requirements Conflict Detector")
st.caption("Loss function simulation suite · AFC = Adaptive Focal Confidence")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · Focal Gradient",
    "2 · Class Weights",
    "3 · Conf Penalty",
    "4 · KL Divergence",
    "5 · Dynamic γ",
])


# ─────────────────────────────────────────────────────────────────
# SIM 1 — Focal loss gradient vs predicted probability
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Focal Loss Gradient vs Predicted Probability")
    st.markdown(
        "Shows where the loss has gradient signal and where it saturates. "
        "Low p = hard example (model uncertain). High p = easy example (model confident). "
        "Higher γ suppresses easy examples more aggressively."
    )

    gammas = st.multiselect(
        "Select γ values to plot",
        options=[0.5, 1.0, 2.0, 5.0, 10.0],
        default=[0.5, 1.0, 2.0, 5.0],
    )

    ps = np.linspace(0.01, 0.99, 500)
    fig = go.Figure()

    for i, g in enumerate(sorted(gammas)):
        loss = (1 - ps) ** g * -np.log(ps)
        fig.add_trace(go.Scatter(
            x=ps, y=loss, mode='lines', name=f'γ = {g}',
            line=dict(color=COLORS['gamma'][i % len(COLORS['gamma'])], width=2.5)
        ))

    fig.add_vrect(x0=0, x1=0.4, fillcolor=COLORS['conflict'], opacity=0.05,
                  annotation_text="HARD ZONE", annotation_position="top left",
                  annotation_font_color=COLORS['conflict'])
    fig.add_vrect(x0=0.7, x1=1.0, fillcolor=COLORS['independent'], opacity=0.05,
                  annotation_text="EASY ZONE", annotation_position="top right",
                  annotation_font_color=COLORS['independent'])

    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis_title="Predicted probability p (of true class)",
        yaxis_title="Focal loss value",
        yaxis_range=[0, 6],
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='insight-box'>
        <strong>What to look for:</strong> At γ=2, a model predicting p=0.9 on an easy example gets ~10x less gradient than at γ=0.
        If your conflict pairs are sparse and hard, γ≥2 is necessary to stop independent pairs from drowning out the signal.
        If γ is too high, gradient vanishes on everything late in training — even genuinely hard examples stop contributing.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIM 2 — Class weight sensitivity
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Class Weight Sensitivity")
    st.markdown(
        "Set your expected class distribution. Compare how different weight schemes "
        "affect the loss contribution per class at different confidence levels."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        conflict_pct = st.slider("Conflict %", 1, 25, 5)
    with col2:
        max_dup = 100 - conflict_pct - 1
        dup_pct = st.slider("Duplicate %", 1, max_dup, min(5, max_dup))
    with col3:
        ind_pct = 100 - conflict_pct - dup_pct
        st.metric("Independent %", f"{ind_pct}%")

    gamma_w = st.slider("γ (focusing parameter)", 0.5, 5.0, 2.0, 0.5, key="gamma_w")

    freqs = np.array([conflict_pct, dup_pct, ind_pct]) / 100.0
    uniform  = np.array([1/3, 1/3, 1/3])
    inv_freq = (1 / freqs) / (1 / freqs).sum()
    sqrt_inv = (1 / np.sqrt(freqs)) / (1 / np.sqrt(freqs)).sum()

    schemes = {'Uniform': uniform, 'Inv-freq': inv_freq, 'Sqrt-inv': sqrt_inv}
    class_names = ['conflict', 'duplicate', 'independent']
    class_colors = [COLORS['conflict'], COLORS['duplicate'], COLORS['independent']]

    ps = np.linspace(0.01, 0.99, 300)

    # Weight table
    st.markdown("**Effective weight multipliers (relative to uniform):**")
    cols = st.columns(len(schemes))
    for col, (name, w) in zip(cols, schemes.items()):
        with col:
            st.markdown(f"**{name}**")
            for cls, wi in zip(class_names, w):
                ratio = wi / (1/3)
                st.markdown(f"`{cls}`: {ratio:.2f}x")

    fig = go.Figure()
    dash_styles = {'Uniform': 'dash', 'Inv-freq': 'solid', 'Sqrt-inv': 'dot'}

    for scheme_name, weights in schemes.items():
        for ci, (cls, w, col) in enumerate(zip(class_names, weights, class_colors)):
            focal = w * (1 - ps) ** gamma_w * -np.log(ps)
            fig.add_trace(go.Scatter(
                x=ps, y=focal, mode='lines',
                name=f'{scheme_name} · {cls}',
                line=dict(color=col, width=2, dash=dash_styles[scheme_name]),
                legendgroup=scheme_name,
            ))

    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis_title="Predicted probability p",
        yaxis_title="Weighted focal loss",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='insight-box'>
        <strong>What to look for:</strong> With inv-freq weights at 5/5/90, conflict gets ~18x the weight of independent.
        If that gap looks too extreme for your data, sqrt-inv is a softer alternative.
        The goal is that a hard conflict pair contributes meaningfully even in batches dominated by independent pairs.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIM 3 — Confidence penalty vs focal loss interaction
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Confidence Penalty vs Focal Loss Interaction")
    st.markdown(
        "L_Conf penalizes overconfident predictions. Focal loss already suppresses high-confidence easy examples. "
        "Check they aren't fighting each other — the combined loss on a correct confident prediction should still be low, "
        "but combined loss on a wrong confident prediction should stay high."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        alpha = st.slider("α (focal weight)", 0.1, 3.0, 1.0, 0.1)
    with col2:
        beta = st.slider("β (conf penalty weight)", 0.0, 2.0, 0.5, 0.1)
    with col3:
        gamma_c = st.slider("γ", 0.5, 5.0, 2.0, 0.5, key="gamma_c")

    ps = np.linspace(0.34, 0.99, 300)
    p_other = (1 - ps) / 2

    focal_correct   = (1 - ps) ** gamma_c * -np.log(ps)
    conf            = ps * np.log(ps) + 2 * p_other * np.log(p_other + 1e-9)
    combined_correct = alpha * focal_correct + beta * conf

    focal_wrong      = (1 - p_other) ** gamma_c * -np.log(p_other + 1e-9)
    conf_wrong       = ps * np.log(ps) + 2 * p_other * np.log(p_other + 1e-9)
    combined_wrong   = alpha * focal_wrong + beta * conf_wrong

    ratio_at_95 = combined_wrong[-10] / (combined_correct[-10] + 1e-9)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ps, y=focal_correct, mode='lines', name='Focal only (correct)',
        line=dict(color=COLORS['accent'], width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=ps, y=combined_correct, mode='lines', name='Combined (correct)',
        line=dict(color=COLORS['duplicate'], width=2.5)))
    fig.add_trace(go.Scatter(x=ps, y=combined_wrong, mode='lines', name='Combined (wrong)',
        line=dict(color=COLORS['conflict'], width=2.5)))
    fig.add_hline(y=0, line_color='#334155', line_width=1)

    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis_title="Confidence in correct class (p)",
        yaxis_title="Loss value",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    status = "✅ Good" if ratio_at_95 > 3 else ("⚠️ Marginal" if ratio_at_95 > 1 else "❌ Bad — β too large")
    st.metric("Wrong/correct loss ratio at p=0.95", f"{ratio_at_95:.1f}x", help="Should be >> 1")
    st.caption(status)

    st.markdown("""<div class='insight-box'>
        <strong>What to look for:</strong> Combined (correct) should drop toward 0 as confidence increases.
        Combined (wrong) should stay high. If β is too large, both curves flatten —
        the model gets penalized equally for being right and wrong at high confidence. That's the failure mode.
        The wrong/correct ratio at p=0.95 should be well above 3x.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIM 4 — KL divergence term sensitivity
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("KL Divergence Term Sensitivity")
    st.markdown(
        "L_Domain = KL(q ∥ p_avg). Shows how the domain loss blows up when batches contain no conflict pairs — "
        "a real risk with sparse classes. The prior q is what you believe the true distribution should be."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        q_conflict = st.slider("Prior q: conflict %", 1, 20, 5) / 100
    with col2:
        q_dup = st.slider("Prior q: duplicate %", 1, 20, 5) / 100
    with col3:
        eps = st.select_slider("ε smoothing on p_avg", options=[0.0, 0.001, 0.005, 0.01, 0.05], value=0.001)

    q_ind = 1 - q_conflict - q_dup
    q = np.array([q_conflict, q_dup, q_ind])

    batch_conflict = np.linspace(1e-6, 0.25, 500)
    batch_dup      = np.minimum(batch_conflict, (1 - batch_conflict) * 0.1)
    batch_ind      = 1 - batch_conflict - batch_dup

    kl_raw      = []
    kl_smoothed = []

    for bc, bd, bi in zip(batch_conflict, batch_dup, batch_ind):
        pavg = np.array([bc, bd, bi])
        kl_r = float(np.sum(q * np.log(q / (pavg + 1e-9))))
        kl_raw.append(min(kl_r, 30))

        if eps > 0:
            pavg_s = np.array([bc + eps, bd + eps, bi])
            pavg_s /= pavg_s.sum()
            kl_s = float(np.sum(q * np.log(q / (pavg_s + 1e-9))))
            kl_smoothed.append(min(kl_s, 30))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=batch_conflict * 100, y=kl_raw, mode='lines', name='No smoothing',
        line=dict(color=COLORS['conflict'], width=2.5)
    ))
    if eps > 0:
        fig.add_trace(go.Scatter(
            x=batch_conflict * 100, y=kl_smoothed, mode='lines', name=f'ε={eps} smoothing',
            line=dict(color=COLORS['duplicate'], width=2.5, dash='dash')
        ))
    fig.add_vline(x=q_conflict * 100, line_color=COLORS['accent'], line_dash='dot',
                  annotation_text=f"q_conflict={q_conflict*100:.0f}%",
                  annotation_font_color=COLORS['accent'])

    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis_title="Actual conflict % in batch",
        yaxis_title="KL divergence",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='insight-box'>
        <strong>What to look for:</strong> KL spikes sharply when batch conflict % → 0 and q_conflict is non-trivial.
        A batch with no conflict pairs generates a massive domain loss that swamps the focal term.
        Fix options: (1) add ε smoothing to p_avg — use the slider above to see how much it helps,
        (2) stratified batching to guarantee ≥1 conflict/duplicate pair per batch,
        (3) cap KL with λ · min(KL, threshold).
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIM 5 — Dynamic gamma trajectory
# ─────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Dynamic γ Trajectory Over Training")
    st.markdown(
        "γ = γ_base + η · Accuracy_val. As the model improves, γ grows, sharpening focus on hard examples. "
        "Check that γ doesn't grow so large that gradients vanish on real hard examples late in training."
    )

    col1, col2 = st.columns(2)
    with col1:
        gamma_base = st.slider("γ_base (initial value)", 0.1, 2.0, 0.5, 0.1)
        p_hard_val = st.slider("p for hard example", 0.1, 0.5, 0.3, 0.05,
                               help="Model confidence on a genuinely hard pair early in training")
    with col2:
        eta = st.slider("η (modulation factor)", 0.1, 5.0, 2.0, 0.1)
        p_easy_val = st.slider("p for easy example", 0.6, 0.99, 0.9, 0.05,
                               help="Model confidence on an easy independent pair")

    accuracies = np.linspace(0, 1, 300)
    gammas_dyn = gamma_base + eta * accuracies

    loss_hard = (1 - p_hard_val) ** gammas_dyn * -np.log(p_hard_val)
    loss_easy = (1 - p_easy_val) ** gammas_dyn * -np.log(p_easy_val)
    ratio     = loss_hard / (loss_easy + 1e-9)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=accuracies, y=loss_hard, mode='lines', name=f'Hard (p={p_hard_val})',
        line=dict(color=COLORS['conflict'], width=2.5)))
    fig.add_trace(go.Scatter(x=accuracies, y=loss_easy, mode='lines', name=f'Easy (p={p_easy_val})',
        line=dict(color=COLORS['independent'], width=2.5)))
    fig.add_trace(go.Scatter(x=accuracies, y=gammas_dyn, mode='lines', name='γ value',
        line=dict(color=COLORS['gamma'][0], width=1.5, dash='dot'),
        yaxis='y2'))

    fig.update_layout(
        **PLOT_LAYOUT,
        xaxis_title="Validation accuracy",
        yaxis_title="Focal loss value",
        yaxis2=dict(
            title="γ value", overlaying='y', side='right',
            gridcolor='#1e293b', color=COLORS['gamma'][0]
        ),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=accuracies, y=ratio, mode='lines', name='Hard/Easy loss ratio',
        line=dict(color=COLORS['duplicate'], width=2.5),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.08)'))
    fig2.add_hline(y=1, line_color='#334155', line_dash='dash')
    fig2.update_layout(
        **PLOT_LAYOUT,
        xaxis_title="Validation accuracy",
        yaxis_title="Hard / Easy loss ratio",
        height=280,
        title=dict(text="How much more gradient hard examples get vs easy", font=dict(color='white', size=13))
    )
    st.plotly_chart(fig2, use_container_width=True)

    ratio_at_90 = ratio[int(0.9 * 300)]
    loss_hard_at_100 = loss_hard[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("γ at acc=1.0", f"{gammas_dyn[-1]:.2f}")
    col2.metric("Hard/easy ratio at acc=0.9", f"{ratio_at_90:.1f}x", help="Should stay >> 1")
    col3.metric("Hard loss at acc=1.0", f"{loss_hard_at_100:.4f}", help="Should not approach 0")

    status = "✅ Good" if ratio_at_90 > 5 and loss_hard_at_100 > 0.01 else \
             ("⚠️ Watch η — ratio dropping" if ratio_at_90 < 5 else "❌ Hard loss vanishing — reduce η")
    st.caption(status)

    st.markdown("""<div class='insight-box'>
        <strong>What to look for:</strong> The hard/easy ratio should stay well above 1 throughout training.
        If η is too high, γ grows so large that even genuinely hard examples (p=0.3) get their loss suppressed.
        The hard loss at acc=1.0 should not approach zero — if it does, reduce η.
        A good η keeps the separation between hard and easy loss curves wide without killing absolute gradient magnitude.
    </div>""", unsafe_allow_html=True)
