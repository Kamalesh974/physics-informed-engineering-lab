"""
architecture_plot.py -- network diagram of the actual PINN.

Layer sizes come from the configuration (before training) or from the live
model; once a model exists, each drawn connection is coloured by the real
weight value in that model (blue negative, orange positive, opacity = |w|),
refreshed with every prediction snapshot during training.
"""
import numpy as np
import plotly.graph_objects as go

from .theme import BLUE, GREEN, MUTED, ORANGE, TEXT, base_layout

MAX_DRAWN = 10   # neurons drawn per layer (the rest are summarised as "+k")


def _drawn(n):
    return list(range(min(n, MAX_DRAWN)))


def _ypos(k, n_drawn):
    if n_drawn == 1:
        return 0.5
    return 0.08 + 0.84 * k / (n_drawn - 1)


def architecture_figure(sizes, activation, output_names, weights=None, height=300, n_params=None):
    L = len(sizes)
    xs = np.linspace(0.06, 0.94, L)
    fig = go.Figure()

    # connections
    bins = {}
    for li in range(L - 1):
        a, b = _drawn(sizes[li]), _drawn(sizes[li + 1])
        W = weights[li] if weights is not None and li < len(weights) else None
        wmax = float(np.max(np.abs(W))) if W is not None and W.size else 1.0
        for j in b:
            for i in a:
                if W is not None:
                    w = float(W[j, i]) / (wmax or 1.0)
                    key = ("pos" if w >= 0 else "neg", min(int(abs(w) * 4), 3))
                else:
                    key = ("none", 0)
                seg = bins.setdefault(key, ([], []))
                seg[0].extend([xs[li], xs[li + 1], None])
                seg[1].extend([_ypos(i, len(a)), _ypos(j, len(b)), None])
    for (sign, level), (sx, sy) in bins.items():
        color = {"pos": ORANGE, "neg": BLUE, "none": "#3a4a61"}[sign]
        alpha = [0.10, 0.25, 0.45, 0.8][level] if sign != "none" else 0.35
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=color, width=0.6 + 0.5 * level), opacity=alpha))

    # neurons
    for li, n in enumerate(sizes):
        d = _drawn(n)
        color = GREEN if li == 0 else (ORANGE if li == L - 1 else "#8fb8ff")
        ys = [_ypos(k, len(d)) for k in d]
        if li == L - 1:
            hover = [output_names[k] if k < len(output_names) else "" for k in d]
        elif li == 0:
            hover = ["input t (normalised to τ ∈ [-1, 1])"]
        else:
            hover = [f"hidden {li}, neuron {k + 1}" for k in d]
        fig.add_trace(go.Scatter(x=[xs[li]] * len(d), y=ys, mode="markers", showlegend=False,
                                 marker=dict(size=11 if li in (0, L - 1) else 8, color=color,
                                             line=dict(color="#0b1018", width=1.5)),
                                 hovertext=hover, hoverinfo="text"))
        if n > MAX_DRAWN:
            fig.add_annotation(x=xs[li], y=-0.02, text=f"+{n - MAX_DRAWN}", showarrow=False,
                               font=dict(size=10, color=MUTED))
        if li == 0:
            label = "Input<br>t"
        elif li == L - 1:
            label = "Output<br>" + ", ".join(output_names[:3]) + ("…" if len(output_names) > 3 else "")
        else:
            label = f"Hidden {li}<br>{n} · {activation}"
        fig.add_annotation(x=xs[li], y=1.02, text=label, showarrow=False, yanchor="bottom",
                           font=dict(size=10, color=TEXT))
        if li == L - 1:
            for k in d:
                if k < len(output_names):
                    fig.add_annotation(x=xs[li] + 0.012, y=_ypos(k, len(d)), text=output_names[k], xanchor="left",
                                       showarrow=False, font=dict(size=9, color=MUTED))

    note = "connections coloured by live weights (orange +, blue −)" if weights is not None else \
        "configured network (weights shown once the model is built)"
    if n_params is not None:
        note = f"{n_params:,} trainable parameters · " + note
    fig.add_annotation(x=0.5, y=-0.1, xref="paper", yref="paper", text=note, showarrow=False,
                       font=dict(size=10, color=MUTED))
    fig.update_layout(**base_layout(height=height, margin=dict(l=8, r=40, t=44, b=30), showlegend=False,
                                    uirevision=None))
    fig.update_xaxes(visible=False, range=[0, 1.04], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.08, 1.12], fixedrange=True)
    return fig


def count_parameters(sizes):
    return int(sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1)))
