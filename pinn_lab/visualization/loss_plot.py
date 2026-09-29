"""Live loss curves (log scale) built from the trainer's real loss history."""
import numpy as np
import plotly.graph_objects as go

from .theme import LOSS_COLORS, MUTED, axis, base_layout, empty_figure

SERIES = [("total", "Total"), ("data", "Data"), ("physics", "Physics"), ("bc", "BC"), ("ic", "IC"),
          ("val", "Validation")]
MAX_POINTS = 1500


def _downsample(x, n=MAX_POINTS):
    if len(x) <= n:
        return np.arange(len(x))
    return np.unique(np.linspace(0, len(x) - 1, n).astype(int))


def loss_figure(history, log_y=True, height=360):
    if not history or not history.get("epoch"):
        return empty_figure("Loss curves appear here once training starts", height)
    ep = np.asarray(history["epoch"])
    idx = _downsample(ep)
    fig = go.Figure()
    for key, name in SERIES:
        vals = np.array([np.nan if v is None else v for v in history.get(key, [])], dtype=float)
        if vals.size != ep.size:
            continue
        finite = vals[np.isfinite(vals)]
        if finite.size == 0 or (log_y and np.all(finite <= 0)):
            continue                     # e.g. BC loss when no boundary conditions exist
        y = vals[idx]
        if log_y:
            y = np.where(y > 0, y, np.nan)
        fig.add_trace(go.Scattergl(
            x=ep[idx], y=y, name=name, mode="lines",
            line=dict(color=LOSS_COLORS[key], width=2.2 if key == "total" else 1.5,
                      dash="dot" if key == "val" else "solid"),
            hovertemplate=f"{name}: %{{y:.3e}}<extra>epoch %{{x}}</extra>",
        ))
    fig.update_layout(**base_layout(height=height, margin=dict(l=62, r=16, t=34, b=42)))
    fig.update_xaxes(**axis(title_text="epoch"))
    fig.update_yaxes(**axis(title_text="loss (log scale)" if log_y else "loss",
                            type="log" if log_y else "linear", exponentformat="power"))
    fig.update_layout(hovermode="x unified")
    return fig


def lr_figure(history, height=120):
    if not history or not history.get("epoch"):
        return empty_figure("", height)
    ep = np.asarray(history["epoch"])
    idx = _downsample(ep)
    fig = go.Figure(go.Scattergl(x=ep[idx], y=np.asarray(history["lr"])[idx], mode="lines",
                                 line=dict(color=MUTED, width=1.3), name="learning rate",
                                 hovertemplate="lr %{y:.2e}<extra>epoch %{x}</extra>"))
    fig.update_layout(**base_layout(height=height, margin=dict(l=62, r=16, t=6, b=30), showlegend=False))
    fig.update_xaxes(**axis())
    fig.update_yaxes(**axis(type="log", title_text="LR", exponentformat="power"))
    return fig
