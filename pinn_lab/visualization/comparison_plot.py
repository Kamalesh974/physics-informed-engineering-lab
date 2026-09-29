"""
comparison_plot.py -- reference vs PINN prediction, error curve and metrics.

The metrics are computed from real arrays: the PINN prediction evaluated at
the reference times (or at the held-out measurement times) against the
reference ODE solution (or the measured values).
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import BLUE, MUTED, ORANGE, PINK, axis, base_layout, empty_figure

REF_COLOR = "#e6edf7"


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return None
    err = y_pred - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(y_true.size),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "nrmse": float(np.sqrt(np.mean(err ** 2)) / (np.std(y_true) or 1.0)),
    }


def metrics_table(system, predict, ref, dataset, source):
    """One row per observable. predict(t_array) -> {obs: array}."""
    rows = []
    if source == "reference" and ref is not None:
        p = predict(ref["t"])
        for o in system.observables:
            m = metrics(ref["observables"][o], p[o])
            if m:
                rows.append(dict(m, observable=o, label=system.label(o)))
    elif source == "validation" and dataset is not None and dataset.has_val:
        p = predict(dataset.t_val)
        for o, y in dataset.y_val.items():
            m = metrics(y, p[o])
            if m:
                rows.append(dict(m, observable=o, label=system.label(o)))
    return rows


def comparison_figure(system, obs, pred, ref, dataset, height=380):
    if system is None or obs is None:
        return empty_figure("Compile the equations first", height)
    if pred is None:
        return empty_figure("No PINN prediction yet — start training or load a checkpoint", height)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    if ref is not None:
        fig.add_trace(go.Scatter(x=ref["t"], y=ref["observables"][obs], mode="lines", name="Reference (ODE solver)",
                                 line=dict(color=REF_COLOR, width=1.6, dash="dash"),
                                 hovertemplate="ref %{y:.5g}<extra></extra>"), row=1, col=1)
    if dataset is not None and obs in dataset.y_train:
        fig.add_trace(go.Scatter(x=dataset.t_train, y=dataset.y_train[obs], mode="markers", name="Training data",
                                 marker=dict(color=BLUE, size=6, line=dict(color="#0b1018", width=1)),
                                 hovertemplate="data %{y:.5g}<extra></extra>"), row=1, col=1)
    if dataset is not None and obs in dataset.y_val:
        fig.add_trace(go.Scatter(x=dataset.t_val, y=dataset.y_val[obs], mode="markers", name="Validation data",
                                 marker=dict(color="rgba(0,0,0,0)", size=8, line=dict(color=PINK, width=1.6)),
                                 hovertemplate="val %{y:.5g}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pred["t"], y=pred["observables"][obs], mode="lines",
                             name=f"PINN (epoch {pred['epoch']})", line=dict(color=ORANGE, width=2.4),
                             hovertemplate="PINN %{y:.5g}<extra></extra>"), row=1, col=1)
    if ref is not None:
        p_at_ref = np.interp(ref["t"], pred["t"], pred["observables"][obs])
        err = p_at_ref - ref["observables"][obs]
        fig.add_trace(go.Scatter(x=ref["t"], y=err, mode="lines", name="Error (PINN − ref)",
                                 line=dict(color=PINK, width=1.4), fill="tozeroy", fillcolor="rgba(255,111,174,0.12)",
                                 hovertemplate="err %{y:.3e}<extra></extra>"), row=2, col=1)
    fig.update_layout(**base_layout(height=height, margin=dict(l=66, r=14, t=34, b=40), hovermode="x unified"))
    fig.update_yaxes(**axis(title_text=system.label(obs)), row=1, col=1)
    fig.update_yaxes(**axis(title_text="error", exponentformat="e"), row=2, col=1)
    fig.update_xaxes(**axis(), row=1, col=1)
    fig.update_xaxes(**axis(title_text="t"), row=2, col=1)
    if ref is None:
        fig.add_annotation(text="no reference solution available", x=0.5, y=0.1, xref="paper", yref="paper",
                           showarrow=False, font=dict(color=MUTED))
    return fig
