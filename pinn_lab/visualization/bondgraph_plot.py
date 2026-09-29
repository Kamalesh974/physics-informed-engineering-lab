"""Bond graph drawing (nodes, half-arrow bonds, causality strokes) for a BondLab template."""
import math

import plotly.graph_objects as go

from .theme import BLUE, MUTED, ORANGE, base_layout, empty_figure

NODE_COLORS = {"0": "#8fb8ff", "1": "#8fb8ff", "Se": ORANGE, "Sf": ORANGE, "MSe": ORANGE, "MSf": ORANGE,
               "I": "#3ddc97", "C": "#3ddc97", "R": "#b18cff", "GY": "#ffd166", "TF": "#ffd166"}


def bondgraph_figure(drawing, height=260):
    if not drawing:
        return empty_figure("No bond graph attached — equations were typed or pasted directly", height)
    nodes = {n["id"]: n for n in drawing["nodes"]}
    lay = drawing.get("layout") or {}
    pos = {}
    for k, nid in enumerate(nodes):
        p = lay.get(nid) or {"x": 150 * (k % 5), "y": 120 * (k // 5)}
        pos[nid] = (float(p["x"]), -float(p["y"]))
    fig = go.Figure()
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    r = 0.07 * span   # node "radius" for trimming bond ends
    for e in drawing["edges"]:
        (x0, y0), (x1, y1) = pos[e["from"]], pos[e["to"]]
        d = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        ax, ay, bx, by = x0 + ux * r, y0 + uy * r, x1 - ux * r, y1 - uy * r
        signal = e.get("signal")
        fig.add_shape(type="line", x0=ax, y0=ay, x1=bx, y1=by,
                      line=dict(color=MUTED if signal else "#9fb3cc", width=1.6, dash="dash" if signal else "solid"))
        if not signal:
            # half arrow at the power-receiving end (one barb, on the left side)
            h = 0.045 * span
            fig.add_shape(type="line", x0=bx, y0=by, x1=bx - ux * h - uy * h * 0.7, y1=by - uy * h + ux * h * 0.7,
                          line=dict(color="#9fb3cc", width=1.6))
            # causality stroke at the end that RECEIVES effort
            giver = e.get("effort_giver")
            if giver:
                end = (bx, by) if giver == e["from"] else (ax, ay)
                s = 0.04 * span
                fig.add_shape(type="line", x0=end[0] - uy * s, y0=end[1] + ux * s, x1=end[0] + uy * s,
                              y1=end[1] - ux * s, line=dict(color=ORANGE, width=2.4))
        else:
            h = 0.04 * span
            fig.add_shape(type="line", x0=bx, y0=by, x1=bx - ux * h - uy * h * 0.5, y1=by - uy * h + ux * h * 0.5,
                          line=dict(color=MUTED, width=1.4))
            fig.add_shape(type="line", x0=bx, y0=by, x1=bx - ux * h + uy * h * 0.5, y1=by - uy * h - ux * h * 0.5,
                          line=dict(color=MUTED, width=1.4))
    labels, hover, colors = [], [], []
    for nid, n in nodes.items():
        t = n["type"]
        name = n.get("param_symbol_name") or n.get("expr_param_name") or ""
        labels.append(f"<b>{t}</b>" + (f"<br><span style='font-size:9px'>{name}</span>" if name else ""))
        hover.append(f"{nid}: {n.get('label', '')} ({n.get('domain', '')})")
        colors.append(NODE_COLORS.get(t, BLUE))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=labels,
        textfont=dict(color="#0b1018", size=11), hovertext=hover, hoverinfo="text",
        marker=dict(size=34, color=colors, line=dict(color="#0b1018", width=2), symbol="circle"),
        showlegend=False))
    pad = 0.14 * span
    fig.update_layout(**base_layout(height=height, margin=dict(l=6, r=6, t=8, b=6), uirevision=None))
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad], fixedrange=True)
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad], fixedrange=True, scaleanchor="x")
    fig.add_annotation(text="half-arrow = power direction · orange stroke = causality (effort-receiving end) · "
                            "dashed = signal", x=0.5, y=0.0, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=9, color=MUTED), yanchor="bottom")
    return fig
