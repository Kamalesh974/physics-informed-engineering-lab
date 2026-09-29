"""Shared dark theme for every Plotly figure in the dashboard."""
BG = "#0b1018"
PANEL = "#0f1622"
GRID = "#1c2635"
AXIS = "#2a3a50"
TEXT = "#d7e0ec"
MUTED = "#8394ab"
BLUE = "#4f9dff"
BLUE_DIM = "#2b5d9c"
ORANGE = "#ff9f43"
GREEN = "#3ddc97"
PURPLE = "#b18cff"
PINK = "#ff6fae"
RED = "#ff5d5d"

LOSS_COLORS = {"total": "#eef3fa", "data": BLUE, "physics": ORANGE, "bc": PURPLE, "ic": GREEN, "val": PINK}

FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace"


def base_layout(**kw):
    layout = dict(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family=FONT, color=TEXT, size=12),
        margin=dict(l=56, r=18, t=30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED), orientation="h",
                    yanchor="bottom", y=1.0, x=0, xanchor="left"),
        hoverlabel=dict(bgcolor="#162132", bordercolor=AXIS, font=dict(family=MONO, size=11, color=TEXT)),
        uirevision="keep",
    )
    layout.update(kw)
    return layout


def axis(**kw):
    a = dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED, size=11),
             title_font=dict(color=MUTED, size=11), showline=True, mirror=False, ticks="outside",
             tickcolor=AXIS, ticklen=4)
    a.update(kw)
    return a


def empty_figure(message, height=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(**base_layout(height=height, margin=dict(l=10, r=10, t=10, b=10)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    return fig


def lerp_color(stops, u):
    """Piecewise-linear colour map; stops = [(pos, '#rrggbb'), ...], u in [0, 1]."""
    u = min(max(float(u), 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if u <= p1:
            f = 0.0 if p1 == p0 else (u - p0) / (p1 - p0)
            a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
            b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
            return "#%02x%02x%02x" % tuple(round(x + (y - x) * f) for x, y in zip(a, b))
    return stops[-1][1]


THERMAL = [(0.0, "#1d4ed8"), (0.35, "#22d3ee"), (0.6, "#facc15"), (0.8, "#fb923c"), (1.0, "#ef4444")]
DIVERGING = [(0.0, "#4f9dff"), (0.5, "#2a3444"), (1.0, "#ff9f43")]
