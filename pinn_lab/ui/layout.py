"""
layout.py -- PINNlab page structure.

Main page (kept clean): current model + status, training controls and live numbers, the animated
physics view from the PINN prediction, live loss curves, prediction vs reference with RMSE/MAE/R2,
and compact one-line summaries of the network, data and checkpoint state.

Everything detailed lives in the SETTINGS panel (always mounted, shown/hidden with CSS so every
control keeps its value): A. PINN Architecture, B. Measurement Data, C. Checkpoints, D. Training,
E. Physics.
"""
from dash import dcc, html

from pinn.trainer import TrainConfig, cuda_available, gpu_name

SECTIONS = [("arch", "A · PINN Architecture"), ("data", "B · Measurement Data"), ("ckpt", "C · Checkpoints"),
            ("train", "D · Training"), ("phys", "E · Physics")]


def on(flag):
    return ["on"] if flag else []


# config field id -> (TrainConfig attribute, kind)
CFG_FIELDS = [
    ("cfg-network", "network", str), ("cfg-fourier", "fourier_features", int), ("cfg-layers", "hidden_layers", int),
    ("cfg-neurons", "neurons", int), ("cfg-act", "activation", str), ("norm-mode", "normalization", str),
    ("cfg-device", "device", str), ("cfg-precision", "precision", str),
    ("cfg-lr", "learning_rate", float), ("cfg-epochs", "epochs", int), ("cfg-optimizer", "optimizer", str),
    ("cfg-wd", "weight_decay", float), ("cfg-scheduler", "scheduler", str), ("cfg-decay", "lr_decay", float),
    ("cfg-stepsize", "step_size", int), ("cfg-batch", "batch_size", int), ("cfg-early", "early_stopping", bool),
    ("cfg-patience", "patience", int), ("cfg-mindelta", "min_delta", float), ("cfg-ckpt", "checkpoint_every", int),
    ("cfg-eval", "eval_every", int), ("cfg-seed", "seed", int),
    ("cfg-lphys", "lambda_physics", float), ("cfg-ldata", "lambda_data", float), ("cfg-lic", "lambda_ic", float),
    ("cfg-lbc", "lambda_bc", float), ("cfg-colloc", "n_collocation", int),
    ("cfg-sampling", "collocation_sampling", str), ("cfg-resscale", "residual_scaling", str),
    ("cfg-normdata", "normalize_data", bool),
]
CFG_IDS = [f for f, _, _ in CFG_FIELDS]


def config_values(c):
    return [on(getattr(c, a)) if kind is bool else getattr(c, a) for _, a, kind in CFG_FIELDS]


def config_from_values(vals):
    kw = {}
    for (fid, attr, kind), v in zip(CFG_FIELDS, vals):
        if kind is bool:
            kw[attr] = bool(v)
            continue
        if v is None or v == "":
            raise ValueError(f"setting '{attr.replace('_', ' ')}' is empty")
        kw[attr] = kind(v)
    return TrainConfig(**kw).validate()


# --------------------------------------------------------------- helpers
def card(title, children, cls="", right=None, icon=None, cid=None):
    head = [html.Span(icon, className="card-icon") if icon else None, html.H3(title)]
    kw = {"id": cid} if cid else {}
    return html.Section(className=f"card {cls}", **kw, children=[
        html.Header(className="card-head", children=[html.Div(head, className="card-title"), right]),
        html.Div(children, className="card-body"),
    ])


def field(label, component, hint=None):
    return html.Label(className="field", children=[html.Span(label, className="field-label"), component,
                                                   html.Span(hint, className="field-hint") if hint else None])


def num(id_, value, step="any", min_=None, max_=None, cls="num"):
    return dcc.Input(id=id_, type="number", value=value, step=step, min=min_, max=max_, debounce=True, className=cls)


def dd(id_, value, options, **kw):
    return dcc.Dropdown(id=id_, className="dd", clearable=False, value=value, options=options, **kw)


def check(id_, label, value):
    return dcc.Checklist(id=id_, className="check inline", options=[{"label": f" {label}", "value": "on"}],
                         value=value)


def metric(id_, label, cls=""):
    return html.Div(className=f"metric {cls}", children=[html.Span(label, className="m-label"),
                                                         html.Span("—", id=id_, className="m-value")])


def chip(id_, label):
    return html.Div(className="chip", children=[html.Span(label, className="chip-k"),
                                               html.Span("—", id=id_, className="chip-v")])


def device_options():
    gpu = gpu_name()
    return [
        {"label": f"Auto — {'GPU: ' + gpu if gpu else 'CPU (no CUDA GPU found)'}", "value": "auto"},
        {"label": f"GPU (CUDA){': ' + gpu if gpu else ' — not available'}", "value": "cuda",
         "disabled": not cuda_available()},
        {"label": "CPU", "value": "cpu"},
    ]


# --------------------------------------------------------------- main page
def topbar(links):
    nav = [html.A("⌂ Home", href=links["home"], className="nav-link")] if links.get("home") else []
    if links.get("bondlab"):
        nav.append(html.A("BondLab", href=links["bondlab"], className="nav-link"))
    nav.append(html.Span("PINNlab", className="nav-link active"))
    return html.Header(className="topbar", children=[
        html.Div(className="brand", children=[
            html.Div(className="logo", children=[html.Span("∂"), html.Span("x")]),
            html.Div([html.H1("PINNLAB"),
                      html.P("Physics-Informed Engineering Lab · bond-graph equations → PyTorch PINN")]),
        ]),
        html.Nav(className="nav", children=nav),
        html.Div(className="top-right", children=[
            html.Div(id="status-pill", className="status-pill idle", children=[html.Span(className="dot"), "Idle"]),
            html.Button("⚙ SETTINGS", id="btn-settings", className="btn settings-btn"),
        ]),
    ])


def model_bar(links):
    custom = (html.A("CUSTOM → BondLab", id="lnk-custom", href=links["bondlab_custom"], className="btn")
              if links.get("bondlab_custom") else html.Button("CUSTOM (edit equations)", id="btn-custom", className="btn"))
    return html.Section(className="card span-12 modelbar", children=[
        html.Div(className="mb-left", children=[
            html.Div(className="workflow", children=[
                html.Span("Workflow", className="field-label"),
                html.Div(className="seg", children=[html.Button("DEMO", id="btn-demo", className="btn"), custom]),
            ]),
            html.Div(className="proj-pick", children=[
                html.Span("Project", className="field-label"),
                dcc.Dropdown(id="project-select", className="dd", clearable=False, searchable=False, options=[]),
            ]),
        ]),
        html.Div(className="mb-main", children=[
            html.Div(className="mb-title-row", children=[html.Span(id="model-badge", className="badge"),
                                                          html.H2(id="model-title", className="model-title")]),
            html.Div(id="model-meta", className="model-meta"),
            html.Div(className="chips", children=[
                chip("sum-arch", "PINN"), chip("sum-data", "Measurement Data"), chip("sum-ckpt", "Checkpoint"),
                chip("sum-device", "Device"),
            ]),
            html.Details(className="eq-details", children=[
                html.Summary("View received equations"),
                html.Div(className="eq-grid", children=[
                    dcc.Markdown(id="eq-view", mathjax=True, className="residuals"),
                    dcc.Graph(id="fig-bondgraph", config={"displayModeBar": False}),
                ]),
            ]),
        ]),
    ])


def training_bar():
    return html.Section(className="card span-12 trainbar", children=[
        html.Div(className="controls", children=[
            html.Button("▶ Start training", id="btn-start", className="btn primary"),
            html.Button("❚❚ Pause", id="btn-pause", className="btn"),
            html.Button("▶ Resume", id="btn-resume", className="btn"),
            html.Button("⤓ Save checkpoint", id="btn-save", className="btn"),
            html.Div(className="continue", children=[
                html.Button("⟳ Continue training", id="btn-continue", className="btn accent"),
                html.Span("for", className="muted"),
                dcc.Input(id="cont-epochs", type="number", value=2000, min=1, step=1, className="num small"),
                html.Span("more epochs", className="muted"),
            ]),
            html.Button("⟲ Reset", id="btn-reset", className="btn danger"),
        ]),
        html.Div(className="metrics", children=[
            metric("m-epoch", "Epoch", "wide"), metric("m-total", "Total loss", "hl"), metric("m-data", "Data loss"),
            metric("m-phys", "Physics loss"), metric("m-bc", "BC loss"), metric("m-ic", "IC loss"),
            metric("m-val", "Validation"), metric("m-lr", "Learning rate"), metric("m-time", "Training time"),
            metric("m-speed", "Speed"), metric("m-device", "Device", "wide2"),
        ]),
        html.Div(className="progress", children=[html.Div(id="progress-fill", className="progress-fill")]),
        html.Div(id="action-msg", className="action-msg"),
    ])


def physics_card():
    return card("Physics Visualization — PINN prediction", icon="◈", cls="span-7", children=[
        html.Div(id="anim-root", className="anim-root"),
        html.Div(className="note", children=[
            "Drawn from the network's own output: positions, levels and colours are the PINN prediction; particle "
            "speed, streams and arrows are predicted flows (d/dt by torch.autograd) and the input signals. "
            "Dashed = reference solution."]),
        dcc.Store(id="anim-data"), dcc.Store(id="anim-ack"),
    ])


def loss_card():
    return card("Live Loss Curves", icon="∿", cls="span-5", right=dcc.Checklist(
        id="log-y", options=[{"label": " log y", "value": "log"}], value=["log"], className="check inline"),
        children=[
            dcc.Graph(id="fig-loss", config={"displaylogo": False}),
            dcc.Graph(id="fig-lr", config={"displayModeBar": False}),
            html.Div("L_total = λ_data·L_data + λ_physics·L_physics + λ_BC·L_BC + λ_IC·L_IC — L_physics is the mean "
                     "squared residual of the transferred bond-graph equations.", className="note"),
        ])


def comparison_card():
    return card("Prediction vs Reference", icon="≈", cls="span-12", right=html.Div(className="row gap", children=[
        dcc.Dropdown(id="cmp-obs", className="dd narrow", clearable=False, options=[]),
        dcc.RadioItems(id="cmp-src", className="radio inline", value="reference", options=[
            {"label": "vs reference ODE", "value": "reference"},
            {"label": "vs validation data", "value": "validation"}]),
    ]), children=[
        html.Div(className="cmp-grid", children=[
            dcc.Graph(id="fig-cmp", config={"displaylogo": False}),
            html.Div(id="cmp-table"),
        ]),
    ])


# --------------------------------------------------------------- settings
def arch_section(c):
    return html.Div(id="set-arch", className="set-section", children=[
        html.H4("A · PINN Architecture"),
        html.Div(id="arch-io", className="note"),
        html.Div(className="grid4", children=[
            field("Network type", dd("cfg-network", c.network, [
                {"label": "MLP (fully connected)", "value": "mlp"},
                {"label": "Residual MLP (skip connections)", "value": "resnet"},
                {"label": "Fourier-feature MLP", "value": "fourier"}])),
            field("Fourier features", num("cfg-fourier", c.fourier_features, 1, 1, 64), "fourier network only"),
            field("Hidden layers", num("cfg-layers", c.hidden_layers, 1, 1, 12)),
            field("Neurons / layer", num("cfg-neurons", c.neurons, 1, 2, 512)),
            field("Activation", dd("cfg-act", c.activation, [{"label": a, "value": a} for a in
                                                               ("sin", "tanh", "silu", "gelu", "softplus")])),
            field("Normalization (output scaling)", dd("norm-mode", c.normalization, [
                {"label": "Standardize (μ, σ)", "value": "standard"},
                {"label": "Min-max → [-1, 1]", "value": "minmax"},
                {"label": "None (raw units)", "value": "none"}])),
            field("Device", dd("cfg-device", c.device, device_options())),
            field("Precision", dd("cfg-precision", c.precision, [
                {"label": "Auto (f32 GPU / f64 CPU)", "value": "auto"},
                {"label": "float32", "value": "float32"}, {"label": "float64", "value": "float64"}])),
        ]),
        dcc.Graph(id="fig-arch", config={"displayModeBar": False}),
        html.Div(id="arch-flow", className="flow"),
    ])


def data_section(st, c):
    return html.Div(id="set-data", className="set-section", style={"display": "none"}, children=[
        html.H4("B · Measurement Data"),
        dcc.RadioItems(id="data-source", className="radio inline", value=st.get("source", "synthetic"), options=[
            {"label": "Synthetic (sampled from the reference + noise)", "value": "synthetic"},
            {"label": "CSV file", "value": "csv"}, {"label": "None — physics only", "value": "none"}]),
        html.Div(id="syn-box", children=[
            html.Div(className="grid3", children=[
                field("Measured quantities (targets)", dcc.Checklist(id="syn-obs", className="check", options=[],
                                                                     value=st.get("observables") or [])),
                field("Points", num("syn-n", st.get("n_points", 30), 1, 2, 5000)),
                field("Noise (% of std)", num("syn-noise", st.get("noise_pct", 1.0), "any", 0, 100)),
            ]),
        ]),
        html.Div(id="csv-box", style={"display": "none"}, children=[
            dcc.Upload(id="upload-csv", className="upload", accept=".csv,text/csv",
                       children=html.Div(["Drop a CSV here or ", html.B("browse")])),
            html.Div(id="csv-status", className="note"),
            html.Div(className="grid2", children=[
                field("Input feature (time column)", dcc.Dropdown(id="csv-time", className="dd", options=[],
                                                                  value=st.get("time_column"))),
                html.Div(id="csv-map"),
            ]),
        ]),
        html.Div(className="grid3", children=[
            field("Train / validation split", dcc.Slider(id="val-split", min=0, max=0.5, step=0.05,
                                                         value=st.get("val_fraction", 0.2),
                                                         marks={0: "0", 0.25: "25%", 0.5: "50%"},
                                                         tooltip={"placement": "bottom"})),
            field("Normalize", check("cfg-normdata", "divide misfit by each signal's scale", on(c.normalize_data))),
        ]),
        html.Div(className="row gap", children=[
            html.Button("Apply / replace dataset", id="btn-data", className="btn primary"),
            html.Button("Remove dataset", id="btn-data-remove", className="btn danger"),
            html.Button("Download reference CSV", id="btn-dl-ref", className="btn ghost"),
            dcc.Download(id="dl-ref"),
        ]),
        html.Div(id="data-summary", className="note"),
        html.Div(id="data-preview", className="preview"),
    ])


def ckpt_section():
    return html.Div(id="set-ckpt", className="set-section", style={"display": "none"}, children=[
        html.H4("C · Checkpoints"),
        html.Div(id="ckpt-info", className="ckpt-info"),
        html.Div(className="grid-ck", children=[
            field("Checkpoint file", dcc.Dropdown(id="ckpt-select", className="dd", options=[],
                                                  placeholder="Select a checkpoint…")),
            html.Div(className="row gap ck-actions", children=[
                html.Button("Load", id="btn-load", className="btn"),
                html.Button("Load & resume", id="btn-ck-resume", className="btn accent"),
                html.Button("Delete", id="btn-ck-delete", className="btn danger"),
                html.Button("Save now", id="btn-ck-save", className="btn"),
            ]),
        ]),
        html.Div(id="ckpt-list", className="ckpt-list"),
        dcc.Upload(id="upload-ckpt", className="upload small", accept=".pt",
                   children=html.Div(["Import a .pt checkpoint into this project: drop or ", html.B("browse")])),
        html.Div("Each file holds the model and optimizer state_dicts, LR-scheduler state, epoch, every loss term, "
                 "best validation loss, the configuration, equations, dataset and loss history. Files are read with "
                 "torch.load(weights_only=True).", className="note"),
    ])


def train_section(c):
    return html.Div(id="set-train", className="set-section", style={"display": "none"}, children=[
        html.H4("D · Training"),
        html.Div(className="grid4", children=[
            field("Learning rate", num("cfg-lr", c.learning_rate, "any", 1e-6, 1)),
            field("Epochs", num("cfg-epochs", c.epochs, 1, 1, 1_000_000)),
            field("Optimizer", dd("cfg-optimizer", c.optimizer, [
                {"label": "Adam", "value": "adam"}, {"label": "AdamW", "value": "adamw"},
                {"label": "RMSprop", "value": "rmsprop"}, {"label": "SGD + momentum", "value": "sgd"},
                {"label": "L-BFGS (full batch, lr≈1)", "value": "lbfgs"}])),
            field("Weight decay", num("cfg-wd", c.weight_decay, "any", 0)),
            field("Scheduler", dd("cfg-scheduler", c.scheduler, [
                {"label": "None", "value": "none"}, {"label": "Exponential", "value": "exponential"},
                {"label": "Cosine annealing", "value": "cosine"}, {"label": "Step (×0.5)", "value": "step"},
                {"label": "Reduce on plateau", "value": "plateau"}])),
            field("Exp. decay γ / epoch", num("cfg-decay", c.lr_decay, "any", 0.9, 1)),
            field("Step size (epochs)", num("cfg-stepsize", c.step_size, 1, 1)),
            field("Batch size", num("cfg-batch", c.batch_size, 1, 0, 20000), "collocation pts/step; 0 = all"),
            field("Early stopping", check("cfg-early", "stop when validation stalls", on(c.early_stopping))),
            field("Patience (evaluations)", num("cfg-patience", c.patience, 1, 1)),
            field("Min. rel. improvement", num("cfg-mindelta", c.min_delta, "any", 0)),
            field("Checkpoint every (epochs)", num("cfg-ckpt", c.checkpoint_every, 1, 1)),
            field("Evaluate every (epochs)", num("cfg-eval", c.eval_every, 1, 1)),
            field("Seed", num("cfg-seed", c.seed, 1, 0)),
        ]),
    ])


def phys_section(c):
    return html.Div(id="set-phys", className="set-section", style={"display": "none"}, children=[
        html.H4("E · Physics"),
        html.Div(id="phys-origin", className="note"),
        dcc.Markdown(id="eq-view-full", mathjax=True, className="residuals"),
        html.H5("Physics-loss weights"),
        html.Div(className="grid4", children=[
            field("λ physics", num("cfg-lphys", c.lambda_physics, "any", 0)),
            field("λ data", num("cfg-ldata", c.lambda_data, "any", 0)),
            field("λ IC", num("cfg-lic", c.lambda_ic, "any", 0)),
            field("λ BC", num("cfg-lbc", c.lambda_bc, "any", 0)),
        ]),
        html.H5("Physical parameters, initial and boundary conditions"),
        html.Div(className="grid2", children=[
            html.Div(id="phys-params"),
            html.Div([html.Div(id="phys-ic"),
                      field("Boundary / point constraints (one per line: variable(t) = value)",
                            dcc.Textarea(id="phys-bc", className="code small-code", spellCheck=False)),
                      field("Time horizon t_end", num("phys-tend", None, "any"))]),
        ]),
        html.Div(className="row gap", children=[
            html.Button("Apply physics changes", id="btn-phys-apply", className="btn primary"),
            html.Span(id="phys-status", className="compile-status"),
        ]),
        html.H5("Residual configuration"),
        html.Div(className="grid4", children=[
            field("Collocation points", num("cfg-colloc", c.n_collocation, 1, 16, 20000)),
            field("Collocation sampling", dd("cfg-sampling", c.collocation_sampling, [
                {"label": "Fixed grid (random mini-batches)", "value": "grid"},
                {"label": "Fresh random points each epoch", "value": "random"}])),
            field("Residual scaling", dd("cfg-resscale", c.residual_scaling, [
                {"label": "Normalized (dx/dt − f)/(σ/Δt)", "value": "normalized"},
                {"label": "Physical units dx/dt − f", "value": "physical"}])),
        ]),
        html.Details(className="eq-details", children=[
            html.Summary("Advanced: edit the model text directly"),
            dcc.Textarea(id="model-text", className="code", spellCheck=False),
            html.Div(className="row gap", children=[
                html.Button("Compile equations → residuals", id="btn-compile", className="btn"),
                html.Span(id="compile-status", className="compile-status"),
            ]),
        ]),
    ])


def settings_overlay(c, st):
    return html.Div(id="settings-overlay", className="settings-overlay", children=[
        html.Div(className="settings-panel", children=[
            html.Header(className="settings-head", children=[
                html.H3("⚙ Settings"),
                dcc.RadioItems(id="settings-nav", className="settings-nav", value="arch",
                               options=[{"label": lab, "value": key} for key, lab in SECTIONS]),
                html.Div(className="row gap", children=[
                    html.Button("Save settings to project", id="btn-save-settings", className="btn primary"),
                    html.Button("✕", id="btn-settings-close", className="btn icon", title="Close"),
                ]),
            ]),
            html.Div(id="settings-msg", className="settings-msg"),
            html.Div(className="settings-body", children=[
                arch_section(c), data_section(st, c), ckpt_section(), train_section(c), phys_section(c)]),
        ]),
    ])


def build_layout(lab, links):
    lab.ensure_loaded()
    c, st = lab.config, lab.data_settings
    return html.Div(className="app", children=[
        dcc.Location(id="url", refresh=False),
        topbar(links),
        html.Main(className="grid", children=[
            model_bar(links), training_bar(), physics_card(), loss_card(), comparison_card()]),
        settings_overlay(c, st),
        html.Footer(className="foot", children=[
            "Every number, curve and picture comes from the running PyTorch model, the equations transferred from "
            "BondLab, or files in the project folder. Reference: SciPy solve_ivp on the same equations."]),
        dcc.Store(id="compiled-rev", data=0), dcc.Store(id="data-rev", data=0),
        dcc.Store(id="seen-version", data=-1), dcc.Store(id="loss-seen", data=None),
        dcc.Store(id="ckpt-rev", data=""), dcc.Store(id="url-clean"), dcc.Store(id="url-clean-ack"),
        dcc.Interval(id="tick", interval=700),
    ])
