"""
callbacks.py -- everything PINNlab does when the user clicks, types or waits.

Every visible number comes from LAB (ui/state.py): the compiled equations of the current project,
the live TrainingSession, or files in the project folder.
"""
import base64
import math
import time
from dataclasses import asdict
from urllib.parse import parse_qs

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from bondgraph.equations import ModelError, update_model_text
from data import dataset_loader as dl
from pinn import checkpoint as ckpt
from pinn.model import NETWORK_LABELS
from pinn.trainer import TrainingSession
from storage import project_store as store
from visualization.architecture_plot import architecture_figure, count_parameters
from visualization.bondgraph_plot import bondgraph_figure
from visualization.comparison_plot import comparison_figure, metrics_table
from visualization.loss_plot import loss_figure, lr_figure
from visualization.physics_visualization import animation_payload

from .layout import CFG_FIELDS, CFG_IDS, SECTIONS, config_from_values, config_values
from .state import DEFAULT_DATA, LAB

STATUS_TEXT = {
    None: ("Idle", "idle"), "ready": ("Ready", "idle"), "training": ("Training", "training"),
    "paused": ("Paused", "paused"), "completed": ("Completed", "completed"), "stopped": ("Stopped", "idle"),
    "checkpoint_loaded": ("Checkpoint Loaded", "st-loaded"), "error": ("Error", "error"),
}
CFG_STATES = [State(i, "value") for i in CFG_IDS]
CFG_OUT = [Output(i, "value", allow_duplicate=True) for i in CFG_IDS]
CFG_OUT_PRIMARY = [Output(i, "value") for i in CFG_IDS]


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{digits}e}"


def fmt_time(s):
    s = float(s or 0)
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def msg(text, kind="info"):
    return html.Div(text, className=f"msg {kind}", key=str(time.time()))


def residual_markdown(system, compact=False):
    lines = [] if compact else [
        "Each equation received from BondLab becomes a residual that PyTorch evaluates on the network output with "
        "automatic differentiation; the mean square of all residuals is **L_physics**.", ""]
    for _s, eq, res in system.residual_latex():
        lines += [f"$${eq}$$"] + ([] if compact else [f"$${res}$$"]) + [""]
    lines += ["Parameters: " + ", ".join(f"`{k} = {v:g}`" for k, v in system.params.items())]
    if system.inputs:
        lines += ["", "Inputs: " + ", ".join(f"`{k}(t) = {v}`" for k, v in system.inputs.items())]
    lines += ["", "Initial conditions (L_IC): " + ", ".join(f"`{k}(t0) = {v:g}`" for k, v in system.initial.items())]
    if system.boundary:
        lines += ["", "Boundary / point constraints (L_BC): "
                  + ", ".join(f"`{o}({tb:g}) = {v:g}`" for o, tb, v in system.boundary)]
    else:
        lines += ["", "No boundary constraints: L_BC = 0."]
    return "\n".join(lines)


def _data_controls(st, system):
    obs = st.get("observables") or (LAB.default_observables(system) if system else [])
    return [st.get("source", "synthetic"), obs, st.get("n_points", 30), st.get("noise_pct", 1.0),
            st.get("val_fraction", 0.2), st.get("time_column")]


DATA_OUT = [Output("data-source", "value"), Output("syn-obs", "value"), Output("syn-n", "value"),
            Output("syn-noise", "value"), Output("val-split", "value"), Output("csv-time", "value")]


def register(app, links):
    integrated = bool(links.get("bondlab"))

    # ================================================================ project router
    @app.callback(
        Output("action-msg", "children", allow_duplicate=True), Output("compiled-rev", "data"),
        Output("url-clean", "data"), *CFG_OUT_PRIMARY, *DATA_OUT,
        Input("url", "search"), Input("btn-demo", "n_clicks"), Input("project-select", "value"),
        State("compiled-rev", "data"), prevent_initial_call="initial_duplicate")
    def router(search, _demo, selected, rev):
        trig = ctx.triggered_id
        LAB.ensure_loaded()
        current = LAB.project.id if LAB.project is not None else None
        n_out = 3 + len(CFG_FIELDS) + len(DATA_OUT)
        if trig == "project-select" and (not selected or selected == current):
            return (no_update,) * n_out          # the dropdown was only being synced to the current project
        if trig == "btn-demo":
            pid = store.DEMO_ID
        elif trig == "project-select":
            pid = selected
        else:   # page load (url): ?project=<id> is how PINNSIM hands a model over
            q = parse_qs((search or "").lstrip("?"))
            pid = (q.get("project") or [None])[0]
        if not pid or (pid == current and trig != "url"):
            vals = config_values(LAB.config) + _data_controls(LAB.data_settings, LAB.system)
            return (no_update, no_update, no_update, *vals)
        if pid == current:      # arrived from PINNSIM for the project that is already open (just refreshed)
            try:
                p = LAB.load_project(pid) if not LAB.training_alive else LAB.project
                note = _connected_msg(p) if p.source == "bondlab" else None
            except Exception as e:  # noqa: BLE001
                note = msg(f"✗ Could not reload project '{pid}': {e}", "error")
            vals = config_values(LAB.config) + _data_controls(LAB.data_settings, LAB.system)
            return (note or no_update, (rev or 0) + 1, pid, *vals)
        try:
            p = LAB.load_project(pid)
            note = _connected_msg(p) if p.source == "bondlab" else msg(f"Loaded project “{p.name}”.", "ok")
        except Exception as e:  # noqa: BLE001
            note = msg(f"✗ Could not open project '{pid}': {e}", "error")
        vals = config_values(LAB.config) + _data_controls(LAB.data_settings, LAB.system)
        return (note, (rev or 0) + 1, pid, *vals)

    def _connected_msg(p):
        eq = p.equations() or {}
        return msg(f"BondLab Model Connected ✓ — “{p.name}”: {len(eq.get('equations', []))} governing equation(s), "
                   f"{len(eq.get('parameters', []))} parameter(s), {len(eq.get('initial_conditions', []))} IC(s), "
                   f"{len(eq.get('boundary_conditions', []))} BC(s) received {eq.get('transferred', '')}. They now "
                   "define L_physics.", "ok")

    app.clientside_callback(
        "function(p){ if (p) { try { history.replaceState(null, '', window.location.pathname); } catch(e){} }"
        " return window.dash_clientside.no_update; }",
        Output("url-clean-ack", "data"), Input("url-clean", "data"))

    # ================================================================ views of the project
    @app.callback(
        Output("model-title", "children"), Output("model-badge", "children"), Output("model-badge", "className"),
        Output("model-meta", "children"), Output("eq-view", "children"), Output("fig-bondgraph", "figure"),
        Output("project-select", "options"), Output("project-select", "value"),
        Output("syn-obs", "options"), Output("cmp-obs", "options"), Output("cmp-obs", "value"),
        Output("csv-map", "children"), Output("phys-params", "children"), Output("phys-ic", "children"),
        Output("phys-bc", "value"), Output("phys-tend", "value"), Output("model-text", "value"),
        Output("eq-view-full", "children"), Output("phys-origin", "children"), Output("arch-io", "children"),
        Input("compiled-rev", "data"), State("cmp-obs", "value"))
    def project_views(_rev, cmp_obs):
        LAB.ensure_loaded()
        p = LAB.project
        system, ref, _ds = LAB.view()
        src = p.source if p else ""
        if src == "bondlab":
            badge, bcls = "BondLab Model Connected ✓", "badge ok"
        elif src == "demo":
            badge, bcls = "Demo Physics Problem", "badge demo"
        else:
            badge, bcls = "Custom equations", "badge"
        meta = (f"{len(system.states)} state(s): {', '.join(system.states)} · {len(system.params)} parameter(s) · "
                f"t ∈ [{system.t_start:g}, {system.t_end:g}]")
        if p and p.meta().get("modified_in_pinnlab"):
            meta += " · edited in PINNlab"
        if LAB.ref_error:
            meta += f" · reference solver failed: {LAB.ref_error}"
        opts = [{"label": ("★ " if q.source == "demo" else "") + f"{q.name}  ({q.source})", "value": q.id}
                for q in store.list_projects()]
        obs_opts = [{"label": system.label(o), "value": o} for o in system.observables]
        cmp_val = cmp_obs if cmp_obs in system.observables else system.states[0]
        params = [html.Div("Parameters", className="field-label")] + [
            html.Div(className="map-row", children=[
                html.Span(system.label(k), className="map-name"),
                dcc.Input(id={"type": "par", "name": k}, type="number", value=v, debounce=True, className="num")])
            for k, v in system.params.items()]
        ics = [html.Div("Initial conditions (t = t_start)", className="field-label")] + [
            html.Div(className="map-row", children=[
                html.Span(system.label(s), className="map-name"),
                dcc.Input(id={"type": "ic", "name": s}, type="number", value=system.initial.get(s, 0.0),
                          debounce=True, className="num")])
            for s in system.states]
        bc_text = "\n".join(f"{o}({tb:g}) = {v:g}" for o, tb, v in system.boundary)
        if src == "bondlab":
            eq = p.equations() or {}
            origin = [html.B("Source: BondLab. "),
                      f"Derived by the {eq.get('derived_by', 'BondLab engine')} from the transferred bond graph "
                      f"(bondgraph/model.json) and stored in bondgraph/equations.json at {eq.get('transferred', '')}."]
        elif src == "demo":
            origin = [html.B("Source: built-in demo. "),
                      "These are BondLab's equations for the mass-spring-damper bond graph."]
        else:
            origin = "Equations entered directly."
        io = [html.B("Input: "), "t (normalised to τ ∈ [−1, 1]).  ", html.B("Outputs: "),
              ", ".join(system.label(s) for s in system.states), ".  Extra readouts from the outputs: ",
              ", ".join(system.outputs) or "none", "."]
        return (system.title, badge, bcls, meta, residual_markdown(system, compact=True),
                bondgraph_figure(LAB.drawing), opts, p.id if p else None, obs_opts, obs_opts, cmp_val,
                _csv_map_children(system), params, ics, bc_text, system.t_end, system.source_text,
                residual_markdown(system), origin, io)

    if integrated:
        @app.callback(Output("lnk-custom", "href"), Input("compiled-rev", "data"))
        def custom_link(_rev):
            p = LAB.project
            base = links["bondlab_custom"]
            return f"{base}&open={p.id}" if p is not None and p.source == "bondlab" else base

    def _csv_map_children(system):
        cols = list(LAB.csv_df.columns) if LAB.csv_df is not None else []
        mapping = (LAB.data_settings or {}).get("mapping") or {}
        rows = [html.Div("Target variables (map CSV columns → states/outputs)", className="field-label")]
        for o in system.observables:
            guess = mapping.get(o) or (o if o in cols else None)
            rows.append(html.Div(className="map-row", children=[
                html.Span(system.label(o), className="map-name"),
                dcc.Dropdown(id={"type": "map", "obs": o}, className="dd", options=cols, value=guess,
                             placeholder="not measured"),
            ]))
        return rows

    # compact summaries on the dashboard
    @app.callback(Output("sum-arch", "children"), Output("sum-data", "children"), Output("sum-ckpt", "children"),
                  Output("sum-device", "children"),
                  Input("compiled-rev", "data"), Input("data-rev", "data"), Input("ckpt-rev", "data"),
                  Input("seen-version", "data"))
    def summaries(*_):
        s = LAB.session
        c = s.config if s is not None else LAB.config
        arch = (s.model.summary() if s is not None else
                f"{c.hidden_layers}-layer {NETWORK_LABELS[c.network]} ({c.neurons}, {c.activation})")
        ds = LAB.dataset
        data = (f"Loaded ✓ ({ds.n_train} train / {ds.n_val} val, {ds.source})" if ds.has_data
                else "None — physics only")
        if s is not None and s.dataset is not ds:
            sd = s.dataset
            data += f" · current run: {sd.n_train}/{sd.n_val}" if sd.has_data else " · current run: physics only"
        if s is not None and s.loaded_from:
            lf = s.loaded_from
            ck = f"Loaded ✓ epoch {lf['epoch']:,} ({lf['file']})"
            if s.saved:
                ck += f" · last saved {s.saved[-1]}"
        elif s is not None and s.saved:
            ck = f"Saved ✓ {s.saved[-1]} ({len(s.saved)} this run)"
        else:
            n = len(ckpt.list_checkpoints(LAB.project.checkpoints_dir)) if LAB.project else 0
            ck = f"{n} in project" if n else "None yet"
        dev = s.device_label() if s is not None else {"auto": "Auto", "cuda": "GPU", "cpu": "CPU"}[c.device]
        return f"{arch}", data, ck, dev

    # ================================================================ settings panel
    @app.callback(Output("settings-overlay", "className"), Output("settings-nav", "value"),
                  Input("btn-settings", "n_clicks"), Input("btn-settings-close", "n_clicks"),
                  *([Input("btn-custom", "n_clicks")] if not integrated else []),
                  State("settings-nav", "value"), prevent_initial_call=True)
    def toggle_settings(*args):
        trig, nav = ctx.triggered_id, args[-1]
        if trig == "btn-settings-close":
            return "settings-overlay", no_update
        if trig == "btn-custom":
            return "settings-overlay open", "phys"
        return "settings-overlay open", nav or "arch"

    @app.callback(*[Output(f"set-{k}", "style") for k, _ in SECTIONS], Input("settings-nav", "value"))
    def show_section(nav):
        return [{"display": "block" if k == nav else "none"} for k, _ in SECTIONS]

    @app.callback(Output("settings-msg", "children"), Input("btn-save-settings", "n_clicks"), *CFG_STATES,
                  prevent_initial_call=True)
    def save_settings(_n, *vals):
        try:
            cfg = config_from_values(vals)
        except (ValueError, TypeError) as e:
            return msg(f"✗ {e}", "error")
        LAB.config = cfg
        if LAB.project is not None:
            LAB.project.save_configuration(asdict(cfg))
        extra = " The running session keeps its own settings until the next Start." if LAB.training_alive else ""
        return msg(f"Settings saved to {LAB.project.id}/pinn/configuration.json.{extra}", "ok")

    @app.callback(Output("fig-arch", "figure"), Output("arch-flow", "children"),
                  Input("cfg-network", "value"), Input("cfg-layers", "value"), Input("cfg-neurons", "value"),
                  Input("cfg-act", "value"), Input("cfg-fourier", "value"),
                  Input("seen-version", "data"), Input("compiled-rev", "data"))
    def arch(network, layers, neurons, act, nf, _v, _r):
        s = LAB.session
        system, _, _ = LAB.view()
        names = list(system.states)
        if s is not None:
            sizes, act_, weights, n_params = s.model.layer_sizes(), s.model.activation, s.weights_view(), s.model.n_parameters()
            label = f"live model: {s.model.summary()} (epoch {s.epoch})"
        else:
            try:
                first = 2 * int(nf) if network == "fourier" else 1
                sizes = [first] + [int(neurons)] * int(layers) + [len(names)]
            except (TypeError, ValueError):
                return no_update, no_update
            act_, weights, n_params = act, None, count_parameters(sizes)
            label = f"configured {NETWORK_LABELS.get(network, network)} (built when training starts)"
        flow = []
        for i, n in enumerate(sizes):
            if i == 0:
                lab = "Input t" if sizes[0] == 1 else f"Fourier features ({n})"
            elif i == len(sizes) - 1:
                lab = "Output " + ", ".join(names)
            else:
                lab = f"Hidden {i} ({n}, {act_})"
            flow += [html.Span(lab, className="flow-node")] + ([html.Span("→", className="flow-arrow")]
                                                               if i < len(sizes) - 1 else [])
        flow.append(html.Span(f" · {label} · d/dt by torch.autograd", className="muted"))
        return architecture_figure(sizes, act_, names, weights, n_params=n_params), flow

    # ================================================================ measurement data
    @app.callback(Output("syn-box", "style"), Output("csv-box", "style"), Input("data-source", "value"))
    def data_boxes(src):
        return ({"display": "block" if src == "synthetic" else "none"},
                {"display": "block" if src == "csv" else "none"})

    @app.callback(Output("csv-status", "children"), Output("csv-time", "options"),
                  Output("csv-time", "value", allow_duplicate=True), Output("csv-map", "children", allow_duplicate=True),
                  Input("upload-csv", "contents"), State("upload-csv", "filename"), prevent_initial_call=True)
    def csv_upload(contents, name):
        try:
            raw = base64.b64decode(contents.split(",", 1)[1])
            df = dl.read_csv_bytes(raw)
        except Exception as e:  # noqa: BLE001
            return msg(f"✗ {e}", "error"), no_update, no_update, no_update
        LAB.csv_df, LAB.csv_name, LAB.csv_bytes = df, name, raw
        cols = list(df.columns)
        tguess = next((c for c in cols if c.lower() in ("t", "time", "time_s", "t_s")), cols[0])
        system, _, _ = LAB.view()
        return (msg(f"✓ {name}: {len(df)} rows × {len(cols)} numeric columns — pick the time column and targets, "
                    "then Apply.", "ok"), cols, tguess, _csv_map_children(system))

    @app.callback(Output("data-rev", "data"), Output("settings-msg", "children", allow_duplicate=True),
                  Output("data-source", "value", allow_duplicate=True),
                  Input("btn-data", "n_clicks"), Input("btn-data-remove", "n_clicks"),
                  State("data-source", "value"), State("syn-obs", "value"), State("syn-n", "value"),
                  State("syn-noise", "value"), State("val-split", "value"), State("csv-time", "value"),
                  State({"type": "map", "obs": ALL}, "value"), State({"type": "map", "obs": ALL}, "id"),
                  State("data-rev", "data"), prevent_initial_call=True)
    def apply_data(_a, _r, src, syn_obs, syn_n, syn_noise, val_split, csv_time, map_vals, map_ids, rev):
        if ctx.triggered_id == "btn-data-remove":
            LAB.remove_dataset()
            return (rev or 0) + 1, msg("Dataset removed — physics-only training (L_data = 0).", "ok"), "none"
        mapping = {i["obs"]: v for i, v in zip(map_ids or [], map_vals or []) if v}
        st = dict(DEFAULT_DATA, source=src, observables=syn_obs, n_points=syn_n, noise_pct=syn_noise,
                  val_fraction=val_split, time_column=csv_time, mapping=mapping)
        try:
            ds = LAB.apply_dataset(st)
        except (dl.DataError, ValueError) as e:
            return no_update, msg(f"✗ {e}", "error"), no_update
        note = " The running/loaded session keeps its own data until the next Start." if LAB.session else ""
        where = f" Saved to {LAB.project.id}/measurement_data/." if LAB.project else ""
        return (rev or 0) + 1, msg(f"Dataset applied: {ds.n_train} train / {ds.n_val} validation points.{where}{note}",
                                   "ok"), no_update

    @app.callback(Output("data-summary", "children"), Output("data-preview", "children"),
                  Input("data-rev", "data"), Input("compiled-rev", "data"))
    def data_preview(*_):
        ds = LAB.dataset
        if not ds.has_data:
            return html.Span(["No measurement data: ", html.B("L_data = 0"), " (physics only)."]), None
        head = html.Tr([html.Th("split"), html.Th("t")] + [html.Th(o) for o in ds.y_train])
        rows = []
        for split, t, ys in (("train", ds.t_train, ds.y_train), ("val", ds.t_val, ds.y_val)):
            for i in range(min(6, len(t))):
                rows.append(html.Tr([html.Td(split), html.Td(f"{t[i]:.4g}")] +
                                    [html.Td(f"{ys[o][i]:.4g}") for o in ds.y_train]))
        return (html.Span([html.B(f"{ds.n_train} train / {ds.n_val} validation points. "), ds.description]),
                html.Table([html.Thead(head), html.Tbody(rows)], className="tbl small"))

    @app.callback(Output("dl-ref", "data"), Input("btn-dl-ref", "n_clicks"), prevent_initial_call=True)
    def download_ref(_n):
        system, ref, _ = LAB.view()
        if ref is None:
            return no_update
        return dict(content=dl.reference_to_csv(ref), filename=f"reference_{LAB.project.id if LAB.project else 'model'}.csv")

    # ================================================================ physics settings
    @app.callback(Output("phys-status", "children"), Output("compiled-rev", "data", allow_duplicate=True),
                  Input("btn-phys-apply", "n_clicks"),
                  State({"type": "par", "name": ALL}, "value"), State({"type": "par", "name": ALL}, "id"),
                  State({"type": "ic", "name": ALL}, "value"), State({"type": "ic", "name": ALL}, "id"),
                  State("phys-bc", "value"), State("phys-tend", "value"), State("compiled-rev", "data"),
                  prevent_initial_call=True)
    def apply_physics(_n, pvals, pids, ivals, iids, bc_text, t_end, rev):
        system = LAB.system
        try:
            params = {i["name"]: float(v) for i, v in zip(pids, pvals) if v is not None}
            initial = {i["name"]: float(v) for i, v in zip(iids, ivals) if v is not None}
            bcs = []
            for line in (bc_text or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                name, rest = line.split("(", 1)
                tb, val = rest.split(")", 1)
                bcs.append((name.strip(), float(tb), float(val.split("=", 1)[1])))
            text = update_model_text(system.source_text, params=params, initial=initial, boundary=bcs,
                                     t_end=float(t_end) if t_end else None)
            LAB.compile_text(text, "physics settings changed in PINNlab")
        except (ValueError, IndexError, ModelError) as e:
            return msg(f"✗ {e}", "error"), no_update
        note = " (the current session keeps its own equations until Reset/Start)" if LAB.session else ""
        return msg(f"✓ Applied and saved to equations.json — the next training uses these physics.{note}", "ok"), (rev or 0) + 1

    @app.callback(Output("compile-status", "children"), Output("compiled-rev", "data", allow_duplicate=True),
                  Input("btn-compile", "n_clicks"), State("model-text", "value"), State("compiled-rev", "data"),
                  prevent_initial_call=True)
    def compile_model(_n, text, rev):
        try:
            system = LAB.compile_text(text or "", "model text edited in PINNlab")
        except ModelError as e:
            return msg(f"✗ {e}", "error"), no_update
        return msg(f"✓ {len(system.states)} state(s): {', '.join(system.states)}", "ok"), (rev or 0) + 1

    # ================================================================ training controls
    @app.callback(
        Output("action-msg", "children"), Output("compiled-rev", "data", allow_duplicate=True),
        Output("ckpt-rev", "data", allow_duplicate=True), Output("settings-msg", "children", allow_duplicate=True),
        *CFG_OUT,
        Input("btn-start", "n_clicks"), Input("btn-pause", "n_clicks"), Input("btn-resume", "n_clicks"),
        Input("btn-save", "n_clicks"), Input("btn-ck-save", "n_clicks"), Input("btn-load", "n_clicks"),
        Input("btn-ck-resume", "n_clicks"), Input("btn-ck-delete", "n_clicks"), Input("btn-continue", "n_clicks"),
        Input("btn-reset", "n_clicks"), Input("upload-ckpt", "contents"),
        State("upload-ckpt", "filename"), State("ckpt-select", "value"), State("cont-epochs", "value"),
        State("compiled-rev", "data"), *CFG_STATES, prevent_initial_call=True)
    def controls(*args):
        up_contents = args[10]
        (up_name, ck_path, cont_epochs, rev), cfg_vals = args[11:15], args[15:]
        trig = ctx.triggered_id
        sess = LAB.session

        def reply(m, bump=False, ck=False, settings=None, cfg=None):
            return (m, (rev or 0) + 1 if bump else no_update, str(time.time()) if ck else no_update,
                    settings if settings is not None else no_update,
                    *(cfg if cfg is not None else [no_update] * len(CFG_FIELDS)))

        def load(path, resume=False):
            new = TrainingSession.from_checkpoint(path, on_complete=LAB.results_writer(LAB.project))
            if sess is not None:
                sess.stop()
            new.project_id = LAB.project.id
            LAB.session = new
            LAB.config = new.config
            lf = new.loaded_from
            text = (f"Loaded checkpoint {lf['file']} — epoch {lf['epoch']:,}, total loss {fmt(lf['total_loss'])} "
                    f"on {new.device_label()}.")
            if resume:
                new.continue_training(int(cont_epochs or 1000))
                text += f" Resumed: training epochs {lf['epoch']:,} → {new.target_epochs:,} with the saved optimizer state."
            else:
                text += f" Press “Continue training” to go on from epoch {lf['epoch']:,}."
            m = msg(text, "ok")
            return reply(m, bump=True, ck=True, settings=m, cfg=config_values(new.config))

        try:
            if trig == "btn-start":
                if sess is not None and sess.is_alive:
                    return reply(msg("Training is already running — pause or reset first.", "warn"))
                cfg = config_from_values(cfg_vals)
                if sess is not None:
                    sess.stop()
                p = LAB.project
                LAB.config = cfg
                p.save_configuration(asdict(cfg))
                new = TrainingSession(LAB.system, LAB.dataset, cfg, reference=LAB.reference,
                                      checkpoint_root=p.checkpoints_dir, on_complete=LAB.results_writer(p))
                new.project_id = p.id
                LAB.session = new
                new.start()
                note = f" {new.device_note}" if new.device_note else ""
                return reply(msg(f"Training started on {new.device_label()}: {cfg.epochs:,} epochs → "
                                 f"{p.id}/pinn/checkpoints/{new.run_dir.name}.{note}", "warn" if note else "ok"),
                             bump=True, ck=True)
            if trig == "btn-pause":
                if sess is None or not sess.is_alive:
                    return reply(msg("Nothing is training.", "warn"))
                sess.pause()
                return reply(msg(f"Pausing at epoch {sess.epoch:,}…"))
            if trig == "btn-resume":
                if sess is None or not sess.is_alive or sess.status != "paused":
                    return reply(msg("Resume continues a paused run; use Continue training for a finished or loaded "
                                     "one.", "warn"))
                sess.resume()
                return reply(msg(f"Resumed at epoch {sess.epoch:,}.", "ok"))
            if trig in ("btn-save", "btn-ck-save"):
                if sess is None:
                    return reply(msg("No model to save yet — start training first.", "warn"))
                path = sess.save_checkpoint()
                m = msg(f"Saved {path.name} (epoch {sess.epoch:,}) in {LAB.project.id}/pinn/checkpoints/{path.parent.name}.", "ok")
                return reply(m, ck=True, settings=m)
            if trig in ("btn-load", "btn-ck-resume", "upload-ckpt"):
                if trig == "upload-ckpt":
                    raw = base64.b64decode(up_contents.split(",", 1)[1])
                    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in (up_name or "upload.pt"))
                    target = LAB.project.checkpoints_dir / "imported" / safe
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    ck_path = str(target)
                if not ck_path:
                    return reply(msg("Select a checkpoint first (Settings → C · Checkpoints).", "warn"),
                                 settings=msg("Select a checkpoint first.", "warn"))
                if sess is not None and sess.is_alive:
                    return reply(msg("Pause and Reset the running training before loading a checkpoint.", "warn"),
                                 settings=msg("Stop the running training first.", "warn"))
                return load(ck_path, resume=trig == "btn-ck-resume")
            if trig == "btn-ck-delete":
                if not ck_path:
                    return reply(no_update, settings=msg("Select a checkpoint to delete.", "warn"))
                ckpt.delete(ck_path)
                return reply(no_update, ck=True, settings=msg(f"Deleted {ck_path.split(chr(92))[-1].split('/')[-1]}.", "ok"))
            if trig == "btn-continue":
                if sess is None:
                    return reply(msg("Load a checkpoint or train a model first.", "warn"))
                if sess.is_alive:
                    return reply(msg("Already training (use Resume if it is paused).", "warn"))
                start_ep = sess.epoch
                sess.continue_training(int(cont_epochs or 1000))
                return reply(msg(f"Continuing from epoch {start_ep:,} to {sess.target_epochs:,} with the saved model "
                                 "and optimizer state.", "ok"))
            if trig == "btn-reset":
                if sess is not None:
                    sess.stop()
                LAB.session = None
                return reply(msg("Reset: training stopped and the model cleared (checkpoints stay in the project).",
                                 "info"), bump=True, ck=True)
        except Exception as e:  # noqa: BLE001
            return reply(msg(f"✗ {type(e).__name__}: {e}", "error"), settings=msg(f"✗ {e}", "error"))
        return reply(no_update)

    # ================================================================ live tick
    @app.callback(
        Output("status-pill", "children"), Output("status-pill", "className"),
        Output("m-epoch", "children"), Output("m-total", "children"), Output("m-data", "children"),
        Output("m-phys", "children"), Output("m-bc", "children"), Output("m-ic", "children"),
        Output("m-val", "children"), Output("m-lr", "children"), Output("m-time", "children"),
        Output("m-speed", "children"), Output("m-device", "children"), Output("progress-fill", "style"),
        Output("fig-loss", "figure"), Output("fig-lr", "figure"), Output("loss-seen", "data"),
        Output("seen-version", "data"), Output("ckpt-rev", "data"),
        Output("btn-start", "disabled"), Output("btn-pause", "disabled"), Output("btn-resume", "disabled"),
        Output("btn-save", "disabled"), Output("btn-continue", "disabled"), Output("btn-continue", "children"),
        Output("btn-load", "disabled"), Output("btn-ck-resume", "disabled"),
        Input("tick", "n_intervals"), Input("log-y", "value"), Input("compiled-rev", "data"),
        State("loss-seen", "data"), State("seen-version", "data"), State("ckpt-rev", "data"))
    def tick(_n, logy, _rev, loss_seen, seen_version, ckpt_rev):
        s = LAB.session
        if s is None:
            if loss_seen == "idle" and ctx.triggered_id == "tick":
                return (no_update,) * 27
            text, cls = STATUS_TEXT[None]
            return ([html.Span(className="dot"), text], f"status-pill {cls}", *(["—"] * 11), {"width": "0%"},
                    loss_figure(None, log_y="log" in (logy or [])), lr_figure(None), "idle",
                    -1 if seen_version != -1 else no_update, no_update,
                    False, True, True, True, True, "⟳ Continue training", False, False)
        snap = s.snapshot()
        last, status = snap["last"], snap["status"]
        text, cls = STATUS_TEXT.get(status, (status, "idle"))
        if status == "error":
            text = f"Error — {snap['message'][:90]}"
        elif status == "completed" and snap["message"].startswith("Early stopping"):
            text = "Completed (early stop)"
        pct = 100.0 * snap["epoch"] / max(snap["target_epochs"], 1)
        speed = snap["epoch"] / snap["elapsed"] if snap["elapsed"] > 0 else 0
        key = [snap["epoch"], "log" in (logy or []), snap["version"], id(s)]
        if key != loss_seen or ctx.triggered_id in ("log-y", "compiled-rev"):
            hist = s.history_copy()
            lf, lrf = loss_figure(hist, log_y="log" in (logy or [])), lr_figure(hist)
        else:
            lf = lrf = no_update
        ver = [id(s), snap["version"]]
        ckey = f"{id(s)}:{len(s.saved)}:{s.saved[-1] if s.saved else ''}"
        alive, paused = s.is_alive, status == "paused"
        return ([html.Span(className="dot"), text], f"status-pill {cls}",
                f"{snap['epoch']:,} / {snap['target_epochs']:,}", fmt(last["total"]), fmt(last["data"]),
                fmt(last["physics"]), fmt(last["bc"]), fmt(last["ic"]), fmt(last["val"]), fmt(snap["lr"], 2),
                fmt_time(snap["elapsed"]), f"{speed:,.0f} ep/s" if speed else "—", snap["device"],
                {"width": f"{pct:.1f}%"}, lf, lrf, key,
                ver if ver != seen_version else no_update, ckey if ckey != ckpt_rev else no_update,
                alive, (not alive) or paused, not paused, False, alive,
                f"⟳ Continue training from epoch {snap['epoch']:,}", alive, alive)

    # ================================================================ physics animation
    @app.callback(Output("anim-data", "data"), Input("seen-version", "data"), Input("compiled-rev", "data"))
    def anim_data(_v, _r):
        system, ref, _ = LAB.view()
        s_ = LAB.session
        return animation_payload(system, s_.prediction() if s_ is not None else None, ref)

    app.clientside_callback(
        "function(d){ if (window.PINNAnim) { window.PINNAnim.update(d); } return window.dash_clientside.no_update; }",
        Output("anim-ack", "data"), Input("anim-data", "data"))

    # ================================================================ prediction vs reference
    @app.callback(Output("fig-cmp", "figure"), Output("cmp-table", "children"),
                  Input("cmp-obs", "value"), Input("cmp-src", "value"), Input("seen-version", "data"),
                  Input("compiled-rev", "data"), Input("data-rev", "data"))
    def compare(obs, src, *_):
        system, ref, dataset = LAB.view()
        s = LAB.session
        pred = s.prediction() if s is not None else None
        if obs not in system.observables:
            obs = system.states[0]
        fig = comparison_figure(system, obs, pred, ref, dataset, height=360)
        if s is None:
            return fig, html.Div("RMSE, MAE and R² appear once a model exists (start training or load a checkpoint).",
                                 className="note")
        rows = metrics_table(system, s.predict, ref, dataset, src)
        if not rows:
            what = "reference solution" if src == "reference" else "validation data"
            return fig, html.Div(f"No {what} available for metrics.", className="note")
        head = html.Tr([html.Th(h) for h in ("Quantity", "n", "RMSE", "MAE", "R²")])
        body = [html.Tr([html.Td(r["label"]), html.Td(r["n"]), html.Td(fmt(r["rmse"])), html.Td(fmt(r["mae"])),
                         html.Td(f"{r['r2']:.5f}", className=_r2_class(r["r2"]))]) for r in rows]
        src_txt = ("PINN at the reference solver's time points" if src == "reference"
                   else "PINN at the held-out validation measurements")
        return fig, html.Div([html.Table([html.Thead(head), html.Tbody(body)], className="tbl"),
                              html.Div(f"{src_txt} · epoch {s.epoch:,}.", className="note")])

    def _r2_class(r2):
        if not math.isfinite(r2):
            return ""
        return "good" if r2 >= 0.99 else ("ok" if r2 >= 0.9 else "bad")

    # ================================================================ checkpoints (settings C)
    @app.callback(Output("ckpt-select", "options"), Output("ckpt-list", "children"), Output("ckpt-info", "children"),
                  Input("ckpt-rev", "data"), Input("compiled-rev", "data"))
    def ckpt_list(*_):
        items = ckpt.list_checkpoints(LAB.project.checkpoints_dir) if LAB.project else []
        opts = [{"label": f"{c['run']} / {c['file']}  (epoch {c.get('epoch') if c.get('epoch') is not None else '?'},"
                          f" loss {fmt(c.get('total_loss'))})", "value": c["path"]} for c in items]
        s = LAB.session
        loaded = s.loaded_from["file"] if s is not None and s.loaded_from else None
        cur_run = s.run_dir.name if s is not None and s.run_dir else None
        rows = []
        for c in items[:40]:
            status = "loaded" if (c["file"] == loaded and c["run"] == cur_run) else (
                "current run" if c["run"] == cur_run else "")
            rows.append(html.Tr([html.Td(c["run"], className="mono"), html.Td(c["file"], className="mono"),
                                 html.Td(c.get("kind", "")),
                                 html.Td(f"{c['epoch']:,}" if c.get("epoch") is not None else "?"),
                                 html.Td(fmt(c.get("total_loss"))), html.Td(fmt(c.get("val_loss"))),
                                 html.Td(fmt(c.get("best_val_loss"))), html.Td(c.get("saved_at") or ""),
                                 html.Td(status, className="good" if status else "")]))
        table = (html.Table([html.Thead(html.Tr([html.Th(h) for h in (
            "Run", "File", "Kind", "Epoch", "Loss", "Val loss", "Best val", "Date", "Status")])), html.Tbody(rows)],
            className="tbl small") if rows else html.Div("No checkpoints in this project yet.", className="note"))
        info = None
        if s is not None and s.loaded_from:
            lf = s.loaded_from
            info = html.Div(className="loaded", children=[
                html.Div("Loaded checkpoint", className="loaded-title"),
                html.Div([html.Span("File "), html.B(f"{lf['run']}/{lf['file']}")]),
                html.Div([html.Span("Epoch "), html.B(f"{lf['epoch']:,}"), html.Span("   Total loss "),
                          html.B(fmt(lf['total_loss']))]),
                html.Div(f"saved {lf.get('saved_at') or ''}", className="muted"),
            ])
        where = html.Div(f"Folder: {LAB.project.checkpoints_dir}" if LAB.project else "", className="note")
        return opts, [where, table], info
