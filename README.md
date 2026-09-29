# Physics-Informed Engineering Lab

**Bond Graph Modeling → Governing Equations → PINNs → Real-Time Simulation**

One integrated application with three connected parts, served by a single server:

| URL | Part |
|---|---|
| `/` | Home — launch BondLab or PINNlab |
| `/bondlab/` | **BondLab** — build, import and edit bond graphs; parameters, initial and boundary conditions; automatic causality and symbolic governing equations; **PINNSIM →** |
| `/pinnlab/` | **PINNlab** — PyTorch physics-informed neural network whose physics loss is built from the BondLab equations (DEMO or CUSTOM) |

## Run

```bash
python -m pip install -r requirements.txt
python -m integrated.server                 # http://127.0.0.1:8000
python -m integrated.server --host 0.0.0.0 --port 8080 --device cuda
```

`--device auto|cuda|cpu` selects the training device (auto uses a CUDA GPU when PyTorch finds one).
Set `GEMINI_API_KEY` in the environment to enable photo import in BondLab (optional).

## Workflow

1. **BondLab**: build a bond graph (or open one of the 11 templates), set parameter values, initial
   conditions, the time horizon and optional boundary conditions. The equations are derived automatically
   (SCAP causality assignment + SymPy).
2. **PINNSIM →** saves the model and transfers the structured bond graph, equations, variables, parameters,
   ICs and BCs to PINNlab. The server re-derives the equations from the graph, checks that they compile and
   stores them in the project folder.
3. **PINNlab** turns every equation `dx_i/dt = f_i(x, t)` into a residual evaluated with PyTorch automatic
   differentiation:

   `L_total = L_data + λ_physics·L_physics + λ_BC·L_BC + λ_IC·L_IC`

   Training runs in a background thread with live loss curves, an animated physics view drawn from the
   network's predictions, prediction vs reference with RMSE / MAE / R², and checkpoints with resume.
   **⚙ SETTINGS** holds the PINN architecture, measurement data, checkpoints, training and physics options.
4. **CUSTOM** in PINNlab returns to BondLab with the current model; **DEMO** loads the built-in example.

## Project folders

Every model is stored in `workspace/projects/<project>/` (created at runtime):

```
project.json
bondgraph/model.json        BondLab document: graph, layout, values, sources, ICs, BCs
bondgraph/equations.json    derived equations, states, parameters, inputs, ICs, BCs
pinn/configuration.json     PINN and training settings
pinn/checkpoints/<run>/     PyTorch checkpoints (model/optimizer/scheduler state, losses, config)
measurement_data/           uploaded CSV + dataset settings
results/                    metrics.json and prediction.csv of the last completed run
```

Set `LAB_WORKSPACE` to store projects elsewhere.

## Layout

```
integrated/          single server (FastAPI): home page, BondLab, PINNlab, PINNSIM endpoints
app/backend/         BondLab API: templates, derivation, simulation, photo import
app/frontend/dist/   BondLab web app (built)
*.py (root)          bond-graph engine: causality (scap_causality), equation derivation (equation_gen), ...
pinn_lab/            PINNlab: PINN model, physics loss, trainer, checkpoints, data, project store, UI
```

PINNlab can also run on its own: `python pinn_lab/main.py` (http://127.0.0.1:8050).
