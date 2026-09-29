# PINN Lab — Bond Graph → Physics-Informed Neural Network

A local research dashboard in which the governing equations derived by **BondLab** from a bond graph are
compiled into the **physics loss of a PyTorch PINN**, trained live, visualised and checkpointed.

```
Bond graph (BondLab) → governing equations → residuals r_i = dx_i/dt − f_i(x,t) → PINN (PyTorch autograd)
→ training in a background thread → live loss curves → physical view from the PINN output
→ checkpoints (.pt) → prediction vs reference (RMSE, MAE, R²)
```

Everything on screen comes from the running model, the compiled equations or files on disk. The reference
curve is SciPy `solve_ivp` (LSODA) run on **the same compiled equations**; it is drawn dashed for comparison
and is never fed to the network unless you choose it as the data source ("synthetic measurements").

## Two ways to run

* **Integrated (recommended):** `python -m integrated.server` from the repository root → Home at `/`,
  BondLab at `/bondlab/`, PINNlab at `/pinnlab/`, with PINNSIM transfer and the shared project folders in
  `workspace/projects/`. See the top-level README.
* **Standalone PINNlab:** the commands below (CUSTOM then opens the equation editor instead of BondLab).

PINNlab's main page shows only the current model/status (“BondLab Model Connected ✓” + an expandable view
of the received equations and bond graph), training controls with live numbers, the animated physics view,
live loss curves and prediction vs reference with RMSE/MAE/R²; one-line summaries show the network, data,
checkpoint and device. Everything else is under **⚙ SETTINGS**: A · PINN Architecture (network type MLP /
residual MLP / Fourier-feature MLP, layers, neurons, activation, normalization, device, precision), B ·
Measurement Data (synthetic/CSV/none, targets, split, normalize, preview, replace/remove), C · Checkpoints
(load, load & resume, delete, save, import; epoch/loss/val/date/status), D · Training (learning rate,
epochs, optimizer Adam/AdamW/RMSprop/SGD/L-BFGS, scheduler, batch size, early stopping, intervals), E ·
Physics (received equations, λ weights, parameters, ICs, BCs, t_end, collocation sampling, residual scaling,
advanced text editor). Physics edits are written back to the project's `equations.json`.

## Run standalone

```bash
python -m pip install -r pinn_lab/requirements.txt
python pinn_lab/main.py            # open http://127.0.0.1:8050
```

`--port`, `--host`, `--threads`, `--device {auto,cuda,cpu}` are available (`--threads 1` is fastest for these
small networks on CPU).

## GPU (CUDA)

Pick **Device** and **Precision** in the *PINN Architecture* card (or `--device cuda` on the command line).
`auto` uses the CUDA GPU when PyTorch sees one; precision `auto` = float32 on the GPU, float64 on the CPU
(consumer GPUs run float64 30-60x slower). The *Device* tile shows where training runs and the GPU memory in use.
Checkpoints are always saved with CPU tensors, so a GPU run loads and continues on a CPU-only machine.

Measured on an RTX 4050 Laptop GPU vs this laptop's CPU (demo problem, 5000 epochs, same R² 0.99999):

| setting | CPU float64 | GPU float32 |
|---|---|---|
| default: 600 collocation pts, batch 256, 4×48 | **~124 ep/s** | ~62 ep/s |
| large: 4000 pts full batch, 4×96 | ~7 ep/s | **~64 ep/s (9× faster)** |
| raw step, 10000 pts, 4×128 | 501 ms | **8.6 ms (58× faster)** |

The GPU costs a fixed ~15 ms per step (kernel-launch overhead, not maths), so small problems are faster on the
CPU; the GPU wins as soon as you raise collocation points or network width — on the GPU those are almost free,
so use them for accuracy. Resume is bit-identical on the CPU; GPUs do not guarantee bitwise determinism.
(The PyTorch in this environment is `2.6.0+cu124`; a CPU-only PyTorch install simply shows "no CUDA GPU".)

## Using it

1. **Bond Graph & Governing Equations** — pick the *Demo Physics Problem*, any BondLab template (equations
   are derived live by the BondLab engine in `app/backend`), or *Custom* and paste your own. Press
   **Compile**. The *Physics residuals* tab shows exactly what goes into `L_physics`; *Bond graph* draws the
   graph with causality strokes when it came from BondLab.
2. **Measurement Data** — synthetic points sampled from the reference (+ noise), a **CSV** (pick the time
   column and map columns to states/outputs), or none (physics only). Validation split and scaling here.
3. **PINN Architecture** — layers, neurons, activation, learning rate, epochs, loss weights λ, collocation
   points, batch size, LR decay, checkpoint interval, seed. The diagram shows the real network; once it
   exists, connections are coloured by the live weights.
4. **Start training** — Pause / Resume, Save checkpoint, Load checkpoint, Continue training (from the loaded
   epoch with the saved optimizer state), Reset.
5. **Physics Visualization** — a 60 fps canvas animation (`ui/assets/physics_anim.js`) with play/pause, speed
   (0.25×–4×) and scrubbing, refreshed live while training. Everything that moves comes from the network:
   *thermal* = reservoir at the source temperature → conduction rod with the temperature gradient and heat
   particles whose speed ∝ (T_src − T) → body coloured by the PINN temperature, heat shimmer, thermometer, and
   the net heat inflow C·dT/dt taken by autograd; *mass_spring* = stretching spring, damper piston, force arrow
   F(t), velocity arrow dx/dt; *rotor* = disc turned by ∫ω dt of the prediction; *tank* = level with a surface
   and an inflow/outflow stream sized by the predicted net flow; *gauge* = analog dial. When the predicted
   flows are zero the picture is still. Dashed outlines/needles are the reference solution.

## Equation format

```ini
[meta]
title = My model
bondlab_template = mass_spring_damper   # optional, draws the bond graph

[equations]                             # one per state; BondLab's text output works as-is
d(f_M1(t))/dt = (C_k*(F - c*f_M1(t)) - q_K1(t))/(C_k*m)
d(q_K1(t))/dt = f_M1(t)

[parameters]                            # name [unit] = number   # label
m [kg] = 1.0  # mass

[inputs]                                # functions of t: sin cos exp log sqrt tanh Abs Heaviside Min Max Mod pi
F [N] = 1.0*Heaviside(t)

[initial]                               # values at t_start  ->  L_IC
f_M1 [m/s] = 0
q_K1 [m] = 0

[outputs]                               # optional readouts (can be measured / plotted)
F_spring [N] = q_K1/C_k

[boundary]                              # optional point constraints  ->  L_BC
q_K1(3.0) = 0.01

[visual]                                # mass_spring | rotor | thermal | tank | gauge
mass_spring = q_K1

[time]
t_start = 0
t_end = 3
```

Formulas are parsed without access to Python builtins (whitelisted functions only), so pasted text cannot
execute code.

## Loss

```
L_total = λ_data·L_data + λ_physics·L_physics + λ_BC·L_BC + λ_IC·L_IC
```

* `L_physics` — mean square of the normalised residuals `dŷ_i/dτ − (Δt/2σ_i)·f_i(x,t)` at collocation points,
  with `dŷ/dτ` from `torch.autograd.grad` (network input τ ∈ [−1, 1], output x = μ + σ·ŷ).
* `L_data` — measurement misfit, each quantity divided by its scale.
* `L_IC` — initial conditions; `L_BC` — `[boundary]` constraints (0 when none are defined).
* Validation loss = held-out data misfit, or (without validation data) physics+IC loss on points the
  optimizer never sees. `best_model.pt` tracks it.

## Checkpoints

`pinn_lab/checkpoints/<run>/checkpoint_epoch_N.pt`, `best_model.pt`, `index.json`. Each file holds the model
and optimizer `state_dict`s, LR-scheduler state, epoch, all loss terms, best validation loss, the training
configuration, the equation text, scaling, dataset, loss history and the sampler RNG state. Files are loaded
with `torch.load(weights_only=True)`. A test checks that *train 60 epochs* and *train 30 → save → load →
continue 30* give bit-identical weights.

## Replacing the demo with your BondLab equations

Choose a BondLab template (or build a graph in BondLab, copy the equations from its Equations tab) and paste
them under `[equations]`, with values under `[parameters]`, `[inputs]` and `[initial]`. Nothing else in the code
needs to change: `bondgraph/equations.py` compiles whatever is there into the residuals.

## Honest limits

* Speed: ~70–130 epochs/s on the CPU for the default 4×48 network (5000 epochs ≈ 40–75 s); see the GPU table.
* Stiff systems (widely separated time scales, e.g. the DC-motor template) are hard for plain PINNs:
  physics-only training reached R² ≈ 0.74 there in 5000 epochs. Add measurement data, more epochs, or a
  larger λ_IC. Smooth and oscillatory systems (demo, thermal, hydraulic, RLC) reach R² ≥ 0.996 physics-only.
* Time-only (ODE) problems — which is what a lumped bond graph produces. Spatial PDE fields are out of scope.

## Layout

```
pinn_lab/
  main.py                          entry point (Dash server)
  ui/dashboard.py, ui/assets/      layout + callbacks, dark theme
  pinn/model.py                    MLP with built-in non-dimensionalisation
  pinn/physics_loss.py             L_data, L_physics (autograd residuals), L_BC, L_IC
  pinn/trainer.py                  TrainingSession: background thread, pause/resume, checkpoints
  pinn/checkpoint.py               save / load / index
  bondgraph/parser.py              safe formula parser + torch/numpy evaluators
  bondgraph/equations.py           model text -> ODESystem, residuals, reference solver, demo problem
  bondgraph/bondlab_bridge.py      BondLab template -> model text, bond-graph drawing
  data/dataset_loader.py           CSV / synthetic data, train/val split, scaling
  visualization/*.py               loss, physics, architecture, comparison, bond-graph figures
```
