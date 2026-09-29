"""
trainer.py -- real PINN training in a background thread.

TrainingSession owns the model, optimizer, scheduler, loss and history.
`start()` launches a worker thread that runs Adam steps on the physics loss;
the UI only ever reads `snapshot()` (a cheap copy of the latest numbers), so
the interface stays responsive while training runs.

Controls: start / pause / resume / stop, save_checkpoint (any time, thread
safe), from_checkpoint + continue_training (restores model, optimizer and
scheduler state and continues the epoch counter).

Device: TrainConfig.device = "auto" (CUDA GPU when available, else CPU),
"cuda" or "cpu"; precision "auto" = float32 on the GPU (consumer GPUs run
float64 ~30-60x slower) and float64 on the CPU. Checkpoints are always
written with CPU tensors, so a GPU-trained model loads on a CPU-only machine
and vice versa.
"""
import copy
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from bondgraph.equations import parse_model
from data.dataset_loader import Dataset, compute_scaling, empty_dataset

from . import checkpoint as ckpt
from .model import PINN
from .physics_loss import LossWeights, PhysicsLoss

HISTORY_KEYS = ("epoch", "total", "data", "physics", "bc", "ic", "val", "lr", "time")


@dataclass
class TrainConfig:
    hidden_layers: int = 4
    neurons: int = 48
    activation: str = "sin"
    learning_rate: float = 2e-3
    epochs: int = 5000
    lambda_data: float = 1.0
    lambda_physics: float = 1.0
    lambda_bc: float = 1.0
    lambda_ic: float = 10.0
    n_collocation: int = 600
    batch_size: int = 256            # collocation points per step; 0 = all (full batch)
    lr_decay: float = 0.9993         # ExponentialLR gamma per epoch (1.0 = constant LR)
    min_lr: float = 1e-5
    checkpoint_every: int = 1000
    eval_every: int = 100
    log_every: int = 5
    normalization: str = "standard"  # standard | minmax | none
    seed: int = 1234
    device: str = "auto"             # auto | cuda | cpu
    precision: str = "auto"          # auto | float32 | float64
    network: str = "mlp"             # mlp | resnet | fourier
    fourier_features: int = 8
    optimizer: str = "adam"          # adam | adamw | rmsprop | sgd | lbfgs
    weight_decay: float = 0.0
    scheduler: str = "exponential"   # none | exponential | cosine | step | plateau
    step_size: int = 1000            # StepLR period (epochs), gamma 0.5
    early_stopping: bool = False     # stop when the validation loss stops improving
    patience: int = 10               # evaluations without improvement before stopping
    min_delta: float = 0.0           # relative improvement that counts
    collocation_sampling: str = "grid"   # grid (fixed grid, random mini-batches) | random (fresh points each epoch)
    residual_scaling: str = "normalized"  # normalized | physical
    normalize_data: bool = True      # divide data/BC misfits by each quantity's scale

    @classmethod
    def from_dict(cls, d):
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def validate(self):
        errs = []
        if not (1 <= self.hidden_layers <= 12):
            errs.append("hidden layers must be 1-12")
        if not (2 <= self.neurons <= 512):
            errs.append("neurons per layer must be 2-512")
        if not (1e-6 <= self.learning_rate <= 1.0):
            errs.append("learning rate must be between 1e-6 and 1")
        if not (1 <= self.epochs <= 1_000_000):
            errs.append("epochs must be 1-1,000,000")
        if not (16 <= self.n_collocation <= 20000):
            errs.append("collocation points must be 16-20000")
        if self.batch_size < 0:
            errs.append("batch size must be >= 0")
        if not (0.9 <= self.lr_decay <= 1.0):
            errs.append("LR decay must be in [0.9, 1]")
        if min(self.lambda_data, self.lambda_physics, self.lambda_bc, self.lambda_ic) < 0:
            errs.append("loss weights must be >= 0")
        if self.checkpoint_every < 1 or self.eval_every < 1:
            errs.append("checkpoint/eval intervals must be >= 1")
        if self.device not in ("auto", "cuda", "cpu"):
            errs.append("device must be auto, cuda or cpu")
        if self.precision not in ("auto", "float32", "float64"):
            errs.append("precision must be auto, float32 or float64")
        if self.network not in ("mlp", "resnet", "fourier"):
            errs.append("network must be mlp, resnet or fourier")
        if not (1 <= self.fourier_features <= 64):
            errs.append("Fourier features must be 1-64")
        if self.optimizer not in OPTIMIZERS:
            errs.append("optimizer must be one of " + ", ".join(OPTIMIZERS))
        if self.scheduler not in SCHEDULERS:
            errs.append("scheduler must be one of " + ", ".join(SCHEDULERS))
        if self.step_size < 1 or self.patience < 1 or self.min_delta < 0 or self.weight_decay < 0:
            errs.append("step size / patience must be >= 1, min delta and weight decay >= 0")
        if self.collocation_sampling not in ("grid", "random"):
            errs.append("collocation sampling must be grid or random")
        if self.residual_scaling not in ("normalized", "physical"):
            errs.append("residual scaling must be normalized or physical")
        if errs:
            raise ValueError("; ".join(errs))
        return self


OPTIMIZERS = ("adam", "adamw", "rmsprop", "sgd", "lbfgs")
SCHEDULERS = ("none", "exponential", "cosine", "step", "plateau")


def make_optimizer(cfg, params, device):
    lr, wd, cuda = cfg.learning_rate, cfg.weight_decay, device.type == "cuda"
    if cfg.optimizer == "adam":      # fused = one CUDA kernel for the whole update (~12% faster on the GPU)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, fused=cuda)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=cuda)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.LBFGS(params, lr=lr, max_iter=1, history_size=50, line_search_fn="strong_wolfe")


def make_scheduler(cfg, opt):
    s = cfg.scheduler
    if s == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.lr_decay)
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.min_lr)
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=0.5)
    if s == "plateau":   # stepped with the validation loss at every evaluation
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=cfg.min_lr)
    return None


def cuda_available():
    return torch.cuda.is_available()


def gpu_name():
    return torch.cuda.get_device_name(0) if cuda_available() else None


def resolve_device(name):
    """-> (torch.device, note). A request for CUDA without a usable GPU falls back to the CPU."""
    if name == "cpu":
        return torch.device("cpu"), ""
    if cuda_available():
        return torch.device("cuda"), ""
    if name == "cuda":
        return torch.device("cpu"), "CUDA was requested but no usable GPU was found; training on the CPU."
    return torch.device("cpu"), ""


def resolve_dtype(precision, device):
    if precision == "float32":
        return torch.float32
    if precision == "float64":
        return torch.float64
    return torch.float32 if device.type == "cuda" else torch.float64


def _cpu(obj):
    """Deep copy of a (nested) state dict with every tensor on the CPU."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu").clone()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _np(x):
    return x.detach().to("cpu", torch.float64).numpy().copy()


class TrainingSession:
    STATUSES = ("ready", "training", "paused", "completed", "stopped", "checkpoint_loaded", "error")

    def __init__(self, system, dataset, config, reference=None, run_dir=None, scaling=None,
                 checkpoint_root=None, on_complete=None):
        """checkpoint_root: folder for new run directories (the project's pinn/checkpoints);
        on_complete(session): called from the worker thread after a run finishes normally."""
        self.system = system
        dataset = dataset if dataset is not None else empty_dataset()      # None = physics only
        self.checkpoint_root = checkpoint_root
        self.on_complete = on_complete
        self.dataset = dataset
        self.config = config.validate()
        self.reference = reference
        self.device, self.device_note = resolve_device(config.device)
        self.dtype = resolve_dtype(config.precision, self.device)
        torch.manual_seed(config.seed)   # weights are initialised on the CPU, so the start is device-independent
        self.scaling = scaling or compute_scaling(system, reference, dataset, config.normalization)
        self.model = PINN(len(system.states), config.hidden_layers, config.neurons, config.activation,
                          system.t_start, system.t_end, self.scaling["mu"], self.scaling["sigma"],
                          dtype=self.dtype, network=config.network,
                          fourier_features=config.fourier_features).to(self.device)
        self.optimizer = make_optimizer(config, self.model.parameters(), self.device)
        self.scheduler = make_scheduler(config, self.optimizer)
        self.loss_fn = PhysicsLoss(system, self.scaling, dataset, dtype=self.dtype, device=self.device,
                                   residual_scaling=config.residual_scaling, normalize_data=config.normalize_data)
        self.weights = LossWeights(config.lambda_data, config.lambda_physics, config.lambda_bc, config.lambda_ic)
        self.gen = torch.Generator().manual_seed(config.seed)

        n = config.n_collocation
        dd = {"dtype": self.dtype, "device": self.device}
        self.t_colloc = torch.linspace(system.t_start, system.t_end, n, **dd)
        # validation collocation points: midpoints of the training grid (never trained on)
        self.t_colloc_val = 0.5 * (self.t_colloc[1:] + self.t_colloc[:-1])
        self.t_eval = torch.linspace(system.t_start, system.t_end, 400, **dd)

        self.run_dir = run_dir
        self.epoch = 0
        self.target_epochs = config.epochs
        self.elapsed = 0.0
        self.best_val = math.inf
        self.evals_since_best = 0
        self.last = {"total": None, "data": None, "physics": None, "bc": None, "ic": None, "val": None}
        self.history = {k: [] for k in HISTORY_KEYS}
        self.status = "ready"
        self.message = ""
        self.loaded_from = None
        self.saved = []                   # recent checkpoint file names (for the UI)

        self.lock = threading.RLock()     # guards model/optimizer between worker and UI
        self._run = threading.Event()     # set = running, clear = paused
        self._stop = threading.Event()
        self._thread = None
        self._pred = None
        self._weights_view = None
        self._t_mark = None
        self.version = 0                  # bumps whenever a new prediction snapshot exists
        self._refresh_prediction()

    # ------------------------------------------------------------ controls
    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, target_epochs=None):
        if self.is_alive:
            raise RuntimeError("training is already running")
        if target_epochs is not None:
            self.target_epochs = int(target_epochs)
        if self.epoch >= self.target_epochs:
            raise RuntimeError(f"already at epoch {self.epoch}; choose more epochs")
        if self.run_dir is None:
            self.run_dir = ckpt.new_run_dir(self.system.title, self.checkpoint_root)
        self._stop.clear()
        self._early = False
        self._run.set()
        self.status = "training"
        self.message = ""
        self._thread = threading.Thread(target=self._worker, name="pinn-trainer", daemon=True)
        self._thread.start()

    def pause(self):
        if self.is_alive and self._run.is_set():
            self._run.clear()

    def resume(self):
        if self.is_alive and not self._run.is_set():
            self._run.set()
            self.status = "training"

    def stop(self, wait=True):
        self._stop.set()
        self._run.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=30)

    def continue_training(self, extra_epochs):
        extra = int(extra_epochs)
        if extra < 1:
            raise ValueError("continue for at least 1 epoch")
        self.start(self.epoch + extra)

    # --------------------------------------------------------------- worker
    def _sample(self):
        n, b = self.t_colloc.numel(), self.config.batch_size
        if self.config.optimizer == "lbfgs":          # L-BFGS needs the same (deterministic) objective every step
            return self.t_colloc
        if self.config.collocation_sampling == "random":
            m = n if b <= 0 else min(b, n)
            u = torch.rand(m, generator=self.gen, dtype=torch.float64)
            t = self.system.t_start + u * (self.system.t_end - self.system.t_start)
            return t.to(dtype=self.dtype, device=self.device)
        if b <= 0 or b >= n:
            return self.t_colloc
        idx = torch.randperm(n, generator=self.gen)[:b]      # CPU generator: reproducible + checkpointable
        return self.t_colloc[idx.to(self.device)]

    def _step(self):
        t_c = self._sample()
        if self.config.optimizer == "lbfgs":
            box = {}

            def closure():
                self.optimizer.zero_grad(set_to_none=True)
                box["terms"] = self.loss_fn(self.model, t_c, self.weights)
                box["terms"]["total"].backward()
                return box["terms"]["total"]
            self.optimizer.step(closure)
            terms = box["terms"]
        else:
            self.optimizer.zero_grad(set_to_none=True)
            terms = self.loss_fn(self.model, t_c, self.weights)
            terms["total"].backward()
            self.optimizer.step()
        sch = self.scheduler
        if sch is not None and not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if not (self.config.scheduler == "exponential" and self.optimizer.param_groups[0]["lr"] <= self.config.min_lr):
                sch.step()
        keys = list(terms)
        vals = torch.stack([terms[k].detach() for k in keys]).double().cpu().tolist()   # one GPU sync per step
        return dict(zip(keys, vals))

    def _worker(self):
        cfg = self.config
        t_mark = self._t_mark = time.perf_counter()
        try:
            while self.epoch < self.target_epochs and not self._stop.is_set() and not self._early:
                if not self._run.is_set():
                    self.elapsed += time.perf_counter() - t_mark
                    self._t_mark = None
                    self.status = "paused"
                    self._run.wait()
                    t_mark = self._t_mark = time.perf_counter()
                    if self._stop.is_set():
                        break
                    self.status = "training"
                with self.lock:
                    terms = self._step()
                    self.epoch += 1
                    if not all(math.isfinite(v) for v in terms.values()):
                        raise FloatingPointError(f"loss became non-finite at epoch {self.epoch}; "
                                                 "try a smaller learning rate")
                    self.last.update(terms)
                    now = self.elapsed + time.perf_counter() - t_mark
                    if self.epoch % cfg.eval_every == 0 or self.epoch == self.target_epochs:
                        self._evaluate()
                    if self.epoch == 1 or self.epoch % cfg.log_every == 0 or self.epoch == self.target_epochs:
                        self._log(now)
                    if self.epoch % cfg.checkpoint_every == 0:
                        self._save(f"checkpoint_epoch_{self.epoch}.pt", "periodic", now)
            self.elapsed += time.perf_counter() - t_mark
            self._t_mark = None
            with self.lock:
                self._evaluate()
                if self._stop.is_set():
                    self.status = "stopped"
                else:
                    if self.epoch % cfg.checkpoint_every != 0:
                        self._save(f"checkpoint_epoch_{self.epoch}.pt", "final", self.elapsed)
                    self.status = "completed"
                    if self._early:
                        self.message = (f"Early stopping at epoch {self.epoch}: no validation improvement for "
                                        f"{cfg.patience} evaluations (best {self.best_val:.3e}).")
            if self.status == "completed" and self.on_complete is not None:
                try:
                    self.on_complete(self)
                except Exception as e:  # noqa: BLE001  -- results export must never kill the session
                    self.message = (self.message + " " if self.message else "") + f"(results not saved: {e})"
        except Exception as e:  # noqa: BLE001
            self.elapsed += time.perf_counter() - t_mark
            self._t_mark = None
            self.status = "error"
            self.message = f"{type(e).__name__}: {e}"

    def _log(self, now):
        h = self.history
        h["epoch"].append(self.epoch)
        for k in ("total", "data", "physics", "bc", "ic", "val"):
            h[k].append(self.last[k])
        h["lr"].append(self.optimizer.param_groups[0]["lr"])
        h["time"].append(now)

    def _val_loss(self):
        """Held-out data loss if validation data exist, else the physics+IC loss
        on collocation points the optimizer never sees (grid midpoints)."""
        if self.loss_fn.val:
            return float(self.loss_fn.val_data_loss(self.model).detach())
        phys = self.loss_fn.physics(self.model, self.t_colloc_val)
        return float((phys + self.loss_fn.initial(self.model)).detach())

    def _evaluate(self):
        val = self._val_loss()
        self.last["val"] = val
        cfg = self.config
        improved = val < self.best_val * (1.0 - cfg.min_delta) if math.isfinite(self.best_val) else True
        if val < self.best_val:
            self.best_val = val
            if self.run_dir is not None and self.epoch > 0:
                self._save("best_model.pt", "best", self.elapsed)
        self.evals_since_best = 0 if improved else self.evals_since_best + 1
        if cfg.early_stopping and self.evals_since_best >= cfg.patience and self.epoch > 0:
            self._early = True
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.epoch > 0:
            self.scheduler.step(val)
        self._refresh_prediction()

    def _refresh_prediction(self):
        """Prediction snapshot for the UI: every observable and its time derivative d/dt, both from the
        network (the derivative by autograd, the same way the physics residual is formed)."""
        with torch.enable_grad():
            t = self.t_eval.detach().clone().requires_grad_(True)
            x = self.model(t)
            xs = [x[:, i] for i in range(x.shape[1])]
            obs, rates = {}, {}
            for o in self.system.observables:
                v = self.system.observable_torch(o, t, xs)
                obs[o] = _np(v)
                g = torch.autograd.grad(v.sum(), t, retain_graph=True, allow_unused=True)[0]
                rates[o] = _np(g) if g is not None else np.zeros(len(obs[o]))
        self._pred = {"epoch": self.epoch, "t": _np(t), "observables": obs, "rates": rates}
        self._weights_view = [_np(m.weight) for m in self.model.linear_layers()]
        self.version += 1

    # ----------------------------------------------------------- inference
    def predict(self, t):
        with self.lock, torch.no_grad():
            tt = torch.as_tensor(np.asarray(t, float), dtype=self.dtype).to(self.device)
            x = self.model(tt)
            xs = [x[:, i] for i in range(x.shape[1])]
            return {o: _np(self.system.observable_torch(o, tt, xs)) for o in self.system.observables}

    def snapshot(self):
        """Cheap, thread-safe copy of what the dashboard shows."""
        mark = self._t_mark
        return {
            "status": self.status, "message": self.message, "epoch": self.epoch,
            "target_epochs": self.target_epochs, "last": dict(self.last),
            "lr": self.optimizer.param_groups[0]["lr"],
            "elapsed": self.elapsed + (time.perf_counter() - mark if mark else 0.0),
            "best_val": self.best_val, "version": self.version, "run_dir": str(self.run_dir or ""),
            "saved": list(self.saved[-6:]), "loaded_from": self.loaded_from,
            "device": self.device_label(),
        }

    def device_label(self):
        prec = "float32" if self.dtype == torch.float32 else "float64"
        if self.device.type == "cuda":
            mb = torch.cuda.memory_allocated(self.device) / 2 ** 20
            return f"GPU · {gpu_name()} · {prec} · {mb:.0f} MB"
        return f"CPU · {prec}"

    def history_copy(self):
        return {k: list(v) for k, v in self.history.items()}

    def prediction(self):
        return self._pred

    def weights_view(self):
        return self._weights_view

    # --------------------------------------------------------- checkpoints
    def _payload(self, kind):
        return {
            "kind": kind, "title": self.system.title,
            "model_state_dict": _cpu(self.model.state_dict()),
            "optimizer_state_dict": _cpu(self.optimizer.state_dict()),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else {},
            "epoch": self.epoch, "target_epochs": self.target_epochs,
            "total_loss": self.last["total"], "data_loss": self.last["data"],
            "physics_loss": self.last["physics"], "bc_loss": self.last["bc"], "ic_loss": self.last["ic"],
            "val_loss": self.last["val"],
            "best_val_loss": None if math.isinf(self.best_val) else self.best_val,
            "config": asdict(self.config), "model_text": self.system.source_text,
            "scaling": {"mode": self.scaling["mode"], "mu": list(self.scaling["mu"]),
                        "sigma": list(self.scaling["sigma"]), "obs_scale": dict(self.scaling["obs_scale"]),
                        "source": self.scaling.get("source", "")},
            "dataset": self.dataset.to_dict(), "history": self.history_copy(),
            "elapsed_s": self.elapsed, "sampler_rng_state": self.gen.get_state(),
        }

    def _save(self, name, kind, now=None):
        path = ckpt.save(self._payload(kind), self.run_dir / name)
        self.saved.append(path.name)
        return path

    def save_checkpoint(self):
        """Manual save from the UI (safe while training: waits for the current step)."""
        with self.lock:
            if self.run_dir is None:
                self.run_dir = ckpt.new_run_dir(self.system.title)
            return self._save(f"checkpoint_epoch_{self.epoch}.pt", "manual")

    @classmethod
    def from_checkpoint(cls, path, reference=None, device=None, precision=None, on_complete=None):
        """device/precision override the values stored in the checkpoint (e.g. load a GPU run on a CPU)."""
        ck = ckpt.load(path)
        system = parse_model(ck["model_text"])
        dataset = Dataset.from_dict(ck["dataset"])
        config = TrainConfig.from_dict(ck["config"])
        if device is not None:
            config.device = device
        if precision is not None:
            config.precision = precision
        if reference is None:
            try:
                reference = system.solve_reference()
            except Exception:  # noqa: BLE001
                reference = None
        s = cls(system, dataset, config, reference=reference, run_dir=Path(path).parent,
                scaling=ck["scaling"], checkpoint_root=Path(path).parent.parent, on_complete=on_complete)
        s.model.load_state_dict(ck["model_state_dict"])
        s.optimizer.load_state_dict(ck["optimizer_state_dict"])
        if s.scheduler is not None and ck.get("scheduler_state_dict"):
            s.scheduler.load_state_dict(ck["scheduler_state_dict"])
        s.gen.set_state(ck["sampler_rng_state"])
        s.epoch = int(ck["epoch"])
        s.target_epochs = int(ck.get("target_epochs") or s.epoch)
        s.elapsed = float(ck.get("elapsed_s") or 0.0)
        s.best_val = ck["best_val_loss"] if ck.get("best_val_loss") is not None else math.inf
        s.history = {k: list(ck["history"].get(k, [])) for k in HISTORY_KEYS}
        s.last = {"total": ck["total_loss"], "data": ck["data_loss"], "physics": ck["physics_loss"],
                  "bc": ck.get("bc_loss"), "ic": ck.get("ic_loss"), "val": ck.get("val_loss")}
        s.status = "checkpoint_loaded"
        s.loaded_from = {"file": Path(path).name, "run": Path(path).parent.name, "epoch": s.epoch,
                         "total_loss": ck["total_loss"], "saved_at": ck.get("saved_at")}
        s._refresh_prediction()
        return s
