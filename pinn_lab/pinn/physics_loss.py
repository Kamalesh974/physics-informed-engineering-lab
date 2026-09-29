"""
physics_loss.py -- the PINN loss

    L_total = lambda_data * L_data + lambda_physics * L_physics
            + lambda_BC * L_BC + lambda_IC * L_IC

L_physics is built directly from the bond-graph equations: for every state
x_i the residual

    r_i(t) = d x_i / dt - f_i(x(t), t)

is evaluated at collocation points, with d x_i / dt obtained by PyTorch
autograd through the network. In normalised variables (x = mu + sigma*y_hat,
t = t_start + (tau + 1) * dt_dtau) this is

    r_i / (sigma_i / dt_dtau) = d y_hat_i / d tau - (dt_dtau / sigma_i) * f_i

so every residual is O(1) regardless of the physical units.
"""
from dataclasses import dataclass

import torch

from bondgraph.parser import DTYPE


@dataclass
class LossWeights:
    data: float = 1.0
    physics: float = 1.0
    bc: float = 1.0
    ic: float = 10.0


class PhysicsLoss:
    def __init__(self, system, scaling, dataset=None, dtype=DTYPE, device="cpu",
                 residual_scaling="normalized", normalize_data=True):
        """residual_scaling: "normalized" = each residual divided by sigma_i/dt_dtau (O(1) terms, default);
        "physical" = raw residual dx/dt - f in physical units.
        normalize_data: divide data/BC misfits by each quantity's scale (else raw units)."""
        if residual_scaling not in ("normalized", "physical"):
            raise ValueError("residual_scaling must be 'normalized' or 'physical'")
        self.system = system
        self.dtype, self.device = dtype, torch.device(device)
        self.n = len(system.states)
        self.residual_scaling = residual_scaling
        self.obs_scale = {o: (float(v) if normalize_data else 1.0) for o, v in scaling["obs_scale"].items()}
        self.x0 = self._t([system.initial.get(s, 0.0) for s in system.states])

        # measurement data -> tensors, one entry per observable (NaNs dropped)
        self.data = self._pack(dataset.t_train, dataset.y_train) if dataset is not None and dataset.has_data else []
        self.val = self._pack(dataset.t_val, dataset.y_val) if dataset is not None and dataset.has_val else []
        # boundary / point constraints grouped the same way
        self.bc = [(o, self._t([tb]), self._t([v])) for o, tb, v in system.boundary]

    def _t(self, values):
        return torch.as_tensor(values, dtype=self.dtype).to(self.device)

    def _pack(self, t, ys):
        out = []
        for o, y in ys.items():
            if o not in self.system.observables:
                continue
            y = torch.as_tensor(y, dtype=torch.float64)
            m = torch.isfinite(y)
            if int(m.sum()) == 0:
                continue
            out.append((o, self._t(torch.as_tensor(t, dtype=torch.float64)[m]), self._t(y[m])))
        return out

    # ------------------------------------------------------------------ terms
    def residuals(self, model, t_colloc):
        """Residuals of the bond-graph equations, shape (N, n_states). t_colloc: (N,) physical time."""
        tau = model.to_tau(t_colloc.reshape(-1, 1)).detach().requires_grad_(True)
        y_hat = model.forward_tau(tau)                                  # (N, S)
        x = model.mu + model.sigma * y_hat
        t = t_colloc.reshape(-1)
        f = self.system.rhs_torch(t, [x[:, i] for i in range(self.n)])
        cols = []
        for i in range(self.n):
            dy = torch.autograd.grad(y_hat[:, i].sum(), tau, create_graph=True)[0][:, 0]
            if self.residual_scaling == "physical":        # dx/dt - f  in physical units
                cols.append(dy * model.sigma[i] / model.dt_dtau - f[i])
            else:                                          # (dx/dt - f) / (sigma_i / dt_dtau)
                cols.append(dy - (model.dt_dtau / model.sigma[i]) * f[i])
        return torch.stack(cols, dim=1)

    def physics(self, model, t_colloc):
        return (self.residuals(model, t_colloc) ** 2).mean()

    def initial(self, model):
        x0_hat = model(model.t_start.reshape(1))[0]
        return (((x0_hat - self.x0) / model.sigma) ** 2).mean()

    def _obs_mse(self, model, groups):
        if not groups:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        total, count = torch.zeros((), dtype=self.dtype, device=self.device), 0
        for o, t, y in groups:
            x = model(t)
            pred = self.system.observable_torch(o, t, [x[:, i] for i in range(self.n)])
            total = total + (((pred - y) / self.obs_scale[o]) ** 2).sum()
            count += y.numel()
        return total / count

    def data_loss(self, model):
        return self._obs_mse(model, self.data)

    def val_data_loss(self, model):
        return self._obs_mse(model, self.val)

    def boundary(self, model):
        return self._obs_mse(model, self.bc)

    def __call__(self, model, t_colloc, w: LossWeights):
        terms = {
            "data": self.data_loss(model),
            "physics": self.physics(model, t_colloc),
            "bc": self.boundary(model),
            "ic": self.initial(model),
        }
        terms["total"] = (w.data * terms["data"] + w.physics * terms["physics"]
                          + w.bc * terms["bc"] + w.ic * terms["ic"])
        return terms
