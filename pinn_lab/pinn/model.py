"""
model.py -- the PINN: a network x_hat(t) with built-in non-dimensionalisation.

    t  --(affine)-->  tau in [-1, 1]  -->  network  -->  y_hat (dimensionless)
    x(t) = mu + sigma * y_hat           (physical state, per state variable)

Network types
    mlp      plain fully connected network (the default)
    resnet   same widths, but every hidden layer after the first adds a skip
             connection  h <- h + act(W h + b)  (easier to train when deep)
    fourier  the input tau is first expanded into fixed Fourier features
             [sin(k*pi*tau/2), cos(k*pi*tau/2)], k = 1..m, then an MLP; helps
             with oscillatory solutions

mu, sigma, t_start and t_end are registered buffers, so they are stored in
the state_dict, move with model.to(device) and are restored exactly by a
checkpoint. The dtype (float64 or float32) is chosen by the trainer. The
"mlp" parameter names (net.0.weight, ...) are unchanged from earlier
versions, so older checkpoints still load.
"""
import math

import torch
from torch import nn

from bondgraph.parser import DTYPE


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "sin": Sine,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
}
NETWORKS = ("mlp", "resnet", "fourier")
NETWORK_LABELS = {"mlp": "MLP", "resnet": "Residual MLP", "fourier": "Fourier-feature MLP"}


class PINN(nn.Module):
    def __init__(self, n_out, hidden_layers=4, neurons=64, activation="tanh",
                 t_start=0.0, t_end=1.0, mu=None, sigma=None, dtype=DTYPE, network="mlp", fourier_features=8):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        if network not in NETWORKS:
            raise ValueError(f"unknown network type {network!r}")
        if hidden_layers < 1 or neurons < 1:
            raise ValueError("need at least one hidden layer with at least one neuron")
        self.n_out = int(n_out)
        self.hidden_layers = int(hidden_layers)
        self.neurons = int(neurons)
        self.activation = activation
        self.network = network
        self.fourier_features = int(fourier_features) if network == "fourier" else 0
        act = ACTIVATIONS[activation]

        in_width = 2 * self.fourier_features if network == "fourier" else 1
        if network == "fourier":
            k = torch.arange(1, self.fourier_features + 1, dtype=dtype)
            self.register_buffer("freqs", k * math.pi / 2.0)

        if network in ("mlp", "fourier"):
            layers, width = [], in_width
            for _ in range(self.hidden_layers):
                layers += [nn.Linear(width, self.neurons), act()]
                width = self.neurons
            layers.append(nn.Linear(width, self.n_out))
            self.net = nn.Sequential(*layers)
        else:  # resnet
            self.inp = nn.Linear(in_width, self.neurons)
            self.hidden = nn.ModuleList(nn.Linear(self.neurons, self.neurons) for _ in range(self.hidden_layers - 1))
            self.out = nn.Linear(self.neurons, self.n_out)
            self.act = act()
        self._init_weights()

        self.register_buffer("t_start", torch.tensor(float(t_start), dtype=dtype))
        self.register_buffer("t_end", torch.tensor(float(t_end), dtype=dtype))
        self.register_buffer("mu", torch.tensor(mu if mu is not None else [0.0] * n_out, dtype=dtype))
        self.register_buffer("sigma", torch.tensor(sigma if sigma is not None else [1.0] * n_out, dtype=dtype))
        self.to(dtype)

    def linear_layers(self):
        """Linear layers in forward order (used by the architecture diagram)."""
        if self.network == "resnet":
            return [self.inp, *self.hidden, self.out]
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def _init_weights(self):
        gain = nn.init.calculate_gain("tanh") if self.activation == "tanh" else 1.0
        layers = self.linear_layers()
        for m in layers:
            nn.init.xavier_normal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)
        if self.network == "resnet":       # start residual branches small so the net starts near an MLP
            for m in self.hidden:
                with torch.no_grad():
                    m.weight.mul_(0.5)
        if self.activation == "sin" and self.network != "fourier":
            # SIREN-style first layer so sin() sees a useful frequency range
            with torch.no_grad():
                layers[0].weight.uniform_(-math.pi, math.pi)

    # time <-> normalised time
    def to_tau(self, t):
        return 2.0 * (t - self.t_start) / (self.t_end - self.t_start) - 1.0

    @property
    def dt_dtau(self):
        return 0.5 * (self.t_end - self.t_start)

    def forward_tau(self, tau):
        """Dimensionless output y_hat for tau of shape (N, 1)."""
        if self.network == "fourier":
            z = tau * self.freqs
            return self.net(torch.cat([torch.sin(z), torch.cos(z)], dim=1))
        if self.network == "resnet":
            h = self.act(self.inp(tau))
            for layer in self.hidden:
                h = h + self.act(layer(h))
            return self.out(h)
        return self.net(tau)

    def forward(self, t):
        """Physical states x(t) for t of shape (N,) or (N, 1); returns (N, n_out)."""
        t = t.reshape(-1, 1).to(dtype=self.mu.dtype, device=self.mu.device)
        return self.mu + self.sigma * self.forward_tau(self.to_tau(t))

    @property
    def device(self):
        return self.mu.device

    @property
    def dtype(self):
        return self.mu.dtype

    def layer_sizes(self):
        first = 2 * self.fourier_features if self.network == "fourier" else 1
        return [first] + [self.neurons] * self.hidden_layers + [self.n_out]

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        return f"{self.hidden_layers}-layer {NETWORK_LABELS[self.network]} ({self.neurons} neurons, {self.activation})"

    def config(self):
        return {"n_out": self.n_out, "hidden_layers": self.hidden_layers, "neurons": self.neurons,
                "activation": self.activation, "network": self.network}
