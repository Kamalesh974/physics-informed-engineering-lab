"""
dashboard.py -- builds the PINNlab Dash app.

    create_app()                                   standalone (python pinn_lab/main.py)
    create_app(prefix="/pinnlab/", links=...)      inside the integrated Physics-Informed Engineering Lab
                                                   server (integrated/server.py), next to BondLab

The layout is a function, so every page load shows the current project and settings (state is kept on
the server, e.g. while you go to BondLab and come back).
"""
from pathlib import Path

import dash

from . import callbacks
from .layout import build_layout
from .state import LAB

ASSETS = Path(__file__).resolve().parent / "assets"


def create_app(prefix="/", links=None, default_device="auto"):
    links = dict(links or {})
    LAB.default_device = default_device
    kw = {}
    if prefix != "/":
        # mounted under a sub-path by the integrated server: Flask routes stay at "/", the browser
        # requests them under the prefix
        kw.update(requests_pathname_prefix=prefix, routes_pathname_prefix="/")
    app = dash.Dash(__name__, assets_folder=str(ASSETS), title="PINNlab — Physics-Informed Engineering Lab",
                    update_title=None, suppress_callback_exceptions=True, **kw)
    app.layout = lambda: build_layout(LAB, links)
    callbacks.register(app, links)
    return app
