"""
PINN Lab -- Bond graph (BondLab) -> governing equations -> physics-informed neural network.

Run from anywhere:
    python pinn_lab/main.py            # then open http://127.0.0.1:8050
    python pinn_lab/main.py --port 8060 --threads 2
    python pinn_lab/main.py --device cuda      # default the UI to the GPU
"""
import argparse
import faulthandler
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description="PINN laboratory for bond-graph equations")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                    help="default training device shown in the UI (auto = CUDA GPU when available)")
    ap.add_argument("--threads", type=int, default=1,
                    help="PyTorch CPU threads for training (1 is fastest for these small networks)")
    args = ap.parse_args()

    # print the Python stack of every thread if native code (torch/numpy/scipy) ever crashes
    faulthandler.enable()

    # Load pandas (and the pyarrow DLLs it pulls in) BEFORE torch. On Windows, importing pyarrow after
    # torch raises a native access violation while the DLLs load (seen with torch 2.6 + pyarrow 24);
    # the reverse order is clean.
    import pandas  # noqa: F401
    import torch
    torch.set_num_threads(max(1, args.threads))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    from ui.dashboard import create_app
    app = create_app(default_device=args.device)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"PINN Lab running on http://{args.host}:{args.port}  (CUDA GPU: {gpu}; Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
