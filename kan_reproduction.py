"""
Reproduce the KAN paper's toy function fitting experiment with official pykan.

Paper:
  Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson,
  Marin Soljacic, Thomas Y. Hou, Max Tegmark. "KAN: Kolmogorov-Arnold
  Networks." ICLR 2025; arXiv:2404.19756, 2024.

This script reproduces the official pykan "Hello KAN" experiment on:
  f(x, y) = exp(sin(pi*x) + y^2), x,y in [-1, 1].

It uses the official KAN implementation with cubic B-spline edge functions,
LBFGS training, sparsity regularization, pruning, and optional symbolic
regression. MLP baselines are included only for the course-report comparison.

Run:
  python kan_reproduction.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import sympy as sp
import torch
from kan import KAN
from kan.utils import create_dataset
from torch import nn


SEED = 2026
OUT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int = SEED, torch_threads: int | None = None) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)


def resolve_device(device: str) -> str:
    requested = device.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def target_function(x: torch.Tensor) -> torch.Tensor:
    """Toy function used in the official pykan introduction."""
    return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)


def make_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    device: str,
) -> dict[str, torch.Tensor]:
    return create_dataset(
        target_function,
        n_var=2,
        ranges=[-1, 1],
        train_num=n_train,
        test_num=n_test,
        normalize_input=False,
        normalize_label=False,
        device=device,
        seed=seed,
    )


def save_dataset_csv(dataset: dict[str, torch.Tensor], data_path: Path) -> None:
    with data_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "x", "y", "target"])
        for split, inputs_key, labels_key in [
            ("train", "train_input", "train_label"),
            ("test", "test_input", "test_label"),
        ]:
            x_data = dataset[inputs_key].detach().cpu()
            y_data = dataset[labels_key].detach().cpu()
            for x, y in zip(x_data, y_data):
                writer.writerow([
                    split,
                    f"{x[0].item():.10f}",
                    f"{x[1].item():.10f}",
                    f"{y.item():.10f}",
                ])


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_ckpt_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    history = path / "history.txt"
    if not history.exists():
        history.write_text("", encoding="utf-8")


def evaluate_torch_model(model: nn.Module, dataset: dict[str, torch.Tensor]) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_score = rmse(model(dataset["train_input"]), dataset["train_label"])
        test_score = rmse(model(dataset["test_input"]), dataset["test_label"])
    return train_score, test_score


def fit_official_kan(
    dataset: dict[str, torch.Tensor],
    out_dir: Path,
    seed: int,
    width: list[int],
    grid: int,
    spline_order: int,
    steps: int,
    prune_steps: int,
    symbolic_steps: int,
    lamb: float,
    lamb_entropy: float,
    device: str,
    clean_ckpt: bool,
) -> tuple[KAN, dict[str, Any]]:
    original_width = list(width)
    ckpt_path = out_dir / "pykan_checkpoints"
    prepare_ckpt_dir(ckpt_path, clean=clean_ckpt)

    model = KAN(
        width=list(original_width),
        grid=grid,
        k=spline_order,
        seed=seed,
        device=device,
        symbolic_enabled=True,
        auto_save=False,
        ckpt_path=str(ckpt_path),
    )

    start = time.perf_counter()
    history: dict[str, Any] = {}
    history["initial_params"] = count_params(model)
    history["stage1"] = model.fit(
        dataset,
        opt="LBFGS",
        steps=steps,
        lamb=lamb,
        lamb_entropy=lamb_entropy,
        log=1,
    )

    pruned_model = model.prune()
    history["pruned_params"] = count_params(pruned_model)

    if prune_steps > 0:
        history["after_prune"] = pruned_model.fit(
            dataset,
            opt="LBFGS",
            steps=prune_steps,
            lamb=0.0,
            log=1,
        )

    symbolic_info: dict[str, Any] = {"attempted": False, "formula": None, "error": None}
    if symbolic_steps > 0:
        symbolic_info["attempted"] = True
        try:
            pruned_model.auto_symbolic(lib=["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "abs"])
            history["after_symbolic"] = pruned_model.fit(
                dataset,
                opt="LBFGS",
                steps=symbolic_steps,
                lamb=0.0,
                log=1,
            )
            x, y = sp.symbols("x y")
            symbolic_info["formula"] = str(pruned_model.symbolic_formula(var=[x, y])[0][0])
        except Exception as exc:  # pykan symbolic conversion can fail on some environments.
            symbolic_info["error"] = f"{type(exc).__name__}: {exc}"

    train_rmse, test_rmse = evaluate_torch_model(pruned_model, dataset)
    stats = {
        "model": f"Official-pykan-KAN-width-{original_width}-grid-{grid}-k-{spline_order}",
        "optimizer": "LBFGS",
        "params": count_params(pruned_model),
        "best_train_rmse": train_rmse,
        "best_test_rmse": test_rmse,
        "seconds": time.perf_counter() - start,
        "details": {
            "implementation": "pykan",
            "initial_params": history["initial_params"],
            "pruned_params": history["pruned_params"],
            "width": original_width,
            "grid": grid,
            "spline_order": spline_order,
            "lamb": lamb,
            "lamb_entropy": lamb_entropy,
            "steps": steps,
            "prune_steps": prune_steps,
            "symbolic_steps": symbolic_steps,
            "post_prune_lamb": 0.0,
            "symbolic": symbolic_info,
        },
    }
    return pruned_model, stats


class MLP(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    model: nn.Module,
    dataset: dict[str, torch.Tensor],
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_test = float("inf")
    best_train = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(dataset["train_input"])
        loss = loss_fn(pred, dataset["train_label"])
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == epochs:
            train_score, test_score = evaluate_torch_model(model, dataset)
            if test_score < best_test:
                best_test = test_score
                best_train = train_score
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "optimizer": "AdamW",
        "params": count_params(model),
        "best_train_rmse": best_train,
        "best_test_rmse": best_test,
        "seconds": time.perf_counter() - start,
    }


def write_outputs(
    out_dir: Path,
    results: list[dict[str, Any]],
    best_model: nn.Module,
    dataset: dict[str, torch.Tensor],
    prediction_samples: int,
    run_config: dict[str, Any],
) -> None:
    csv_rows = [
        {
            "model": row["model"],
            "optimizer": row["optimizer"],
            "params": row["params"],
            "best_train_rmse": row["best_train_rmse"],
            "best_test_rmse": row["best_test_rmse"],
            "seconds": row["seconds"],
        }
        for row in results
    ]

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    best_model.eval()
    with torch.no_grad():
        sample_count = min(prediction_samples, len(dataset["test_input"]))
        sample_x = dataset["test_input"][:sample_count]
        sample_y = dataset["test_label"][:sample_count]
        sample_pred = best_model(sample_x)

    with (out_dir / "prediction_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "target", "prediction", "abs_error"])
        for xi, yi, pi in zip(sample_x.cpu(), sample_y.cpu(), sample_pred.detach().cpu()):
            writer.writerow([
                f"{xi[0].item():.6f}",
                f"{xi[1].item():.6f}",
                f"{yi.item():.6f}",
                f"{pi.item():.6f}",
                f"{abs(yi.item() - pi.item()):.6f}",
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official pykan reproduction of the KAN paper's toy function experiment."
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for dataset generation and training.")
    parser.add_argument("--n-train", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--n-test", type=int, default=2000, help="Number of test samples.")
    parser.add_argument("--width", type=int, nargs="+", default=[2, 5, 1], help="KAN width, e.g. 2 5 1.")
    parser.add_argument("--grid", type=int, default=5, help="Number of KAN grid intervals.")
    parser.add_argument("--spline-order", type=int, default=3, help="B-spline order k used by pykan.")
    parser.add_argument("--kan-steps", type=int, default=20, help="Initial pykan LBFGS steps.")
    parser.add_argument("--prune-steps", type=int, default=50, help="Additional LBFGS steps after pruning.")
    parser.add_argument("--symbolic-steps", type=int, default=50, help="Optional LBFGS steps after auto_symbolic.")
    parser.add_argument("--lamb", type=float, default=0.01, help="pykan sparsity regularization strength.")
    parser.add_argument("--lamb-entropy", type=float, default=10.0, help="pykan entropy regularization strength.")
    parser.add_argument("--mlp-epochs", type=int, default=2500, help="AdamW epochs for each MLP baseline.")
    parser.add_argument("--mlp-lr", type=float, default=0.01, help="AdamW learning rate for MLP baselines.")
    parser.add_argument("--prediction-samples", type=int, default=20, help="Rows to write into prediction_samples.csv.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:0. Defaults to auto.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Optional CPU thread limit. Leave unset to use PyTorch defaults.",
    )
    parser.add_argument("--clean-ckpt", action="store_true", help="Clean pykan_checkpoints before running.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for dataset and result files. Defaults to the script directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    set_seed(args.seed, torch_threads=args.torch_threads)
    print(f"Using torch device: {device}")
    if device.startswith("cuda"):
        print(f"CUDA device: {torch.cuda.get_device_name(torch.device(device))}")

    dataset = make_dataset(args.seed, args.n_train, args.n_test, device)
    save_dataset_csv(dataset, out_dir / "kan_paper_toy_dataset.csv")

    kan_model, kan_stats = fit_official_kan(
        dataset=dataset,
        out_dir=out_dir,
        seed=args.seed,
        width=args.width,
        grid=args.grid,
        spline_order=args.spline_order,
        steps=args.kan_steps,
        prune_steps=args.prune_steps,
        symbolic_steps=args.symbolic_steps,
        lamb=args.lamb,
        lamb_entropy=args.lamb_entropy,
        device=device,
        clean_ckpt=args.clean_ckpt,
    )

    experiments: list[tuple[str, nn.Module]] = [
        ("MLP-16-hidden", MLP(hidden=16).to(device)),
        ("MLP-24-hidden", MLP(hidden=24).to(device)),
    ]

    results = [kan_stats]
    models: dict[str, nn.Module] = {kan_stats["model"]: kan_model}
    for name, model in experiments:
        stats = train_mlp(model, dataset, epochs=args.mlp_epochs, lr=args.mlp_lr)
        row = {"model": name, **stats, "details": {"implementation": "torch.nn.MLP"}}
        results.append(row)
        models[name] = model

    for row in results:
        print(
            f"{row['model']:42s} params={row['params']:5d} "
            f"train_rmse={row['best_train_rmse']:.6f} "
            f"test_rmse={row['best_test_rmse']:.6f} "
            f"time={row['seconds']:.2f}s"
        )

    best_name = min(results, key=lambda r: r["best_test_rmse"])["model"]
    run_config = {
        "paper_function": "f(x,y)=exp(sin(pi*x)+y^2), x,y in [-1,1]",
        "paper_code_path": "official pykan / Hello KAN example",
        "seed": args.seed,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "width": args.width,
        "grid": args.grid,
        "spline_order": args.spline_order,
        "kan_steps": args.kan_steps,
        "prune_steps": args.prune_steps,
        "symbolic_steps": args.symbolic_steps,
        "lamb": args.lamb,
        "lamb_entropy": args.lamb_entropy,
        "mlp_epochs": args.mlp_epochs,
        "mlp_lr": args.mlp_lr,
        "requested_device": args.device,
        "resolved_device": device,
        "cuda_device": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "torch_threads": args.torch_threads,
        "best_model": best_name,
    }
    write_outputs(
        out_dir=out_dir,
        results=results,
        best_model=models[best_name],
        dataset=dataset,
        prediction_samples=args.prediction_samples,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
