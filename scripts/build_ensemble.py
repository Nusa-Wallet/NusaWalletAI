"""Phase 12: build the FX ensemble + evaluate the fee-aware decision engine.

Discovers the latest Phase 9/10 backtest run per model, derives per-pair weights from
validation error, combines the saved predictions, runs the decision engine, and writes
artifacts/fx_ensemble/fx_ensemble_metadata.json (weights + decision metrics). No model is
re-run — it combines the existing predictions.parquet files.

    python scripts/build_ensemble.py
    python scripts/build_ensemble.py --risk CONSERVATIVE --amount 5000
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.fx.dataset.config import PRIMARY_PAIRS  # noqa: E402
from app.fx.decision.config import MODEL_VERSION, DecisionConfig  # noqa: E402
from app.fx.decision.ensemble import combine_predictions  # noqa: E402
from app.fx.decision.evaluate import evaluate_decisions, model_net_gain, summarize  # noqa: E402
from app.fx.decision.weights import derive_weights  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKTESTS = ROOT / "artifacts" / "fx_backtests"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def discover_runs(models: tuple[str, ...]) -> dict[str, Path]:
    """Latest backtest run directory per model name."""
    found: dict[str, Path] = {}
    for directory in sorted(BACKTESTS.glob("*/"), key=lambda p: p.stat().st_mtime):
        meta = directory / "metadata.json"
        if not meta.exists():
            continue
        for model in json.loads(meta.read_text())["models"]:
            if model["name"] in models:
                found[model["name"]] = directory
    return found


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build FX ensemble + decision engine metadata.")
    p.add_argument("--models", default=",".join(DecisionConfig().models))
    p.add_argument("--pairs", default=",".join(PRIMARY_PAIRS))
    p.add_argument("--amount", type=float, default=1000.0)
    p.add_argument("--risk", default="MODERATE")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "fx_ensemble")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    config = DecisionConfig(models=models)

    runs = discover_runs(models)
    missing = [m for m in models if m not in runs]
    if missing:
        print(f"WARNING: no backtest run found for: {missing} (run scripts/backtest_fx.py first)")
    if not runs:
        print("No model runs found under artifacts/fx_backtests. Aborting.")
        return 1

    per_model = {m: pd.read_parquet(runs[m] / "predictions.parquet") for m in runs}
    per_metrics = {m: json.loads((runs[m] / "metrics.json").read_text()) for m in runs}

    weights = derive_weights(per_metrics, tuple(runs), pairs)
    ensemble = combine_predictions(per_model, weights, tuple(runs))
    decisions = evaluate_decisions(ensemble, config, amount=args.amount, risk_preference=args.risk)
    summary = summarize(decisions)

    metadata = {
        "model_version": MODEL_VERSION,
        "git_commit": _git_commit(),
        "models": list(runs),
        "source_runs": {m: runs[m].name for m in runs},
        "pairs": pairs,
        "weights": weights,
        "decision_config": {"fee_rate": config.fee_rate, "risk_preference": args.risk, "amount": args.amount},
        "decision_metrics": summary,
        "model_val_net_gain": model_net_gain(per_model, "val"),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "fx_ensemble_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Ensemble models: {list(runs)}")
    print(f"Weights (val error-based) e.g. {pairs[0]}: {weights.get(pairs[0])}")
    print("\nDecision engine — net gain vs immediate (fee-aware):")
    for split in ("val", "test"):
        s = summary["by_split"].get(split)
        if s:
            print(f"  {split}: total={s['total_net_gain']:.1f} mean={s['mean_net_gain']:.1f} "
                  f"max_regret={s['max_regret']:.1f} conf={s['mean_confidence']:.2f} actions={s['action_mix']}")
    print("\nvs individual models (val total net gain):", metadata["model_val_net_gain"])
    print(f"\nSaved -> {args.output_root / 'fx_ensemble_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
