"""Generate exact Phase 15 fraud evaluation artifacts without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.fraud.training.data import load_dataset, time_split, to_xy  # noqa: E402
from app.fraud.training.pipeline import load_bundle, predict_risk  # noqa: E402


def build_report(root: Path = ROOT) -> dict:
    artifacts = root / "artifacts"
    metadata = json.loads((artifacts / "fraud_metadata.json").read_text(encoding="utf-8"))
    dataset = root / "data" / "synthetic" / f"{metadata['dataset_version']}.parquet"
    frame = load_dataset(str(dataset))
    split = metadata["split"]
    test = time_split(frame, split["train_frac"], split["val_frac"]).test
    _, y_test = to_xy(test)
    model, isolation, ensemble, _ = load_bundle(artifacts)
    risk = predict_risk(model, isolation, ensemble, test)
    pred = risk >= float(metadata["threshold"])
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    observed, predicted = calibration_curve(y_test, risk, n_bins=10, strategy="quantile")
    return {
        "phase": 15,
        "evaluation_policy": "untouched chronological test split; no retraining",
        "model_version": metadata["model_version"],
        "dataset_version": metadata["dataset_version"],
        "threshold": metadata["threshold"],
        "test_rows": int(len(test)),
        "confusion_matrix": {"labels": ["normal", "fraud"], "tn": int(tn),
                             "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration_curve": [
            {"mean_predicted_probability": round(float(p), 6),
             "observed_positive_fraction": round(float(o), 6)}
            for p, o in zip(predicted, observed)
        ],
        "metrics": metadata["metrics"]["full-ensemble"],
        "shap_mean_abs_top": metadata["shap_global_summary_top"],
    }


def calibration_svg(report: dict) -> str:
    points = report["calibration_curve"]
    coords = " ".join(f"{60+p['mean_predicted_probability']*400:.1f},{440-p['observed_positive_fraction']*400:.1f}" for p in points)
    circles = "".join(f'<circle cx="{60+p["mean_predicted_probability"]*400:.1f}" cy="{440-p["observed_positive_fraction"]*400:.1f}" r="4"/>' for p in points)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="520" height="500"><style>text{{font:14px sans-serif}}.a{{stroke:#333}}.i{{stroke:#999;stroke-dasharray:6 4}}.c{{fill:none;stroke:#2563eb;stroke-width:3}}circle{{fill:#2563eb}}</style><text x="150" y="24" font-size="18">Fraud calibration curve (test)</text><line class="a" x1="60" y1="440" x2="460" y2="440"/><line class="a" x1="60" y1="440" x2="60" y2="40"/><line class="i" x1="60" y1="440" x2="460" y2="40"/><polyline class="c" points="{coords}"/>{circles}<text x="180" y="480">Mean predicted probability</text><text transform="translate(18 340) rotate(-90)">Observed fraud fraction</text><text x="54" y="460">0</text><text x="450" y="460">1</text><text x="38" y="445">0</text><text x="38" y="45">1</text></svg>'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "phase15")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (args.output_dir / "fraud_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "fraud_calibration.svg").write_text(calibration_svg(report), encoding="utf-8")
    print(f"Phase 15 evaluation written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
