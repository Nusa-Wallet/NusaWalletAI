"""Demo CLI: explain flagged transactions from the trained fraud bundle (Phase 6).

Loads the saved artifacts, scores the dataset, and prints 3-5 Indonesian risk factors
for one flagged example per anomaly type. Useful for the demo scenarios and for a
quick sanity check that explanations match feature values.

    python -m scripts.explain_fraud
    python -m scripts.explain_fraud --max-factors 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.fraud.explain import explain_if_flagged  # noqa: E402
from app.fraud.explain.shap_explain import shap_matrix, shap_row_dict  # noqa: E402
from app.fraud.feature_spec import MODEL_FEATURES  # noqa: E402
from app.fraud.simulation.config import DATASET_VERSION  # noqa: E402
from app.fraud.training.data import to_xy  # noqa: E402
from app.fraud.training.pipeline import load_bundle, predict_risk  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Explain flagged fraud transactions.")
    p.add_argument("--dataset", type=str, default=str(_ROOT / "data" / "synthetic" / f"{DATASET_VERSION}.parquet"))
    p.add_argument("--artifacts-dir", type=str, default=str(_ROOT / "artifacts"))
    p.add_argument("--max-factors", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model, isolation, ensemble, _ = load_bundle(Path(args.artifacts_dir))

    df = pd.read_parquet(args.dataset)
    risk = predict_risk(model, isolation, ensemble, df)
    df = df.assign(_risk=risk, _flagged=risk >= ensemble.threshold)

    x, _ = to_xy(df)
    print(f"Threshold={ensemble.threshold:.3f}  flagged={int(df['_flagged'].sum()):,}/{len(df):,}\n")

    shown = 0
    for atype, group in df[df["_flagged"]].groupby("anomaly_type"):
        row = group.sort_values("_risk", ascending=False).iloc[0]
        shap_row = shap_row_dict(shap_matrix(model, x.loc[[row.name]])[0], list(MODEL_FEATURES))
        explanation = explain_if_flagged(
            row["_risk"], ensemble.threshold, row.to_dict(), shap_row, max_factors=args.max_factors
        )
        print(f"[{atype}] {row['transaction_id']}  risk={row['_risk']:.3f}  "
              f"amount={row['amount']:.2f} {row['currency']} from {row['origin_country']}")
        for i, factor in enumerate(explanation.factors, 1):
            print(f"    {i}. {factor}")
        print()
        shown += 1
    print(f"Explained {shown} flagged anomaly types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
