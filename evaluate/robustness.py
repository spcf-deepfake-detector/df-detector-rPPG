
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from evaluate.metrics import compute_all_metrics


class RobustnessEvaluator:
    """Evaluate FalseFacet performance across different video conditions.

    Labels:
        0 = REAL
        1 = FAKE

    Scores must be predicted probabilities for the FAKE class.
    """

    DEFAULT_ATTRIBUTES = [
        "quality",
        "lighting",
        "compression",
        "motion",
        "camera_id",
        "generator",
        "subject_id",
    ]

    def __init__(self, metadata_df: pd.DataFrame):
        if metadata_df.empty:
            raise ValueError("The metadata DataFrame is empty.")

        self.df = metadata_df.copy()

        # Normalize ground-truth labels.
        label_col = (
            "true_label"
            if "true_label" in self.df.columns
            else "label" if "label" in self.df.columns
            else None
        )

        if label_col is None:
            raise KeyError(
                "Metadata must contain a 'true_label' or 'label' column."
            )

        def normalize_label(value):
            if pd.isna(value):
                return np.nan

            if isinstance(value, str):
                value = value.strip().upper()
                if value == "REAL":
                    return 0
                if value == "FAKE":
                    return 1

            if value in (0, 1):
                return int(value)

            return np.nan

        self.df["y_true"] = self.df[label_col].map(normalize_label)

        # Support either score column name.
        if "score" not in self.df.columns:
            if "prediction_score" in self.df.columns:
                self.df["score"] = self.df["prediction_score"]
            else:
                raise KeyError(
                    "Metadata must contain 'score' or 'prediction_score'."
                )

        self.df["score"] = pd.to_numeric(
            self.df["score"], errors="coerce"
        )

        # Remove rows with invalid labels or scores, and report how many.
        valid_mask = (
            self.df["y_true"].notna()
            & self.df["score"].notna()
            & np.isfinite(self.df["score"])
            & self.df["score"].between(0.0, 1.0)
        )

        invalid_count = int((~valid_mask).sum())

        if invalid_count:
            print(
                f"[!] Excluding {invalid_count} rows with invalid "
                "labels or prediction scores."
            )

        self.df = self.df.loc[valid_mask].copy()
        self.df["y_true"] = self.df["y_true"].astype(int)

        if self.df.empty:
            raise ValueError(
                "No valid samples remain after validating labels and scores."
            )

        print(f"[i] Valid samples loaded: {len(self.df)}")

    def evaluate_attribute(
        self,
        attribute_col: str,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """Calculate metrics separately for each attribute group."""

        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        if attribute_col not in self.df.columns:
            print(
                f"[!] Attribute '{attribute_col}' not found. Skipping."
            )
            return pd.DataFrame()

        results = []

        # Include missing attribute values so their samples are not omitted.
        grouped = self.df.groupby(attribute_col, dropna=False)

        for attr_value, group in grouped:
            y_true = group["y_true"].to_numpy(dtype=int)
            y_scores = group["score"].to_numpy(dtype=float)

            metrics = compute_all_metrics(
                y_true,
                y_scores,
                threshold=threshold
            )

            metrics[attribute_col] = (
                "MISSING" if pd.isna(attr_value) else attr_value
            )
            metrics["sample_count"] = len(group)
            metrics["real_count"] = int((y_true == 0).sum())
            metrics["fake_count"] = int((y_true == 1).sum())

            results.append(metrics)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)

        leading_cols = [
            attribute_col,
            "sample_count",
            "real_count",
            "fake_count",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc",
        ]

        remaining_cols = [
            col for col in result_df.columns
            if col not in leading_cols
        ]

        return result_df[leading_cols + remaining_cols]

    def generate_full_robustness_report(
        self,
        attribute_cols: Optional[List[str]] = None,
        threshold: float = 0.5,
        save_dir: str = "results"
    ) -> Dict[str, pd.DataFrame]:
        """Generate and save a CSV report for each available attribute."""

        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        os.makedirs(save_dir, exist_ok=True)

        if attribute_cols is None:
            attribute_cols = self.DEFAULT_ATTRIBUTES

        reports = {}

        print("=" * 65)
        print("        FALSEFACET — ROBUSTNESS ANALYSIS REPORT")
        print(f"        Samples: {len(self.df)} | Threshold: {threshold:.2f}")
        print("=" * 65)

        for col in attribute_cols:
            if col not in self.df.columns:
                print(f"[!] Attribute '{col}' not present. Skipping.")
                continue

            report_df = self.evaluate_attribute(
                col,
                threshold=threshold
            )

            if report_df.empty:
                print(f"[!] No report generated for '{col}'.")
                continue

            reports[col] = report_df

            out_path = os.path.join(
                save_dir,
                f"robustness_{col}.csv"
            )
            report_df.to_csv(out_path, index=False)

            display_cols = [
                col, "sample_count", "accuracy",
                "precision", "recall", "f1", "auc"
            ]

            print(f"\n--- Condition Breakdown: {col.upper()} ---")
            print(report_df[display_cols].to_string(index=False))
            print(f"[+] Saved report to: {out_path}")

        print("\n" + "=" * 65)
        print(f"[+] Completed reports: {len(reports)}")
        print("=" * 65)

        return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FalseFacet Robustness Evaluation"
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="CSV containing ground-truth labels, predictions and metadata"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory for saved robustness reports"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for classifying FAKE probabilities"
    )

    args = parser.parse_args()

    if args.metadata_csv:
        if not os.path.isfile(args.metadata_csv):
            raise FileNotFoundError(
                f"Metadata CSV not found: {args.metadata_csv}"
            )

        print(f"[i] Loading metadata from: {args.metadata_csv}")
        metadata = pd.read_csv(args.metadata_csv)

    else:
        print("[i] No CSV supplied. Running synthetic smoke test.")

        rng = np.random.default_rng(42)
        n = 30

        metadata = pd.DataFrame({
            "video_path": [
                f"video_{i:03d}.mp4" for i in range(n)
            ],
            "true_label": [0] * 15 + [1] * 15,
            "score": np.concatenate([
                rng.uniform(0.05, 0.45, 15),
                rng.uniform(0.55, 0.95, 15),
            ]),
            "quality": rng.choice(
                ["high", "medium", "low"], n
            ),
            "lighting": rng.choice(
                ["normal", "low_light", "backlit"], n
            ),
            "compression": rng.choice(
                ["c23_low", "c40_high"], n
            ),
            "generator": (
                ["camera"] * 15
                + rng.choice(
                    ["faceswap", "deepfake_v2", "gan"], 15
                ).tolist()
            ),
        })

    evaluator = RobustnessEvaluator(metadata)
    evaluator.generate_full_robustness_report(
        threshold=args.threshold,
        save_dir=args.save_dir
    )
#python -m evaluate.robustness --metadata_csv results/predictions_metadata.csv