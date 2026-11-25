import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factor_store import load_factor_source_metadata, resolve_factor_path  # noqa: E402


class FactorStoreTest(unittest.TestCase):
    def test_three_strict_factors_are_resolved_from_separate_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "alphas_point_in_time"
            exact = root / "alphas_point_in_time_exact_signal_anchor"
            base.mkdir()
            exact.mkdir()
            for name in ("alpha004", "alpha067", "alpha088"):
                (base / f"{name}.parquet").touch()
                expected = exact / f"{name}.parquet"
                expected.touch()
                self.assertEqual(resolve_factor_path(base, name), expected)

    def test_other_factor_remains_in_base_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "alphas_point_in_time"
            base.mkdir()
            expected = base / "alpha005.parquet"
            expected.touch()
            self.assertEqual(resolve_factor_path(base, "alpha_005"), expected)

    def test_missing_strict_file_does_not_fall_back_to_old_approximation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "alphas_point_in_time"
            base.mkdir()
            (base / "alpha004.parquet").touch()
            with self.assertRaisesRegex(FileNotFoundError, "严格信号日锚定"):
                resolve_factor_path(base, "alpha004")

    def test_strict_metadata_overrides_old_approximation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "alphas_point_in_time"
            exact = root / "alphas_point_in_time_exact_signal_anchor"
            base.mkdir()
            exact.mkdir()
            pd.DataFrame(
                [
                    {
                        "factor_name": "alpha004",
                        "calculation_method": "causal_screening_approximation",
                        "is_signal_anchor_exact": False,
                    },
                    {
                        "factor_name": "alpha005",
                        "calculation_method": "signal_date_anchor_exact_verified",
                        "is_signal_anchor_exact": True,
                    },
                ]
            ).to_csv(base / "point_in_time_run_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "factor_name": "alpha004",
                        "calculation_method": "signal_date_anchor_exact_per_signal",
                        "is_signal_anchor_exact": True,
                    }
                ]
            ).to_csv(exact / "exact_signal_anchor_run_summary.csv", index=False)
            metadata = load_factor_source_metadata(base).set_index("factor_name")
            self.assertEqual(
                metadata.at["alpha004", "calculation_method"],
                "signal_date_anchor_exact_per_signal",
            )
            self.assertTrue(metadata.at["alpha004", "is_signal_anchor_exact"])


if __name__ == "__main__":
    unittest.main()
