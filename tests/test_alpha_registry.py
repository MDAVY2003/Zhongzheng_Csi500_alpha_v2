from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import alpha_registry as registry  # noqa: E402
import alphas  # noqa: E402
from factor_data import FactorData  # noqa: E402
from run_all_alphas import audit_factor_outputs, run_registered_alphas  # noqa: E402


class AlphaRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2025-01-02", periods=3, freq="D")
        columns = ["AAA", "BBB", "CCC"]
        close = pd.DataFrame(
            [[10.0, 20.0, 30.0], [11.0, 19.0, 31.0], [12.0, 18.0, 32.0]],
            index=dates,
            columns=columns,
        )
        volume = pd.DataFrame(
            [[100.0, 200.0, 300.0], [110.0, 190.0, 320.0], [90.0, 210.0, 330.0]],
            index=dates,
            columns=columns,
        )
        self.data = FactorData(close=close, volume=volume)

    def test_registry_covers_all_101_alphas(self) -> None:
        self.assertEqual(sorted(registry.ALPHA_REGISTRY), list(range(1, 102)))
        self.assertEqual(len(registry.registry_records()), 101)

    def test_key_dependency_metadata(self) -> None:
        alpha001 = registry.get_alpha_spec("alpha001")
        self.assertEqual(alpha001.required_fields, ("close", "returns"))
        self.assertEqual(alpha001.max_lookback, 20)

        alpha056 = registry.get_alpha_spec(56)
        self.assertTrue(alpha056.uses_market_cap)
        self.assertIn("cap", alpha056.required_fields)

        alpha067 = registry.get_alpha_spec("alpha_067")
        self.assertEqual(alpha067.industry_levels, ("sector", "subindustry"))
        self.assertIn("volume", registry.get_alpha_spec(61).required_fields)

        self.assertFalse(registry.get_alpha_spec(61).is_approximate)
        self.assertFalse(registry.get_alpha_spec(58).is_approximate)
        self.assertIn("小数窗口", registry.get_alpha_spec(61).notes)
        self.assertFalse(registry.get_alpha_spec(12).is_approximate)

    def test_existing_functions_are_discovered_without_if_statements(self) -> None:
        implemented = {
            spec.alpha_id
            for spec in registry.select_alpha_specs(implemented_only=True)
        }
        self.assertEqual(implemented, set(alphas.IMPLEMENTED_ALPHA_IDS))
        self.assertEqual(len(implemented), 101)
        self.assertIs(registry.get_alpha_spec(12).function, alphas.alpha012)

    def test_canonical_function_uses_factor_data(self) -> None:
        result = alphas.alpha012(self.data)
        self.assertTrue(result.index.equals(self.data.index))
        self.assertTrue(result.columns.equals(self.data.columns))
        self.assertTrue(result.iloc[0].isna().all())

    def test_runner_continues_after_one_factor_fails(self) -> None:
        def failing_alpha(data: FactorData) -> pd.DataFrame:
            raise RuntimeError("人工构造的失败")

        failed_spec = registry.AlphaSpec(
            alpha_id=1,
            function_name="alpha001",
            required_fields=("close",),
            industry_levels=(),
            uses_market_cap=False,
            max_lookback=20,
            implementation_batch=3,
            status="implemented",
            function=failing_alpha,
        )
        success_spec = registry.get_alpha_spec(12)

        with tempfile.TemporaryDirectory() as directory:
            summary = run_registered_alphas(
                [failed_spec, success_spec],
                data=self.data,
                output_dir=Path(directory),
                overwrite=True,
            )
            statuses = summary.set_index("alpha_id")["run_status"]
            self.assertEqual(statuses.loc[1], "failed")
            self.assertEqual(statuses.loc[12], "success")
            self.assertTrue((Path(directory) / "alpha012.parquet").exists())
            quality = audit_factor_outputs([success_spec], Path(directory))
            self.assertEqual(quality.loc[0, "rows"], 3)
            self.assertEqual(quality.loc[0, "inf_values"], 0)

    def test_factor_data_rejects_misaligned_panels(self) -> None:
        shifted = self.data.close.copy()
        shifted.index = shifted.index + pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "日期索引不一致"):
            FactorData(close=self.data.close, volume=shifted)


if __name__ == "__main__":
    unittest.main()
