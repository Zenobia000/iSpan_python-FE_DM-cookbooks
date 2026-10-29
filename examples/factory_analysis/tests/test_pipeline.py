import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from examples.factory_analysis.pipeline import (
    add_proxies, associations, business_scenarios, make_demo, run, wilson_interval,
)


class PipelineTests(unittest.TestCase):
    def test_rate_intervals_stay_within_probability_bounds(self):
        for positive, n in [(0, 15), (15, 15), (3, 15)]:
            low, high = wilson_interval(positive, n)
            self.assertGreaterEqual(low, 0)
            self.assertLessEqual(high, 1)
            self.assertLessEqual(low, positive / n)
            self.assertGreaterEqual(high, positive / n)

    def test_future_values_do_not_change_past_features(self):
        spec = [{"name": "std", "source": "current", "operation": "rolling_std",
                 "window": 3, "rationale": "負載波動", "available_at_decision": True},
                {"name": "change", "source": "current", "operation": "pct_change",
                 "rationale": "負載變化", "available_at_decision": True}]
        original = pd.DataFrame({"current": [1., 2., 3., 4., 5., 6.]})
        modified = original.copy()
        modified.loc[4:, "current"] = [1000., 2000.]
        before = add_proxies(original, spec, "yield", "timestamp")
        after = add_proxies(modified, spec, "yield", "timestamp")
        pd.testing.assert_frame_equal(before.iloc[:4], after.iloc[:4])

    def test_target_cannot_be_proxy_source(self):
        with self.assertRaises(ValueError):
            add_proxies(pd.DataFrame({"yield": [0.9, 0.8]}),
                        [{"name": "leak", "source": "yield"}], "yield", "timestamp")

    def test_effect_and_multiple_comparisons(self):
        frame = pd.DataFrame({"target": np.arange(40), "signal": np.arange(40),
                              "constant": np.ones(40)})
        result = associations(frame, "target", ["signal", "constant"])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["spearman_rho"], 1)
        self.assertGreaterEqual(result[0]["q_value_bh"], result[0]["p_value"])

    def test_cost_period_and_zero_investment(self):
        costs = {"monthly_baseline_loss": 100, "monthly_action_cost": 10,
                 "initial_cost": 20, "months": 3, "assumed_reductions": [0.5],
                 "currency": "TWD"}
        scenario = business_scenarios(costs)["scenarios"][0]
        self.assertEqual(scenario["avoided_loss"], 150)
        self.assertEqual(scenario["investment"], 50)
        self.assertEqual(scenario["net_benefit"], 100)
        self.assertEqual(scenario["roi"], 2)
        costs.update(monthly_action_cost=0, initial_cost=0)
        self.assertIsNone(business_scenarios(costs)["scenarios"][0]["roi"])
        costs["assumed_reductions"] = [1.1]
        with self.assertRaises(ValueError):
            business_scenarios(costs)

    def test_end_to_end_and_holdout_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            data_path, config_path = make_demo(first)
            run(data_path, config_path, first)
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertLess(manifest["split"]["train_end"], manifest["split"]["holdout_start"])
            # Held-out outcomes must not affect discovery or fitted boundaries.
            frame = pd.read_csv(data_path)
            frame.loc[280:, "yield_rate"] = 0.1
            changed = second / "changed.csv"
            frame.to_csv(changed, index=False)
            run(changed, config_path, second)
            self.assertEqual((first / "eda.json").read_text(), (second / "eda.json").read_text())
            r1 = json.loads((first / "rules.json").read_text())
            r2 = json.loads((second / "rules.json").read_text())
            self.assertEqual([r["conditions"] for r in r1["rules"]],
                             [r["conditions"] for r in r2["rules"]])
            self.assertEqual([r["train"] for r in r1["rules"]],
                             [r["train"] for r in r2["rules"]])
            self.assertNotEqual(r1["holdout_event_rate"], r2["holdout_event_rate"])
            self.assertEqual(sum(r["holdout"]["n"] for r in r1["rules"]), r1["holdout_n"])
            self.assertEqual(sum(r1["holdout_confusion"].values()), r1["holdout_n"])
            for name in ["proxies.json", "eda.json", "rules.json", "impact.json", "manifest.json"]:
                json.loads((first / name).read_text())
            self.assertTrue((first / "eda-1.png").is_file())
            self.assertIn("假設情境", (first / "report.md").read_text())

    def test_cli_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "keep.md"
            marker.write_text("original")
            result = subprocess.run([sys.executable, "-m", "examples.factory_analysis", "demo", "--out", temp],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(marker.read_text(), "original")

    def test_duplicate_timestamps_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            data, config = make_demo(temp)
            frame = pd.read_csv(data)
            frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
            frame.to_csv(data, index=False)
            with self.assertRaises(ValueError):
                run(data, config, temp)


if __name__ == "__main__":
    unittest.main()
