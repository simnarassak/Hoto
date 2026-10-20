import unittest
from pathlib import Path

from hoto_rural import config
from hoto_rural.simulate import run_monthly_simulation, summer_month_summary
from hoto_rural.world import build_water_body, load_world


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rural"


def _world():
    world = load_world(DATA)
    world["water_bodies"] = build_water_body()
    return world


class RuralMonthlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = _world()
        cls.baseline = run_monthly_simulation(
            cls.world,
            scenario="baseline",
            n_months=24,
            seed=config.RANDOM_SEED,
            history=True,
        )
        cls.drought = run_monthly_simulation(
            cls.world,
            scenario="drought",
            n_months=24,
            seed=config.RANDOM_SEED + 3,
            history=True,
        )

    def test_water_balance_near_zero(self):
        residual = self.baseline["water_balance_residual"].abs().max()
        self.assertLess(residual, 1e-4)

    def test_food_balance_near_zero(self):
        residual = self.baseline["food_balance_residual"].abs().max()
        self.assertLess(residual, 1e-4)

    def test_storage_within_capacity(self):
        capacity = float(self.world["water_bodies"].iloc[0]["capacity_m3"])
        self.assertTrue((self.baseline["storage_m3"] >= -1e-6).all())
        self.assertTrue((self.baseline["storage_m3"] <= capacity + 1e-6).all())

    def test_food_import_export_nonnegative(self):
        self.assertTrue((self.baseline["food_import_t"] >= -1e-9).all())
        self.assertTrue((self.baseline["food_export_t"] >= -1e-9).all())

    def test_labour_match_identity(self):
        for role in ("permanent", "seasonal", "immigrant"):
            demand = self.baseline[f"labour_demand_{role}"]
            employed = self.baseline[f"labour_employed_{role}"]
            unmet = self.baseline[f"labour_unmet_{role}"]
            idle = self.baseline[f"labour_idle_{role}"]
            supply = self.baseline[f"labour_supply_{role}"]
            self.assertTrue(((employed + unmet) - demand).abs().max() < 1e-6)
            self.assertTrue(((employed + idle) - supply).abs().max() < 1e-6)

    def test_summer_months_marked(self):
        summer = self.baseline.loc[self.baseline["is_summer"] == 1]
        self.assertGreater(len(summer), 0)
        self.assertTrue(summer["calendar_month"].isin(["Dec", "Jan", "Feb"]).all())

    def test_summer_has_higher_seasonal_labour_than_other(self):
        summary = summer_month_summary(self.baseline)
        self.assertGreater(summary["summer_labour_demand"], summary["other_labour_demand"])

    def test_summer_has_higher_irrigation_than_other(self):
        summary = summer_month_summary(self.baseline)
        self.assertGreater(summary["summer_irrig_m3"], summary["other_irrig_m3"])

    def test_drought_more_stressed_than_baseline(self):
        self.assertGreater(
            float(self.drought["water_stress_index"].mean()),
            float(self.baseline["water_stress_index"].mean()),
        )

    def test_some_months_import_or_export(self):
        imports = int((self.baseline["food_import_t"] > 1e-6).sum())
        exports = int((self.baseline["food_export_t"] > 1e-6).sum())
        self.assertGreater(imports + exports, 0)

    def test_drought_imports_more_than_baseline(self):
        self.assertGreater(
            float(self.drought["food_import_t"].sum()),
            float(self.baseline["food_import_t"].sum()),
        )



if __name__ == "__main__":
    unittest.main()
