"""Configurable assumptions for the Hoto regional simulation."""

from dataclasses import asdict, dataclass, field, replace

from hoto.config import REGIONS, WORLD


# Illustrative hazard profiles (synthetic design choices, not empirical estimates).
HAZARD_PROFILES = {
    "Economic stress": {
        "description": "Regional socioeconomic pressure shock.",
        "default_regions": ("Industrial",),
        "default_stress_shock_pct": 10.0,
        "demand_beta_multiplier": 1.0,
        "capacity_loss_pct": 0.0,
        "mean_amount_uplift_pct": 0.0,
        "allows_region_pick": True,
    },
    "Flood": {
        "description": "Coastal/river flooding raises claims and reduces local capacity.",
        "default_regions": ("Coastal", "Rural"),
        "default_stress_shock_pct": 25.0,
        "demand_beta_multiplier": 1.4,
        "capacity_loss_pct": 15.0,
        "mean_amount_uplift_pct": 8.0,
        "allows_region_pick": True,
    },
    "Earthquake": {
        "description": "Sudden structural/injury claims with temporary capacity loss.",
        "default_regions": ("Capital", "Metro", "Industrial"),
        "default_stress_shock_pct": 35.0,
        "demand_beta_multiplier": 1.6,
        "capacity_loss_pct": 25.0,
        "mean_amount_uplift_pct": 12.0,
        "allows_region_pick": True,
    },
    "Pandemic": {
        "description": "Nationwide health and disruption pressure across all regions.",
        "default_regions": ("ALL",),
        "default_stress_shock_pct": 20.0,
        "demand_beta_multiplier": 1.2,
        "capacity_loss_pct": 10.0,
        "mean_amount_uplift_pct": 5.0,
        "allows_region_pick": False,
    },
    "Storm": {
        "description": "Severe storm damage focused on exposed regions.",
        "default_regions": ("Coastal", "Rural"),
        "default_stress_shock_pct": 22.0,
        "demand_beta_multiplier": 1.3,
        "capacity_loss_pct": 12.0,
        "mean_amount_uplift_pct": 7.0,
        "allows_region_pick": True,
    },
    "Tsunami": {
        "description": "Coastal inundation with sharp demand spike and capacity loss.",
        "default_regions": ("Coastal",),
        "default_stress_shock_pct": 45.0,
        "demand_beta_multiplier": 1.8,
        "capacity_loss_pct": 35.0,
        "mean_amount_uplift_pct": 15.0,
        "allows_region_pick": True,
    },
    "All-region stress": {
        "description": "Uniform socioeconomic stress applied to every region.",
        "default_regions": ("ALL",),
        "default_stress_shock_pct": 15.0,
        "demand_beta_multiplier": 1.0,
        "capacity_loss_pct": 0.0,
        "mean_amount_uplift_pct": 0.0,
        "allows_region_pick": False,
    },
}

CAPACITY_MODES = {
    "Normal": {
        "headroom": 1.08,
        "description": "About 8% spare monthly capacity in baseline.",
    },
    "Tight": {
        "headroom": 1.02,
        "description": "About 2% spare capacity — backlog appears more easily.",
    },
    "Very tight": {
        "headroom": 0.98,
        "description": "Capacity below baseline demand — backlog demos clearly.",
    },
}

REGION_CHOICES = tuple(REGIONS.keys()) + ("All profile regions",)


@dataclass
class SimulationConfig:
    """Illustrative synthetic assumptions — not real-world forecasts."""

    horizon_months: int = 6
    baseline_stress_index: float = 100.0
    stress_to_demand_beta: float = 0.25
    hazard_type: str = "Economic stress"
    shocked_region: str = "Industrial"
    stress_shock_pct: float = 10.0
    prevention_budget_pct: float = 0.0
    prevention_beta_dampener: float = 0.5
    capacity_mode: str = "Normal"
    capacity_scale: float = 0.02
    backlog_clearance_share: float = 1.0
    random_seed: int = field(default_factory=lambda: WORLD["random_seed"])
    regions: tuple = field(default_factory=lambda: tuple(REGIONS.keys()))

    def hazard_profile(self):
        if self.hazard_type not in HAZARD_PROFILES:
            raise ValueError(f"Unknown hazard_type: {self.hazard_type}")
        return HAZARD_PROFILES[self.hazard_type]

    def capacity_profile(self):
        if self.capacity_mode not in CAPACITY_MODES:
            raise ValueError(f"Unknown capacity_mode: {self.capacity_mode}")
        return CAPACITY_MODES[self.capacity_mode]

    def affected_regions(self):
        profile = self.hazard_profile()
        defaults = profile["default_regions"]
        if defaults == ("ALL",) or self.hazard_type in {
            "Pandemic",
            "All-region stress",
        }:
            return tuple(self.regions)

        if self.shocked_region == "All profile regions":
            return tuple(r for r in defaults if r in self.regions)

        # User-picked single region (still valid for most hazards).
        if self.shocked_region in self.regions:
            return (self.shocked_region,)
        return tuple(r for r in defaults if r in self.regions)

    def effective_beta(self):
        damp = 1.0 - (
            self.prevention_beta_dampener
            * max(0.0, min(self.prevention_budget_pct, 100.0))
            / 100.0
        )
        return (
            self.stress_to_demand_beta
            * damp
            * self.hazard_profile()["demand_beta_multiplier"]
        )

    def capacity_headroom(self):
        return float(self.capacity_profile()["headroom"])

    def assumptions_text(self):
        profile = self.hazard_profile()
        cap = self.capacity_profile()
        affected = ", ".join(self.affected_regions())
        return (
            f"Hazard: {self.hazard_type}. {profile['description']} "
            f"Affected regions: {affected}. "
            f"Stress shock={self.stress_shock_pct:.1f}%, beta={self.stress_to_demand_beta:.2f} "
            f"(effective beta={self.effective_beta():.3f}). "
            f"Capacity mode: {self.capacity_mode} ({cap['description']}). "
            f"Hazard capacity loss={profile['capacity_loss_pct']:.0f}%, "
            f"mean claim uplift={profile['mean_amount_uplift_pct']:.0f}%. "
            "Regional claims budgets are population shares of the island-wide "
            "baseline demand-cost envelope. "
            "Outputs are synthetic projections under explicit rules, not forecasts. "
            "ML risk models are separate and do not drive demand."
        )

    def to_dict(self):
        payload = asdict(self)
        payload["affected_regions"] = list(self.affected_regions())
        payload["effective_beta"] = self.effective_beta()
        return payload


def default_config(**overrides):
    cfg = SimulationConfig()
    if "hazard_type" in overrides and "stress_shock_pct" not in overrides:
        hazard = overrides["hazard_type"]
        if hazard in HAZARD_PROFILES:
            overrides = {
                **overrides,
                "stress_shock_pct": HAZARD_PROFILES[hazard][
                    "default_stress_shock_pct"
                ],
            }
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def hazard_choices():
    return list(HAZARD_PROFILES.keys())


def capacity_mode_choices():
    return list(CAPACITY_MODES.keys())
