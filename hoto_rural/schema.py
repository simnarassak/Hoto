REGION_ID = "Rural"
CURRENCY = "HotoDollar"
TIME_UNIT = "month"
VOLUME_UNIT_M3 = "m3"
MASS_UNIT_KG = "kg"
AREA_UNIT_HA = "ha"

PERSON_FIELDS = (
    "person_id",
    "region_id",
    "population_group",
    "age_band",
    "immigrant_status",
    "employment_status",
    "worker_role",
    "household_id",
)

HOUSEHOLD_FIELDS = (
    "household_id",
    "region_id",
    "n_members",
    "monthly_income",
    "food_spend_share",
)

FARM_FIELDS = (
    "farm_id",
    "region_id",
    "farm_type",
    "area_ha",
    "water_entitlement_m3_month",
    "operator_household_id",
)

HERD_FIELDS = (
    "herd_id",
    "farm_id",
    "species",
    "n_animals",
    "feed_need_kg_month_per_animal",
    "manure_kg_month_per_animal",
    "product",
)

CROP_FIELDS = (
    "crop_id",
    "farm_id",
    "archetype",
    "commodity",
    "area_ha",
    "yield_t_ha_year",
    "water_m3_ha_month",
)

WATER_BODY_FIELDS = (
    "water_body_id",
    "region_id",
    "kind",
    "capacity_m3",
    "initial_storage_m3",
    "environmental_flow_m3_month",
)

MONTHLY_OUTPUT_FIELDS = (
    "region_id",
    "month",
    "rainfall_mm",
    "inflow_m3",
    "storage_m3",
    "irrigation_use_m3",
    "household_water_m3",
    "livestock_water_m3",
    "unmet_water_m3",
    "water_stress_index",
    "manure_kg",
    "nutrient_load_kg",
    "river_nutrient_mg_l",
    "nutrient_exceedance",
    "crop_supply_t",
    "farm_revenue",
    "farm_opex",
    "farm_profit",
    "labour_demand_permanent",
    "labour_demand_seasonal",
    "labour_demand_immigrant",
    "food_price_index",
    "subsidy_need",
)

CONTRACT_EXPORT_FIELDS = (
    "region_id",
    "month",
    "crop_supply",
    "food_price_index",
    "labour_demand_seasonal",
    "immigrant_labour_demand",
    "water_stress_index",
    "nutrient_exceedance",
    "farm_profit",
    "subsidy_need",
)

FARM_TYPES = (
    "pastoral_dairy",
    "pastoral_sheep",
    "orchard",
    "horticulture",
    "paddy_forage",
)

CROP_ARCHETYPES = {
    "pastoral_forage": {"commodities": ("forage",), "water_m3_ha_month": 80.0, "yield_t_ha_year": 8.0},
    "orchard": {
        "commodities": ("avocado", "kiwi", "apple", "grapes", "berries"),
        "water_m3_ha_month": 120.0,
        "yield_t_ha_year": 12.0,
    },
    "horticulture": {
        "commodities": ("carrot", "cucumber"),
        "water_m3_ha_month": 100.0,
        "yield_t_ha_year": 25.0,
    },
    "paddy_forage": {"commodities": ("paddy_feed",), "water_m3_ha_month": 200.0, "yield_t_ha_year": 6.0},
}

SPECIES = {
    "cow": {
        "feed_need_kg_month": 450.0,
        "manure_kg_month": 900.0,
        "water_l_day": 70.0,
        "product": "milk",
        "revenue_per_animal_month": 280.0,
    },
    "sheep": {
        "feed_need_kg_month": 45.0,
        "manure_kg_month": 40.0,
        "water_l_day": 8.0,
        "product": "meat_wool",
        "revenue_per_animal_month": 35.0,
    },
}

WORKER_ROLES = (
    "permanent_farm",
    "seasonal_farm",
    "immigrant_farm",
    "non_farm",
    "not_in_labour_force",
)

SCENARIO_NAMES = (
    "baseline",
    "drought",
    "efficiency",
    "allocation",
    "combined",
)
