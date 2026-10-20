from pathlib import Path

import numpy as np
import pandas as pd

from hoto_rural import config
from hoto_rural.schema import CROP_ARCHETYPES, FARM_TYPES, REGION_ID, SPECIES


def _categorical(n, mapping, rng):
    labels = list(mapping.keys())
    probs = np.array(list(mapping.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(labels, size=n, p=probs)


def build_households(n_people, rng):
    sizes = rng.choice([1, 2, 3, 4, 5], size=max(1, n_people // 3), p=[0.2, 0.3, 0.25, 0.15, 0.1])
    rows = []
    remaining = n_people
    hid = 1
    for size in sizes:
        if remaining <= 0:
            break
        members = min(int(size), remaining)
        rows.append(
            {
                "household_id": f"RH{hid:05d}",
                "region_id": REGION_ID,
                "n_members": members,
                "monthly_income": float(rng.uniform(1800, 6500)),
                "food_spend_share": float(rng.uniform(0.18, 0.35)),
            }
        )
        remaining -= members
        hid += 1
    while remaining > 0:
        members = min(2, remaining)
        rows.append(
            {
                "household_id": f"RH{hid:05d}",
                "region_id": REGION_ID,
                "n_members": members,
                "monthly_income": float(rng.uniform(1800, 6500)),
                "food_spend_share": float(rng.uniform(0.18, 0.35)),
            }
        )
        remaining -= members
        hid += 1
    return pd.DataFrame(rows)


def build_people(households, rng):
    rows = []
    pid = 1
    for _, hh in households.iterrows():
        for _ in range(int(hh["n_members"])):
            group = _categorical(1, config.POPULATION_GROUPS, rng)[0]
            age = _categorical(1, config.AGE_BANDS, rng)[0]
            emp = _categorical(1, config.EMPLOYMENT_DISTRIBUTION, rng)[0]
            if age == "0-17" or emp in {"Student", "Retired"}:
                role = "not_in_labour_force"
            elif emp == "Unemployed":
                role = "non_farm"
            else:
                role = rng.choice(
                    ["permanent_farm", "seasonal_farm", "immigrant_farm", "non_farm"],
                    p=[0.25, 0.2, 0.15, 0.4],
                )
            immigrant_status = "Citizen"
            if group == "Immigrant":
                immigrant_status = _categorical(1, config.IMMIGRANT_STATUS_DISTRIBUTION, rng)[0]
                if role == "immigrant_farm" and immigrant_status == "Citizen":
                    immigrant_status = "Work Visa"
            rows.append(
                {
                    "person_id": f"RP{pid:06d}",
                    "region_id": REGION_ID,
                    "population_group": group,
                    "age_band": age,
                    "immigrant_status": immigrant_status,
                    "employment_status": emp,
                    "worker_role": role,
                    "household_id": hh["household_id"],
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def build_farms(households, rng):
    n = config.N_FARMS
    farm_types = rng.choice(
        list(FARM_TYPES),
        size=n,
        p=[0.22, 0.18, 0.28, 0.18, 0.14],
    )
    operators = rng.choice(households["household_id"].to_numpy(), size=n, replace=True)
    rows = []
    for i, (ftype, op) in enumerate(zip(farm_types, operators), start=1):
        area = float(rng.uniform(8, 120))
        entitlement = area * float(rng.uniform(70, 160))
        rows.append(
            {
                "farm_id": f"RF{i:04d}",
                "region_id": REGION_ID,
                "farm_type": ftype,
                "area_ha": area,
                "water_entitlement_m3_month": entitlement,
                "operator_household_id": op,
            }
        )
    return pd.DataFrame(rows)


def build_herds(farms, rng):
    rows = []
    hid = 1
    for _, farm in farms.iterrows():
        if farm["farm_type"] == "pastoral_dairy":
            species = "cow"
            n_animals = int(rng.integers(40, 220))
        elif farm["farm_type"] == "pastoral_sheep":
            species = "sheep"
            n_animals = int(rng.integers(200, 1200))
        else:
            continue
        spec = SPECIES[species]
        rows.append(
            {
                "herd_id": f"RHRD{hid:04d}",
                "farm_id": farm["farm_id"],
                "species": species,
                "n_animals": n_animals,
                "feed_need_kg_month_per_animal": spec["feed_need_kg_month"],
                "manure_kg_month_per_animal": spec["manure_kg_month"],
                "product": spec["product"],
            }
        )
        hid += 1
    return pd.DataFrame(rows)


def build_crops(farms, rng):
    rows = []
    cid = 1
    for _, farm in farms.iterrows():
        ftype = farm["farm_type"]
        if ftype in {"pastoral_dairy", "pastoral_sheep"}:
            archetype = "pastoral_forage"
            commodity = "forage"
            area = farm["area_ha"] * 0.85
        elif ftype == "orchard":
            archetype = "orchard"
            commodity = rng.choice(CROP_ARCHETYPES["orchard"]["commodities"])
            area = farm["area_ha"]
        elif ftype == "horticulture":
            archetype = "horticulture"
            commodity = rng.choice(CROP_ARCHETYPES["horticulture"]["commodities"])
            area = farm["area_ha"]
        else:
            archetype = "paddy_forage"
            commodity = "paddy_feed"
            area = farm["area_ha"]
        meta = CROP_ARCHETYPES[archetype]
        rows.append(
            {
                "crop_id": f"RC{cid:04d}",
                "farm_id": farm["farm_id"],
                "archetype": archetype,
                "commodity": commodity,
                "area_ha": float(area),
                "yield_t_ha_year": float(meta["yield_t_ha_year"] * rng.uniform(0.85, 1.15)),
                "water_m3_ha_month": float(meta["water_m3_ha_month"]),
            }
        )
        cid += 1
    return pd.DataFrame(rows)


def build_water_body():
    return pd.DataFrame(
        [
            {
                "water_body_id": "RW001",
                "region_id": REGION_ID,
                "kind": "reservoir",
                "capacity_m3": config.BASE_STORAGE_M3,
                "initial_storage_m3": config.BASE_STORAGE_M3 * 0.62,
                "environmental_flow_m3_month": 280_000.0,
            }
        ]
    )


def generate_rural_world(seed=None, n_people=None):
    seed = config.RANDOM_SEED if seed is None else seed
    n_people = config.RURAL_POPULATION if n_people is None else n_people
    rng = np.random.default_rng(seed)
    households = build_households(n_people, rng)
    people = build_people(households, rng)
    farms = build_farms(households, rng)
    herds = build_herds(farms, rng)
    crops = build_crops(farms, rng)
    water = build_water_body()
    return {
        "people": people,
        "households": households,
        "farms": farms,
        "herds": herds,
        "crops": crops,
        "water_bodies": water,
        "meta": {
            "region_id": REGION_ID,
            "population": int(len(people)),
            "n_farms": int(len(farms)),
            "seed": int(seed),
        },
    }


def save_world(world, data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    for key in ("people", "households", "farms", "herds", "crops", "water_bodies"):
        world[key].to_csv(data_dir / f"{key}.csv", index=False)
    pd.DataFrame([world["meta"]]).to_csv(data_dir / "meta.csv", index=False)
    return data_dir


def load_world(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    return {
        "people": pd.read_csv(data_dir / "people.csv"),
        "households": pd.read_csv(data_dir / "households.csv"),
        "farms": pd.read_csv(data_dir / "farms.csv"),
        "herds": pd.read_csv(data_dir / "herds.csv"),
        "crops": pd.read_csv(data_dir / "crops.csv"),
        "water_bodies": pd.read_csv(data_dir / "water_bodies.csv"),
        "meta": pd.read_csv(data_dir / "meta.csv").iloc[0].to_dict(),
    }
