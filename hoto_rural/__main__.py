from hoto_rural.monte_carlo import build_forecast_bundle
from hoto_rural.world import generate_rural_world, save_world


def main():
    world = generate_rural_world()
    path = save_world(world)
    bundle = build_forecast_bundle(world, scenario="drought", n_paths=40)
    bundle["history"].to_csv(path / "history_baseline.csv", index=False)
    bundle["exports"].to_csv(path / "contract_exports_drought.csv", index=False)
    print(
        f"Rural world ready: people={world['meta']['population']}, "
        f"farms={world['meta']['n_farms']}, data={path}"
    )


if __name__ == "__main__":
    main()
