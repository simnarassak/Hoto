from hoto.simulation import load_sim_inputs, regional_impact_summary, run_baseline_and_scenario
from hoto.simulation_charts import region_impact_map
from hoto.simulation_config import default_config, hazard_choices

def main():
    claims, providers = load_sim_inputs('data')
    economic = run_baseline_and_scenario(claims, providers, config=default_config(hazard_type='Economic stress', shocked_region='Industrial', stress_shock_pct=10.0, stress_to_demand_beta=0.25, capacity_mode='Normal'))
    summary = regional_impact_summary(economic['comparison'])
    industrial = summary.loc[summary['region'].eq('Industrial')].iloc[0]
    assert 2.0 <= industrial['mean_pct_change_demand'] <= 3.0
    other = summary.loc[summary['region'].ne('Industrial'), 'abs_change_claims_demand']
    assert (other.abs() < 1e-09).all()
    all_region = run_baseline_and_scenario(claims, providers, config=default_config(hazard_type='All-region stress', stress_shock_pct=15.0, stress_to_demand_beta=0.25))
    all_summary = regional_impact_summary(all_region['comparison'])
    assert (all_summary['abs_change_claims_demand'] > 0).all()
    tsunami = run_baseline_and_scenario(claims, providers, config=default_config(hazard_type='Tsunami', shocked_region='Coastal', capacity_mode='Very tight', horizon_months=6))
    coastal = tsunami['scenario'].loc[tsunami['scenario']['region'].eq('Coastal')]
    assert coastal['backlog'].max() > 0
    for hazard in hazard_choices():
        result = run_baseline_and_scenario(claims, providers, config=default_config(hazard_type=hazard, capacity_mode='Tight'))
        assert not result['comparison'].empty
        assert region_impact_map(regional_impact_summary(result['comparison'])) is not None
    print('industrial_pct', round(industrial['mean_pct_change_demand'], 3))
    print('coastal_max_backlog', round(float(coastal['backlog'].max()), 3))
    print('SIM_OK')
if __name__ == '__main__':
    main()
