from hoto.simulation import load_sim_inputs, regional_impact_summary, run_baseline_and_scenario
from hoto.simulation_config import default_config

def show(hazard, region):
    claims, providers = load_sim_inputs('data')
    result = run_baseline_and_scenario(claims, providers, config=default_config(hazard_type=hazard, shocked_region=region, stress_shock_pct=10, stress_to_demand_beta=0.25, capacity_mode='Normal'))
    summary = regional_impact_summary(result['comparison'])
    print('===', hazard, region, '===')
    print(summary[['region', 'mean_pct_change_demand', 'abs_change_claims_demand', 'abs_change_expenditure']].to_string(index=False))
    print('month1 scenario')
    print(result['scenario'].loc[result['scenario']['month'].eq(1), ['region', 'stress_index', 'claims_demand', 'claims_served', 'claims_expenditure']].to_string(index=False))
    print('month1 baseline')
    print(result['baseline'].loc[result['baseline']['month'].eq(1), ['region', 'stress_index', 'claims_demand', 'claims_served', 'claims_expenditure']].to_string(index=False))
    print('comparison pct demand sample')
    print(result['comparison'].loc[result['comparison']['month'].eq(1), ['region', 'baseline_claims_demand', 'scenario_claims_demand', 'pct_change_claims_demand', 'abs_change_claims_expenditure']].to_string(index=False))
if __name__ == '__main__':
    show('Economic stress', 'Industrial')
    show('All-region stress', 'Industrial')
    show('Tsunami', 'Coastal')
