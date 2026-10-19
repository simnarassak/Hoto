from pathlib import Path
import numpy as np
import pandas as pd
from hoto.config import REGIONS, WORLD
from hoto.simulation_config import default_config

def regional_populations(config=None):
    config = default_config() if config is None else config
    shares = pd.Series({region: float(REGIONS.get(region, 0.0)) for region in config.regions}, dtype=float)
    share_total = float(shares.sum())
    if share_total <= 0:
        equal = 1.0 / max(len(config.regions), 1)
        shares = pd.Series(equal, index=list(config.regions), dtype=float)
        share_total = float(shares.sum())
    population = WORLD['population'] * shares / share_total
    return population

def population_budget_shares(config=None):
    population = regional_populations(config)
    total = float(population.sum())
    if total <= 0:
        n = max(len(population), 1)
        return pd.Series(1.0 / n, index=population.index, dtype=float)
    return population / total

def build_regional_baselines(claims, providers, config=None):
    config = default_config() if config is None else config
    claims = claims.copy()
    claims['submission_date'] = pd.to_datetime(claims['submission_date'])
    region_col = 'claimant_region' if 'claimant_region' in claims.columns else 'region'
    monthly = claims.groupby([region_col, claims['submission_date'].dt.to_period('M')]).size().rename('claims').reset_index()
    baseline_demand = monthly.groupby(region_col)['claims'].mean().reindex(config.regions).fillna(0.0)
    capacity_raw = providers.groupby('region')['service_capacity'].sum().reindex(config.regions).fillna(0.0)
    scaled_capacity = capacity_raw * config.capacity_scale
    headroom = config.capacity_headroom()
    monthly_capacity = baseline_demand.copy().astype(float)
    for region in config.regions:
        demand = float(baseline_demand.loc[region])
        scaled = float(scaled_capacity.loc[region])
        if demand <= 0:
            monthly_capacity.loc[region] = max(scaled, 1.0)
            continue
        headroom_capacity = demand * headroom
        if headroom >= 1.0:
            monthly_capacity.loc[region] = max(headroom_capacity, min(scaled, demand * 1.25) if scaled > 0 else headroom_capacity)
        else:
            monthly_capacity.loc[region] = headroom_capacity
    approved = claims.loc[claims['claim_status'].eq('Approved')]
    if approved.empty or 'approved_amount' not in approved.columns:
        mean_amount = pd.Series(claims['claimed_amount'].mean(), index=config.regions)
    else:
        mean_amount = approved.groupby(region_col)['approved_amount'].mean().reindex(config.regions)
        mean_amount = mean_amount.fillna(claims['claimed_amount'].mean())
    population = regional_populations(config).reindex(config.regions).fillna(0.0)
    baseline_demand_cost = baseline_demand.astype(float) * mean_amount.astype(float)
    national_envelope = float(baseline_demand_cost.sum())
    budget_shares = population_budget_shares(config).reindex(config.regions).fillna(0.0)
    monthly_budget = national_envelope * budget_shares
    return pd.DataFrame({'region': list(config.regions), 'population': population.to_numpy(dtype=float), 'budget_share': budget_shares.to_numpy(dtype=float), 'baseline_monthly_demand': baseline_demand.to_numpy(dtype=float), 'monthly_capacity': monthly_capacity.to_numpy(dtype=float), 'mean_claim_amount': mean_amount.to_numpy(dtype=float), 'monthly_budget': monthly_budget.to_numpy(dtype=float), 'baseline_monthly_demand_cost': baseline_demand_cost.to_numpy(dtype=float)}).reset_index(drop=True)

def _is_affected(region, config, apply_scenario):
    return apply_scenario and region in config.affected_regions()

def _stress_for_region(region, month_index, config, apply_scenario):
    if _is_affected(region, config, apply_scenario) and month_index < config.horizon_months:
        return config.baseline_stress_index * (1.0 + config.stress_shock_pct / 100.0)
    return config.baseline_stress_index

def _demand_from_stress(baseline_demand, stress_index, config):
    beta = config.effective_beta()
    return baseline_demand * (1.0 + beta * (stress_index - config.baseline_stress_index) / 100.0)

def _scenario_capacity(base_capacity, region, config, apply_scenario):
    capacity = float(base_capacity)
    if not _is_affected(region, config, apply_scenario):
        return capacity
    loss = config.hazard_profile()['capacity_loss_pct'] / 100.0
    return capacity * (1.0 - loss)

def _scenario_mean_amount(base_amount, region, config, apply_scenario):
    amount = float(base_amount)
    if not _is_affected(region, config, apply_scenario):
        return amount
    uplift = config.hazard_profile()['mean_amount_uplift_pct'] / 100.0
    return amount * (1.0 + uplift)

def run_simulation(claims, providers, config=None, apply_scenario=False, baselines=None):
    config = default_config() if config is None else config
    if baselines is None:
        baselines = build_regional_baselines(claims, providers, config)
    rows = []
    backlog = {region: 0.0 for region in config.regions}
    affected = set(config.affected_regions()) if apply_scenario else set()
    for month in range(config.horizon_months):
        for _, base in baselines.iterrows():
            region = base['region']
            stress = _stress_for_region(region, month, config, apply_scenario)
            new_demand = _demand_from_stress(float(base['baseline_monthly_demand']), stress, config)
            incoming = new_demand + backlog[region] * config.backlog_clearance_share
            capacity = _scenario_capacity(base['monthly_capacity'], region, config, apply_scenario)
            served = min(incoming, capacity)
            unmet = max(0.0, incoming - served)
            backlog[region] = unmet
            utilisation = served / capacity if capacity > 0 else np.nan
            mean_amount = _scenario_mean_amount(base['mean_claim_amount'], region, config, apply_scenario)
            expenditure = served * mean_amount
            demand_cost = new_demand * mean_amount
            regional_budget = float(base['monthly_budget'])
            budget_gap = demand_cost - regional_budget
            budget_utilisation = demand_cost / regional_budget if regional_budget > 0 else np.nan
            prevention_spend = expenditure * max(0.0, config.prevention_budget_pct) / 100.0 if region in affected else 0.0
            rows.append({'month': month + 1, 'region': region, 'hazard_type': config.hazard_type if apply_scenario else 'Baseline', 'affected': int(region in affected), 'stress_index': stress, 'claims_demand': new_demand, 'incoming_workload': incoming, 'claims_served': served, 'backlog': backlog[region], 'provider_capacity': capacity, 'provider_utilisation': utilisation, 'mean_claim_amount': mean_amount, 'claims_expenditure': expenditure, 'demand_cost': demand_cost, 'regional_budget': regional_budget, 'budget_gap': budget_gap, 'budget_utilisation': budget_utilisation, 'prevention_budget': prevention_spend, 'unmet_demand': unmet, 'path': 'scenario' if apply_scenario else 'baseline'})
    result = pd.DataFrame(rows)
    result.attrs['config'] = config.to_dict()
    result.attrs['assumptions'] = config.assumptions_text()
    return result

def compare_scenarios(baseline, scenario):
    keys = ['month', 'region']
    value_cols = ['stress_index', 'claims_demand', 'claims_served', 'backlog', 'provider_utilisation', 'claims_expenditure', 'demand_cost', 'regional_budget', 'budget_gap', 'budget_utilisation', 'prevention_budget', 'unmet_demand', 'provider_capacity', 'mean_claim_amount']
    left = baseline[keys + value_cols].rename(columns={c: f'baseline_{c}' for c in value_cols})
    right = scenario[keys + value_cols].rename(columns={c: f'scenario_{c}' for c in value_cols})
    merged = left.merge(right, on=keys, how='inner')
    for col in value_cols:
        b = merged[f'baseline_{col}']
        s = merged[f'scenario_{col}']
        merged[f'abs_change_{col}'] = s - b
        merged[f'pct_change_{col}'] = np.where(b == 0, np.where(s == 0, 0.0, np.nan), (s - b) / b * 100.0)
    return merged

def run_baseline_and_scenario(claims, providers, config=None):
    config = default_config() if config is None else config
    baselines = build_regional_baselines(claims, providers, config)
    baseline = run_simulation(claims, providers, config=config, apply_scenario=False, baselines=baselines)
    scenario = run_simulation(claims, providers, config=config, apply_scenario=True, baselines=baselines)
    comparison = compare_scenarios(baseline, scenario)
    return {'config': config, 'baselines': baselines, 'baseline': baseline, 'scenario': scenario, 'comparison': comparison, 'assumptions': config.assumptions_text()}

def regional_impact_summary(comparison):
    summary = comparison.groupby('region', as_index=False).agg(abs_change_claims_demand=('abs_change_claims_demand', 'sum'), abs_change_claims_served=('abs_change_claims_served', 'sum'), abs_change_backlog=('abs_change_backlog', 'mean'), abs_change_served_expenditure=('abs_change_claims_expenditure', 'sum'), abs_change_demand_cost=('abs_change_demand_cost', 'sum'), abs_change_budget_gap=('abs_change_budget_gap', 'sum'), baseline_demand_cost=('baseline_demand_cost', 'sum'), scenario_demand_cost=('scenario_demand_cost', 'sum'), regional_budget=('scenario_regional_budget', 'sum'), baseline_budget_gap=('baseline_budget_gap', 'sum'), scenario_budget_gap=('scenario_budget_gap', 'sum'), mean_budget_utilisation=('scenario_budget_utilisation', 'mean'), mean_pct_change_demand=('pct_change_claims_demand', 'mean'), mean_pct_change_utilisation=('pct_change_provider_utilisation', 'mean'), mean_pct_change_served_expenditure=('pct_change_claims_expenditure', 'mean'), mean_pct_change_demand_cost=('pct_change_demand_cost', 'mean'), mean_pct_change_backlog=('pct_change_backlog', 'mean')).sort_values('abs_change_claims_demand', ascending=False)
    summary['over_budget'] = summary['scenario_budget_gap'] > 1e-06
    summary['abs_change_expenditure'] = summary['abs_change_demand_cost']
    summary['mean_pct_change_expenditure'] = summary['mean_pct_change_demand_cost']
    return summary

def load_sim_inputs(data_dir='data'):
    data_dir = Path(data_dir)
    claims = pd.read_csv(data_dir / 'hoto_claims_synthetic.csv')
    providers = pd.read_csv(data_dir / 'hoto_providers.csv')
    return (claims, providers)
