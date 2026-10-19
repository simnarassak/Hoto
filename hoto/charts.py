import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def _empty_fig(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5)
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def population_by_region(population):
    if population is None or population.empty:
        return _empty_fig('Population data unavailable')
    counts = population.groupby('region', as_index=False).size().rename(columns={'size': 'people'}).sort_values('people', ascending=False)
    fig = px.bar(counts, x='region', y='people', title='Population by region', labels={'people': 'People', 'region': 'Region'})
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360)
    return fig

def age_by_group_heatmap(population):
    if population is None or population.empty:
        return _empty_fig('Population data unavailable')
    table = pd.crosstab(population['age_band'], population['population_group'])
    age_order = ['0-17', '18-34', '35-49', '50-64', '65+']
    table = table.reindex([a for a in age_order if a in table.index])
    fig = px.imshow(table, text_auto=True, aspect='auto', title='Age band by population group', labels=dict(x='Population group', y='Age band', color='People'), color_continuous_scale='Blues')
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=380)
    return fig

def employment_mix(population):
    if population is None or population.empty:
        return _empty_fig('Population data unavailable')
    counts = population.groupby('employment_status', as_index=False).size().rename(columns={'size': 'people'})
    fig = px.pie(counts, names='employment_status', values='people', title='Employment status mix', hole=0.35)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360)
    return fig

def claims_over_time(claims):
    if claims is None or claims.empty:
        return _empty_fig('Claims data unavailable')
    frame = claims.copy()
    frame['submission_date'] = pd.to_datetime(frame['submission_date'])
    frame['month'] = frame['submission_date'].dt.to_period('M').astype(str)
    monthly = frame.groupby('month', as_index=False).size().rename(columns={'size': 'claims'}).sort_values('month')
    fig = px.line(monthly, x='month', y='claims', markers=True, title='Claims submitted over time', labels={'month': 'Month', 'claims': 'Claims'})
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360)
    return fig

def claim_type_mix(claims):
    if claims is None or claims.empty:
        return _empty_fig('Claims data unavailable')
    counts = claims.groupby('claim_type', as_index=False).size().rename(columns={'size': 'claims'}).sort_values('claims', ascending=False)
    fig = px.bar(counts, x='claims', y='claim_type', orientation='h', title='Claim type mix', labels={'claims': 'Claims', 'claim_type': 'Claim type'})
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360)
    return fig

def claims_by_region(claims):
    if claims is None or claims.empty:
        return _empty_fig('Claims data unavailable')
    region_col = 'claimant_region' if 'claimant_region' in claims.columns else 'region'
    if region_col not in claims.columns:
        return _empty_fig('Region column unavailable')
    counts = claims.groupby(region_col, as_index=False).size().rename(columns={'size': 'claims', region_col: 'region'}).sort_values('claims', ascending=False)
    fig = px.bar(counts, x='region', y='claims', title='Claims by claimant region', labels={'claims': 'Claims', 'region': 'Region'})
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360)
    return fig

def providers_by_type_region(providers):
    if providers is None or providers.empty:
        return _empty_fig('Provider data unavailable')
    counts = providers.groupby(['provider_type', 'region'], as_index=False).size().rename(columns={'size': 'providers'})
    fig = px.bar(counts, x='region', y='providers', color='provider_type', barmode='stack', title='Providers by region and type', labels={'providers': 'Providers', 'region': 'Region'})
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=380)
    return fig

def approved_amount_year_on_year(claims):
    if claims is None or claims.empty:
        return _empty_fig('Claims data unavailable')
    frame = claims.copy()
    approved = frame.loc[frame['claim_status'].eq('Approved')].copy()
    if approved.empty or 'approved_amount' not in approved.columns:
        return _empty_fig('No approved amounts available')
    approved['decision_date'] = pd.to_datetime(approved['decision_date'], errors='coerce')
    approved = approved.dropna(subset=['decision_date'])
    approved['year'] = approved['decision_date'].dt.year.astype(int)
    yearly = approved.groupby('year', as_index=False).agg(approved_amount=('approved_amount', 'sum'), approved_claims=('claim_id', 'size')).sort_values('year')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly['year'].astype(str), y=yearly['approved_amount'], name='Approved amount', marker_color='#2F6F8F'))
    fig.add_trace(go.Scatter(x=yearly['year'].astype(str), y=yearly['approved_claims'], name='Approved claims', yaxis='y2', mode='lines+markers', line=dict(color='#C45C26')))
    fig.update_layout(title='Approved claims year on year', yaxis=dict(title='Approved amount'), yaxis2=dict(title='Approved claims', overlaying='y', side='right'), legend=dict(orientation='h'), margin=dict(l=20, r=20, t=50, b=20), height=400)
    return fig

def score_histogram(scored, score_column='ensemble_score', threshold=0.35):
    if scored is None or scored.empty or score_column not in scored.columns:
        return _empty_fig('Score data unavailable')
    fig = px.histogram(scored, x=score_column, nbins=40, title=f'Distribution of {score_column}', labels={score_column: 'Score'})
    fig.add_vline(x=threshold, line_dash='dash', line_color='#C45C26', annotation_text=f'threshold {threshold:.2f}')
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360)
    return fig

def provider_risk_bubble(provider_risk, top_n=40):
    if provider_risk is None or provider_risk.empty:
        return _empty_fig('Provider risk unavailable')
    frame = provider_risk.head(top_n).copy()
    size_col = 'total_claimed_amount' if 'total_claimed_amount' in frame.columns else 'claim_count'
    fig = px.scatter(frame, x='claim_count', y='mean_ensemble_score', size=size_col, color='provider_risk_index', hover_name='provider_id', title='Provider risk landscape', labels={'claim_count': 'Claim count', 'mean_ensemble_score': 'Mean ensemble score', 'provider_risk_index': 'Risk index'}, color_continuous_scale='YlOrRd')
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=400)
    return fig
