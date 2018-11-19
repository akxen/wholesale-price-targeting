
# coding: utf-8

# # Process Results
# Process data from DCOPF and MPPDC models.
# 
# ## Import packages

# In[1]:


import os
import re
import pickle

import numpy as np
import pandas as pd


# ## Declare paths to files

# In[2]:


# Identifier used to update paths depending on the number of scenarios investigated
number_of_scenarios = '100_scenarios'

# Core data directory
data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data')

# Operating scenario data
operating_scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, '1_create_scenarios')

# Model output directory
parameter_selector_dir = os.path.join(os.path.curdir, os.path.pardir, '2_parameter_selector', 'output', number_of_scenarios)

# Output directory
output_dir = os.path.join(os.path.curdir, 'output', number_of_scenarios)


# ## Import data

# In[3]:


# NOTE: SRMCs were adjusted for generators and node reindexed in the parameter_selector notebook.
# Therefore data should be loaded from the parameter_selector output folder

# Generator data
with open(os.path.join(parameter_selector_dir, 'df_g.pickle'), 'rb') as f:
    df_g = pickle.load(f)
    
# Node data
with open(os.path.join(parameter_selector_dir, 'df_n.pickle'), 'rb') as f:
    df_n = pickle.load(f)
    
# Scenario data
with open(os.path.join(parameter_selector_dir, 'df_scenarios.pickle'), 'rb') as f:
    df_scenarios = pickle.load(f)
    
# DCOPF results for BAU scenario - (baseline=0, permit price=0)
with open(os.path.join(parameter_selector_dir, 'DCOPF-FIXED_PARAMETERS-PERMIT_PRICE_0-BASELINE_0.pickle'), 'rb') as f:
    df_dcopf = pickle.load(f)
    
# MPPDC results for BAU scenario - (baseline=0, permit price=0)
with open(os.path.join(parameter_selector_dir, 'MPPDC-FIXED_PARAMETERS-BASELINE_0-PERMIT_PRICE_0.pickle'), 'rb') as g:
    df_mppdc = pickle.load(g) 

# Function used to collate DataFrames for different modelling scenarios
def collate_data(filename_contains):
    """Collate data for different scenarios into a single DataFrame
    
    Parameters
    ----------
    filename_contains : str
        Partial filename used to filter files in parameter_selector output directory
        
    Returns
    -------
    df_o : pandas DataFrame
        Collated results for specified model type    
    """
    
    # Filtered files
    files = [os.path.join(parameter_selector_dir, f) for f in os.listdir(os.path.join(parameter_selector_dir)) if filename_contains in f]
    
    # Container for inidividual scenarios
    dfs = []
    
    # Loop through files and load results objects
    for f in files:
        # Load results
        with open(f, 'rb') as g:
            df = pickle.load(g)

            # Filter column names
            filtered_columns = [i for i in df.columns if i not in ['Gap', 'Status', 'Message', 'Problem', 'Objective', 'Constraint']]

            # Filter DataFrame
            df_filtered = df.loc[df.index.str.contains(r'\.P\[|\.H\[|\.vang\[|\.lambda_var\[|phi'), filtered_columns]

            # Append to container which will later be concatenated
            dfs.append(df_filtered)
    
    # Concatenate results
    df_o = pd.concat(dfs)
    
    return df_o    


# ## Compare DCOPF and MPPDC models

# Compare primal variables between DCOPF and MPPDC models.

# In[4]:


def compare_primal_variables(df_dcopf, df_mppdc):
    """Verify that MPPDC and DCOPF results match
    
    Parameters
    ----------
    df_dcopf : pandas DataFrame
        DataFrame containing results for DCOPF model
    
    df_mppdc : pandas DataFrame
        DataFrame containing results for MPPDC model
        
    Returns
    -------
    df_max_difference : pandas DataFrame
        Max difference across all scenarios for each primal variable    
    """
    
    # Process DCOPF DataFrame
    # -----------------------
    df_dcopf = df_dcopf.reset_index()
    df_dcopf = df_dcopf[df_dcopf['index'].str.contains(r'\.P\[|\.H\[|\.vang\[')]
    
    # Extract values for primal variables
    df_dcopf['Value'] = df_dcopf.apply(lambda x: x['Variable']['Value'], axis=1)

    # Extract scenario ID
    df_dcopf['SCENARIO_ID'] = df_dcopf['SCENARIO_ID'].astype(int)
    
    # Extract variable names and indices
    df_dcopf['variable_index_1'] = df_dcopf['index'].str.extract(r'\.(.+)\[')
    df_dcopf['variable_index_2'] = df_dcopf['index'].str.extract(r'\.[A-Za-z]+\[(.+)\]$')
    
    # Reset index
    df_dcopf = df_dcopf.set_index(['SCENARIO_ID', 'variable_index_1', 'variable_index_2'])['Value']
    
    
    # Process MPPDC DataFrame
    # -----------------------
    df_mppdc = df_mppdc.reset_index()
    
    # Extract values for primal variables
    df_mppdc = df_mppdc[df_mppdc['index'].str.contains(r'\.P\[|\.H\[|\.vang\[')]
    df_mppdc['Value'] = df_mppdc.apply(lambda x: x['Variable']['Value'], axis=1)

    # Extract scenario ID
    df_mppdc['SCENARIO_ID'] = df_mppdc['index'].str.extract(r'\[(\d+)\]\.')
    df_mppdc['SCENARIO_ID'] = df_mppdc['SCENARIO_ID'].astype(int)

    # Extract variable names and indices
    df_mppdc['variable_index_1'] = df_mppdc['index'].str.extract(r'\.(.+)\[')
    df_mppdc['variable_index_2'] = df_mppdc['index'].str.extract(r'\.[A-Za-z]+\[(.+)\]$')
    
    # Reset index
    df_mppdc = df_mppdc.set_index(['SCENARIO_ID', 'variable_index_1', 'variable_index_2'])['Value']
    
    # Max difference across all time intervals and primal variables
    df_max_difference = (df_dcopf.subtract(df_mppdc)
                         .reset_index()
                         .groupby(['variable_index_1'])['Value']
                         .apply(lambda x: x.abs().max()))
    
    return df_max_difference

# Find max difference in primal variables between DCOPF and MPPDC models over all scenarios
compare_primal_variables(df_dcopf, df_mppdc)


# Note that voltage angles and HVDC flows do not correspond exactly, but power output and prices do. This is likely due to the introduction of an additional degree of freedom when adding HVDC links into the network analysis. Having HVDC links allows power to flow over either the HVDC or AC networks. So long as branch flows are within limits for constrained links, different combinations of these flows may be possible. This results in different intra-zonal flows (hence different voltage angles), but net inter-zonal flows are the same as the DCOPF case.
# 
# These differences are likely due to the way in which the solver approaches a solution; different feasible HVDC and intra-zonal AC flows yield the same least-cost dispatch. Consequently, DCOPF and MPPDC output corresponds, but HVDC flows and node voltage angles (which relate to AC power flows) do not. As an additional check average and nodal prices are compared between the DCOPF and MPPDC models.

# In[5]:


def get_dcopf_average_price(df, variable_name_contains):
    "Find average price for DCOPF BAU scenario"
    
    df_tmp = df.reset_index().copy()

    # Filter price records
    df_tmp = df_tmp[df_tmp['index'].str.contains(r'\.{0}\['.format(variable_name_contains))]
    
    # Extract values
    df_tmp['Value'] = df_tmp.apply(lambda x: x['Constraint']['Dual'], axis=1)

    # Extract node and scenario IDs
    df_tmp['NODE_ID'] = df_tmp['index'].str.extract(r'\.{0}\[(\d+)\]'.format(variable_name_contains)).astype(int)
    df_tmp['SCENARIO_ID'] = df_tmp['SCENARIO_ID'].astype(int)

    # Merge demand for each node and scenario
    df_demand = df_scenarios.loc[:, ('demand')].T
    df_demand.index = df_demand.index.astype(int)
    df_demand = df_demand.reset_index().melt(id_vars=['NODE_ID']).rename(columns={'value': 'demand'})
    df_demand['closest_centroid'] = df_demand['closest_centroid'].astype(int)
    df_tmp = pd.merge(df_tmp, df_demand, left_on=['SCENARIO_ID', 'NODE_ID'], right_on=['closest_centroid', 'NODE_ID'])

    # Merge duration information for each scenario
    df_duration = df_scenarios.loc[:, ('hours', 'duration')].to_frame()
    df_duration.columns = df_duration.columns.droplevel()
    df_tmp = pd.merge(df_tmp, df_duration, left_on='SCENARIO_ID', right_index=True)

    # Compute total revenue and total energy demand
    total_revenue = df_tmp.apply(lambda x: x['Value'] * x['demand'] * x['duration'], axis=1).sum()
    total_demand = df_tmp.apply(lambda x: x['demand'] * x['duration'], axis=1).sum()

    # Find average price (national)
    average_price = total_revenue / total_demand
    
    return average_price

dcopf_average_price = get_dcopf_average_price(df_dcopf, 'POWER_BALANCE')
mppdc_average_price = df_mppdc['AVERAGE_PRICE'].unique()[0]

# Save BAU average price - useful when constructing plots
with open(os.path.join(output_dir, 'mppdc_bau_average_price.pickle'), 'wb') as f:
    pickle.dump(mppdc_average_price, f)

print('Absolute difference between DCOPF and MPPDC average prices: {0} [$/MWh]'.format(abs(dcopf_average_price - mppdc_average_price)))


# Compare nodal prices between DCOPF and MPPDC models.

# In[6]:


def compare_nodal_prices(df_dcopf, df_mppdc):
    """Find max absolute difference in nodal prices between DCOPF and MPPDC models
    
    Parameters
    ----------
    df_dcopf : pandas DataFrame
        Results from DCOPF model
    
    df_mppdc : pandas DataFrame
        Results from MPPDC model
    
    Returns
    -------
    max_price_difference : float
        Maximum difference between nodal prices for DCOPF and MPPDC models
        over all nodes and scenarios    
    """

    # DCOPF model
    # -----------
    df_tmp_1 = df_dcopf.reset_index().copy()

    # Filter price records
    df_tmp_1 = df_tmp_1[df_tmp_1['index'].str.contains(r'\.POWER_BALANCE\[')]

    # Extract values
    df_tmp_1['Value'] = df_tmp_1.apply(lambda x: x['Constraint']['Dual'], axis=1)

    # Extract node and scenario IDs
    df_tmp_1['NODE_ID'] = df_tmp_1['index'].str.extract(r'\.POWER_BALANCE\[(\d+)\]').astype(int)
    df_tmp_1['SCENARIO_ID'] = df_tmp_1['SCENARIO_ID'].astype(int)

    # Prices at each node for each scenario
    df_dcopf_prices = df_tmp_1.set_index(['SCENARIO_ID', 'NODE_ID'])['Value']


    # MPPDC model
    # -----------
    df_tmp_2 = df_mppdc.reset_index().copy()

    # Filter price records
    df_tmp_2 = df_tmp_2[df_tmp_2['index'].str.contains(r'\.lambda_var\[')]

    # Extract values
    df_tmp_2['Value'] = df_tmp_2.apply(lambda x: x['Variable']['Value'], axis=1)

    # Extract node and scenario IDs
    df_tmp_2['NODE_ID'] = df_tmp_2['index'].str.extract(r'\.lambda_var\[(\d+)\]').astype(int)
    df_tmp_2['SCENARIO_ID'] = df_tmp_2['index'].str.extract(r'LL_DUAL\[(\d+)\]').astype(int)

    # Prices at each node for each scenario
    df_mppdc_prices = df_tmp_2.set_index(['SCENARIO_ID', 'NODE_ID'])['Value']

    # Compute difference between models
    # ---------------------------------
    max_price_difference = df_dcopf_prices.subtract(df_mppdc_prices).abs().max()
    print('Maximum difference between nodal prices over all nodes and scenarios: {0}'.format(max_price_difference))

    return max_price_difference

# Find max nodal price difference between DCOPF and MPPDC models
compare_nodal_prices(df_dcopf=df_dcopf, df_mppdc=df_mppdc)


# Close correspondence of price and output results between MPPDC and DCOPF representations suggests that the MPPDC has been formulated correctly.

# ## Organise data for plotting
# Using collated model results data, find:
# 1. emissions intensity baselines that target average wholesale prices for different permit price scenarios;
# 2. scheme revenue that corresponds with price targeting baselines and permit price scenarios;
# 3. average regional and national prices under a REP scheme with different average wholesale price targets;
# 4. average regional and national prices under a carbon tax.
# 
# 
# ### Price targeting baselines for different average price targets and fixed permit prices
# Collated data.

# In[7]:


# Price targeting baseline results - unprocessed
df_price_targeting_baseline = collate_data('MPPDC-FIND_PRICE_TARGETING_BASELINE')


# #### Baselines for different average price targets

# In[8]:


# Price targeting baseline as a function of permit price
df_baseline_vs_permit_price = df_price_targeting_baseline.groupby(['FIXED_TAU', 'TARGET_PRICE_BAU_MULTIPLE']).apply(lambda x: x.loc['phi', 'Variable']['Value']).unstack()

with open(os.path.join(output_dir, 'df_baseline_vs_permit_price.pickle'), 'wb') as f:
    pickle.dump(df_baseline_vs_permit_price, f)


# #### Scheme revenue for different average price targets

# In[9]:


def get_revenue(group):
    "Get scheme revenue for each permit price - price target scenario"
    
    # Get baseline for each group
    baseline = group.loc['phi', 'Variable']['Value']
    
    # Filter power output records
    group = group[group.index.str.contains(r'LL_PRIM\[\d+\].P\[')].copy()
    
    # Extract DUID - used to merge generator information
    group['DUID'] = group.apply(lambda x: re.findall(r'P\[(.+)\]', x.name)[0], axis=1)
    
    # Extract scenario ID - used to get duration of scenario
    group['SCENARIO_ID'] = group.apply(lambda x: re.findall(r'LL_PRIM\[(\d+)\]', x.name)[0], axis=1)

    # Duration of each scenario [hours]
    scenario_id = int(group['SCENARIO_ID'].unique()[0])
    scenario_duration = df_scenarios.loc[scenario_id, ('hours', 'duration')]
    
    # Generator power output
    group['Value'] = group.apply(lambda x: x['Variable']['Value'], axis=1)
    
    # Merge SRMCs and emissions intensities
    group = pd.merge(group, df_g[['SRMC_2016-17', 'EMISSIONS']], how='left', left_on='DUID', right_index=True)
    
    # Compute revenue SUM ((emissions_intensity - baseline) * power_output * duration)
    revenue = group['EMISSIONS'].subtract(baseline).mul(group['Value']).mul(group['FIXED_TAU']).mul(scenario_duration).sum() / 8760
    
    return revenue

# Get revenue for each permit price and wholesale average price target scenario
df_baseline_vs_revenue = df_price_targeting_baseline.groupby(['FIXED_TAU', 'TARGET_PRICE_BAU_MULTIPLE']).apply(get_revenue).unstack()

with open(os.path.join(output_dir, 'df_baseline_vs_revenue.pickle'), 'wb') as f:
    pickle.dump(df_baseline_vs_revenue, f)


# #### Regional and national average wholesale electricity prices for different average price targets

# In[10]:


def get_average_prices(group):
    "Compute regional and national average wholesale prices"
    
    # Filter price records
    group = group[group.index.str.contains(r'\.lambda_var\[')].reset_index().copy()
    
    # Extract nodal prices
    group['Value'] = group.apply(lambda x: x['Variable']['Value'], axis=1)

    # Extract node and scenario IDs
    group['NODE_ID'] = group['index'].str.extract(r'lambda_var\[(\d+)\]')[0].astype(int)
    group['SCENARIO_ID'] = group['index'].str.extract(r'LL_DUAL\[(\d+)\]')[0].astype(int)

    # Scenario Demand
    df_demand = df_scenarios.loc[:, ('demand')].reset_index().melt(id_vars=['closest_centroid'])
    df_demand['NODE_ID'] = df_demand['NODE_ID'].astype(int)
    df_demand = df_demand.rename(columns={'value': 'demand'})
    group = pd.merge(group, df_demand, how='left', left_on=['NODE_ID', 'SCENARIO_ID'], right_on=['NODE_ID', 'closest_centroid'])

    # Scenario duration
    df_duration = df_scenarios.loc[:, ('hours', 'duration')].to_frame()
    df_duration.columns = df_duration.columns.droplevel()
    group = pd.merge(group, df_duration, how='left', left_on='SCENARIO_ID', right_index=True)

    # NEM regions
    group = pd.merge(group, df_n[['NEM_REGION']], how='left', left_on='NODE_ID', right_index=True)

    # Compute node energy demand [MWh]
    group['total_demand'] = group['demand'].mul(group['duration'])

    # Compute node revenue [$]
    group['total_revenue'] = group['total_demand'].mul(group['Value'])

    # National average price
    national_average_price = group['total_revenue'].sum() / group['total_demand'].sum()

    # Regional averages
    df_average_prices = group.groupby('NEM_REGION')[['total_demand', 'total_revenue']].sum().apply(lambda x: x['total_revenue'] / x['total_demand'], axis=1)
    df_average_prices.loc['NATIONAL'] = national_average_price

    return df_average_prices

# REP Scheme average prices
df_rep_average_prices = df_price_targeting_baseline.groupby(['FIXED_TAU', 'TARGET_PRICE_BAU_MULTIPLE']).apply(get_average_prices)

with open(os.path.join(output_dir, 'df_rep_average_prices.pickle'), 'wb') as f:
    pickle.dump(df_rep_average_prices, f)


# ### Average prices under a carbon tax
# Collate data

# In[11]:


# Carbon tax scenario results
df_carbon_tax = collate_data('MPPDC-FIXED_PARAMETERS-BASELINE_0')


# #### Average prices under carbon tax

# In[12]:


# Average regional and national prices under a carbon tax
df_carbon_tax_average_prices = df_carbon_tax.groupby(['FIXED_TAU', 'TARGET_PRICE_BAU_MULTIPLE']).apply(get_average_prices)

with open(os.path.join(output_dir, 'df_carbon_tax_average_prices.pickle'), 'wb') as f:
    pickle.dump(df_carbon_tax_average_prices, f)


# #### Average system emissions intensity for different permit prices
# Average system emissions intensity is invariant to the emissions intensity baseline (the emissions intensity baseline doesn't affect the relative costs of generators). Therefore the emissions outcomes will be the same for the REP and carbon tax scenarios when permit prices are fixed.
# 
# **Note: this equivalence in outcomes assumes that demand is perfectly inelastic.**

# In[13]:


def get_average_emissions_intensity(group):
    "Get average emissions intensity for each permit price scenario"
    
    # Reset index
    df_tmp = group.reset_index()
    df_tmp = df_tmp[df_tmp['index'].str.contains(r'\.P\[')].copy()
    
    # Extract generator IDs
    df_tmp['DUID'] = df_tmp['index'].str.extract(r'\.P\[(.+)\]')
    
    # Extract and format scenario IDs
    df_tmp['SCENARIO_ID'] = df_tmp['index'].str.extract(r'LL_PRIM\[(.+)\]\.')
    df_tmp['SCENARIO_ID'] = df_tmp['SCENARIO_ID'].astype(int)
    
    # Extract power output values
    df_tmp['value'] = df_tmp.apply(lambda x: x['Variable']['Value'], axis=1)

    # Duration of each scenario
    df_duration = df_scenarios.loc[:, ('hours', 'duration')].to_frame()
    df_duration.columns = df_duration.columns.droplevel(0)

    # Merge scenario duration information
    df_tmp = pd.merge(df_tmp, df_duration, how='left', left_on='SCENARIO_ID', right_index=True)

    # Merge emissions intensity information
    df_tmp = pd.merge(df_tmp, df_g[['EMISSIONS']], how='left', left_on='DUID', right_index=True)

    # Total emissions [tCO2]
    total_emissions = df_tmp['value'].mul(df_tmp['duration']).mul(df_tmp['EMISSIONS']).sum()
    
    # Total demand [MWh]
    total_demand = df_scenarios.loc[:, ('demand')].sum(axis=1).mul(df_duration['duration']).sum()    

    # Average emissions intensity
    average_emissions_intensity = total_emissions / total_demand
    
    return average_emissions_intensity

# Average emissions intensity for different permit price scenarios
df_average_emissions_intensities = df_carbon_tax.groupby('FIXED_TAU').apply(get_average_emissions_intensity)

with open(os.path.join(output_dir, 'df_average_emissions_intensities.pickle'), 'wb') as f:
    pickle.dump(df_average_emissions_intensities, f)

