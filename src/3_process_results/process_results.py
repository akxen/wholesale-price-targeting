
# coding: utf-8

# # Emissions Intensity Scheme (EIS) Parameter Selection
# Process data from DCOPF and MPPDC models.
# 
# ## Import packages

# In[1]:


import os
import re
import pickle

import numpy as np
import pandas as pd
import datetime as dt

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Set text options for plots

# In[2]:


matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.serif'] = ['Helvetica']
plt.rc('text', usetex=True)


# ## Declare paths to files

# In[3]:


# Core data directory
data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data')

# Operating scenario data
operating_scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, '1_create_scenarios')

# Model output directory
parameter_selector_dir = os.path.join(os.path.curdir, os.path.pardir, '2_parameter_selector', 'output', '48_scenarios')

# Output directory
output_dir = os.path.join(os.path.curdir, 'output', '48_scenarios')


# ## Import data

# In[4]:


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

# In[5]:


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

# In[6]:


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

print('Absolute difference between DCOPF and MPPDC average prices: {0} [$/MWh]'.format(abs(dcopf_average_price - mppdc_average_price)))


# Compare nodal prices between DCOPF and MPPDC models.

# In[7]:


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

# In[8]:


# Price targeting baseline results - unprocessed
df_price_targeting_baseline = collate_data('MPPDC-FIND_PRICE_TARGETING_BASELINE')

# Price targeting baseline as a function of permit price
df_baseline_vs_permit_price = df_price_targeting_baseline.groupby(['FIXED_TAU', 'TARGET_PRICE_BAU_MULTIPLE']).apply(lambda x: x.loc['phi', 'Variable']['Value']).unstack()


# ### Scheme revenue for different average price targets and fixed permit prices

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


# ### Regional and national average for different average price targets and fixed permit prices

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

# Carbon tax average prices
df_carbon_tax = collate_data('MPPDC-FIXED_PARAMETERS-BASELINE_0')
df_carbon_tax_average_prices = df_carbon_tax.groupby(['FIXED_TAU', 'TARGET_PRICE_BAU_MULTIPLE']).apply(get_average_prices)


# ## Plotting
# 
# Figures to plot:
# 1. Merit order plots showing:
#     1. how emissions intensive plant move down the merit order as the permit price increases (subplot a), and the net liability faced by different generators if dispatched (subplot b);
#     2. short-run marginal costs of generators under a REP scheme and a carbon tax.
#     
# 2. Plot showing baselines that target average wholesale prices for different price targets over different permit price scenarios (subplot a). Plot showing scheme revenue arising from different baseline permit price combinations (subplot b). Plot showing final average wholesale prices (subplot c).
# 
# 3. BAU targeting baseline and scheme revenue over a range of permit prices
# 
# 4. Average emissions intensity as a function of permit price
# 
# 5. Average regional prices under a BAU average wholesale price targeting REP scheme (subplot a), and a carbon tax (subplot b) for different permit prices.
# 
# Conversion factors used to format figure size.

# In[11]:


# Millimeters to inches
mmi = 0.0393701


# ### Merit order plots

# In[12]:


def plot_merit_order():
    "Shows how merit order is affected w.r.t emissions intensities, SRMCs, and net liability under a REP scheme"

    # Only consider fossil units
    df_gp = df_g[df_g['FUEL_CAT']=='Fossil'].copy()

    # Permit prices
    permit_prices = range(2, 71, 2)

    # Number of rectanlges
    n = len(permit_prices)

    # Gap as a fraction of rectangle height
    gap_fraction = 1 / 10

    # Rectangle height
    rectangle_height = 1 / (gap_fraction * (n - 1) + n)

    # Gap between rectangles
    y_gap = rectangle_height * gap_fraction

    # Initial y offset
    y_offset = 0

    # Container for rectangle patches
    rectangles = []

    # Container for colours corresponding to patches
    colours_emissions_intensity = []
    colours_net_liability = []
    colours_srmc_rep = []
    colours_srmc_carbon_tax = []

    # Construct rectangles to plot for each permit price scenario
    for permit_price in permit_prices:

        # Baseline corresponding to BAU price targeting scenario
        baseline = df_baseline_vs_permit_price.loc[permit_price, 1]

        # Net liability faced by generator under REP scheme
        df_gp['NET_LIABILITY'] = (df_gp['EMISSIONS'] - baseline) * permit_price

        # Compute updated SRMC and sort from least cost to most expensive (merit order)
        df_gp['SRMC_REP'] = df_gp['SRMC_2016-17'] + df_gp['NET_LIABILITY']
        df_gp.sort_values('SRMC_REP', inplace=True)

        # Carbon tax SRMCs (baseline = 0 for all permit price scenarios)
        df_gp['SRMC_TAX'] = df_gp['SRMC_2016-17'] + (df_gp['EMISSIONS'] * permit_price)

        # Normalising registered capacities
        df_gp['REG_CAP_NORM'] = (df_gp['REG_CAP'] / df_gp['REG_CAP'].sum())

        x_offset = 0

        # Plotting rectangles
        for index, row in df_gp.iterrows():
            rectangles.append(patches.Rectangle((x_offset, y_offset), row['REG_CAP_NORM'], rectangle_height))

            # Colour for emissions intensity plot
            colours_emissions_intensity.append(row['EMISSIONS'])

            # Colour for net liability under REP scheme for each generator
            colours_net_liability.append(row['NET_LIABILITY'])

            # Colour for net generator SRMCs under REP scheme
            colours_srmc_rep.append(row['SRMC_REP'])

            # Colour for SRMCs under carbon tax
            colours_srmc_carbon_tax.append(row['SRMC_TAX'])

            # Offset for placement of next rectangle
            x_offset += row['REG_CAP_NORM']
        y_offset += rectangle_height + y_gap

    # Merit order emissions intensity patches
    patches_emissions_intensity = PatchCollection(rectangles, cmap='Reds')
    patches_emissions_intensity.set_array(np.array(colours_emissions_intensity))

    # Net liability under REP scheme patches
    patches_net_liability = PatchCollection(rectangles, cmap='bwr')
    patches_net_liability.set_array(np.array(colours_net_liability))

    # SRMCs under REP scheme patches
    patches_srmc_rep = PatchCollection(rectangles, cmap='Reds')
    patches_srmc_rep.set_array(np.array(colours_srmc_rep))

    # SRMCs under carbon tax patches
    patches_srmc_carbon_tax = PatchCollection(rectangles, cmap='Reds')
    patches_srmc_carbon_tax.set_array(np.array(colours_srmc_carbon_tax))


    # Format tick positions
    # ---------------------
    # y-ticks
    # -------
    # Minor ticks
    yminorticks = []
    for counter, permit_price in enumerate(permit_prices):
        if counter == 0:
            position = rectangle_height / 2
        else:
            position = yminorticks[-1] + y_gap + rectangle_height
        yminorticks.append(position)
    yminorlocator = FixedLocator(yminorticks)

    # Major ticks
    ymajorticks = []
    for counter in range(0, 7):
        if counter == 0:
            position = (4.5 * rectangle_height) + (4 * y_gap)
        else:
            position = ymajorticks[-1] + (5 * rectangle_height) + (5 * y_gap)
        ymajorticks.append(position)
    ymajorlocator = FixedLocator(ymajorticks)

    # x-ticks
    # -------
    # Minor locator
    xminorlocator = LinearLocator(21)

    # Major locator
    xmajorlocator = LinearLocator(6)


    # Emissions intensity and net liability figure
    # --------------------------------------------
    plt.clf()

    # Initialise figure
    fig1 = plt.figure()

    # Axes on which to construct plots
    ax1 = plt.axes([0.065, 0.185, 0.40, .79])
    ax2 = plt.axes([0.57, 0.185, 0.40, .79])

    # Add emissions intensity patches
    ax1.add_collection(patches_emissions_intensity)

    # Add net liability patches
    patches_net_liability.set_clim([-35, 35])
    ax2.add_collection(patches_net_liability)

    # Add colour bars with labels
    cbar1 = fig1.colorbar(patches_emissions_intensity, ax=ax1, pad=0.015, aspect=30)
    cbar1.set_label('Emissions intensity (tCO${_2}$/MWh)', fontsize=8, fontname='Helvetica')

    cbar2 = fig1.colorbar(patches_net_liability, ax=ax2, pad=0.015, aspect=30)
    cbar2.set_label('Net liability ($/MWh)', fontsize=8, fontname='Helvetica')

    # Label axes
    ax1.set_ylabel('Permit price (\$/tCO$_{2}$)', fontsize=9, fontname='Helvetica')
    ax1.set_xlabel('Normalised cumulative capacity\n(a)', fontsize=9, fontname='Helvetica')

    ax2.set_ylabel('Permit price (\$/tCO$_{2}$)', fontsize=9, fontname='Helvetica')
    ax2.set_xlabel('Normalised cumulative capacity\n(a)', fontsize=9, fontname='Helvetica')


    # Format ticks
    # ------------
    # y-axis
    ax1.yaxis.set_minor_locator(yminorlocator)
    ax1.yaxis.set_major_locator(ymajorlocator)
    ax2.yaxis.set_minor_locator(yminorlocator)
    ax2.yaxis.set_major_locator(ymajorlocator)

    # y-tick labels
    ax1.yaxis.set_ticklabels(['10', '20', '30', '40', '50', '60', '70'])
    ax2.yaxis.set_ticklabels(['10', '20', '30', '40', '50', '60', '70'])

    # x-axis
    ax1.xaxis.set_minor_locator(xminorlocator)
    ax1.xaxis.set_major_locator(xmajorlocator)
    ax2.xaxis.set_minor_locator(xminorlocator)
    ax2.xaxis.set_major_locator(xmajorlocator)

    # Format figure size
    width = 180 * mmi
    height = 75 * mmi
    fig1.set_size_inches(width, height)
    
    # Save figure
    fig1.savefig(os.path.join(output_dir, 'figures', 'emissions_liability_merit_order.pdf'))


    # SRMCs under REP and carbon tax
    # ------------------------------
    # Initialise figure
    fig2 = plt.figure()

    # Axes on which to construct plots
    ax3 = plt.axes([0.065, 0.185, 0.40, .79])
    ax4 = plt.axes([0.57, 0.185, 0.40, .79])

    # Add REP SRMCs
    patches_srmc_rep.set_clim([25, 200])
    ax3.add_collection(patches_srmc_rep)

    # Add carbon tax net liability
    patches_srmc_carbon_tax.set_clim([25, 200])
    ax4.add_collection(patches_srmc_carbon_tax)

    # Add colour bars with labels
    cbar3 = fig2.colorbar(patches_srmc_rep, ax=ax3, pad=0.015, aspect=30)
    cbar3.set_label('SRMC (\$/MWh)', fontsize=8, fontname='Helvetica')

    cbar4 = fig2.colorbar(patches_srmc_carbon_tax, ax=ax4, pad=0.015, aspect=30)
    cbar4.set_label('SRMC (\$/MWh)', fontsize=8, fontname='Helvetica')

    # Label axes
    ax3.set_ylabel('Permit price (\$/tCO$_{2}$)', fontsize=9, fontname='Helvetica')
    ax3.set_xlabel('Normalised cumulative capacity\n(a)', fontsize=9, fontname='Helvetica')

    ax4.set_ylabel('Permit price (\$/tCO$_{2}$)', fontsize=9, fontname='Helvetica')
    ax4.set_xlabel('Normalised cumulative capacity\n(a)', fontsize=9, fontname='Helvetica')


    # Format ticks
    # ------------
    # y-axis
    ax3.yaxis.set_minor_locator(yminorlocator)
    ax3.yaxis.set_major_locator(ymajorlocator)
    ax4.yaxis.set_minor_locator(yminorlocator)
    ax4.yaxis.set_major_locator(ymajorlocator)

    # y-tick labels
    ax3.yaxis.set_ticklabels(['10', '20', '30', '40', '50', '60', '70'])
    ax4.yaxis.set_ticklabels(['10', '20', '30', '40', '50', '60', '70'])

    # x-axis
    ax3.xaxis.set_minor_locator(xminorlocator)
    ax3.xaxis.set_major_locator(xmajorlocator)
    ax4.xaxis.set_minor_locator(xminorlocator)
    ax4.xaxis.set_major_locator(xmajorlocator)

    # Format figure size
    width = 180 * mmi
    height = 75 * mmi
    fig2.set_size_inches(width, height)

    # Save figure
    fig2.savefig(os.path.join(output_dir, 'figures', 'srmc_merit_order.pdf'))

    plt.show()
    
# Create figure
plot_merit_order()


# ### Price targeting baselines and corresponding scheme revenue

# In[13]:


def plot_price_targeting_baselines_and_scheme_revenue():
    "Plot baselines that target given wholesale prices and scheme revenue that corresponds to these scenarios"

    # Initialise figure 
    plt.clf()
    fig = plt.figure()

    # Axes on which to construct plots
#     ax1 = plt.axes([0.08, 0.175, 0.41, 0.77])
#     ax2 = plt.axes([0.585, 0.175, 0.41, 0.77])
    
    ax1 = plt.axes([0.07, 0.21, 0.25, 0.72])
    ax2 = plt.axes([0.40, 0.21, 0.25, 0.72])
    ax3 = plt.axes([0.74, 0.21, 0.25, 0.72])

    # Price targets
    price_target_colours = {0.8: '#b50e43', 0.9: '#af92cc', 1: '#45a564', 1.1: '#a59845', 1.2: '#f27b2b'}

    
    # Price targeting baselines
    # -------------------------
    for col in df_baseline_vs_permit_price.columns:
        ax1.plot(df_baseline_vs_permit_price[col], '-x', markersize=1.5, linewidth=0.9, label=col, color=price_target_colours[col])

    # Label axes
    ax1.set_ylabel('Emissions intensity baseline\nrelative to BAU', fontsize=9)
    ax1.set_xlabel('Permit price (\$/tCO${_2}$)\n(a)', fontsize=9)

    # Format ticks
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.xaxis.set_minor_locator(MultipleLocator(2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

    
    # Scheme revenue
    # --------------
    for col in df_baseline_vs_revenue.columns:
        ax2.plot(df_baseline_vs_revenue[col], '-x', markersize=1.5, linewidth=0.9, label=col, color=price_target_colours[col])

    # Label axes
    ax2.set_xlabel('Permit price (\$/tCO${_2}$)\n(b)', fontsize=9)
    ax2.set_ylabel('Scheme revenue (\$/h)', labelpad=0, fontsize=9)

    # Format axes
    ax2.ticklabel_format(axis='y', useMathText=True, style='sci', scilimits=(1, 5))
    ax2.xaxis.set_major_locator(MultipleLocator(10))
    ax2.xaxis.set_minor_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(20000))
    
    
    # Average prices
    # --------------
    # BAU average price
    bau_average_price = df_price_targeting_baseline.loc[df_price_targeting_baseline['TARGET_PRICE_BAU_MULTIPLE']==1, 'TARGET_PRICE'].unique()[0]

    # Final average price under different REP scenarios
    df_final_prices = df_price_targeting_baseline.drop_duplicates(subset=['AVERAGE_PRICE', 'FIXED_TAU', 'TARGET_PRICE_BAU_MULTIPLE']).pivot(index='FIXED_TAU', columns='TARGET_PRICE_BAU_MULTIPLE', values='AVERAGE_PRICE').div(bau_average_price)
    
    for col in df_final_prices.columns:
        ax3.plot(df_final_prices[col], '-x', markersize=1.5, linewidth=0.9, label=col, color=price_target_colours[col])
              
    # Label axes
    ax3.set_xlabel('Permit price (\$/tCO${_2}$)\n(c)', fontsize=9)
    ax3.set_ylabel('Average price\nrelative to BAU', labelpad=0, fontsize=9)
    
    # Format ticks
    ax3.xaxis.set_major_locator(MultipleLocator(10))
    ax3.xaxis.set_minor_locator(MultipleLocator(2))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.02))

    # Create legend
    legend = ax2.legend(title='Price target\nrelative to BAU', ncol=1, loc='upper center', bbox_to_anchor=(-0.61, 1.01), fontsize=9)
    legend.get_title().set_fontsize('9')

    # Format figure size
    fig = ax2.get_figure()
    width = 180 * mmi
    height = 70 * mmi
    fig.set_size_inches(width, height)

    # Save figure
    fig.savefig(os.path.join(output_dir, 'figures', 'baseline_revenue_price_subplot.pdf'))
    plt.show()

# Create plot
plot_price_targeting_baselines_and_scheme_revenue()


# In[14]:


def plot_bau_price_target_and_baseline():
    "Plot baseline that targets BAU prices and scheme revenue on same figure"
    
    # Initialise figure
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    # Plot emission intensity baseline and scheme revenue
    df_bau_baseline_revenue = df_baseline_vs_permit_price[1].to_frame().rename(columns={1: 'baseline'}).join(df_baseline_vs_revenue[1].to_frame().rename(columns={1: 'revenue'}), how='left')
    df_bau_baseline_revenue['baseline'].plot(ax=ax1, color='#dd4949', markersize=1.5, linewidth=1, marker='o', linestyle='-')
    df_bau_baseline_revenue['revenue'].plot(ax=ax2, color='#4a63e0', markersize=1.5, linewidth=1, marker='o', linestyle='-')

    # Format axes labels
    ax1.set_xlabel('Permit price (\$/tCO$_{2}$)', fontsize=9)
    ax1.set_ylabel('Emissions intensity baseline\nrelative to BAU', fontsize=9)
    ax2.set_ylabel('Scheme revenue (\$/h)', fontsize=9)

    # Format ticks
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax2.xaxis.set_major_locator(MultipleLocator(10))
    ax2.xaxis.set_minor_locator(MultipleLocator(2))
    ax2.ticklabel_format(axis='y', useMathText=True, style='sci', scilimits=(0, 1))

    # Format legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    l1 = ['Baseline']
    l2 = ['Revenue']
    ax1.legend(h1+h2, l1+l2, loc=0, bbox_to_anchor=(0.405, .25), fontsize=8)

    # Format figure size
    width = 85 * mmi
    height = 65 * mmi
    fig.subplots_adjust(left=0.22, bottom=0.16, right=0.8, top=.93)
    fig.set_size_inches(width, height)

    # Save figure
    fig.savefig(os.path.join(output_dir, 'figures', 'bau_price_target_baseline_and_revenue.pdf'))
    plt.show()

# Create figure
plot_bau_price_target_and_baseline()


# ### System emissions intensity as a function of permit price.

# In[15]:


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


# Create figure

# In[16]:


def plot_permit_price_vs_emissions_intensity():
    "Plot average emissions intensity as a function of permit price"
    
    # Initialise figure
    plt.clf()
    fig, ax = plt.subplots()

    # Plot figure
    df_average_emissions_intensities.div(df_average_emissions_intensities.iloc[0]).plot(linestyle='-', marker='x', markersize=2, linewidth=0.8, color='#c11111', ax=ax)

    # Format axis labels
    ax.set_xlabel('Permit price (\$/tCO${_2}$)', fontsize=9)
    ax.set_ylabel('Emissions intensity\nrelative to BAU', fontsize=9)

    # Format ticks
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))

    # Format figure size
    width = 85 * mmi
    height = width / 1.2
    fig.subplots_adjust(left=0.2, bottom=0.14, right=.98, top=0.98)
    fig.set_size_inches(width, height)
    
    # Save figure
    fig.savefig(os.path.join(output_dir, 'figures', 'permit_price_vs_emissions_intensity_normalised.pdf'))
    plt.show()
    
# Create figure
plot_permit_price_vs_emissions_intensity()


# ### Regional prices under REP scheme and carbon tax

# In[17]:


def plot_regional_prices_rep_and_tax():
    "Plot average regional prices under REP and carbon tax scenarios"
    
    # Initialise figure
    plt.clf()
    fig = plt.figure()
    ax1 = plt.axes([0.068, 0.18, 0.41, 0.8])
    ax2 = plt.axes([0.58, 0.18, 0.41, 0.8])

    # Regional prices under REP scheme
    df_rep_prices = df_rep_average_prices.loc[(slice(None), 1), :]
    df_rep_prices.index = df_rep_prices.index.droplevel(1)
    df_rep_prices.drop('NATIONAL', axis=1).plot(marker='o', linestyle='-', markersize=1.5, linewidth=1, cmap='tab10', ax=ax1)

    # Format labels
    ax1.set_xlabel('Permit price (\$/tCO${_2}$)\n(a)', fontsize=9)
    ax1.set_ylabel('Average wholesale price (\$/MWh)', fontsize=9)     

    # Format axes
    ax1.minorticks_on()
    ax1.xaxis.set_minor_locator(MultipleLocator(2))
    ax1.xaxis.set_major_locator(MultipleLocator(10))

    # Add legend
    legend1 = ax1.legend()
    legend1.remove()

    # Plot prices under a carbon tax (baseline=0)
    df_carbon_tax_prices = df_carbon_tax_average_prices.copy()
    df_carbon_tax_prices.index = df_carbon_tax_prices.index.droplevel(1)
    
    # Rename columns - remove '1' at end of NEM region name
    new_column_names = {i: i.split('_')[-1].replace('1','') for i in df_carbon_tax_prices.columns}
    df_carbon_tax_prices = df_carbon_tax_prices.rename(columns=new_column_names)
    
    df_carbon_tax_prices.drop('NATIONAL', axis=1).plot(marker='o', linestyle='-', markersize=1.5, linewidth=1, cmap='tab10', ax=ax2)

    # Format axes labels
    ax2.set_ylabel('Average wholesale price (\$/MWh)', fontsize=9)
    ax2.set_xlabel('Permit price (\$/tCO${_2}$)\n(b)', fontsize=9)

    # Format ticks
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(MultipleLocator(2))
    ax2.xaxis.set_major_locator(MultipleLocator(10))

    # Create legend
    legend2 = ax2.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.708, 0.28), fontsize=9)

    # Format figure size
    width = 180 * mmi
    height = 80 * mmi
    fig.set_size_inches(width, height)

    # Save figure
    fig.savefig(os.path.join(output_dir, 'figures', 'regional_wholesale_prices.pdf'))
    plt.show()
    
# Create figure
plot_regional_prices_rep_and_tax()

