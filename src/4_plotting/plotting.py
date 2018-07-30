
# coding: utf-8

# # Emissions Intensity Scheme (EIS) Parameter Selection
# Analysis and plotting
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PatchCollection
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator


# Set text options for plots.

# In[2]:


matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.serif'] = ['Helvetica']
plt.rc('text', usetex=True)


# ## Declare paths to files

# In[3]:


# Core data directory
data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data'))

# Model output directory
parameter_selector_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '2_parameter_selector'))

# Processed results directory
processed_results_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '3_process_results'))

# Output directory
output_dir = os.path.abspath(os.path.join(os.path.curdir, 'output'))


# ## Import model data

# In[4]:


# Generator data
with open(os.path.join(parameter_selector_dir, 'output', 'df_g.pickle'), 'rb') as f:
    df_g = pickle.load(f)


# ## Import summary of model results

# In[5]:


class PlotMPPDC(object):
    
    def __init__(self, mppdc_summary_path, df_g):
        
        # Load MPPDC summary data and convert to dataframe
        # ------------------------------------------------
        with open(mppdc_summary_path, 'rb') as f:
            mppdc_summary = pickle.load(f)
            
            # Columns in summary data
        cols = mppdc_summary[0].keys()

        # Records for each scenario
        r = [[s[c] for c in cols] for s in mppdc_summary]

        # Construct dataframe
        df_s = pd.DataFrame(data=r, columns=cols)

        # Expand regional average prices from dictionary
        df_regional_average_prices = df_s['regional_average_prices'].apply(pd.Series).add_prefix('regional_average_price_')
        df_s = df_s.join(df_regional_average_prices)

        # Expand regional average emissions intensities from dictionary
        df_regional_average_emissions_intensity = df_s['regional_average_emissions_intensity'].apply(pd.Series).add_prefix('regional_average_emissions_intensity_')
        df_s = df_s.join(df_regional_average_emissions_intensity)

        # Expand national fossil generation proportions from dictionary
        df_national_fossil_generation_proportions = df_s['national_fossil_generation_proportions'].apply(pd.Series).add_prefix('national_fossil_generation_proportion_')
        df_s = df_s.join(df_national_fossil_generation_proportions)
        
        # Round national average and target prices
        df_s['national_average_price'] = df_s['national_average_price'].round(4)
        df_s['target_price'] = df_s['target_price'].round(4)
        
        
        # Instantiate class
        # -----------------
        # Dataframe summarising results from MPPDC scenarios
        self.df_s = df_s
        
        # Generator information dataframe
        self.df_g = df_g
        
        # Conversion from cm to inches
        self.mmi = 0.0393701
        
        
    def get_national_bau_price(self):
        "Get average national business-as-usual price"

        # Consider scenario when permit price is fixed to zero
        mask = (mppdc_plots.df_s['mode'] == 'CalcFixed') & (mppdc_plots.df_s['tau'] == 0)

        # Round national average price to 4 decimal places
        return np.around(mppdc_plots.df_s.loc[mask, 'national_average_price'].values[0], 4)
        
    
    def price_targeting_baselines(self, filename, normalise=False):
        "Plot price targetting baselines for given permit prices"
        
        # Business-as-usual price
        bau_price = self.get_national_bau_price()
        
        # Scenario data
        df = self.df_s.loc[self.df_s['mode'] == 'CalcPhi'].pivot(index='tau', columns='target_price', values='phi')

        # BAU emissions inensity
        mask = (self.df_s['mode'] == 'CalcFixed') & (self.df_s['tau'] == 0)
        bau_emissions_intensity = self.df_s.loc[mask, 'national_average_emissions_intensity'].values[0]
        
        if normalise:
            # Rename columns
            df = df.rename(columns={i : '{0:.1f}'.format(i / bau_price) for i in df.columns})
            df = df / bau_emissions_intensity
        else:
            # Rename columns
            df = df.rename(columns={i: '{0:.2f}'.format(i) for i in df.columns})

        # Plot axes
        ax = df.plot(cmap='Accent', style='o-', linewidth=1, markersize=2.8)
        fig = ax.get_figure()
        
        # Name legend and axes
        if normalise:
            ax.legend(title='Price target relative to BAU', ncol=5)
            ax.set_ylabel('Price targeting baseline relative to BAU emissions intensity')
            
        else:
            ax.legend(title='Price target', ncol=5)
            ax.set_ylabel('Price targeting baseline (tCO$_{2}$/MWh)')

        ax.set_xlabel('Permit price (\$/tCO${_2}$)')
        
        # Formatting
        ax.minorticks_on()
        width = 180 * self.mmi
        height = 112.5 * self.mmi
        fig.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.97)
        fig.set_size_inches(width, height)

        # Save figure
        fig.savefig(filename)
        plt.show()
        
        
    def scheme_revenue(self, filename, relative=False, ax=None):
        "Plot scheme revenue for different permit price and price targetting baseline combinations"
        
        # Business-as-usual price
        bau_price = self.get_national_bau_price()
        
        df = self.df_s.loc[self.df_s['mode'] == 'CalcPhi'].pivot(index='tau', columns='target_price', values='scheme_revenue')
        
        # Rename columns
        if relative:
            df = df.rename(columns={i : '{0:.1f}'.format(i / bau_price) for i in df.columns})
        else:
            df = df.rename(columns={i: '{0:.2f}'.format(i) for i in df.columns})
        
        # Plot data
        if ax:
            df.plot(cmap='tab10', style='o-', markersize=3.5, ax=ax)
        else:
            ax = df.plot(cmap='tab10', style='o-', markersize=3.5)
            
        fig = ax.get_figure()
        
        # Format labels
        if relative:
            ax.legend(title='Price target relative to BAU', ncol=5)
        else:
            ax.legend(title='Price target ', ncol=5)
        
        ax.set_xlabel('Permit price (\$/tCO${_2}$)')
        ax.set_ylabel('Scheme revenue (\$/h)')
        
        # Format axes
        ax.ticklabel_format(axis='y', useMathText=True, style='sci', scilimits=(1, 5))
        ax.minorticks_on()
        width = 180 * self.mmi
        height = 112.5 * self.mmi
        fig.subplots_adjust(left=0.09, bottom=0.1, right=0.98, top=0.95)
        fig.set_size_inches(width, height)

        # Save figure
        fig.savefig(filename)
        plt.show()
        
        return ax
        
    
    def emissions_vs_permit_price(self, filename, normalise=False):
        "Plot relationship between emissions intensity and permit price"
        
        # Scenario data
        df = self.df_s
        
        # Filter records
        df = df.loc[self.df_s['mode'] == 'CalcFixed']
        
        # Sort by permit price
        df = df.sort_values('tau')

        if normalise:
            # BAU emissions intensity
            mask_rows = df['tau'] == 0
            bau_emissions_intensity = df.loc[mask_rows, 'national_average_emissions_intensity'].values[0]      
            
            # Normalise emissions intensities relative to BAU scenario
            df['national_average_emissions_intensity_normalised'] = df['national_average_emissions_intensity'] / bau_emissions_intensity
            
            ax = df.plot(x='tau', y='national_average_emissions_intensity_normalised', marker='o', linestyle='-', color='#dd4949', markersize=3.5, label='')
            ax.set_ylabel('Average emissions intensity relative to BAU')
        
        else:
            # Absolute emissions intensity
            ax = df.plot(x='tau', y='national_average_emissions_intensity', marker='o', linestyle='-', color='#dd4949', markersize=3.5, label='')
            ax.set_ylabel('Average emissions intensity (tCO$_{2}$/MWh)')
        
        # Label x-axis
        ax.set_xlabel('Permit price (\$/tCO${_2}$)')
        fig = ax.get_figure()
               
        # Format axes
        ax.legend_.remove()
        ax.ticklabel_format(axis='y', useMathText=True)
        ax.minorticks_on()
        width = (180 * self.mmi)
        height = (112.5 * self.mmi)
        fig.subplots_adjust(left=0.09, bottom=0.11, right=0.98, top=0.97)
        fig.set_size_inches(width, height)

        # Save figure
        fig.savefig(filename)
        plt.show()
        return df
        
    
    def BAU_baseline_and_revenue(self, filename, normalise=False):
        "Plot BAU price targetting baseline and resulting scheme revenue for different permit prices"
        
        # National business-as-usual price
        bau_price = self.get_national_bau_price()
        
        # BAU emissions inensity
        mask = (self.df_s['mode'] == 'CalcFixed') & (self.df_s['tau'] == 0)
        bau_emissions_intensity = self.df_s.loc[mask, 'national_average_emissions_intensity'].values[0]
        
        # Price targetting baseline data
        df_bas = self.df_s.loc[self.df_s['mode'] == 'CalcPhi'].pivot(index='tau', columns='target_price', values='phi')[[bau_price]]

        # Scheme revenue data
        df_rev = self.df_s.loc[self.df_s['mode'] == 'CalcPhi'].pivot(index='tau', columns='target_price', values='scheme_revenue')[[bau_price]]
        
        # Rename columns
        if normalise:
            df_bas = df_bas.rename(columns={i : '{0:.1f}'.format(i / bau_price) for i in df_bas.columns}) / bau_emissions_intensity
            df_rev = df_rev.rename(columns={i : '{0:.1f}'.format(i / bau_price) for i in df_rev.columns})
        else:
            df_bas = df_bas.rename(columns={i: '{0:.2f}'.format(i) for i in df_bas.columns})
            df_rev = df_rev.rename(columns={i: '{0:.2f}'.format(i) for i in df_rev.columns})
            
        # Initialise figure
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        # Plot price targetting baseline for BAU price scenario
        ln_bas = df_bas.plot(marker='o', linestyle='-', color='#dd4949', markersize=3.5, ax=ax1)
        ln_bas.set_label('baseline')
        
        # Create second y-axis
        ax2 = ax1.twinx()
        ax2.ticklabel_format(axis='y', useMathText=True, style='sci', scilimits=(0, 1))
        
        # Plot scheme revenue on second y-axis for BAU price target scenario
        ln_rev = df_rev.plot(marker='o', linestyle='-', color='#4a63e0', markersize=3.5, ax=ax2)
        ln_rev.set_label('revenue')
        
        # Format axes and labels
        ax1.minorticks_on()
        ax1.set_xlabel('Permit price (\$/tCO$_{2}$)')
        
        if normalise:
            ax1.set_ylabel('Emissions intensity baseline relative to BAU emissions intensity')
        else:        
            ax1.set_ylabel('Emissions intensity baseline (tCO$_{2}$/MWh)')
        
        ax2.minorticks_on()
        ax2.set_ylabel('Scheme revenue (\$/h)')
        
        # Format legend
        ax2.legend_.remove()
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        l1 = ['Baseline']
        l2 = ['Revenue']
        
        ax1.legend(h1+h2, l1+l2, loc=0)
        
        # Line showing revenue neutrality
        break_even = ax2.plot([2, 80], [0,0], linestyle='--', color='#9899a0', label='break even')        

        # Format figure
        width = (180 * self.mmi)
        height = (112.5 * self.mmi)
        fig.subplots_adjust(left=0.09, bottom=0.1, right=0.9, top=0.95)
        fig.set_size_inches(width, height)

        # Save figure
        fig.savefig(filename)
        plt.show()
        
    def merit_order_emissions_intensity(self, filename):
        "Show how emissions intensity of merit order changes as permit price increases"
        
        # Only consider fossil units
        mask = self.df_g['FUEL_CAT'] == 'Fossil'
        df_gp = self.df_g[mask].copy()

        # Initialise figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Permit prices
        permit_prices = range(0, 81, 2)

        # Number of rectanlges
        n = len(permit_prices)

        # Gap as a fraction of rectangle height
        q = 1/10

        # Rectangle height
        rec_height = 1 / (q*(n-1) + n)

        # Gap between rectangles
        y_gap = rec_height * (1 + q)

        # Initial y offset
        y_offset = 0

        rects = []
        colours = []
        for p in permit_prices:
            # Compute update SRMC and sort from least cost to most expensive (merit order)
            df_gp['SRMC_EIS'] = df_g['SRMC_2016-17'] + p*df_g['EMISSIONS']
            df_gp.sort_values('SRMC_EIS', inplace=True)

            # Max and min emissions intensities used for scaling
            max_emint = df_gp['EMISSIONS'].max()
            min_emint = df_gp['EMISSIONS'].min()
            norm = matplotlib.colors.Normalize(vmin=min_emint, vmax=max_emint)

            # Mapping emissions intensities to colours intensities
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
            df_gp['RECT_COLOUR'] = df_gp['EMISSIONS'].map(lambda x: mapper.to_rgba(x))

            # Normalising registered capacities
            df_gp['REG_CAP_NORM'] = (df_gp['REG_CAP'] / df_gp['REG_CAP'].sum())

            x_offset = 0
            # Plotting rectangles
            for index, row in df_gp.iterrows():
                rects.append(patches.Rectangle((x_offset, y_offset), row['REG_CAP_NORM'], rec_height))
                colours.append(row['EMISSIONS'])

                x_offset += row['REG_CAP_NORM']

            y_offset += y_gap

        p = PatchCollection(rects, cmap='Reds')
        p.set_array(np.array(colours))
        ax.add_collection(p)
        fig.colorbar(p, ax=ax, pad=0.015, aspect=18, label='Emissions intensity (tCO${_2}$/MWh)')

        # Format ticks
        rel_dec = 4
        ytick_locs = [rec_height/2 + i*(rec_height * (1 + q)) for i in range(0, n)]
        ytick_locs = [j for i, j in enumerate(ytick_locs) if i % rel_dec == 0]

        ytick_labels = list(permit_prices)
        ytick_labels = [j for i, j in enumerate(ytick_labels) if i % rel_dec == 0]

        ax.set_ylabel('Permit price (\$/tCO$_{2}$)')
        ax.set_xlabel('Normalised cumulative capacity arranged by SRMC')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        plt.yticks(ytick_locs, ytick_labels, fontname='Helvetica')

        # Format figure and save
        width = (180 * self.mmi)
        height = (90 * self.mmi)
        fig.subplots_adjust(left=0.11, bottom=0.15, right=1, top=0.95)
        fig.set_size_inches(width, height)

        # Save figure
        fig.savefig(filename)
        plt.show()
        
    def merit_order_prices(self, filename, metric, cmap='bwr', tax=False):
        "Show how prices change for different permit price combinations"
        
        # Business-as-usual price
        bau_price = self.get_national_bau_price()

        # Only consider fossil units
        mask = self.df_g['FUEL_CAT'] == 'Fossil'
        df_gp = self.df_g[mask].copy()

        # Initialise figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Alternative scenario construction procedure
        mask = (self.df_s['mode'] == 'CalcPhi') & (self.df_s['target_price'] == bau_price)
        df_scenarios = self.df_s.loc[mask, ['tau', 'phi']]
        df_scenarios.sort_values('tau', inplace=True)
        df_scenarios        
        
        # Number of rectanlges
        n = len(df_scenarios)

        # Gap as a fraction of rectangle height
        q = 1/10

        # Rectangle height
        rec_height = 1 / (q*(n-1) + n)

        # Gap between rectangles
        y_gap = rec_height * (1 + q)

        # Initial y offset
        y_offset = 0

        rects = []
        colours = []

        for index, row in df_scenarios.iterrows():
            p = row['tau'] # permit price
            
            if tax:
                baseline = 0
            else:
                baseline = row['phi'] # baseline
            
            # Compute net liability under scheme and update SRMCs
            df_gp['NET_LIABILITY'] = p * (self.df_g['EMISSIONS'] - baseline)
            df_gp['SRMC_EIS'] = self.df_g['SRMC_2016-17'] + df_gp['NET_LIABILITY']
            
            # Sort from least cost to most expensive (merit order)
            df_gp.sort_values('SRMC_EIS', inplace=True)

            # Metric to display. Either SRMC or net liability
            if metric not in ['SRMC_EIS', 'NET_LIABILITY']:
                raise(Exception('Metric must be in [SRMC_EIS, NET_LIABILITY]'))
                        
            # Max and min values used for scaling
            max_emint = df_gp[metric].max()
            min_emint = df_gp[metric].min()
            norm = matplotlib.colors.Normalize(vmin=min_emint, vmax=max_emint)

            # Normalising registered capacities
            df_gp['REG_CAP_NORM'] = (df_gp['REG_CAP'] / df_gp['REG_CAP'].sum())
           
            x_offset = 0
            # Plotting rectangles
            for index, row in df_gp.iterrows():
                rects.append(patches.Rectangle((x_offset, y_offset), row['REG_CAP_NORM'], rec_height))
                
                # Append colour for patch
                colours.append(row[metric])

                x_offset += row['REG_CAP_NORM']

            y_offset += y_gap

        p = PatchCollection(rects, cmap=cmap)
        p.set_array(np.array(colours))
        ax.add_collection(p)
        
        
        if metric is 'NET_LIABILITY':
            cbar = fig.colorbar(p, ax=ax, pad=0.015, aspect=18, label='Net liability (\$/MWh)')
        if metric is 'SRMC_EIS':
            cbar = fig.colorbar(p, ax=ax, pad=0.015, aspect=18, label='SRMC (\$/MWh)')
       
        # Format ticks
        rel_dec = 2
        ytick_locs = [rec_height/2 + i*(rec_height * (1 + q)) for i in range(0, n)]
        ytick_locs = [j for i, j in enumerate(ytick_locs) if i % rel_dec == 0]

        # Permit price ticks
        ytick_labels = list(df_scenarios['tau'])
        ytick_labels = [j for i, j in enumerate(ytick_labels) if i % rel_dec == 0]

        # Format labels
        ax.set_ylabel('Permit price (\$/tCO$_{2}$)')
        ax.set_xlabel('Normalised cumulative capacity arranged by SRMC')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.yticks(ytick_locs, ytick_labels, fontname='Helvetica')

        # Format figure and save
        width = (180 * self.mmi)
        height = (90 * self.mmi)
        fig.subplots_adjust(left=0.11, bottom=0.15, right=1, top=0.95)
        fig.set_size_inches(width, height)

        # Save figure
        fig.savefig(filename)

        plt.show()
        
        return df_gp
    
    def get_BAU_regional_price_series(self):
        "Extract series containing BAU prices for each NEM region"
        df = self.df_s

        # Get BAU price series
        mask_rows = (df['mode'] == 'CalcFixed') & (df['tau'] == 0)
        mask_cols = df.columns.str.contains('regional_average_price_')
        bau_price_series = df.loc[mask_rows, mask_cols].T[0]
        
        return bau_price_series
    
    # Normalise regional prices relative to BAU scenario
    @staticmethod
    def normalise_prices(row, bau_price_series):
        "Normalise prices relative to BAU scenario"
        return row / bau_price_series
    
    
    def regional_prices(self, filename, normalise=False):
        "Plot prices for each NEM region"       
        
        # Set price target to business-as-usual price
        price_target = self.get_national_bau_price()
        
        # Contains model results
        df = self.df_s
        
        # Get absolute regional prices for different permit prices when targetting average national BAU price
        mask_cols = df.columns.str.contains('regional_average_price_') | df.columns.str.contains('tau')
        mask_rows = (df['mode'] == 'CalcPhi') & (df['target_price'] == price_target)
        df = df.loc[mask_rows, mask_cols].sort_values('tau').set_index('tau')

        # Rename columns
        new_col_names = {i: i.split('_')[-1].replace('1','') for i in df.columns}
        
        # BAU prices for each region
        bau_price_series = self.get_BAU_regional_price_series()
        
        if normalise:
            # Prices relative to BAU
            ax = df.apply(self.normalise_prices, args=(bau_price_series,), axis=1).rename(columns=new_col_names).plot(marker='o', linestyle='-', markersize=3.5, cmap='tab10')
            ax.set_ylabel('Average regional wholesale price relative to BAU')
        
        else:
            # Absolute prices
            ax = df.rename(columns=new_col_names).plot(marker='o', linestyle='-', markersize=3.5, cmap='tab10')
            ax.set_ylabel('Average regional wholesale price (\$/MWh)')
            
        fig = ax.get_figure()           
        ax.set_xlabel('Permit price (\$/tCO${_2}$)')
        
        # Format axes
        ax.minorticks_on()
        
        # Format figure
        width = (180 * self.mmi)
        height = (112.5 * self.mmi)
        fig.subplots_adjust(left=0.09, bottom=0.1, right=0.95, top=0.95)
        fig.set_size_inches(width, height)

        # Save figure
        fig.savefig(filename)
        plt.show()
        
    def regional_prices_tax(self, filename, normalise=False):
        "Regional prices when system subjected to carbon tax (no refunding)"
        
        # Scenario data
        df = mppdc_plots.df_s
        
        # Filter records
        mask_rows = df['mode'] == 'CalcFixed'
        mask_cols = df.columns.str.contains('regional_average_price_') | df.columns.str.contains('tau')
        df = df.loc[mask_rows, mask_cols].sort_values('tau').set_index('tau')
        
        # New column names
        new_col_names = {i: i.split('_')[-1].replace('1','') for i in df.columns}
        
        # Normalise prices relative to BAU scenario
        if normalise:
            # BAU prices for each NEM region
            bau_price_series = self.get_BAU_regional_price_series()
            
            # Normalise prices relative to BAU prices for each region
            ax = df.apply(self.normalise_prices, args=(bau_price_series,), axis=1).rename(columns=new_col_names).plot(marker='o', linestyle='-', markersize=3.5, cmap='tab10')
            ax.set_ylabel('Average regional wholesale price relative to BAU')
        else:
            # Plot absolute prices under a carbon tax
            ax = df.rename(columns=new_col_names).plot(marker='o', linestyle='-', markersize=3.5, cmap='tab10')
            ax.set_ylabel('Regional wholesale price (\$/MWh)')
            
        fig = ax.get_figure()
        ax.set_xlabel('Permit price (\$/tCO${_2}$)')
    
        # Format axes
        ax.minorticks_on()
        
        # Format figure
        width = (180 * self.mmi)
        height = (112.5 * self.mmi)
        fig.subplots_adjust(left=0.09, bottom=0.1, right=0.95, top=0.95)
        fig.set_size_inches(width, height)

        # Save figure
        fig.savefig(filename)
        plt.show()


# Instantiate class from MPPDC scenario summary data
# --------------------------------------------------
f = os.path.join(processed_results_dir, 'output', 'mppdc_summary.pickle')
mppdc_plots = PlotMPPDC(f, df_g)


# Create plots
# ------------
# # Price targetting baselines vs permit price - absolute prices and emissions intensities
# f = os.path.join(output_dir, 'figures', 'permit_price_vs_baseline_absolute.pdf')
# mppdc_plots.price_targeting_baselines(f)

# # Price targetting baselines vs permit price - prices normalised to BAU scenario
# f = os.path.join(output_dir, 'figures', 'permit_price_vs_baseline_normalised.pdf')
# df = mppdc_plots.price_targeting_baselines(f, normalise=True)

# # Scheme revenue for different price targetting baselines
# f = os.path.join(output_dir, 'figures', 'permit_price_vs_scheme_revenue.pdf')
# ax = mppdc_plots.scheme_revenue(f, relative=True)

# # Emissions intensity as a function of permit price - emissions intensities relative to BAU
# f = os.path.join(output_dir, 'figures', 'permit_price_vs_emissions_intensity_normalised.pdf')
# df = mppdc_plots.emissions_vs_permit_price(f, normalise=True)

# # Emissions intensity as a function of permit price - absolute emissions intensities
# f = os.path.join(output_dir, 'figures', 'permit_price_vs_emissions_intensity_absolute.pdf')
# df = mppdc_plots.emissions_vs_permit_price(f)

# # Scheme revenue and corresponding price targetting baseline for the BAU price target scenario - absolute baselines
# f = os.path.join(output_dir, 'figures', 'bau_revenue_and_baseline_absolute.pdf')
# mppdc_plots.BAU_baseline_and_revenue(f)

# # Scheme revenue and corresponding price targetting baseline for the BAU price target scenario - baselines normalised to BAU emissions intensity
# f = os.path.join(output_dir, 'figures', 'bau_revenue_and_baseline_normalised.pdf')
# mppdc_plots.BAU_baseline_and_revenue(f, normalise=True)

# # Shows how emissions intensity of merit order is affected when permit prices increase
# f = os.path.join(output_dir, 'figures', 'merit_order_emissions_intensity.pdf')
# mppdc_plots.merit_order_emissions_intensity(f)

# # Net liability faced by generators arranged by merit order
# f = os.path.join(output_dir, 'figures', 'merit_order_net_liability.pdf')
# df = mppdc_plots.merit_order_prices(f, metric='NET_LIABILITY', cmap='bwr', tax=False)

# # SRMCs when targetting BAU prices for different permit prices
# f = os.path.join(output_dir, 'figures', 'merit_order_srmc_bau_price_target.pdf')
# df = mppdc_plots.merit_order_prices(f, metric='SRMC_EIS', cmap='Reds', tax=False)

# # Carbon tax scenario (fix baseline to 0)
# f = os.path.join(output_dir, 'figures', 'merit_order_srmc_tax.pdf')
# df = mppdc_plots.merit_order_prices(f, metric='SRMC_EIS', cmap='Reds', tax=True)

# # Regional prices - relative to BAU scenario
# f = os.path.join(output_dir, 'figures', 'regional_prices_normalised.pdf')
# mppdc_plots.regional_prices(f, normalise=True)

# # Regional prices - absolute
# f = os.path.join(output_dir, 'figures', 'regional_prices_absolute.pdf')
# mppdc_plots.regional_prices(f)

# # Regional prices due to carbon tax - relative to BAU regional prices
# f = os.path.join(output_dir, 'figures', 'regional_prices_carbon_tax_normalised.pdf')
# mppdc_plots.regional_prices_tax(f, normalise=True)

# # Regional prices due to carbon tax - absolute
# f = os.path.join(output_dir, 'figures', 'regional_prices_carbon_tax_absolute.pdf')

# mppdc_plots.regional_prices_tax(f)


# ## Manuscript plots
# 
# ### Price targeting baselines and corresponding scheme revenue

# In[20]:


# Figure on which to construct plots
fig = plt.figure()

# Axes on which to construct plots
ax1 = plt.axes([0.08, 0.175, 0.41, 0.77])
ax2 = plt.axes([0.585, 0.175, 0.41, 0.77])

# Business-as-usual price
bau_price = mppdc_plots.get_national_bau_price()


# Price targeting baselines
# -------------------------
# Scenario data
df_baseline = mppdc_plots.df_s.loc[mppdc_plots.df_s['mode'] == 'CalcPhi'].pivot(index='tau', columns='target_price', values='phi')

# BAU emissions inensity
mask = (mppdc_plots.df_s['mode'] == 'CalcFixed') & (mppdc_plots.df_s['tau'] == 0)
bau_emissions_intensity = mppdc_plots.df_s.loc[mask, 'national_average_emissions_intensity'].values[0]

# Rename columns
df_baseline = df_baseline.rename(columns={i : '{0:.1f}'.format(i / bau_price) for i in df_baseline.columns})
df_baseline = df_baseline / bau_emissions_intensity

# Map price targets to colours
price_target_colours = ['#b50e43', '#af92cc', '#45a564', '#a59845', '#f27b2b']
price_target_colour_map = {price_target: colour for price_target, colour in zip(df_baseline.columns.tolist(), price_target_colours)}

# Plotting price targeting baselines
for col in df_baseline.columns:
    ax1.plot(df_baseline[col], '-x', markersize=1.5, linewidth=0.9, label=col, color=price_target_colour_map[col])

# Label axes
ax1.set_ylabel('Emissions intensity baseline\nrelative to BAU', fontsize=9)
ax1.set_xlabel('Permit price (\$/tCO${_2}$)\n(a)', fontsize=9)

# Format axes
ax1.xaxis.set_major_locator(MultipleLocator(10))
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

fig.canvas.draw()
ax1.yaxis.minorTicks[1].set_visible(False)
ax1.xaxis.minorTicks[1].set_visible(False)
ax1.xaxis.minorTicks[-2].set_visible(False)


# Scheme revenue
# --------------
# Data to plot
df_revenue = mppdc_plots.df_s.loc[mppdc_plots.df_s['mode'] == 'CalcPhi'].pivot(index='tau', columns='target_price', values='scheme_revenue')

# Rename columns
df_revenue = df_revenue.rename(columns={i : '{0:.1f}'.format(i / bau_price) for i in df_revenue.columns})

# Plot data
for col in df_revenue.columns:
    ax2.plot(df_revenue[col], '-x', markersize=1.5, linewidth=0.9, label=col, color=price_target_colour_map[col])

# Label axes
ax2.set_xlabel('Permit price (\$/tCO${_2}$)\n(b)', fontsize=9)
ax2.set_ylabel('Scheme revenue (\$/h)', labelpad=0, fontsize=9)
        
# Format axes
ax2.ticklabel_format(axis='y', useMathText=True, style='sci', scilimits=(1, 5))
ax2.xaxis.set_major_locator(MultipleLocator(10))
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax2.yaxis.set_minor_locator(MultipleLocator(10000))

fig.canvas.draw()
ax2.xaxis.minorTicks[1].set_visible(False)
ax2.xaxis.minorTicks[-2].set_visible(False)

# Create legend
legend = ax2.legend(title='Price target\nrelative to BAU', ncol=1, loc='upper center', bbox_to_anchor=(-0.41, 1.), fontsize=9)
legend.get_title().set_fontsize('9')

# Format figure size
mmi = 0.0393701
fig = ax2.get_figure()
width = 180 * mmi
height = 80 * mmi
fig.set_size_inches(width, height)

# Save figure
# fig.savefig('test.pdf')
fig.savefig(os.path.join(output_dir, 'figures', 'baseline_revenue_subplot.pdf'))
plt.show()


# ### Emissions as a function of permit price

# In[7]:


plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

# Scenario data
df = mppdc_plots.df_s

# Filter records
df = df.loc[mppdc_plots.df_s['mode'] == 'CalcFixed']

# Sort by permit price
df = df.sort_values('tau')

# BAU emissions intensity
mask_rows = df['tau'] == 0
bau_emissions_intensity = df.loc[mask_rows, 'national_average_emissions_intensity'].values[0]      

# Normalise emissions intensities relative to BAU scenario
df['national_average_emissions_intensity_normalised'] = df['national_average_emissions_intensity'] / bau_emissions_intensity

# Plot data
ax.plot(df['tau'].tolist(), df['national_average_emissions_intensity_normalised'].tolist(), '-x', markersize=2, linewidth=0.8, color='#c11111')
ax.set_ylabel('Emissions intensity\nrelative to BAU', fontsize=9)

# Label x-axis
ax.set_xlabel('Permit price (\$/tCO${_2}$)', fontsize=9)

# Format axes
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.005))
fig.canvas.draw()
ax.xaxis.minorTicks[1].set_visible(False)
ax.xaxis.minorTicks[-2].set_visible(False)

# Format figure
width = 85 * mmi
height = width / 1.2
fig.subplots_adjust(left=0.2, bottom=0.14, right=.98, top=0.98)
fig.set_size_inches(width, height)

fig.savefig(os.path.join(output_dir, 'figures', 'permit_price_vs_emissions_intensity_normalised.pdf'))

plt.show()


# ## Merit order plots
# ### Emissions intensity and net liability

# In[8]:


# %matplotlib tk
# Emissions intensity
# -------------------
ax = 0
plt.clf()

# Initialise figure
fig = plt.figure()

# Axes on which to construct plots
ax1 = plt.axes([0.065, 0.185, 0.41, .79])
ax2 = plt.axes([0.565, 0.185, 0.41, .79])

# Only consider fossil units
mask = mppdc_plots.df_g['FUEL_CAT'] == 'Fossil'
df_gp = mppdc_plots.df_g[mask].copy()

# Permit prices
permit_prices = range(2, 71, 2)

# Number of rectanlges
n = len(permit_prices)

# Gap as a fraction of rectangle height
q = 1 / 10

# Rectangle height
rec_height = 1 / (q * (n - 1) + n)

# Gap between rectangles
y_gap = rec_height * q

# Initial y offset
y_offset = 0

rects = []
colours = []
for p in permit_prices:
    # Compute update SRMC and sort from least cost to most expensive (merit order)
    df_gp['SRMC_EIS'] = df_g['SRMC_2016-17'] + p*df_g['EMISSIONS']
    df_gp.sort_values('SRMC_EIS', inplace=True)

    # Max and min emissions intensities used for scaling
    max_emint = df_gp['EMISSIONS'].max()
    min_emint = df_gp['EMISSIONS'].min()
    norm = matplotlib.colors.Normalize(vmin=min_emint, vmax=max_emint)

    # Mapping emissions intensities to colours intensities
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    df_gp['RECT_COLOUR'] = df_gp['EMISSIONS'].map(lambda x: mapper.to_rgba(x))

    # Normalising registered capacities
    df_gp['REG_CAP_NORM'] = (df_gp['REG_CAP'] / df_gp['REG_CAP'].sum())

    x_offset = 0
    # Plotting rectangles
    for index, row in df_gp.iterrows():
        rects.append(patches.Rectangle((x_offset, y_offset), row['REG_CAP_NORM'], rec_height))
        colours.append(row['EMISSIONS'])

        x_offset += row['REG_CAP_NORM']

    y_offset += rec_height + y_gap

p = PatchCollection(rects, cmap='Reds')
p.set_array(np.array(colours))
ax1.add_collection(p)
cbar1 = fig.colorbar(p, ax=ax1, pad=0.015, aspect=30)
cbar1.set_label('Emissions intensity (tCO${_2}$/MWh)', fontsize=8, fontname='Helvetica')

# Format ticks
ax1.set_ylabel('Permit price (\$/tCO$_{2}$)', fontsize=9, fontname='Helvetica')
ax1.set_xlabel('Normalised cumulative capacity\n(a)', fontsize=9, fontname='Helvetica')
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

first = (4 * rec_height) + (4 * rec_height * q) + (rec_height / 2)
second = (4 * rec_height) + rec_height + (5 * rec_height * q)
third = second + second
majors = [first, first + second]
minors = np.linspace(0, 1, 71)[1:-1]

for i in range(0, 9):
    majors.append(majors[-1] + second)

ax1.yaxis.set_major_locator(FixedLocator(majors))
ax1.yaxis.set_minor_locator(FixedLocator(minors))
ax1.yaxis.set_ticklabels(['10', '20', '30', '40', '50', '60', '70'])

fig.canvas.draw()
for index, tick in enumerate(ax1.yaxis.get_minor_ticks()):
    if (index + 1) % 2 == 0:
        tick.set_visible(False)
    elif index in [8, 18, 28, 38, 48, 58, 68]:
        tick.set_visible(False)


# Net liability
# -------------
# Business-as-usual price
bau_price = mppdc_plots.get_national_bau_price()

# Only consider fossil units
mask = mppdc_plots.df_g['FUEL_CAT'] == 'Fossil'
df_gp = mppdc_plots.df_g[mask].copy()

# Alternative scenario construction procedure
mask = (mppdc_plots.df_s['mode'] == 'CalcPhi') & (mppdc_plots.df_s['target_price'] == bau_price)
df_scenarios = mppdc_plots.df_s.loc[mask, ['tau', 'phi']]
df_scenarios.sort_values('tau', inplace=True)
df_scenarios        

# Number of rectanlges
n = len(df_scenarios)

# Gap as a fraction of rectangle height
q = 1 / 10

# Rectangle height
rec_height = 1 / (q * (n - 1) + n)

# Gap between rectangles
y_gap = rec_height * (1 + q)

# Initial y offset
y_offset = 0

rects = []
colours = []
for index, row in df_scenarios.iterrows():
    p = row['tau'] # permit price
    baseline = row['phi'] # baseline

    # Compute net liability under scheme and update SRMCs
    df_gp['NET_LIABILITY'] = p * (mppdc_plots.df_g['EMISSIONS'] - baseline)
    df_gp['SRMC_EIS'] = mppdc_plots.df_g['SRMC_2016-17'] + df_gp['NET_LIABILITY']

    # Sort from least cost to most expensive (merit order)
    df_gp.sort_values('SRMC_EIS', inplace=True)

    # Max and min values used for scaling
    max_emint = df_gp['NET_LIABILITY'].max()
    min_emint = df_gp['NET_LIABILITY'].min()
    norm = matplotlib.colors.Normalize(vmin=min_emint, vmax=max_emint)

    # Normalising registered capacities
    df_gp['REG_CAP_NORM'] = (df_gp['REG_CAP'] / df_gp['REG_CAP'].sum())

    x_offset = 0
    # Plotting rectangles
    for index, row in df_gp.iterrows():
        rects.append(patches.Rectangle((x_offset, y_offset), row['REG_CAP_NORM'], rec_height))

        # Append colour for patch
        colours.append(row['NET_LIABILITY'])

        x_offset += row['REG_CAP_NORM']
    y_offset += y_gap

p = PatchCollection(rects, cmap='bwr')
p.set_array(np.array(colours))
ax2.add_collection(p)

# if metric is 'NET_LIABILITY':
cbar = fig.colorbar(p, ax=ax2, pad=0.015, aspect=30)
cbar.set_clim(-40, 40)
cbar.set_label('Net liability (\$/MWh)', fontsize=8, fontname='Helvetica')

# Format ticks
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.yaxis.set_major_locator(FixedLocator(majors))
ax2.yaxis.set_minor_locator(FixedLocator(minors))
ax2.yaxis.set_ticklabels(['10', '20', '30', '40', '50', '60', '70'])

fig.canvas.draw()
for index, tick in enumerate(ax2.yaxis.get_minor_ticks()):
    if (index + 1) % 2 == 0:
        tick.set_visible(False)
    elif index in [8, 18, 28, 38, 48, 58, 68]:
        tick.set_visible(False)

# # Format labels
ax2.set_ylabel('Permit price (\$/tCO$_{2}$)', fontsize=9, fontname='Helvetica')
ax2.set_xlabel('Normalised cumulative capacity\n(b)', fontsize=9, fontname='Helvetica')

# # Format figure and save
width = 180 * mmi
height = 75 * mmi
fig.set_size_inches(width, height)

# Save figure
# fig.savefig('test.pdf')
fig.savefig(os.path.join(output_dir, 'figures', 'emissions_liability_merit_order.pdf'))
plt.show()


# ### SRMCs under REP scheme and carbon tax scenarios

# In[9]:


# Emissions intensity
# -------------------
ax = 0
plt.clf()

# Initialise figure
fig = plt.figure()

# Axes on which to construct plots
ax1 = plt.axes([0.068, 0.185, 0.45, .79])
ax2 = plt.axes([0.535, 0.185, 0.45, .79])

# Net liability
# -------------
# Business-as-usual price
bau_price = mppdc_plots.get_national_bau_price()

# Only consider fossil units
mask = mppdc_plots.df_g['FUEL_CAT'] == 'Fossil'
df_gp = mppdc_plots.df_g[mask].copy()

# Alternative scenario construction procedure
mask = (mppdc_plots.df_s['mode'] == 'CalcPhi') & (mppdc_plots.df_s['target_price'] == bau_price)
df_scenarios = mppdc_plots.df_s.loc[mask, ['tau', 'phi']]
df_scenarios.sort_values('tau', inplace=True)
df_scenarios        

# Number of rectanlges
n = len(df_scenarios)

# Gap as a fraction of rectangle height
q = 1 / 10

# Rectangle height
rec_height = 1 / (q * (n - 1) + n)

# Gap between rectangles
y_gap = rec_height * (1 + q)

# Initial y offset
y_offset = 0

rects = []
colours = []
for index, row in df_scenarios.iterrows():
    p = row['tau'] # permit price
    baseline = row['phi'] # baseline

    # Compute net liability under scheme and update SRMCs
    df_gp['NET_LIABILITY'] = p * (mppdc_plots.df_g['EMISSIONS'] - baseline)
    df_gp['SRMC_EIS'] = mppdc_plots.df_g['SRMC_2016-17'] + df_gp['NET_LIABILITY']

    # Sort from least cost to most expensive (merit order)
    df_gp.sort_values('SRMC_EIS', inplace=True)

    # Max and min values used for scaling
    max_emint = df_gp['SRMC_EIS'].max()
    min_emint = df_gp['SRMC_EIS'].min()
    norm = matplotlib.colors.Normalize(vmin=min_emint, vmax=max_emint)

    # Normalising registered capacities
    df_gp['REG_CAP_NORM'] = (df_gp['REG_CAP'] / df_gp['REG_CAP'].sum())

    x_offset = 0
    # Plotting rectangles
    for index, row in df_gp.iterrows():
        rects.append(patches.Rectangle((x_offset, y_offset), row['REG_CAP_NORM'], rec_height))

        # Append colour for patch
        colours.append(row['SRMC_EIS'])

        x_offset += row['REG_CAP_NORM']
    y_offset += y_gap

p = PatchCollection(rects, cmap='Reds')
p.set_array(np.array(colours))
ax1.add_collection(p)

cbar1 = fig.colorbar(p, ax=ax1, pad=0.015, aspect=30)
cbar1.set_clim(20, 225)
cbar1.remove()

# Format ticks
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax1.yaxis.set_major_locator(FixedLocator(majors))
ax1.yaxis.set_minor_locator(FixedLocator(minors))
ax1.yaxis.set_ticklabels(['10', '20', '30', '40', '50', '60', '70'])

fig.canvas.draw()
for index, tick in enumerate(ax1.yaxis.get_minor_ticks()):
    if (index + 1) % 2 == 0:
        tick.set_visible(False)
    elif index in [8, 18, 28, 38, 48, 58, 68]:
        tick.set_visible(False)

# Format labels
ax1.set_ylabel('Permit price (\$/tCO$_{2}$)', fontsize=9, fontname='Helvetica')
ax1.set_xlabel('Normalised cumulative capacity\n(a)', fontsize=9, fontname='Helvetica')


# Carbon tax
# ----------
# Only consider fossil units
mask = mppdc_plots.df_g['FUEL_CAT'] == 'Fossil'
df_gp = mppdc_plots.df_g[mask].copy()

# Permit prices
permit_prices = range(2, 71, 2)

# Number of rectanlges
n = len(permit_prices)

# Gap as a fraction of rectangle height
q = 1 / 10

# Rectangle height
rec_height = 1 / (q * (n - 1) + n)

# Gap between rectangles
y_gap = rec_height * q

# Initial y offset
y_offset = 0

rects = []
colours = []
for p in permit_prices:
    # Compute update SRMC and sort from least cost to most expensive (merit order)
    df_gp['SRMC_EIS'] = df_g['SRMC_2016-17'] + p*df_g['EMISSIONS']
    df_gp.sort_values('SRMC_EIS', inplace=True)

    # Max and min emissions intensities used for scaling
    max_emint = df_gp['SRMC_EIS'].max()
    min_emint = df_gp['SRMC_EIS'].min()
    norm = matplotlib.colors.Normalize(vmin=min_emint, vmax=max_emint)

    # Mapping emissions intensities to colours intensities
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    df_gp['RECT_COLOUR'] = df_gp['SRMC_EIS'].map(lambda x: mapper.to_rgba(x))

    # Normalising registered capacities
    df_gp['REG_CAP_NORM'] = (df_gp['REG_CAP'] / df_gp['REG_CAP'].sum())

    x_offset = 0
    # Plotting rectangles
    for index, row in df_gp.iterrows():
        rects.append(patches.Rectangle((x_offset, y_offset), row['REG_CAP_NORM'], rec_height))
        colours.append(row['SRMC_EIS'])

        x_offset += row['REG_CAP_NORM']

    y_offset += rec_height + y_gap

p = PatchCollection(rects, cmap='Reds')
p.set_array(np.array(colours))
ax2.add_collection(p)
cbar2 = fig.colorbar(p, ax=ax2, pad=0.015, aspect=30)
cbar2.set_clim(20, 225)
cbar2.set_label('SRMC (\$/MWh)', fontsize=8, fontname='Helvetica')

# Format ticks
ax2.set_ylabel('Permit price (\$/tCO$_{2}$)', fontsize=9, fontname='Helvetica')
ax2.set_xlabel('Normalised cumulative capacity\n(b)', fontsize=9, fontname='Helvetica')
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))

first = (4 * rec_height) + (4 * rec_height * q) + (rec_height / 2)
second = (4 * rec_height) + rec_height + (5 * rec_height * q)
third = second + second
majors = [first, first + second]
minors = np.linspace(0, 1, 71)[1:-1]

for i in range(0, 9):
    majors.append(majors[-1] + second)

ax2.yaxis.set_major_locator(FixedLocator(majors))
ax2.yaxis.set_minor_locator(FixedLocator(minors))
ax2.yaxis.set_ticklabels(['10', '20', '30', '40', '50', '60', '70'])

fig.canvas.draw()
for index, tick in enumerate(ax2.yaxis.get_minor_ticks()):
    if (index + 1) % 2 == 0:
        tick.set_visible(False)
    elif index in [8, 18, 28, 38, 48, 58, 68]:
        tick.set_visible(False)

# Format figure and save
width = 180 * mmi
height = 75 * mmi
fig.set_size_inches(width, height)

# Save figure
# fig.savefig('test.pdf')
fig.savefig(os.path.join(output_dir, 'figures', 'srmc_merit_order.pdf'))
plt.show()


# ### Regional wholesale prices
# Plot prices for each NEM region

# In[10]:


plt.clf()

fig = plt.figure()
ax1 = plt.axes([0.068, 0.17, 0.41, 0.8])
ax2 = plt.axes([0.58, 0.17, 0.41, 0.8])

# Set price target to business-as-usual price
price_target = mppdc_plots.get_national_bau_price()

# Regional prices under price targetting policy
# ---------------------------------------------
# Contains model results
df = mppdc_plots.df_s

# Get absolute regional prices for different permit prices when targetting average national BAU price
mask_cols = df.columns.str.contains('regional_average_price_') | df.columns.str.contains('tau')
mask_rows = (df['mode'] == 'CalcPhi') & (df['target_price'] == price_target)
df = df.loc[mask_rows, mask_cols].sort_values('tau').set_index('tau')

# Rename columns
new_col_names = {i: i.split('_')[-1].replace('1','') for i in df.columns}

# BAU prices for each region
bau_price_series = mppdc_plots.get_BAU_regional_price_series()

# Absolute prices
df.rename(columns=new_col_names).plot(marker='o', linestyle='-', markersize=1.5, linewidth=1, cmap='tab10', ax=ax1)
ax1.set_ylabel('Average wholesale price (\$/MWh)', fontsize=9)     
ax1.set_xlabel('Permit price (\$/tCO${_2}$)\n(a)', fontsize=9)

# Format axes
ax1.minorticks_on()

ax1.xaxis.set_major_locator(MultipleLocator(10))
ax1.xaxis.set_minor_locator(MultipleLocator(2))

fig.canvas.draw()
ax1.xaxis.minorTicks[1].set_visible(False)
ax1.xaxis.minorTicks[-2].set_visible(False)
legend1 = ax1.legend()
legend1.remove()


# Regional prices under carbon tax
# --------------------------------
# Scenario data
df = mppdc_plots.df_s

# Filter records
mask_rows = df['mode'] == 'CalcFixed'
mask_cols = df.columns.str.contains('regional_average_price_') | df.columns.str.contains('tau')
df = df.loc[mask_rows, mask_cols].sort_values('tau').set_index('tau')

# New column names
new_col_names = {i: i.split('_')[-1].replace('1','') for i in df.columns}

# Plot absolute prices under a carbon tax
df.rename(columns=new_col_names).plot(marker='o', linestyle='-', markersize=1.5, linewidth=1, cmap='tab10', ax=ax2)
ax2.set_ylabel('Average wholesale price (\$/MWh)', fontsize=9)
ax2.set_xlabel('Permit price (\$/tCO${_2}$)\n(b)', fontsize=9)
ax2.minorticks_on()
legend2 = ax2.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.7, 0.28), fontsize=9)


ax2.xaxis.set_major_locator(MultipleLocator(10))
ax2.xaxis.set_minor_locator(MultipleLocator(2))

fig.canvas.draw()
ax2.xaxis.minorTicks[1].set_visible(False)
ax2.xaxis.minorTicks[-2].set_visible(False)

# Format figure
width = 180 * mmi
height = 80 * mmi
fig.set_size_inches(width, height)

# Save figure
# fig.savefig('test.pdf')
fig.savefig(os.path.join(output_dir, 'figures', 'regional_wholesale_prices.pdf'))
plt.show()


# ### BAU price targetting baselines scheme revenue
# Plot BAU price targetting baseline and resulting scheme revenue for different permit prices

# In[11]:


plt.clf()
# BAU emissions inensity
mask = (mppdc_plots.df_s['mode'] == 'CalcFixed') & (mppdc_plots.df_s['tau'] == 0)
bau_emissions_intensity = mppdc_plots.df_s.loc[mask, 'national_average_emissions_intensity'].values[0]

# Price targetting baseline data
df_bas = mppdc_plots.df_s.loc[mppdc_plots.df_s['mode'] == 'CalcPhi'].pivot(index='tau', columns='target_price', values='phi')[[bau_price]]

# Scheme revenue data
df_rev = mppdc_plots.df_s.loc[mppdc_plots.df_s['mode'] == 'CalcPhi'].pivot(index='tau', columns='target_price', values='scheme_revenue')[[bau_price]]

# Rename columns
df_bas = df_bas.rename(columns={i : '{0:.1f}'.format(i / bau_price) for i in df_bas.columns}) / bau_emissions_intensity
df_rev = df_rev.rename(columns={i : '{0:.1f}'.format(i / bau_price) for i in df_rev.columns})

# Initialise figure
fig = plt.figure()
ax1 = fig.add_subplot(111)

# Plot price targetting baseline for BAU price scenario
ln_bas = df_bas.plot(marker='o', linestyle='-', color='#dd4949', markersize=1.5, ax=ax1, linewidth=1)
ln_bas.set_label('baseline')

# Create second y-axis
ax2 = ax1.twinx()
ax2.ticklabel_format(axis='y', useMathText=True, style='sci', scilimits=(0, 1))

# Plot scheme revenue on second y-axis for BAU price target scenario
ln_rev = df_rev.plot(marker='o', linestyle='-', color='#4a63e0', markersize=1.5, ax=ax2, linewidth=1)
ln_rev.set_label('revenue')

# Format axes and labels
ax1.minorticks_on()
ax1.set_xlabel('Permit price (\$/tCO$_{2}$)', fontsize=9)
ax1.set_ylabel('Emissions intensity baseline\nrelative to BAU', fontsize=9)

ax2.minorticks_on()
ax2.set_ylabel('Scheme revenue (\$/h)', fontsize=9)

# Format legend
ax2.legend_.remove()
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
l1 = ['Baseline']
l2 = ['Revenue']

ax1.legend(h1+h2, l1+l2, loc=0, bbox_to_anchor=(0.37, .25), fontsize=8)

# Line showing revenue neutrality
# break_even = ax2.plot([2, 70], [0,0], linestyle='--', color='#9899a0', label='break even', linewidth=1)        

# Format ticks
ax2.xaxis.set_major_locator(MultipleLocator(10))
ax2.xaxis.set_minor_locator(MultipleLocator(2))

fig.canvas.draw()
ax2.xaxis.minorTicks[1].set_visible(False)
ax2.xaxis.minorTicks[-2].set_visible(False)
ax1.xaxis.minorTicks[0].set_visible(False)
ax1.xaxis.minorTicks[1].set_visible(False)
ax1.xaxis.minorTicks[-1].set_visible(False)
ax1.xaxis.minorTicks[-2].set_visible(False)

# Format figure
width = 85 * mmi
height = 65 * mmi
fig.subplots_adjust(left=0.22, bottom=0.16, right=0.8, top=.93)
fig.set_size_inches(width, height)

# Save figure
# fig.savefig('test.pdf')
fig.savefig(os.path.join(output_dir, 'figures', 'bau_price_target_baseline_and_revenue.pdf'))
plt.show()

