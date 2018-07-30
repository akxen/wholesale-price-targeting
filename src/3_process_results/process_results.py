
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

import ipyparallel as ipp

import matplotlib.pyplot as plt


# ## Declare paths to files

# In[2]:


# Core data directory
data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data'))

# Compiled model data
compile_data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '1_compile_data'))

# Model output directory
parameter_selector_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '2_parameter_selector'))

# Output directory
output_dir = os.path.abspath(os.path.join(os.path.curdir, 'output'))


# ## Import model data

# In[3]:


# Generator data (with perturbed costs from paramater_selector.ipynb)
with open(os.path.join(parameter_selector_dir, 'output', 'df_g.pickle'), 'rb') as f:
    df_g = pickle.load(f)

# Model data
with open(os.path.join(compile_data_dir, 'output', 'model_data.pickle'), 'rb') as f:
    model_data = pickle.load(f)

# Summary of node data
df_m = model_data['df_m']


# ### Setup parallel processing cluster

# In[4]:


# # Profile for cluster
# client_path = 'C:/Users/eee/AppData/Roaming/SPB_Data/.ipython/profile_parallel/security/ipcontroller-client.json'
# rc = ipp.Client(client_path)
# lview = rc.load_balanced_view()


# ## DCOPF Results
# Summarise results from DCOPF model runs.

# In[5]:


class ProcessDCOPF(object):
    "Process results from DCOPF model runs"
    
    def _init_(self):
        pass
    
    @staticmethod
    def convert_to_dataframe_and_save(f_path):
        """Parse results dictionary for DCOPF runs, construct dataframes, and save to file"""
        
        # Import packages locally in case need to run using ipyparallel
        import os
        import pickle

        import pandas as pd

        # Open pickled results dictionary
        with open(f_path, 'rb') as f:
            results = pickle.load(f)

        # List to store final dataframes
        df_r_list = []

        # Dictionary to store model parameters used in each DCOPF run
        model_params = {}

        # Time stamps for which the DCOPF model has been run
        t_stamps = list(results.keys())
        for i, t_stamp in enumerate(t_stamps):

            if (i + 1) % 10 == 0: print('Finished {0}/{1} timestamps'.format(i + 1, len(t_stamps)))

            # DCOPF output
            # ------------
            # Variables
            df_var = pd.DataFrame.from_dict(results[t_stamp]['Solution'][0]['Variable'])
            df_con = pd.DataFrame.from_dict(results[t_stamp]['Solution'][0]['Constraint'])

            # Use model run timestamp as index
            df_var.index = [pd.to_datetime(t_stamp)]
            df_con.index = [pd.to_datetime(t_stamp)]

            # Combine variable and constraint information into one dataframe
            df_r = df_var.join(df_con)

            # Add model parameters / options to dataframe   
            df_r['fix_phi'] = results[t_stamp]['Fix phi']
            df_r['fix_tau'] = results[t_stamp]['Fix tau']
            df_r['fix_hydro'] = results[t_stamp]['Fix hydro']


            # Append to list of dataframes
            df_r_list.append(df_r)

        # Concatenate list of dataframes
        df_r_c = pd.concat(df_r_list)

        # Write pickled dataframe to file
        f_name_in = f_path.split('\\')[-1]
        f_path_out = os.path.join(output_dir, f_name_in.replace('DCOPF_', 'df_DCOPF_'))

        with open(f_path_out, 'wb') as f:
            pickle.dump(df_r_c.T, f)
        
    @staticmethod
    def get_variable_values(df_in, var_name):
        """Given DCOPF results dataframe, extract values associated with a given variable

        Paramters
        ---------
        df_in : pandas dataframe
            Compiled DCOPF results dataframe
        
        var_name : str
            Name of variable to extract

        Returns
        -------
        df_out : pandas dataframe
            Dataframe consisting of values for variable under investigation   
        """

        # Only keep columns related to the variable under investigation
        var_filter_pattern = ''.join([r'\.', var_name, '\['])
        mask = df_in.columns.str.contains(var_filter_pattern)
        df_out = df_in.loc[:, mask]

        # Extract variable ID from column names
        var_extract_pattern = ''.join([r'\.', var_name, '\[(.+)\]'])
        df_out.columns = df_out.columns.str.extract(var_extract_pattern, expand=False)

        # Try to set to int if possible
        try:
            df_out.columns = df_out.columns.map(int)
        except: 
            pass

        return df_out
    
    @classmethod
    def get_generator_emissions(cls, df_dcopf, df_g):
        "Compute emissions for each generator in each time period"
        
        def emissions(row):
            return df_g.loc[row.index, 'EMISSIONS'] * row

        # Emissions for each generator for each time period
        return cls.get_variable_values(df_dcopf, 'P').apply(emissions, axis=1)
    
    @classmethod
    def get_nodal_revenue(cls, df_dcopf):
        "Compute revenue from wholesale electricity sales for each node"
        
        def revenue(row, df_P_D):
            return df_P_D.loc[row.name] * row
        
        # Demand at each node for each time period
        df_P_D = cls.get_variable_values(df_dcopf, 'P_D')

        # Nodal revenue
        return cls.get_variable_values(df_dcopf, 'power_balance').apply(revenue, args=(df_P_D,), axis=1)
    
    @classmethod
    def get_regional_average_emissions_intensity(cls, df_dcopf, df_g):
        "Get average emissions intensity in each NEM region"
        
        # Total emissions
        emissions = cls.get_generator_emissions(df_dcopf, df_g).T.join(df_g[['NEM_REGION']]).groupby('NEM_REGION').sum().T.sum()

        # Total demand = Total production
        demand = cls.get_variable_values(df_dcopf, 'P_D').T.join(df_m[['NEM_REGION']]).groupby('NEM_REGION').sum().sum(axis=1)

        # Emissions intensity
        emissions_intensity = emissions / demand

        return emissions_intensity.to_dict()
    
    @classmethod
    def get_regional_average_prices(cls, df_dcopf, df_m):
        "Get average prices in for each NEM region"
        
        # Total revenue for each region
        revenue = cls.get_nodal_revenue(df_dcopf).T.join(df_m[['NEM_REGION']]).groupby('NEM_REGION').sum().sum(axis=1)

        # Total demand = total production for each NEM region
        generation = cls.get_variable_values(df_dcopf, 'P_D').T.join(df_m[['NEM_REGION']]).groupby('NEM_REGION').sum().sum(axis=1)
        
        # Average price for each NEM region
        average_price = revenue / generation
        
        return average_price.to_dict()
    
    @classmethod
    def get_national_fossil_generation_proportions(cls, df_dcopf, df_g):
        "Get proportion of electricity generated by gas and coal plant relative to total fossil generation"
        
        # Total generation from fossil plant in each NEM region
        fossil_gen = cls.get_variable_values(df_dcopf, 'P').T.join(df_g[['FUEL_CAT']]).groupby('FUEL_CAT').sum().sum(axis=1).values[0]

        # Some liquid fuel oils, e.g. Diesel oil, are used in gas generators e.g. Mackay Gas Turbine. Same for Kerosene - non aviation.
        # Have added them to the 'gas' generators category, but strictly speaking there is some distincition between plants
        # using liquid fuel oil and plants supplied by a gas pipeline.

        # Gas categories
        gas_cat = ['Natural Gas (Pipeline)', 'Coal seam methane', 'Kerosene - non aviation', 'Diesel oil']

        # Coal categories
        coal_cat = ['Brown coal', 'Black coal']

        # Total gas generation in each NEM region
        gas_gen = cls.get_variable_values(df_dcopf, 'P').T.join(df_g[['FUEL_TYPE']]).groupby('FUEL_TYPE').sum().loc[gas_cat].sum().sum()

        # Gas generation as a proportion of fossil fuel generation
        gas_proportion = cls.get_variable_values(df_dcopf, 'P').T.join(df_g[['FUEL_TYPE']]).groupby(['FUEL_TYPE']).sum().loc[gas_cat].sum().sum() / fossil_gen

        # Coal generation as a proportion of fossil fuel generation
        coal_proportion = cls.get_variable_values(df_dcopf, 'P').T.join(df_g[['FUEL_TYPE']]).groupby(['FUEL_TYPE']).sum().loc[coal_cat].sum().sum() / fossil_gen
        
        return {'coal': coal_proportion, 'gas': gas_proportion}
    
    @classmethod
    def get_national_average_price(cls, df_dcopf):
        "Get average NEM wholesale price"
        
        # Total revenue for each region
        revenue = cls.get_nodal_revenue(df_dcopf).sum().sum()

        # Total demand = total production for each NEM region
        generation = cls.get_variable_values(df_dcopf, 'P_D').sum().sum()

        # Average price for each NEM region
        average_price = revenue / generation
        
        return average_price
    
    @classmethod
    def get_national_average_emissions_intensity(cls, df_dcopf, df_g):
        "Compute average emissions intensity for the NEM"
        
        # Total emissions
        emissions = cls.get_generator_emissions(df_dcopf, df_g).sum().sum()

        # Total demand = Total production
        demand = cls.get_variable_values(df_dcopf, 'P_D').sum().sum()

        # Emissions intensity
        emissions_intensity = emissions / demand
        
        return emissions_intensity       


# Process results files, convert to dataframes, and save pickled output.

# In[6]:


# Paths to files that must be converted
f_paths = [os.path.join(parameter_selector_dir, 'output', f) for f in os.listdir(os.path.join(parameter_selector_dir, 'output')) if 'DCOPF_CalcFixed' in f]

# Run conversion step in parallel
# lview.map_sync(dcopf_results_to_dataframe, fnames)

# Process one file at a time
ProcessDCOPF.convert_to_dataframe_and_save(f_paths[0])


# In[7]:


f_paths


# Compile DCOPF results data.

# In[8]:


# Container to store DCOPF scenario summary data
dcopf_summary = []

# Paths to files
f_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if 'df_DCOPF_CalcFixed' in f]

# Scenario parameters in tuple. First element is baseline, second is permit price
dcopf_scenarios = [(re.findall(r'phi_([\d\.]+)', f)[0], re.findall(r'tau_([\d\.]+)', f)[0]) for f in f_paths]

# Unique permit price scenarios (based on filenames)
for baseline_s, permit_price_s in dcopf_scenarios:
    
    # Container to hold dataframes which will be concatenated
    frames = []

    # Filenames to read-in
    f_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if 'df_DCOPF_CalcFixed_tau_{0}_phi_{1}'.format(permit_price_s, baseline_s) in f]

    # For each file, load it and place in frames list
    for f_path in f_paths:
        with open(f_path, 'rb') as f:
            df = pickle.load(f)
            frames.append(df.T)

    # Concatenate results
    df_dcopf = pd.concat(frames)


    # Initialise dictionary to store results
    # --------------------------------------
    dcopf_scenario = {}
    dcopf_scenario['regional'] = {}
    dcopf_scenario['national'] = {}


    # Information from file name
    # --------------------------
    if (len(df_dcopf['fix_tau'].unique()) != 1) or (len(df_dcopf['fix_phi'].unique()) != 1):
        raise(Exception('Only one unique permit price and baseline can be specified for the scenario'))

    # Permit price and baseline parameters for scenario
    dcopf_scenario['tau'] = df_dcopf['fix_tau'].unique()[0]
    dcopf_scenario['phi'] = df_dcopf['fix_phi'].unique()[0]


    # Regional level summary
    # ----------------------
    # Average emissions intensity in each region
    dcopf_scenario['regional']['emissions_intensity'] = ProcessDCOPF.get_regional_average_emissions_intensity(df_dcopf, df_g)

    # Average prices in each NEM region
    dcopf_scenario['regional']['average_price'] = ProcessDCOPF.get_regional_average_prices(df_dcopf, df_m)


    # National level summary
    # ----------------------
    # Average proportion of gas generation - national
    dcopf_scenario['national']['fossil_fuel_proportions'] = ProcessDCOPF.get_national_fossil_generation_proportions(df_dcopf, df_g)

    # National average emissions intensity
    dcopf_scenario['national']['emissions_intensity'] = ProcessDCOPF.get_national_average_emissions_intensity(df_dcopf, df_g)

    # Average NEM wholesale price
    dcopf_scenario['national']['average_price'] = ProcessDCOPF.get_national_average_price(df_dcopf)
    
    # Append scenario data to summary container
    dcopf_summary.append(dcopf_scenario)


# Save data
# ---------
with open(os.path.join(output_dir, 'dcopf_summary.pickle'), 'wb') as f:
    pickle.dump(dcopf_summary, f)

dcopf_summary


# ## MPPDC Results
# Summarise results for each scenario;

# In[9]:


class ProcessMPPDC(object):
    """Extract information and perform computations on MPPDC model run data"""
    
    def __init__(self, f_path):
        """Convert MPPDC results dictionary into a dataframe and insantiate scenario object"""
    
        with open(f_path, 'rb') as f:
            mppdc_results = pickle.load(f)

            # For each instance construct dataframe and append to frames
            for instance in mppdc_results.keys():
                df = pd.DataFrame.from_dict(mppdc_results[instance]['Solution'][0]['Variable'])

                # Add model settings
                df['instance'] = instance
                df['fix_hydro'] = mppdc_results[instance]['Fix hydro']
                df['target_price'] = mppdc_results[instance]['target_price']

                try:
                    df['R'] = mppdc_results[instance]['R']
                except:
                    pass
                try:
                    df['E'] = mppdc_results[instance]['E']
                except:
                    pass
                try:
                    df['iter_lim_exceeded'] = mppdc_results[instance]['iter_lim_exceeded']
                except:
                    pass

                # Add date range
                df.loc[:, 'date_range'] = pd.Series(dtype='object')
                df.at['Value', 'date_range'] = mppdc_results[instance]['Date range']

        
        # Instantiate MPPDC scenario object
        # ---------------------------------
        # Dataframe containing results
        self.df = df
        
        # Emissions intensity baseline
        self.phi = df['phi'].values[0]
        
        # Permit price
        self.tau = df['tau'].values[0]
        
        # Wholesale price target
        self.target_price = df['target_price'].values[0]
        
        # Tag to indicate if hydro output has been fixed
        self.fix_hydro = df['fix_hydro'].values[0]
        
        # Dates for which model was run
        self.date_range = df['date_range'].values[0]
        
        # Dummy price objective variable 1
        self.x_1 = df['x_1'].values[0]
        
        # Dummy price objective variable 2
        self.x_2 = df['x_2'].values[0]
        

    def get_variable_values(self, var_name):
        """Extract subset of values from MPPDC results dataframe"""
        
        # Construct dataframe from results dictionary
        df_in = self.df
        
        # Identify records to extract from df_mppdc
        date_range = df_in.loc['Value', 'date_range']

        # Only keep columns containing variables of interest
        var_filter_pattern = ''.join([r'\.', var_name, '\['])
        mask = df_in.columns.str.contains(var_filter_pattern)
        df_out = df_in.loc[:, mask].T

        # Mapping between time index and timestamps
        date_map = {i+1: j for i, j in enumerate(date_range)}

        # Extract time index and convert to timestamp
        t_extract_pattern = ''.join([r'\[(.+)\]\.', var_name, '\['])
        df_out['t_index'] = df_out.index.str.extract(t_extract_pattern, expand=False).map(int)
        df_out['t_stamp'] = df_out['t_index'].map(lambda x: date_map[x])

        # Extract variable ID
        var_extract_pattern = ''.join([r'\.', var_name, '\[(.+)\]'])
        df_out['var_id'] = df_out.index.str.extract(var_extract_pattern, expand=False)

        # Try to convert to int
        try:
            df_out['var_id'] = df_out['var_id'].map(int)
        except:
            pass

        # Pivot dataframe such that timestamp is index and variable IDs are columns
        df_out = df_out.pivot(index='t_stamp', columns='var_id', values='Value')

        return df_out
    

    def get_generator_emissions(self, df_g):
        """Compute emissions for each generator in each time period for a given scenario
        
        Parameters
        ----------
        df_g : pandas dataframe
            dataframe containing generator information
        """
               
        def emissions(row):
            return df_g.loc[row.index, 'EMISSIONS'] * row
        
        # Emissions for each generator for each time period
        return self.get_variable_values('P').apply(emissions, axis=1)
        
        
    def get_nodal_revenue(self):
        "Compute revenue from wholesale electricity sales for each node"
        
        def revenue(row, df_P_D):
            return df_P_D.loc[row.name] * row
        
        # Demand at each node for each time period
        df_P_D = self.get_variable_values('P_D')

        # Nodal revenue
        return self.get_variable_values('lambda_var').apply(revenue, args=(df_P_D,), axis=1)

    
    def get_regional_average_emissions_intensity(self, df_g):
        "Get average emissions intensity in each NEM region"
        
        # Total emissions
        emissions = self.get_generator_emissions(df_g).T.join(df_g[['NEM_REGION']]).groupby('NEM_REGION').sum().T.sum()

        # Total demand = Total production
        demand = self.get_variable_values('P_D').T.join(df_m[['NEM_REGION']]).groupby('NEM_REGION').sum().sum(axis=1)

        # Emissions intensity
        emissions_intensity = emissions / demand

        return emissions_intensity.to_dict()


    def get_regional_average_prices(self, df_m):
        "Get average prices in for each NEM region"
        
        # Total revenue for each region
        revenue = self.get_nodal_revenue().T.join(df_m[['NEM_REGION']]).groupby('NEM_REGION').sum().sum(axis=1)

        # Total demand = total production for each NEM region
        generation = self.get_variable_values('P_D').T.join(df_m[['NEM_REGION']]).groupby('NEM_REGION').sum().sum(axis=1)
        
        # Average price for each NEM region
        average_price = revenue / generation
        
        return average_price.to_dict()
    
    
    def get_national_fossil_generation_proportions(self, df_g):
        "Get proportion of electricity generated by gas and coal plant relative to total fossil generation"
        
        # Total generation from fossil plant in each NEM region
        fossil_gen = self.get_variable_values('P').T.join(df_g[['FUEL_CAT']]).groupby('FUEL_CAT').sum().sum(axis=1).values[0]

        # Some liquid fuel oils, e.g. Diesel oil, are used in gas generators e.g. Mackay Gas Turbine. Same for Kerosene - non aviation.
        # Have added them to the 'gas' generators category, but strictly speaking there is some distincition between plants
        # using liquid fuel oil and plants supplied by a gas pipeline.

        # Gas categories
        gas_cat = ['Natural Gas (Pipeline)', 'Coal seam methane', 'Kerosene - non aviation', 'Diesel oil']

        # Coal categories
        coal_cat = ['Brown coal', 'Black coal']

        # Total gas generation in each NEM region
        gas_gen = self.get_variable_values('P').T.join(df_g[['FUEL_TYPE']]).groupby('FUEL_TYPE').sum().loc[gas_cat].sum().sum()

        # Gas generation as a proportion of fossil fuel generation
        gas_proportion = self.get_variable_values('P').T.join(df_g[['FUEL_TYPE']]).groupby(['FUEL_TYPE']).sum().loc[gas_cat].sum().sum() / fossil_gen

        # Coal generation as a proportion of fossil fuel generation
        coal_proportion = self.get_variable_values('P').T.join(df_g[['FUEL_TYPE']]).groupby(['FUEL_TYPE']).sum().loc[coal_cat].sum().sum() / fossil_gen
        
        return {'coal': coal_proportion, 'gas': gas_proportion}
    
    
    def get_national_average_price(self):
        "Get average NEM wholesale price"
        
        # Total revenue for each region
        revenue = self.get_nodal_revenue().sum().sum()

        # Total demand = total production for each NEM region
        generation = self.get_variable_values('P_D').sum().sum()

        # Average price for each NEM region
        average_price = revenue / generation
        
        return average_price
    
    
    def get_national_average_emissions_intensity(self, df_g):
        "Compute average emissions intensity for the NEM"
        
        # Total emissions
        emissions = self.get_generator_emissions(df_g).sum().sum()

        # Total demand = Total production
        demand = self.get_variable_values('P_D').sum().sum()

        # Emissions intensity
        emissions_intensity = emissions / demand
        
        return emissions_intensity
    
    
    def get_scheme_revenue(self, df_g):
        "Revenue for policy scenario"
        
        def compute_scheme_revenue(row, phi, tau):
            "Compute net liability for each generator"
            return (df_g.loc[row.index, 'EMISSIONS'] - phi) * row * tau
        
        # Scheme revenue [$/hr]
        scheme_revenue = self.get_variable_values('P').apply(compute_scheme_revenue, args=(self.phi, self.tau,), axis=1).sum().sum() / len(self.date_range)
    
        return scheme_revenue


# Loop through scenarios, perform computations, and store in dictionary summarising results for scenario. Then save to file.

# In[10]:


# Container used to store summarised results for each scenario
mppdc_summary = []

# Construct paths to MPPDC model results files
f_paths = [os.path.join(parameter_selector_dir, 'output', f) for f in os.listdir(os.path.join(parameter_selector_dir, 'output')) if 'MPPDC' in f]

# Instantiate scenario object
for i, f_path in enumerate(f_paths):
    s = ProcessMPPDC(f_path)

    # Process results for each scenario
    mppdc_scenario = {}

    # Calculation mode specified:
    # CalcPhi = compute baseline given permit price, 
    # CalcFixed = compute primal variables given fixed permit price and baseline (same as DCOPF, used for comparison)
    if 'CalcPhi' in f_path:
        mode = 'CalcPhi'
    elif 'CalcFixed' in f_path:
        mode = 'CalcFixed'
    else:
        raise(Exception('Model run mode not correctly specified. Should be in [CalcPhi, CalcFixed]'))

    mppdc_scenario['mode'] = mode
    mppdc_scenario['regional_average_emissions_intensity'] = s.get_regional_average_emissions_intensity(df_g)
    mppdc_scenario['regional_average_prices'] = s.get_regional_average_prices(df_m)
    mppdc_scenario['national_fossil_generation_proportions'] = s.get_national_fossil_generation_proportions(df_g)
    mppdc_scenario['national_average_price'] = s.get_national_average_price()
    mppdc_scenario['national_average_emissions_intensity'] = s.get_national_average_emissions_intensity(df_g)
    mppdc_scenario['scheme_revenue'] = s.get_scheme_revenue(df_g)
    mppdc_scenario['target_price'] = s.target_price
    mppdc_scenario['objective'] = s.x_1 + s.x_2
    mppdc_scenario['tau'] = s.tau
    mppdc_scenario['phi'] = s.phi
    mppdc_scenario['fix_hydro'] = s.fix_hydro
    mppdc_scenario['date_range'] = s.date_range

    # Append scenario results dictionary to list containing summarised results for each scenario
    mppdc_summary.append(mppdc_scenario)
    
    print('Finished processing {0}, {1}/{2}'.format(f_path, i + 1, len(f_paths)))
    
# Save data
with open(os.path.join(output_dir, 'mppdc_summary.pickle'), 'wb') as f:
    pickle.dump(mppdc_summary, f)


# ## Check results for DCOPF and MPPDC are consistent
# Load base-case files for DCOPF and MPPDC results.

# In[11]:


# Base-case files
dcopf_base_case = 'df_DCOPF_CalcFixed_tau_0_phi_0_FixHydro_True-1.pickle'
mppdc_base_case = 'MPPDC_CalcFixed_tau_0_phi_0_tarp_30_FixHydro_True.pickle'

# Business as usual DCOPF scenario
with open(os.path.join(output_dir, dcopf_base_case), 'rb') as f:
    df_dcopf = pickle.load(f)

# Business as usual MPPDC scenario
mppdc_bau = ProcessMPPDC(os.path.join(parameter_selector_dir, 'output', mppdc_base_case))


# Compute differences between DCOPF and MPPDC results for selected variables.

# In[12]:


# Power output difference between MPPDC and DCOPF results
df_P_diff = mppdc_bau.get_variable_values('P') - ProcessDCOPF.get_variable_values(df_dcopf.T, 'P')

# Power output
df_price_diff = mppdc_bau.get_variable_values('lambda_var') - ProcessDCOPF.get_variable_values(df_dcopf.T, 'power_balance')

# Voltage angle difference
df_vang_diff = mppdc_bau.get_variable_values('vang') - ProcessDCOPF.get_variable_values(df_dcopf.T, 'vang')

# HVDC power flow difference
df_H_diff = mppdc_bau.get_variable_values('H') - ProcessDCOPF.get_variable_values(df_dcopf.T, 'H')

print('Max power output difference over all nodes and time periods: {0} MW'.format(df_P_diff.abs().max().max()))
print('Max price difference over all nodes and time periods: {0} $/MWh'.format(df_price_diff.abs().max().max()))
print('Max voltage angle difference over all branches and time periods: {0} rad'.format(df_vang_diff.abs().max().max()))
print('Max HVDC flow difference over all HVDC links and time periods: {0} MW'.format(df_H_diff.abs().max().max()))


# Note that voltage angles and HVDC flows do not correspond exactly, but power output and prices do. This is likely due to the introduction of an additional degree of freedom when adding HVDC links into the network analysis. Having HVDC links allows power to flow over either the HVDC or AC networks. So long as branch flows are within limits for constrained links, different combinations of these flows may be possible. This results in different intra-zonal flows (hence different voltage angles), but net inter-zonal flows are the same as the DCOPF case. This is illustrated by looking at flows over Basslink connecting Victoria to Tasmania. These HVDC flows correspond exactly, as there is only one path for power to flow between these two regions. For flows between South Australia and Victoria, HVDC flows may differ between the MPPDC and DCOPF cases. This is because these two regions are connected by both HVDC and AC links. Therefore, different combinations of power can flow over the two links between the two regions, but net inter-zonal flows remain the same as the DCOPF case.
# 
# These differences are likely due to the way in which the solver approaches a solution; different feasible HVDC and intra-zonal AC flows yield the same least-cost dispatch. Consequently, DCOPF and MPPDC output and prices correspond, but HVDC and node voltage angles (which relate to AC power flows) do not.

# In[13]:


# Show differences between MPPDC and DCOPF results for HVDC links
df_H_diff

