
# coding: utf-8

# # Emissions Intensity Scheme (EIS) Parameter Selection
# This notebook describes a mathematical framework for selecting EIS parameters. Please be aware of the following key assumptions underlying this model:
# 
# * Generators bid into the market at their short-run marginal cost (SRMC);
# * the market for electricity is perfectly competitive;
# * the policy maker is able to directly control the emissions intensity baseline and permit price.
# 
# Steps taken to conduct the analysis:
# 1. Import packages and declare declare paths to files
# 2. Load data
# 3. Organise data
# 6. Construct model used to select EIS parameters. The model consists of three blocks of equations:
#  * Primal block - contains constraints related to a standard DCOPF model;
#  * Dual block - dual constraints associated with a standard DCOPF model;
#  * Strong duality constraint block - block of constraints linking primal and dual objectives.
# 7. Run DCOPF model to find business-as-usual emissions and wholesale prices.
# 8. Run model used to select EIS parameters, save output
# 
# ## Import packages

# In[1]:


import os
import re
import time
import pickle
import random
from math import pi

import numpy as np
import pandas as pd
import datetime as dt

from pyomo.environ import *

import matplotlib.pyplot as plt

# Seed random number generator
np.random.seed(seed=10)


# ## Declare paths to files

# In[2]:


class DirectoryPaths(object):
    "Paths to relevant directories"
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data')
        self.scenarios_dir = os.path.join(os.path.curdir, os.path.pardir, '1_create_scenarios')
        self.output_dir = os.path.join(os.path.curdir, 'output', '48_scenarios')

paths = DirectoryPaths()


# ## Model data
# ### Input data

# In[3]:


class RawData(object):
    "Collect input data"
    
    def __init__(self):
        
        # Paths to directories
        DirectoryPaths.__init__(self)
        
        
        # Network data
        # ------------
        # Nodes
        self.df_n = pd.read_csv(os.path.join(self.data_dir, 'network_nodes.csv'), index_col='NODE_ID')

        # AC edges
        self.df_e = pd.read_csv(os.path.join(self.data_dir, 'network_edges.csv'), index_col='LINE_ID')

        # HVDC links
        self.df_hvdc_links = pd.read_csv(os.path.join(self.data_dir, 'network_hvdc_links.csv'), index_col='HVDC_LINK_ID')

        # AC interconnector links
        self.df_ac_i_links = pd.read_csv(os.path.join(self.data_dir, 'network_ac_interconnector_links.csv'), index_col='INTERCONNECTOR_ID')

        # AC interconnector flow limits
        self.df_ac_i_limits = pd.read_csv(os.path.join(self.data_dir, 'network_ac_interconnector_flow_limits.csv'), index_col='INTERCONNECTOR_ID')


        # Generators
        # ----------       
        # Generating unit information
        self.df_g = pd.read_csv(os.path.join(self.data_dir, 'generators.csv'), index_col='DUID', dtype={'NODE': int})
        self.df_g['SRMC_2016-17'] = self.df_g['SRMC_2016-17'].map(lambda x: x + np.random.uniform(0, 2))
        
               
        # Operating scenarios
        # -------------------
        with open(os.path.join(paths.scenarios_dir, 'output', '48_scenarios.pickle'), 'rb') as f:
            self.df_scenarios = pickle.load(f)

# Create object containing raw model data
raw_data = RawData() 


# ### Organise data for model

# In[4]:


class OrganiseData(object):
    "Organise data to be used in mathematical program"
    
    def __init__(self):
        # Load model data
        RawData.__init__(self)
        
        def reindex_nodes(self):
            # Original node indices
            df_index_map = self.df_n.index.to_frame().rename(columns={'NODE_ID': 'original'}).reset_index().drop('NODE_ID',axis=1)

            # New node indices
            df_index_map['new'] = df_index_map.apply(lambda x: x.name + 1, axis=1)

            # Create dictionary mapping original node indices to new node indices
            index_map = df_index_map.set_index('original')['new'].to_dict()


            # Network nodes
            # -------------
            # Construct new index and assign to dataframe
            new_index = pd.Index(self.df_n.apply(lambda x: index_map[x.name], axis=1), name=self.df_n.index.name)
            self.df_n.index = new_index


            # Network edges
            # -------------
            # Reindex 'from' and 'to' nodes in network edges dataframe
            def _reindex_from_and_to_nodes(row, order=False):
                """Re-index 'from' and 'to' nodes. If required, change node order such that 'from' node index < 'to' node index"""

                # Original 'from' and 'to' nodes
                n_1 = index_map[row['FROM_NODE']]
                n_2 = index_map[row['TO_NODE']]

                if order:
                    # If original 'from' node index is less than original 'to' node index keep same order, else reverse order
                    if n_1 < n_2:
                        f, t = n_1, n_2
                    else:
                        f, t = n_2, n_1
                    return pd.Series({'FROM_NODE': f, 'TO_NODE': t})
                else:
                    return pd.Series({'FROM_NODE': n_1, 'TO_NODE': n_2})
            self.df_e[['FROM_NODE', 'TO_NODE']] = self.df_e.apply(_reindex_from_and_to_nodes, args=(True,), axis=1)

            # Sort lines by 'from' and 'to' node indices
            self.df_e.sort_values(by=['FROM_NODE', 'TO_NODE'], inplace=True)


            # Generators
            # ----------
            self.df_g['NODE'] = self.df_g['NODE'].map(lambda x: df_index_map.set_index('original')['new'].loc[x])


            # Network HVDC links
            # ------------------
            self.df_hvdc_links[['FROM_NODE', 'TO_NODE']] = self.df_hvdc_links.apply(_reindex_from_and_to_nodes, axis=1)


            # Network interconnectors
            # -----------------------
            self.df_ac_i_links[['FROM_NODE', 'TO_NODE']] = self.df_ac_i_links.apply(_reindex_from_and_to_nodes, axis=1)
            
            # Operating scenarios
            # -------------------
            df_temp = self.df_scenarios.reset_index()
            df_temp['NODE_ID'] = df_temp.apply(lambda x: index_map[x['NODE_ID']] if type(x['NODE_ID']) == int else x['NODE_ID'], axis=1)
            self.df_scenarios = df_temp.set_index(['level', 'NODE_ID']).T     
            
        reindex_nodes(self)    
            
    def get_admittance_matrix(self):
        "Construct admittance matrix for network"

        # Initialise dataframe
        df_Y = pd.DataFrame(data=0j, index=self.df_n.index, columns=self.df_n.index)

        # Off-diagonal elements
        for index, row in self.df_e.iterrows():
            fn, tn = row['FROM_NODE'], row['TO_NODE']
            df_Y.loc[fn, tn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']
            df_Y.loc[tn, fn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']

        # Diagonal elements
        for i in self.df_n.index:
            df_Y.loc[i, i] = - df_Y.loc[i, :].sum()

        # Add shunt susceptance to diagonal elements
        for index, row in self.df_e.iterrows():
            fn, tn = row['FROM_NODE'], row['TO_NODE']
            df_Y.loc[fn, fn] += (row['B_PU'] / 2) * row['NUM_LINES']
            df_Y.loc[tn, tn] += (row['B_PU'] / 2) * row['NUM_LINES']

        return df_Y
    
    def get_HVDC_incidence_matrix(self):
        "Incidence matrix for HVDC links"
        
        # Incidence matrix for HVDC links
        df = pd.DataFrame(index=self.df_n.index, columns=self.df_hvdc_links.index, data=0)

        for index, row in self.df_hvdc_links.iterrows():
            # From nodes assigned a value of 1
            df.loc[row['FROM_NODE'], index] = 1

            # To nodes assigned a value of -1
            df.loc[row['TO_NODE'], index] = -1
        
        return df
    
    def get_reference_nodes(self):
        "Get reference node IDs"
        
        # Filter Regional Reference Nodes (RRNs) in Tasmania and Victoria.
        mask = (model_data.df_n['RRN'] == 1) & (model_data.df_n['NEM_REGION'].isin(['TAS1', 'VIC1']))
        reference_node_ids = model_data.df_n[mask].index
        
        return reference_node_ids
    
    def get_generator_node_map(self, generators):
        "Get set of generators connected to each node"
        generator_node_map = (self.df_g.reindex(index=generators)
                              .reset_index()
                              .rename(columns={'OMEGA_G': 'DUID'})
                              .groupby('NODE').agg(lambda x: set(x))['DUID']
                              .reindex(self.df_n.index, fill_value=set()))
        
        return generator_node_map

# Create object containing organised model data
model_data = OrganiseData()


# Perturb generator SRMCs by a random number uniformly distributed between 0 and 1. Unique SRMCs assist the solver to find a unique solution.

# In[5]:


model_data.df_g['SRMC_2016-17'] = model_data.df_g['SRMC_2016-17'].map(lambda x: x + np.random.uniform(0, 1))
model_data.df_g['SRMC_2016-17'].head()


# Save generator, node, and scenario information so they can be used in later processing and plotting steps.

# In[6]:


with open(os.path.join(paths.output_dir, 'df_g.pickle'), 'wb') as f:
    pickle.dump(model_data.df_g, f)
    
with open(os.path.join(paths.output_dir, 'df_n.pickle'), 'wb') as f:
    pickle.dump(model_data.df_n, f)
    
with open(os.path.join(paths.output_dir, 'df_scenarios.pickle'), 'wb') as f:
    pickle.dump(model_data.df_scenarios, f)


# ## Model
# Wrap optimisation model in function. Pass parameters to solve for different scenarios.

# In[7]:


def run_model(model_type=None, mode=None, tau_list=None, phi_list=None, E_list=None, R_list=None, target_bau_average_price_multiple_list=None, bau_average_price=None, fix_phi=None, fix_tau=None, stream_solver=False):
    """Construct and run model used to calibrate REP scheme parameters
    
    Parameters
    ----------
    model_type : str
        Either DCOPF or MPPDC
    
    mode : str
        Mode in which to run model. E.g. compute baseline given fixed permit price and price target
    
    tau_list : list
        Fixed permit prices for which the model should be run [$/tCO2]
    
    phi_list : list
        Fixed emissions intensity baselines for which the model should be run [tCO2/MWh]
    
    E_list : list
        Average emissions intensity constraints [tCO2/MWh]
        
    R_list : list
        Minimum scheme revenue constraint [$]
        
    target_bau_average_price_multiple_list : list
        Wholesale electricity price target as multiple of the business-as-usual (BAU) price [$/MWh]
        
    bau_average_price : float
        Business-as-usual average wholesale electricity price [$/MWh]
        
    fix_phi : float
        Fixed value of emissions intensity baseline (only applies to DCOPF model)
    
    fix_tau : float
        Fixed value of permit price (only applies to DCOPF model)
        
    stream_solver : bool
        Indicator if solver output should be streamed to terminal    
    """  

    # Checking model options are correctly inputted
    # ---------------------------------------------
    if model_type not in ['DCOPF', 'MPPDC']:
        raise(Exception("Must specify either 'DCOPF' or 'MPPDC' as the model type"))
    
    if model_type is 'MPPDC' and mode not in ['find_price_targeting_baseline', 'fixed_policy_parameters', 'find_permit_price_and_baseline']:
        raise(Exception("If model_type 'MPPDC' specified, must choose from 'find_price_targeting_baseline', 'fixed_policy_parameters', 'find_permit_price_and_baseline'"))

        
    # Initialise model object
    # -----------------------
    model = ConcreteModel(name='DCOPF')


    # Setup solver
    # ------------
    solver = 'cplex'
    solver_io = 'lp'
    keepfiles = False
    solver_opt = {}
    opt = SolverFactory(solver, solver_io=solver_io)


    # Sets
    # ----
    # Operating scenarios
    if model_type is 'DCOPF': model.T = Set(initialize=['DUMMY'])
    if model_type is 'MPPDC': model.T = Set(initialize=model_data.df_scenarios.index)

    # Nodes
    model.I = Set(initialize=model_data.df_n.index)   

    # Network reference nodes (Mainland and Tasmania)
    reference_nodes = model_data.get_reference_nodes()
    model.N = Set(initialize=reference_nodes)

    # AC network edges
    df_Y = model_data.get_admittance_matrix()
    ac_edges = [(df_Y.columns[i], df_Y.columns[j]) for i, j in zip(np.where(df_Y != 0)[0], np.where(df_Y != 0)[1]) if (i < j)]
    model.K = Set(initialize=ac_edges)

    # HVDC links
    hvdc_incidence_matrix = model_data.get_HVDC_incidence_matrix().T
    model.M = Set(initialize=hvdc_incidence_matrix.index)

    # Generators - only non-hydro dispatchable plant
    mask = (model_data.df_g['SCHEDULE_TYPE'] == 'SCHEDULED') & ~(model_data.df_g['FUEL_CAT'] == 'Hydro')
    model.G = Set(initialize=model_data.df_g[mask].index)


    # Parameters
    # ----------
    # Generation lower bound [MW]
    def P_MIN_RULE(model, g):
        return 0
    model.P_MIN = Param(model.G, initialize=P_MIN_RULE)

    # Generation upper bound [MW]
    def P_MAX_RULE(model, g):
        return float(model_data.df_g.loc[g, 'REG_CAP'])
    model.P_MAX = Param(model.G, initialize=P_MAX_RULE)

    # Voltage angle difference between connected nodes i and j lower bound [rad]
    model.VANG_MIN = Param(initialize=float(-pi / 2))

    # Voltage angle difference between connected nodes i and j upper bound [rad]
    model.VANG_MAX = Param(initialize=float(pi / 2))

    # Susceptance matrix [pu]
    def B_RULE(model, i, j):
        return float(np.imag(df_Y.loc[i, j]))
    model.B = Param(model.I, model.I, initialize=B_RULE)

    # HVDC incidence matrix
    def C_RULE(model, m, i):
        return float(hvdc_incidence_matrix.loc[m, i])
    model.C = Param(model.M, model.I, initialize=C_RULE)

    # HVDC reverse flow from node i to j lower bound [MW]
    def H_MIN_RULE(model, m):
        return - float(model_data.df_hvdc_links.loc[m, 'REVERSE_LIMIT_MW'])
    model.H_MIN = Param(model.M, initialize=H_MIN_RULE)

    # HVDC forward flow from node i to j upper bound [MW]
    def H_MAX_RULE(model, m):
        return float(model_data.df_hvdc_links.loc[m, 'FORWARD_LIMIT_MW'])
    model.H_MAX = Param(model.M, initialize=H_MAX_RULE)

    # AC power flow limits on branches
    def F_MIN_RULE(model, i, j):
        return -99999
    model.F_MIN = Param(model.K, initialize=F_MIN_RULE, mutable=True)
    
    def F_MAX_RULE(model, i, j):
        return 99999
    model.F_MAX = Param(model.K, initialize=F_MAX_RULE, mutable=True)

    # Adjust power flow limits for major AC interconnectors
    for index, row in model_data.df_ac_i_links.drop('VIC1-NSW1').iterrows():
        i, j = row['FROM_NODE'], row['TO_NODE']

        # Take into account direction of branch flow
        if i < j:
            model.F_MAX[i, j] = model_data.df_ac_i_limits.loc[index, 'FORWARD_LIMIT_MW']
            model.F_MIN[i, j] = - model_data.df_ac_i_limits.loc[index, 'REVERSE_LIMIT_MW']
        else:
            model.F_MAX[j, i] = model_data.df_ac_i_limits.loc[index, 'REVERSE_LIMIT_MW']
            model.F_MIN[j, i] = - model_data.df_ac_i_limits.loc[index, 'FORWARD_LIMIT_MW']

    # Generator emissions intensities [tCO2/MWh]
    def E_RULE(model, g):
        return float(model_data.df_g.loc[g, 'EMISSIONS'])
    model.E = Param(model.G, initialize=E_RULE)

    # Generator short run marginal costs [$/MWh]
    def A_RULE(model, g):
        return float(model_data.df_g.loc[g, 'SRMC_2016-17'])
    model.A = Param(model.G, initialize=A_RULE)

    # System base power [MVA]
    model.S = Param(initialize=100)
    
    # Revenue constraint [$] - Initialise to very large negative value (loose constraint)
    model.R = Param(initialize=-9e9, mutable=True)
    
    # Target wholsale electricity price [$/MWh]
    model.TARGET_PRICE = Param(initialize=30, mutable=True)
       
    # Target REP scheme revenue
    model.MIN_SCHEME_REVENUE = Param(initialize=-float(5e9), mutable=True)


    # Upper-level program
    # -------------------
    # Primal variables
    # ----------------
    # Emissions intensity baseline [tCO2/MWh]
    model.phi = Var(initialize=0, within=NonNegativeReals)

    # Permit price [$/tCO2]
    model.tau = Var(initialize=0)


    # Lower-level program
    # -------------------
    # Primal block (indexed over T) (model.LL_PRIM)
    # ---------------------------------------------
    def LL_PRIM_RULE(b, t):
        # Parameters
        # ----------
        # Demand at each node (to be set prior to running model)
        b.P_D = Param(model.I, initialize=0, mutable=True)

        # Intermittent power injection at each node (to be set prior to running model)
        b.P_R = Param(model.I, initialize=0, mutable=True)

        # Trading interval length [h] - use 1hr if running DCOPF model
        if t == 'DUMMY':
            b.L = 1
        else:
            b.L = Param(initialize=float(model_data.df_scenarios.loc[t, ('hours', 'duration')]))
    
    
        # Variables
        # ---------
        b.P = Var(model.G)
        b.vang = Var(model.I)
        b.H = Var(model.M)


        # Constraints
        # -----------
        # Power output lower bound
        def P_LB_RULE(b, g):
            return model.P_MIN[g] - b.P[g] <= 0
        b.P_LB = Constraint(model.G, rule=P_LB_RULE)

        # Power output upper bound
        def P_UB_RULE(b, g):
            return b.P[g] - model.P_MAX[g] <= 0
        b.P_UB = Constraint(model.G, rule=P_UB_RULE)

        # Voltage angle difference between connected nodes lower bound
        def VANG_DIFF_LB_RULE(b, i, j):
            return model.VANG_MIN - b.vang[i] + b.vang[j] <= 0
        b.VANG_DIFF_LB = Constraint(model.K, rule=VANG_DIFF_LB_RULE)

        # Voltage angle difference between connected nodes upper bound
        def VANG_DIFF_UB_RULE(b, i, j):
            return b.vang[i] - b.vang[j] - model.VANG_MAX <= 0
        b.VANG_DIFF_UB = Constraint(model.K, rule=VANG_DIFF_UB_RULE)

        # Fix voltage angle = 0 for reference nodes
        def VANG_REF_RULE(b, n):
            return b.vang[n] == 0
        b.VANG_REF = Constraint(model.N, rule=VANG_REF_RULE)

        # Map between nodes and generators connected to each node
        generator_node_map = model_data.get_generator_node_map([g for g in model.G])
        
        # Nodal power balance constraint
        def POWER_BALANCE_RULE(b, i):
            # Branches connected to node i
            K_i = [k for k in model.K if i in k]

            # Nodes connected to node i
            I_i = [ii for branch in K_i for ii in branch if (ii != i)]

            return (-model.S * sum(model.B[i, j] * (b.vang[i] - b.vang[j]) for j in I_i)
                   - sum(model.C[m, i] * b.H[m] for m in model.M)
                   - b.P_D[i]
                   + sum(b.P[g] for g in generator_node_map.loc[i] if g in model.G)
                   + b.P_R[i] == 0)
        b.POWER_BALANCE = Constraint(model.I, rule=POWER_BALANCE_RULE)

        # AC branch flow limits upper bound
        def AC_FLOW_LB_RULE(b, i, j):
            return model.F_MIN[i, j] - model.S * model.B[i, j] * (b.vang[i] - b.vang[j]) <= 0
        b.AC_FLOW_LB = Constraint(model.K, rule=AC_FLOW_LB_RULE)

        # AC branch flow limits lower bound
        def AC_FLOW_UB_RULE(b, i, j):
            return model.S * model.B[i, j] * (b.vang[i] - b.vang[j]) - model.F_MAX[i, j] <= 0
        b.AC_FLOW_UB = Constraint(model.K, rule=AC_FLOW_UB_RULE)

        # HVDC branch flow limits lower bound
        def HVDC_FLOW_LB_RULE(b, m):
            return model.H_MIN[m] - b.H[m] <= 0
        b.HVDC_FLOW_LB = Constraint(model.M, rule=HVDC_FLOW_LB_RULE)

        # HVDC branch flow limits upper bound
        def HVDC_FLOW_UB_RULE(b, m):
            return b.H[m] - model.H_MAX[m] <= 0
        b.HVDC_FLOW_UB = Constraint(model.M, rule=HVDC_FLOW_UB_RULE)


    # Dual block (indexed over T) (model.LL_DUAL)
    # -------------------------------------------
    def LL_DUAL_RULE(b, t):
        # Variables
        # ---------
        b.alpha = Var(model.G, within=NonNegativeReals)
        b.beta = Var(model.G, within=NonNegativeReals)
        b.gamma = Var(model.K, within=NonNegativeReals)
        b.delta = Var(model.K, within=NonNegativeReals)
        b.zeta = Var(model.N)
        b.lambda_var = Var(model.I)
        b.kappa = Var(model.K, within=NonNegativeReals)
        b.eta = Var(model.K, within=NonNegativeReals)
        b.omega = Var(model.M, within=NonNegativeReals)
        b.psi = Var(model.M, within=NonNegativeReals)


        # Constraints
        # -----------
        def DUAL_CONS_1_RULE(b, g):
            # Node at which generator g is located
            f_g = model_data.df_g.loc[g, 'NODE']

            # Don't apply scheme to existing hydro plant
            if model_data.df_g.loc[g, 'FUEL_CAT'] == 'Hydro':
                return model.A[g] - b.alpha[g] + b.beta[g] - b.lambda_var[f_g] == 0
            else:
                return model.A[g] + ((model.E[g] - model.phi) * model.tau) - b.alpha[g] + b.beta[g] - b.lambda_var[f_g] == 0
        b.DUAL_CONS_1 = Constraint(model.G, rule=DUAL_CONS_1_RULE)

        def DUAL_CONS_2_RULE(b, i):
            # Branches connected to node i
            K_i = [k for k in model.K if i in k]

            # Nodes connected to node i
            I_i = [ii for branch in K_i for ii in branch if (ii != i)]

            return (sum( (b.gamma[k] - b.delta[k] + (model.B[k] * model.S * (b.kappa[k] - b.eta[k])) ) * (np.sign(i - k[0]) + np.sign(i - k[1])) for k in K_i)
                    + sum(model.S * ((b.lambda_var[i] * model.B[i, j]) - (b.lambda_var[j] * model.B[j, i])) for j in I_i)
                    + sum(b.zeta[n] for n in model.N if n == i) == 0)
        b.DUAL_CONS_2 = Constraint(model.I, rule=DUAL_CONS_2_RULE)

        def DUAL_CONS_3_RULE(b, m):
            return sum(b.lambda_var[i] * model.C[m, i] for i in model.I) - b.omega[m] + b.psi[m] == 0
        b.DUAL_CONS_3 = Constraint(model.M, rule=DUAL_CONS_3_RULE)


    # Strong duality constraints (indexed over T)
    # -------------------------------------------
    def SD_CONS_RULE(model, t):
        return (sum(model.LL_PRIM[t].P[g] * ( model.A[g] + (model.E[g] - model.phi) * model.tau ) if model_data.df_g.loc[g, 'FUEL_CAT'] == 'Fossil' else model.LL_PRIM[t].P[g] * model.A[g] for g in model.G)
                == sum(model.LL_PRIM[t].P_D[i] * model.LL_DUAL[t].lambda_var[i] - (model.LL_PRIM[t].P_R[i] * model.LL_DUAL[t].lambda_var[i]) for i in model.I)
                + sum((model.LL_DUAL[t].omega[m] * model.H_MIN[m]) - (model.LL_DUAL[t].psi[m] * model.H_MAX[m]) for m in model.M)
                + sum(model.LL_DUAL[t].alpha[g] * model.P_MIN[g] for g in model.G)
                - sum(model.LL_DUAL[t].beta[g] * model.P_MAX[g] for g in model.G)
                + sum((model.VANG_MIN * model.LL_DUAL[t].gamma[k]) - (model.VANG_MAX * model.LL_DUAL[t].delta[k]) + (model.LL_DUAL[t].kappa[k] * model.F_MIN[k]) - (model.LL_DUAL[t].eta[k] * model.F_MAX[k]) for k in model.K))

    # Run DCOPF
    # ---------
    if model_type is 'DCOPF':   
        # Keep dual variables for DCOPF scenario
        model.dual = Suffix(direction=Suffix.IMPORT)
        
        # Build model
        model.LL_PRIM = Block(model.T, rule=LL_PRIM_RULE)

        # Fix phi and tau
        model.phi.fix(fix_phi)
        model.tau.fix(fix_tau)

        # DCOPF OBJECTIVE
        # ---------------
        def DCOPF_OBJECTIVE_RULE(model):
            return sum(model.LL_PRIM[t].P[g] * (model.A[g] + ((model.E[g] - model.phi) * model.tau) ) if model_data.df_g.loc[g, 'FUEL_CAT'] == 'Fossil' else model.LL_PRIM[t].P[g] * model.A[g] for t in model.T for g in model.G)
        model.DCOPF_OBJECTIVE = Objective(rule=DCOPF_OBJECTIVE_RULE, sense=minimize)

        # Container to store results
        results = []
        
        # Solve model for each time period
        for t in model_data.df_scenarios.index:
            # Update demand and intermittent power injections at each node
            for i in model.I:
                # Demand
                model.LL_PRIM['DUMMY'].P_D[i] = float(model_data.df_scenarios.loc[t, ('demand', i)])
                
                # Intermittent injections fixed power injections from hydro plant
                model.LL_PRIM['DUMMY'].P_R[i] = float(model_data.df_scenarios.loc[t, ('intermittent', i)] + model_data.df_scenarios.loc[t, ('hydro', i)])

            # Solve model
            r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)
            print('Finished solving DCOPF for period {}'.format(t))
            
            # Store model output
            model.solutions.store_to(r)
            
            # Convert to DataFrame
            try:
                df_results = pd.DataFrame(r['Solution'][0])
                df_results['SCENARIO_ID'] = t
                df_results['FIXED_PHI'] = fix_phi
                df_results['FIXED_TAU'] = fix_tau

            except:
                df_results = 'infeasible'
            
            # If model not infeasible store results in list to be concatenated 
            if type(df_results) != str:
                results.append(df_results)
            
        return pd.concat(results)
    

    # Run MPPDC
    # ---------
    if model_type is 'MPPDC':
        # Build model
        print('Building primal block')
        model.LL_PRIM = Block(model.T, rule=LL_PRIM_RULE)
        print('Building dual block')
        model.LL_DUAL = Block(model.T, rule=LL_DUAL_RULE)
        print('Building strong duality constraints')
        model.SD_CONS = Constraint(model.T, rule=SD_CONS_RULE)
        print('Finished building blocks')
        
        # Add revenue constraint to model if values of R are specified
        # Note: Exclude existing hydro and renewables from scheme (prevent windfall profits to existing generators)
        eligible_gens = [g for g in model.G if model_data.df_g.loc[g, 'FUEL_CAT'] == 'Fossil']
        model.R_CONS = Constraint(expr=sum((model.E[g] - model.phi) * model.tau * model.LL_PRIM[t].P[g] * model.LL_PRIM[t].L for t in model.T for g in eligible_gens) >= model.R)

        # Dummy variables used to minimise difference between average price and target price
        model.x_1 = Var(within=NonNegativeReals)
        model.x_2 = Var(within=NonNegativeReals)

        # Expressions for total revenue, total demand, and average wholesale price
        model.TOTAL_REVENUE = Expression(expr=sum(model.LL_DUAL[t].lambda_var[i] * model.LL_PRIM[t].L * model.LL_PRIM[t].P_D[i] for t in model.T for i in model.I))
        model.TOTAL_DEMAND = Expression(expr=sum(model.LL_PRIM[t].L * model.LL_PRIM[t].P_D[i] for t in model.T for i in model.I))
        model.AVERAGE_PRICE = Expression(expr=model.TOTAL_REVENUE / model.TOTAL_DEMAND)
        
        # Expression for total emissions and average emissions intensity
        model.TOTAL_EMISSIONS = Expression(expr=sum(model.LL_PRIM[t].P[g] * model.LL_PRIM[t].L * model.E[g] for g in model.G for t in model.T))
        model.AVERAGE_EMISSIONS_INTENSITY = Expression(expr=model.TOTAL_EMISSIONS / model.TOTAL_DEMAND)
        
        # Expression for net scheme revenue
        model.NET_SCHEME_REVENUE = Expression(expr=sum((model.E[g] - model.phi) * model.tau * model.LL_PRIM[t].P[g] * model.LL_PRIM[t].L for g in model.G for t in model.T))

        # Constraints used to minimise difference between average wholesale price and target
        model.x_1_CONS = Constraint(expr=model.x_1 >= model.AVERAGE_PRICE - model.TARGET_PRICE)
        model.x_2_CONS = Constraint(expr=model.x_2 >= model.TARGET_PRICE - model.AVERAGE_PRICE)

        # MPPDC objective function
        def MPPDC_OBJECTIVE_RULE(model):
            return model.x_1 + model.x_2
        model.MPPDC_OBJECTIVE = Objective(rule=MPPDC_OBJECTIVE_RULE, sense=minimize)

        
        # Useful functions
        # ----------------
        def _fix_LLPRIM_vars():
            "Fix generator output, voltage angles, and HVDC power flows."
            for t in model.T:
                for g in model.G:
                    model.LL_PRIM[t].P[g].fix()
                for m in model.M:
                    model.LL_PRIM[t].H[m].fix()
                for i in model.I:
                    model.LL_PRIM[t].vang[i].fix()
                    
        def _unfix_LLPRIM_vars():
            "Unfix generator output, voltage angles, and HVDC power flows"
            for t in model.T:
                for g in model.G:
                    model.LL_PRIM[t].P[g].unfix()
                for m in model.M:
                    model.LL_PRIM[t].H[m].unfix()
                for i in model.I:
                    model.LL_PRIM[t].vang[i].unfix()
                    
        def _store_output(results):
            "Store fixed variable values in solutions set of model object"
            

            print('Storing fixed variables')
            for t in model.T:
                # Store generator output
                for g in model.G:
                    results['Solution'][0]['Variable']['LL_PRIM[{0}].P[{1}]'.format(t, g)] = {'Value': model.LL_PRIM[t].P[g].value}
                
                # Store voltage angles
                for i in model.I:
                    results['Solution'][0]['Variable']['LL_PRIM[{0}].vang[{1}]'.format(t, i)] = {'Value': model.LL_PRIM[t].vang[i].value}
                    
                # Store HVDC power flows
                for m in model.M:
                    results['Solution'][0]['Variable']['LL_PRIM[{0}].H[{1}]'.format(t, m)] = {'Value': model.LL_PRIM[t].H[m].value}

            return results
            
        
        # Solve model for each policy parameter scenario
        # ----------------------------------------------
        print('Updating demand and fixed power injection parameters')
        
        # Initialise dictionary to store model output
        results = {}

        # Loop through scenarios - initialise parameters for demand and fixed power injections
        for t in model.T:
            # Loop through nodes
            for i in model.I:
                # Node demand [MW]
                model.LL_PRIM[t].P_D[i] = float(model_data.df_scenarios.loc[t, ('demand', i)])

                # Power injections from intermittent sources + hydro [MW]
                model.LL_PRIM[t].P_R[i] = float(model_data.df_scenarios.loc[t, ('intermittent', i)] + model_data.df_scenarios.loc[t, ('hydro', i)])

        if mode is 'find_price_targeting_baseline':
            # Loop through permit prices
             for tau in tau_list:
                    # Loop through price targets
                    for target_bau_average_price_multiple in target_bau_average_price_multiple_list:
                        # Loop through revenue targets
                        for R in R_list:     
                            # Start time
                            t0 = time.time()

                            # Fix phi and tau, solve model
                            model.phi.fix(0)
                            model.tau.fix(tau)
                            model.TARGET_PRICE = target_bau_average_price_multiple * bau_average_price
                            model.R = R

                            # Solve model for fixed permit price and baseline
                            r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)
                            print('Finished first stage')

                            # Fix lower level primal variables to their current values
                            _fix_LLPRIM_vars()

                            # Free phi
                            model.phi.unfix()

                            # Re-run model to compute baseline that minimise difference between average price and target
                            r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)                               
                            
                            # Store solutions in results object
                            model.solutions.store_to(r)
                            
                            # Add fixed generator and node data to model object
                            r = _store_output(r)

                            # Convert results object to DataFrame
                            try:
                                df_results = pd.DataFrame(r['Solution'][0])
                                df_results['AVERAGE_PRICE'] = model.AVERAGE_PRICE.expr()
                                df_results['FIXED_TAU'] = tau
                                df_results['TARGET_PRICE'] = model.TARGET_PRICE.value
                                df_results['TARGET_PRICE_BAU_MULTIPLE'] = target_bau_average_price_multiple
                                df_results['REVENUE_CONSTRAINT'] = model.R.value
                                df_results['MODE'] = mode

                            except:
                                df_results = 'infeasible'   
                                print('FAILED TO SAVE DATA')

                            # Construct file name based on parameters
                            fname = 'MPPDC-FIND_PRICE_TARGETING_BASELINE-PERMIT_PRICE_{0}-REVENUE_CONSTRAINT_{1}-TARGET_PRICE_BAU_MULTIPLE_{2}.pickle'.format(tau, int(R), target_bau_average_price_multiple)

                            with open(os.path.join(paths.output_dir, fname), 'wb') as f:
                                pickle.dump(df_results, f)

                            # Unfix lower level problem primal variables
                            _unfix_LLPRIM_vars()

                            print('Finished {0} in {1}s'.format(fname, time.time() - t0))

        elif mode is 'fixed_policy_parameters':
            # Loop through fixed permit price scenarios
            for tau in tau_list:
                # Loop through fixed baseline scenarios
                for phi in phi_list:
                    # Loop through different average wholesale price targets
                    for target_bau_average_price_multiple in target_bau_average_price_multiple_list:
                        # Start time
                        t0 = time.time()

                        # Construct file name based on parameters
                        fname = 'MPPDC-FIXED_PARAMETERS-BASELINE_{0}-PERMIT_PRICE_{1}.pickle'.format(phi, tau)
                        print('Starting scenario {0}'.format(fname))

                        # Fix policy parameters
                        model.tau.fix(tau)
                        model.phi.fix(phi)
                        model.TARGET_PRICE = target_bau_average_price_multiple * bau_average_price

                        # Solve model
                        r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)

                        # Store solutions
                        model.solutions.store_to(r)     

                        # Convert results object to DataFrame
                        try:
                            df_results = pd.DataFrame(r['Solution'][0])
                            df_results['AVERAGE_PRICE'] = model.AVERAGE_PRICE.expr()
                            df_results['FIXED_PHI'] = phi
                            df_results['FIXED_TAU'] = tau
                            df_results['TARGET_PRICE'] = model.TARGET_PRICE.value
                            df_results['TARGET_PRICE_BAU_MULTIPLE'] = target_bau_average_price_multiple
                            df_results['MODE'] = mode

                        except:
                            df_results = 'infeasible'  
                            print('FAILED TO SAVE DATA')

                        # Save DataFrame objects
                        with open(os.path.join(paths.output_dir, fname), 'wb') as f:
                            pickle.dump(df_results, f)

                        print('Finished {0} in {1}s'.format(fname, time.time() - t0))
        
        elif mode is 'find_permit_price_and_baseline':  
            # Add net scheme revenue constraint
            model.SCHEME_REVENUE_CONSTRAINT = Constraint(expr=model.NET_SCHEME_REVENUE >= model.MIN_SCHEME_REVENUE)
                        
            # Emissions intensity target
            for target_average_emissions_intensity in E_list:
                # Average wholesale price targets
                for target_bau_average_price_multiple in target_bau_average_price_multiple_list:
                    # Revenue constraint targets
                    for min_scheme_revenue in R_list:
                        # Start timer for scenario run
                        t0 = time.time()
                        
                        # Construct file name based on parameters
                        fname = 'MPPDC-FIND_PERMIT_PRICE_AND_BASELINE-EMISSIONS_INTENSITY_TARGET_{0}-TARGET_PRICE_BAU_MULTIPLE_{1}-MIN_SCHEME_REVENUE_{2}.pickle'.format(target_average_emissions_intensity, target_bau_average_price_multiple, min_scheme_revenue)
                        print('Starting scenario {0}'.format(fname))
                        
                        # Target average wholesale price
                        model.TARGET_PRICE = target_bau_average_price_multiple * bau_average_price
                        
                        # Scheme revenue constraint (must be greater than or equal to target)
                        model.MIN_SCHEME_REVENUE = min_scheme_revenue
                        
                        # Initialise lower and upper permit price bounds. Fix baseline to zero.
                        tau_up = 100
                        tau_lo = 0
                        model.phi.fix(0)
                        
                        # Fix value of tau to midpoint of upper and lower bounds
                        model.tau.fix((tau_up + tau_lo) / 2)
                        
                        # Iteration counter
                        counter = 0
                        iteration_limit = 7
                        while counter <= iteration_limit:
                            # While tolerance not satisfied (0.01 tCO2/MWh) or iteration not limit exceeded
                            # (Print message if tolerance limit exceeded)

                            # Run model
                            opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)

                            # Compute average emissions intensity
                            average_emissions_intensity = model.AVERAGE_EMISSIONS_INTENSITY.expr()
                            print('Finished iteration {0}, average emissions intensity: {1} tCO2/MWh'.format(counter, average_emissions_intensity))

                            # Check if emissions intensity is sufficiently close to target
                            if abs(target_average_emissions_intensity - average_emissions_intensity) < 0.01:
                                break

                            else:                
                                # If emissions intensity > target, set lower bound to previous guess, recompute new permit price
                                if average_emissions_intensity > target_average_emissions_intensity: 
                                    tau_lo = model.tau.value
                                    model.tau.fix((tau_up + tau_lo) / 2)

                                # If emissions intensity < target, set upper bound to previous guess, recompute new permit price
                                else:
                                    tau_up = model.tau.value
                                    model.tau.fix((tau_up + tau_lo) / 2)

                            # Break loop if iteration limit exceeded. Print message to console.
                            if counter == iteration_limit:
                                print('Iteration limit exceeded. Exiting loop.')   
                            
                            # Increment counter
                            counter += 1

                        # Fix lower level primal variables to their current values
                        _fix_LLPRIM_vars()

                        # Free phi
                        model.phi.unfix()

                        # Re-run model to compute baseline that optimises average price deviation objective
                        r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)
                       
                        # Store solutions
                        model.solutions.store_to(r)     

                        # Convert results object to DataFrame
                        try:
                            df_results = pd.DataFrame(r['Solution'][0])
                            df_results['AVERAGE_PRICE'] = model.AVERAGE_PRICE.expr()
                            df_results['AVERAGE_EMISSIONS_INTENSITY'] = model.AVERAGE_EMISSIONS_INTENSITY.expr()
                            df_results['NET_SCHEME_REVENUE'] = model.NET_SCHEME_REVENUE.expr()
                            df_results['TARGET_EMISSIONS_INTENSITY'] = target_average_emissions_intensity
                            df_results['MIN_SCHEME_REVENUE'] = model.MIN_SCHEME_REVENUE.value
                            df_results['TARGET_PRICE'] = model.TARGET_PRICE.value
                            df_results['TARGET_PRICE_BAU_MULTIPLE'] = target_bau_average_price_multiple
                            df_results['PHI'] = model.phi.value
                            df_results['TAU'] = model.tau.value
                            df_results['MODE'] = mode

                        except:
                            df_results = 'infeasible'  
                            print('FAILED TO SAVE DATA')

                        # Save DataFrame objects
                        with open(os.path.join(paths.output_dir, fname), 'wb') as f:
                            pickle.dump(df_results, f)
                                
                        # Unfix lower level problem primal variables
                        _unfix_LLPRIM_vars()

                        print('Finished {0} in {1}s'.format(fname, time.time() - t0))
        else:
            raise(Warning('Model {0} not recognised'.format(mode)))


# ### DCOPF Results
# Standard DCOPF model used to verify that MPPDC has been formulated correctly.

# In[8]:


# Loop through permit prices
for permit_price in [0]:
    # Loop through baselines
    for baseline in [0]:    
        # Get results for policy scenario
        results = run_model(model_type='DCOPF', stream_solver=False, fix_phi=baseline, fix_tau=permit_price)

        # Construct file name
        fname = 'DCOPF-FIXED_PARAMETERS-PERMIT_PRICE_{0}-BASELINE_{1}.pickle'.format(permit_price, baseline)

        # Save to file
        with open(os.path.join(paths.output_dir, fname), 'wb') as f:
            pickle.dump(results, f)


# ### MPPDC Results
# #### Fixed permit price and baseline
# Fix permit price and baseline. Results should be the same for the DCOPF base-case.

# In[9]:


# Not required, but could potentially impact way in which solver performs, hence the reason for recording this value
target_price_list = [1] 

# Emissions intensity baselines
phi_list = [0]

# Permit prices
tau_list = list(range(0, 71, 2))
tau_list = [0, 10]

# Run model
run_model(model_type='MPPDC', 
          mode='fixed_policy_parameters', 
          stream_solver=True, phi_list=phi_list, 
          tau_list=tau_list, target_bau_average_price_multiple_list=target_price_list, 
          bau_average_price=30)


# #### Compute baseline given fixed permit price
# Calculate emissions intensity baseline that achieves a given average wholesale price target for a given permit price. Set the base-case wholesale price target to be the business-as-usual (BAU) price.

# In[10]:


with open(os.path.join(paths.output_dir, 'MPPDC-FIXED_PARAMETERS-BASELINE_0-PERMIT_PRICE_0.pickle'), 'rb') as f:
        bau_scenario = pickle.load(f)

# Average BAU price
bau_average_price = bau_scenario['AVERAGE_PRICE'].unique()[0]
print('Average price = {0}'.format(bau_average_price))


# Define parameters for policy scenarios.

# In[11]:


# Target average wholesale prices as a multiple of the BAU price
target_bau_average_price_multiple_list = [0.8, 0.9, 1, 1.1, 1.2]
target_bau_average_price_multiple_list = [1.2]

# Permit prices
tau_list = list(range(2, 71, 2))
tau_list = [10]

# Run model
run_model(model_type='MPPDC',
          mode='find_price_targeting_baseline', 
          stream_solver=True, 
          tau_list=tau_list, 
          R_list=[-9e9], 
          target_bau_average_price_multiple_list=target_bau_average_price_multiple_list, 
          bau_average_price=bau_average_price)


# ### Find permit price and baseline given emissions and revenue constraints

# In[12]:


# Scheme revenue constraint (net scheme revenue must be greater than or equal to min_scheme_revenue)
min_scheme_revenue_list = [0]

# Target average emissions intensities
target_average_emissions_intensity_list = [0.88, 0.92]

# Target average wholesale price as a multiple of the BAU average price
target_bau_average_price_multiple_list = [0.9]

# Run model
run_model(model_type='MPPDC',
          mode='find_permit_price_and_baseline', 
          stream_solver=True, 
          tau_list=None, 
          R_list=min_scheme_revenue_list,
          E_list=target_average_emissions_intensity_list,
          target_bau_average_price_multiple_list=target_bau_average_price_multiple_list, 
          bau_average_price=bau_average_price)

