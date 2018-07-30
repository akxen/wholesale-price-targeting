
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
# 2. Load data:
#  * generator data;
#  * network edges;
#  * network HVDC interconnector data;
#  * network AC interconnector data;
#  * network nodes;
#  * NEM demand and dispatch data for 2017.
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

import ipyparallel as ipp

from pyomo.environ import *

import matplotlib.pyplot as plt

# Seed random number generator
np.random.seed(seed=10)


# ## Declare paths to files

# In[2]:


# Core data directory
data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data'))

# Compiled model data directory
model_data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, '1_compile_data'))

# Output directory
output_dir = os.path.abspath(os.path.join(os.path.curdir, 'output'))


# ## Model data

# In[3]:


# Load dictionary created by 'compile_data.ipynb'
with open(os.path.join(model_data_dir, 'output', 'model_data.pickle'), 'rb') as f:
    model_data = pickle.load(f)

# Generator information
df_g = model_data['df_g']

# Admittance matrix
df_Y = model_data['df_Y']

# Node data summary
df_m = model_data['df_m']

# HVDC incidence matrix
df_hvdc_c = model_data['df_hvdc_c']

# Dictionary mapping network zones to node IDs
zones = model_data['zones']

# AC interconnector connection points
df_ac_i = model_data['df_ac_i']

# AC interconnector flow limits
df_ac_i_lim = model_data['df_ac_i_lim']

# HVDC link connection points and flow limits
df_hvdc = model_data['df_hvdc']

# Demand time series for each NEM region
df_regd = model_data['df_regd']

# Time series for power injections from intermittent sources at each node (e.g. wind and solar)
df_inter = model_data['df_inter']

# DUID dispatch signals for 2017 (from MMSDM database)
df_dis = model_data['df_dis']

# Fixed power injections from hydro plant at each node
df_fx_hydro = model_data['df_fx_hydro']


# Perturb generator SRMCs by a random number uniformly distributed between 0 and 1. Unique SRMCs assist the LP to find a unique solution.

# In[4]:


df_g['SRMC_2016-17'] = df_g['SRMC_2016-17'].map(lambda x: x + np.random.uniform(0, 1))
df_g['SRMC_2016-17'].head()


# Save the adjusted generator information dataframe containing updated costs.

# In[5]:


with open(os.path.join(output_dir, 'df_g.pickle'), 'wb') as f:
    pickle.dump(df_g, f)


# ### Setup parallel processing

# In[6]:


# # Profile for cluster
# client_path = 'C:/Users/eee/AppData/Roaming/SPB_Data/.ipython/profile_parallel/security/ipcontroller-client.json'
# c = ipp.Client(client_path)
# v = c[:]


# ## Model
# 
# ### Time horizon
# Construct list of dates over which the model should be run.

# In[7]:


def get_ordinal_list(start_date, end_date):
    """Given start and end dates (as strings) give list of ordinal dates between those days"""

    # Start date (ordinal format)
    ss_ord = dt.datetime.toordinal(dt.datetime.strptime(start_date, '%Y-%m-%d'))

    # End date (ordinal format)
    es_ord = dt.datetime.toordinal(dt.datetime.strptime(end_date, '%Y-%m-%d'))

    # List of ordinal dates between the start and end dates
    ordinal_list = [i for i in range(ss_ord, es_ord + 1)]
    return ordinal_list

def get_load_profile_dates(selected_seasons):
    """Choose random days in each season and construct load profile
    
    Parameters
    ----------
    selected_seasons : list
        List of seasons for which a random day should be selected. 
        Choose from ['summer', 'autumn', 'winter', 'spring']
    
    Returns
    -------
    load_profile : pandas datetime index
        Index consisting of hourly timestamps for the randomly selected
        days in each season specified.    
    """
    
    # Season start and end dates
    seasons = dict()
    seasons['spring'] = {'start': '2017-09-01', 'end': '2017-11-30'}
    seasons['autumn'] = {'start': '2017-03-01', 'end': '2017-08-31'}
    seasons['winter'] = {'start': '2017-06-01', 'end': '2017-08-31'}

    # For each season, get the days (in ordinal format) belonging to that season
    seasons_ord = {key: get_ordinal_list(seasons[key]['start'], seasons[key]['end']) for key in seasons.keys()}

    # Add summer by subtracting the set of days from March - November from all days in the year
    seasons_ord['summer'] = list(set(get_ordinal_list('2017-01-01', '2017-12-31')) - set(get_ordinal_list('2017-03-01', '2017-11-30')))

    # Container for date time indices for days selected in each season.
    dt_indices = []

    for s in selected_seasons:
        if s not in seasons_ord.keys():
            raise(Exception('Can only choose from {0}'.format(seasons.keys())))
            
        # Selected day
        ps = dt.datetime.fromordinal(np.random.choice(seasons_ord[s])) + dt.timedelta(hours=1)
        
        # Time indices over the course of that day (hourly resolution)
        dt_indices.append(pd.date_range(ps, periods=24, freq='1H'))
        
    # Load profile obtained by combining multiple days 
    load_profile = dt_indices[0].union_many(dt_indices[1:])
    return load_profile

load_profile_dates = get_load_profile_dates(['summer', 'winter'])
load_profile_dates    


# ### Mathematical program
# Wrap optimisation model in function. Allows possibility for parallel processing using ipyparrallel.

# In[8]:


def run_model(date_range_list, model_type=None, mode=None, fix_hydro=False, tau_list=None, phi_list=None, E_list=None, R_list=None, target_price_list=None, fix_phi=None, fix_tau=None, stream_solver=False, **model_data):

    # Import packages locally within function to allow possibility of parallel model runs
    import time
    from math import pi

    import numpy as np    
    from pyomo.environ import ConcreteModel, Set, Param, NonNegativeReals, Var, Constraint, Objective, minimize, Suffix, SolverFactory

    # Record start time for simulation run
    t0 = time.time()

    # Checking model options are correctly inputted
    # ---------------------------------------------
    if model_type not in ['dcopf', 'mppdc']:
        raise(Exception("Must specify either 'dcopf' or 'mppdc' as the model type"))
    
    if model_type is 'mppdc' and mode not in ['calc_phi_tau', 'calc_phi', 'calc_fixed']:
        raise(Exception("If model_type 'mppdc' specified, must choose from 'calc_phi_tau', 'calc_phi', 'calc_fixed'"))

    # Data
    df_m = model_data['df_m']
    df_g = model_data['df_g']
    df_Y = model_data['df_Y']
    df_hvdc_c = model_data['df_hvdc_c']
    df_ac_i = model_data['df_ac_i']
    df_ac_i_lim = model_data['df_ac_i_lim']
    df_regd = model_data['df_regd']
    df_inter = model_data['df_inter']
    df_fx_hydro = model_data['df_fx_hydro']


    # Initialise model object
    # -----------------------
    model = ConcreteModel(name='DCOPF')


    # Setup solver
    # ------------
    solver = 'cplex'
    solver_io = 'lp'
    keepfiles = False
    #solver_opt = {'BarHomogeneous': 1, 'FeasibilityTol': 1e-4, 'BarConvTol': 1e-4, 'ScaleFlag': 2, 'NumericFocus': 3}
    solver_opt = {}
    model.dual = Suffix(direction=Suffix.IMPORT)
    opt = SolverFactory(solver, solver_io=solver_io)


    # Sets
    # ----
    # Trading intervals
    if model_type is 'dcopf': model.T = Set(initialize=['DUMMY'])
    if model_type is 'mppdc': model.T = Set(initialize=RangeSet(len(date_range_list[0])))

    # Nodes
    model.I = Set(initialize=df_Y.columns)   

    # Network reference nodes (Mainland and Tasmania)
    model.N = Set(initialize=list(zones.values()))

    # AC network edges
    ac_edges = [(df_Y.columns[i], df_Y.columns[j]) for i, j in zip(np.where(df_Y != 0)[0], np.where(df_Y != 0)[1]) if (i < j)]
    model.K = Set(initialize=ac_edges)

    # HVDC links
    model.M = Set(initialize=df_hvdc_c.index)

    # Generators
    if fix_hydro:
        mask = (df_g['SCHEDULE_TYPE'] == 'SCHEDULED') & ~(df_g['FUEL_CAT'] == 'Hydro')
        model.G = Set(initialize=df_g[mask].index)
    else:
        mask = (df_g['SCHEDULE_TYPE'] == 'SCHEDULED')
        model.G = Set(initialize=df_g[mask].index)


    # Parmaters
    # ---------
    # Trading interval length [h]
    model.L = Param(initialize=1)

    # Generation lower bound [MW]
    def P_MIN_rule(model, g):
        return 0
    model.P_MIN = Param(model.G, initialize=P_MIN_rule)

    # Generation upper bound [MW]
    def P_MAX_rule(model, g):
        return float(df_g.loc[g, 'REG_CAP'])
    model.P_MAX = Param(model.G, initialize=P_MAX_rule)

    # Voltage angle difference between connected nodes i and j lower bound [rad]
    model.VANG_MIN = Param(initialize=float(-pi / 2))

    # Voltage angle difference between connected nodes i and j upper bound [rad]
    model.VANG_MAX = Param(initialize=float(pi / 2))

    # Susceptance matrix [pu]
    def B_rule(model, i, j):
        return float(np.imag(df_Y.loc[i, j]))
    model.B = Param(model.I, model.I, initialize=B_rule)

    # HVDC incidence matrix
    def C_rule(model, m, i):
        return float(df_hvdc_c.loc[m, i])
    model.C = Param(model.M, model.I, initialize=C_rule)

    # HVDC reverse flow from node i to j lower bound [MW]
    def H_MIN_rule(model, m):
        return - float(df_hvdc.loc[m, 'REVERSE_LIMIT_MW'])
    model.H_MIN = Param(model.M, initialize=H_MIN_rule)

    # HVDC forward flow from node i to j upper bound [MW]
    def H_MAX_rule(model, m):
        return float(df_hvdc.loc[m, 'FORWARD_LIMIT_MW'])
    model.H_MAX = Param(model.M, initialize=H_MAX_rule)

    # AC power flow limits on branches
    def F_MAX_rule(model, i, j):
        return 99999
    model.F_MAX = Param(model.K, initialize=F_MAX_rule, mutable=True)

    def F_MIN_rule(model, i, j):
        return -99999
    model.F_MIN = Param(model.K, initialize=F_MIN_rule, mutable=True)

    # Adjust power flow limits for major AC interconnectors
    for index, row in df_ac_i.drop('VIC1-NSW1').iterrows():
        i, j = row['FROM_NODE'], row['TO_NODE']

        # Must take into account direction of branch flow
        if i < j:
            model.F_MAX[i, j] = df_ac_i_lim.loc[index, 'FORWARD_LIMIT_MW']
            model.F_MIN[i, j] = - df_ac_i_lim.loc[index, 'REVERSE_LIMIT_MW']
        else:
            model.F_MAX[j, i] = df_ac_i_lim.loc[index, 'REVERSE_LIMIT_MW']
            model.F_MIN[j, i] = - df_ac_i_lim.loc[index, 'FORWARD_LIMIT_MW']

    # Generator emissions intensities [tCO2/MWh]
    def E_rule(model, g):
        return float(df_g.loc[g, 'EMISSIONS'])
    model.E = Param(model.G, initialize=E_rule)

    # Generator short run marginal cost [$/MWh]
    def A_rule(model, g):
        return float(df_g.loc[g, 'SRMC_2016-17'])
    model.A = Param(model.G, initialize=A_rule)

    # System base [MVA]
    model.S = Param(initialize=100)
    
    # Revenue constraint [$]
    model.R = Param(initialize=0, mutable=True)
    
    # Target wholsale electricity price [$/MWh]
    model.target_price = Param(initialize=30, mutable=True)


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
    def LL_PRIM_rule(b, t):
        # Parameters
        # ----------
        # Demand at each node (to be set prior to running model)
        b.P_D = Param(model.I, initialize=0, mutable=True)

        # Intermittent power injection at each node (to be set prior to running model)
        b.P_R = Param(model.I, initialize=0, mutable=True)


        # Variables
        # ---------
        b.P = Var(model.G)
        b.vang = Var(model.I)
        b.H = Var(model.M)


        # Constraints
        # -----------
        def P_lb_rule(b, g):
            return model.P_MIN[g] - b.P[g] <= 0
        b.P_lb = Constraint(model.G, rule=P_lb_rule)

        def P_ub_rule(b, g):
            return b.P[g] - model.P_MAX[g] <= 0
        b.P_ub = Constraint(model.G, rule=P_ub_rule)

        def vang_diff_lb_rule(b, i, j):
            return model.VANG_MIN - b.vang[i] + b.vang[j] <= 0
        b.vang_diff_lb = Constraint(model.K, rule=vang_diff_lb_rule)

        def vang_diff_ub_rule(b, i, j):
            return b.vang[i] - b.vang[j] - model.VANG_MAX <= 0
        b.vang_diff_ub = Constraint(model.K, rule=vang_diff_ub_rule)

        def vang_ref_rule(b, n):
            return b.vang[n] == 0
        b.vang_ref = Constraint(model.N, rule=vang_ref_rule)

        def power_balance_rule(b, i):
            # Branches connected to node i
            K_i = [k for k in model.K if i in k]

            # Nodes connected to node i
            I_i = [ii for branch in K_i for ii in branch if (ii != i)]

            return (-model.S * sum(model.B[i, j] * (b.vang[i] - b.vang[j]) for j in I_i)
                   - sum(model.C[m, i] * b.H[m] for m in model.M)
                   - b.P_D[i]
                   + sum(b.P[g] for g in df_m.loc[i, 'DUID'] if g in model.G)
                   + b.P_R[i] == 0)
        b.power_balance = Constraint(model.I, rule=power_balance_rule)

        def ac_flow_lb_rule(b, i, j):
            return model.F_MIN[i, j] - model.S * model.B[i, j] * (b.vang[i] - b.vang[j]) <= 0
        b.ac_flow_lb = Constraint(model.K, rule=ac_flow_lb_rule)

        def ac_flow_ub_rule(b, i, j):
            return model.S * model.B[i, j] * (b.vang[i] - b.vang[j]) - model.F_MAX[i, j] <= 0
        b.ac_flow_ub = Constraint(model.K, rule=ac_flow_ub_rule)

        def hvdc_flow_lb_rule(b, m):
            return model.H_MIN[m] - b.H[m] <= 0
        b.hvdc_flow_lb = Constraint(model.M, rule=hvdc_flow_lb_rule)

        def hvdc_flow_ub_rule(b, m):
            return b.H[m] - model.H_MAX[m] <= 0
        b.hvdc_flow_ub = Constraint(model.M, rule=hvdc_flow_ub_rule)


    # Dual block (indexed over T) (model.LL_DUAL)
    # -------------------------------------------
    def LL_DUAL_rule(b, t):
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
        def dual_cons_1_rule(b, g):
            # Node at which generator g is located
            f_g = df_g.loc[g, 'NODE']

            # Don't apply scheme to existing hydro plant
            if df_g.loc[g, 'FUEL_CAT'] == 'Hydro':
                return model.A[g] - b.alpha[g] + b.beta[g] - b.lambda_var[f_g] == 0
            else:
                return model.A[g] + ((model.E[g] - model.phi) * model.tau) - b.alpha[g] + b.beta[g] - b.lambda_var[f_g] == 0
        b.dual_cons_1 = Constraint(model.G, rule=dual_cons_1_rule)

        def dual_cons_2_rule(b, i):
            # Branches connected to node i
            K_i = [k for k in model.K if i in k]

            # Nodes connected to node i
            I_i = [ii for branch in K_i for ii in branch if (ii != i)]

            return (sum( (b.gamma[k] - b.delta[k] + (model.B[k] * model.S * (b.kappa[k] - b.eta[k])) ) * (np.sign(i - k[0]) + np.sign(i - k[1])) for k in K_i)
                    + sum(model.S * ((b.lambda_var[i] * model.B[i, j]) - (b.lambda_var[j] * model.B[j, i])) for j in I_i)
                    + sum(b.zeta[n] for n in model.N if n == i) == 0)
        b.dual_cons_2 = Constraint(model.I, rule=dual_cons_2_rule)

        def dual_cons_4_rule(b, m):
            return sum(b.lambda_var[i] * model.C[m, i] for i in model.I) - b.omega[m] + b.psi[m] == 0
        b.dual_cons_4 = Constraint(model.M, rule=dual_cons_4_rule)


    # Strong duality constraints (indexed over T)
    # -------------------------------------------
    def SD_CONS_rule(model, t):
        return (sum(model.LL_PRIM[t].P[g] * ( model.A[g] + (model.E[g] - model.phi) * model.tau ) if df_g.loc[g, 'FUEL_CAT'] == 'Fossil' else model.LL_PRIM[t].P[g] * model.A[g] for g in model.G)
                == sum(model.LL_PRIM[t].P_D[i] * model.LL_DUAL[t].lambda_var[i] - (model.LL_PRIM[t].P_R[i] * model.LL_DUAL[t].lambda_var[i]) for i in model.I)
                + sum((model.LL_DUAL[t].omega[m] * model.H_MIN[m]) - (model.LL_DUAL[t].psi[m] * model.H_MAX[m]) for m in model.M)
                + sum(model.LL_DUAL[t].alpha[g] * model.P_MIN[g] for g in model.G)
                - sum(model.LL_DUAL[t].beta[g] * model.P_MAX[g] for g in model.G)
                + sum((model.VANG_MIN * model.LL_DUAL[t].gamma[k]) - (model.VANG_MAX * model.LL_DUAL[t].delta[k]) + (model.LL_DUAL[t].kappa[k] * model.F_MIN[k]) - (model.LL_DUAL[t].eta[k] * model.F_MAX[k]) for k in model.K))

    # Run DCOPF
    # ---------
    if model_type is 'dcopf':   
        # Build model
        model.LL_PRIM = Block(model.T, rule=LL_PRIM_rule)

        # Fix phi and tau
        model.phi.fix(fix_phi)
        model.tau.fix(fix_tau)

        # DCOPF objective
        # ---------------
        def dcopf_objective_rule(model):
            return sum(model.LL_PRIM[t].P[g] * (model.A[g] + ((model.E[g] - model.phi) * model.tau) ) if df_g.loc[g, 'FUEL_CAT'] == 'Fossil' else model.LL_PRIM[t].P[g] * model.A[g] for t in model.T for g in model.G)
        model.dcopf_objective = Objective(rule=dcopf_objective_rule, sense=minimize)

        # Dictionary to store results
        results = {}
        
        # Solve model for each time period
        for t in date_range_list:
            # Update demand and intermittent power injections at each node
            for i in model.I:
                # Demand
                model.LL_PRIM['DUMMY'].P_D[i] = float(df_regd.loc[t, df_m.loc[i, 'NEM_REGION']] * df_m.loc[i, 'PROP_REG_D'])

                # Intermittent injections (+ fixed power injections from hydro plant if fix_hydro = True)
                if fix_hydro:
                    model.LL_PRIM['DUMMY'].P_R[i] = float(df_inter.loc[t, i]) + float(df_fx_hydro.loc[t, i])
                else:
                    model.LL_PRIM['DUMMY'].P_R[i] = float(df_inter.loc[t, i])

            # Solve model
            r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)
            print('Finished solving DCOPF for period {}'.format(t))
            
            # Store model output
            model.solutions.store_to(r)
            results[t] = r
            results[t]['fix_hydro'] = fix_hydro
            results[t]['fix_phi'] = fix_phi
            results[t]['fix_tau'] = fix_tau
            
            # Store fixed nodal power injections and demand at each node
            for i in model.I:
                # Fixed nodal power injections
                results[t]['Solution'][0]['Variable']['LL_PRIM[DUMMY].P_R[{0}]'.format(i)] = {'Value': model.LL_PRIM['DUMMY'].P_R[i].value}
                
                # Demand at each node
                results[t]['Solution'][0]['Variable']['LL_PRIM[DUMMY].P_D[{0}]'.format(i)] = {'Value': model.LL_PRIM['DUMMY'].P_D[i].value}
            
        return results

    # Run MPPDC
    # ---------
    if model_type is 'mppdc':
        # Build model
        print('Building primal block')
        model.LL_PRIM = Block(model.T, rule=LL_PRIM_rule)
        print('Building dual block')
        model.LL_DUAL = Block(model.T, rule=LL_DUAL_rule)
        print('Building strong duality constraints')
        model.SD_CONS = Constraint(model.T, rule=SD_CONS_rule)
        print('Finished building blocks')
        
        # Add revenue constraint to model if values of R are specified
        if R_list:
            # Exclude existing hydro and renewable from EIS (prevent windfall profits to existing generators)
            eligible_gens = [g for g in model.G if df_g.loc[g, 'FUEL_CAT'] == 'Fossil']
            model.R_CONS = Constraint(expr=sum((model.E[g] - model.phi) * model.tau * model.LL_PRIM[t].P[g] * model.L for t in model.T for g in eligible_gens) >= model.R)

        # Dummy variables used to minimise difference between average price and target price
        model.x_1 = Var(within=NonNegativeReals)
        model.x_2 = Var(within=NonNegativeReals)

        # Expression for average wholesale price
        model.avg_price = Expression(expr=(sum(model.LL_DUAL[t].lambda_var[i] * model.L * model.LL_PRIM[t].P_D[i] for t in model.T for i in model.I)
                                           / sum(model.L * model.LL_PRIM[t].P_D[i] for t in model.T for i in model.I)))

        # Constraints used to minimise difference between average wholesale price and target
        model.x_1_cons = Constraint(expr=model.x_1 >= model.avg_price - model.target_price)
        model.x_2_cons = Constraint(expr=model.x_2 >= model.target_price - model.avg_price)

        # MPPDC objective function
        def mppdc_objective_rule(model):
            return model.x_1 + model.x_2
        model.mppdc_objective = Objective(rule=mppdc_objective_rule, sense=minimize)

        
        # Useful functions
        # ----------------
        def _fix_LLPRIM_vars():
            """Fix generator output, voltage angles, load shedding and HVDC power flows."""
            for t in model.T:
                for g in model.G:
                    model.LL_PRIM[t].P[g].fix()
                for m in model.M:
                    model.LL_PRIM[t].H[m].fix()
                for i in model.I:
                    model.LL_PRIM[t].vang[i].fix()
                    
        def _unfix_LLPRIM_vars():
            """Unfix generator output, voltage angles, load shedding and HVDC power flows."""
            for t in model.T:
                for g in model.G:
                    model.LL_PRIM[t].P[g].unfix()
                for m in model.M:
                    model.LL_PRIM[t].H[m].unfix()
                for i in model.I:
                    model.LL_PRIM[t].vang[i].unfix()
                    
        def _store_output(results, index):
            """Store parameters and selected variable values from model output"""
            
            # Model options
            results[index]['date_range'] = date_range
            results[index]['fix_hydro'] = fix_hydro

            # Store generator output
            print('Storing generator output')
            for t in model.T:
                for g in model.G:
                    results[index]['Solution'][0]['Variable']['LL_PRIM[{0}].P[{1}]'.format(t, g)] = {'Value': model.LL_PRIM[t].P[g].value}

            # Store fixed nodal power injections and nodal demand
            print('Storing fixed nodal power injections')
            for t in model.T:
                for i in model.I:
                    # Fixed nodal power injections
                    results[index]['Solution'][0]['Variable']['LL_PRIM[{0}].P_R[{1}]'.format(t, i)] = {'Value': model.LL_PRIM[t].P_R[i].value}

                    # Demand
                    results[index]['Solution'][0]['Variable']['LL_PRIM[{0}].P_D[{1}]'.format(t, i)] = {'Value': model.LL_PRIM[t].P_D[i].value}

            results[index]['Solution'][0]['Variable']['tau'] = {'Value': model.tau.value}
            results[index]['Solution'][0]['Variable']['phi'] = {'Value': model.phi.value}
            
            return results

        # Solve model for each policy parameter scenario
        # ----------------------------------------------
        print('Updating demand and fixed power injection parameters')
        for index, date_range in enumerate(date_range_list):
            # Initialise dictionary to store model output
            results = {}
            
            for p in model.T:
                # Time stamp
                t = date_range[p - 1]

                for i in model.I:
                    # Node demand [MW]
                    model.LL_PRIM[p].P_D[i] = float(df_regd.loc[t, df_m.loc[i, 'NEM_REGION']] * df_m.loc[i, 'PROP_REG_D'])

                    # Power injections from intermittent sources [MW] (+ hydro if fix_hydro = True)
                    if fix_hydro:
                        model.LL_PRIM[p].P_R[i] = float(df_inter.loc[t, i]) + float(df_fx_hydro.loc[t, i])
                    else:
                        model.LL_PRIM[p].P_R[i] = float(df_inter.loc[t, i])     
        

            if mode is 'calc_phi_tau':                
                # Emissions intensity targets
                for E in E_list:
                    
                    # Target wholesale electricity price
                    for target_price in target_price_list:
                        model.target_price = target_price
                        
                        # Revenue constraints
                        if not R_list:
                            R_list = [None]
                        
                        for R in R_list:
                            # Start time for scenario run
                            t0 = time.time()
                            
                            # Construct file name based on parameters
                            fname = 'MPPDC_CalcPhiTau_E_{0}_R_{1}_tarp_{2:.3f}_FixHydro_{3}.pickle'.format(E, R, target_price, fix_hydro)
                            print('Starting scenario {0}'.format(fname))
                            
                            # Initialise lower and upper permit price bounds. Fix baseline to zero.
                            tau_up = 100
                            tau_lo = 0
                            model.phi.fix(0)

                            # Fix value of tau to midpoint of upper and lower bounds
                            model.tau.fix((tau_up + tau_lo) / 2)

                            # Iteration counter
                            i = 0
                            iter_lim = 7
                            iter_lim_exceeded = False
                            while i < iter_lim:
                                # While tolerance not satisfied (0.01 tCO2/MWh) or iteration not limit exceeded (7)
                                # (Print message if tolerance limit exceeded)

                                # Run model and compute average emissions intensity
                                r_tau = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)

                                # Average emissions intensity
                                avg_em_int = (sum(model.LL_PRIM[t].P[g].value * df_g.loc[g, 'EMISSIONS'] for t in model.T for g in model.G)
                                              / (sum(model.LL_PRIM[t].P[g].value for t in model.T for g in model.G) + sum(model.LL_PRIM[t].P_R[i].value for t in model.T for i in model.I)))
                                print('Finished iteration {0}, average emissions intensity = {1}'.format(i, avg_em_int))

                                # Check if emissions intensity is sufficiently close to target
                                if abs(E - avg_em_int) < 0.01:
                                    break

                                else:                
                                    # If emissions intensity > target, set lower bound to previous guess, recompute new permit price
                                    if avg_em_int > E: 
                                        tau_lo = model.tau.value
                                        model.tau.fix((tau_up + tau_lo) / 2)

                                    # If emissions intensity < target, set upper bound to previous guess, recompute new permit price
                                    else:
                                        tau_up = model.tau.value
                                        model.tau.fix((tau_up + tau_lo) / 2)

                                i += 1
                                if i == iter_lim:
                                    iter_lim_exceeded = True
                                    print('Iteration limit exceeded. Exiting loop.')

                            # Fix lower level primal variables to their current values
                            _fix_LLPRIM_vars()

                            # Free phi
                            model.phi.unfix()

                            # Re-run model to compute baseline that optimises average price deviation objective
                            r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)

                            # Store solutions
                            model.solutions.store_to(r)
                            results[index] = r
                            results[index]['E'] = E
                            results[index]['R'] = R
                            results[index]['target_price'] = target_price
                            results[index]['iter_lim_exceeded'] = iter_lim_exceeded
                            results[index]['date_range'] = date_range
                            results[index]['fix_hydro'] = fix_hydro
                            
                            # Also store values for some fixed parameters
                            results = _store_output(results, index)
                            
                            with open(os.path.join(output_dir, fname), 'wb') as f:
                                pickle.dump(results, f)

                            # Unfix lower level problem primal variables
                            _unfix_LLPRIM_vars()

                            print('Finished {0} in {1}s'.format(fname, time.time() - t0))

            if mode is 'calc_phi':
                 for tau in tau_list:
                        for target_price in target_price_list:
                            if not R_list:
                                R_list = [None]
                            for R in R_list:     
                                # Start time
                                t0 = time.time()
                                
                                # Fix phi and tau, solve model
                                model.phi.fix(0)
                                model.tau.fix(tau)
                                model.target_price = target_price
                                
                                r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)
                                print('Finished first stage')
                                
                                # Fix lower level primal variables to their current values
                                _fix_LLPRIM_vars()
                                
                                # Free phi
                                model.phi.unfix()

                                # Re-run model to compute baseline that optimises average price deviation objective
                                r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)                               

                                # Store solutions
                                model.solutions.store_to(r)
                                results[index] = r
                                results[index]['R'] = R
                                results[index]['target_price'] = target_price
                                results[index]['date_range'] = date_range
                                results[index]['fix_hydro'] = fix_hydro
                                
                                # Also store values for fixed parameters
                                results = _store_output(results, index)

                                # Construct file name based on parameters
                                fname = 'MPPDC_CalcPhi_tau_{0}_tarp_{1:.3f}_R_{2}_FixHydro_{3}.pickle'.format(tau, target_price, R, fix_hydro)

                                with open(os.path.join(output_dir, fname), 'wb') as f:
                                    pickle.dump(results, f)
                                
                                # Unfix lower level problem primal variables
                                _unfix_LLPRIM_vars()

                                print('Finished {0} in {1}s'.format(fname, time.time() - t0))
            
            if mode is 'calc_fixed':
                for tau in tau_list:
                    for phi in phi_list:
                        for target_price in target_price_list:
                            # Start time
                            t0 = time.time()
                            
                            # Construct file name based on parameters
                            fname = 'MPPDC_CalcFixed_tau_{0}_phi_{1}_tarp_{2}_FixHydro_{3}.pickle'.format(tau, phi, target_price, fix_hydro)
                            print('Starting scenario {0}'.format(fname))

                            # Fix policy parameters
                            model.tau.fix(tau)
                            model.phi.fix(phi)
                            model.target_price = target_price

                            # Solve model
                            r = opt.solve(model, keepfiles=keepfiles, tee=stream_solver, options=solver_opt)

                            # Store solutions
                            model.solutions.store_to(r)
                            results[index] = r
                            results[index]['target_price'] = target_price
                            results[index]['date_range'] = date_range
                            results[index]['fix_hydro'] = fix_hydro
                            
                            # Also store values for fixed parameters
                            results = _store_output(results, index)

                            with open(os.path.join(output_dir, fname), 'wb') as f:
                                pickle.dump(results, f)

                            print('Finished {0} in {1}s'.format(fname, time.time() - t0))                                


# ### DCOPF Results

# In[9]:


# Dates for which the DCOPF model should be executed
# start = '2017-01-01 01:00:00'
# end = '2018-01-01 00:00:00'
# date_range = pd.date_range(start, end, freq='1H')

# Run DCOPF for each period in synthetic time series
# --------------------------------------------------
date_range = load_profile_dates

# Split long date range into n chunks
def create_date_range_chunks(date_range, n):
    """Split date_range into a list of n chunks"""
    for i in range(0, len(date_range), n):
        yield date_range[i:i+n]
date_range_chunks = create_date_range_chunks(date_range, 876)

tau_list = [0]
phi_list = [0]
fix_hydro = True

i = 1
# For each date range chunk, fix the value run DCOPF model for different values of phi and tau
for date_range_chunk in date_range_chunks:
    for fix_tau in tau_list:
        for fix_phi in phi_list:        
            results = run_model(date_range_chunk, model_type='dcopf', fix_hydro=fix_hydro, stream_solver=False, fix_phi=fix_phi, fix_tau=fix_tau, **model_data)
            
            # Construct file name
            fname = 'DCOPF_CalcFixed_tau_{0}_phi_{1}_FixHydro_{2}-{3}.pickle'.format(fix_tau, fix_phi, fix_hydro, i)
            
            # Save to file
            with open(os.path.join(output_dir, fname), 'wb') as f:
                pickle.dump(results, f)
    i += 1


# ### MPPDC Results
# Define time horizon for model.

# In[10]:


date_range_list = [load_profile_dates] # One day randomly selected from summer and winter


# #### Fixed permit price and baseline
# Fix permit price and baseline. Results should be the same for the DCOPF base-case.

# In[11]:


# Policy parameter scenarios
target_price_list = [30] # Not required, but could potentially impact way in which solver performs, hence the reason for recording this value
phi_list = [0]
tau_list = list(range(0, 71, 2))


# FOR TESTING - NEED TO CHANGE LATER
# ----------------------------------
# tau_list = list(range(0, 81, 20))


# Run model
# ---------
run_model(date_range_list, model_type='mppdc', mode='calc_fixed', fix_hydro=True, stream_solver=True, phi_list=phi_list, tau_list=tau_list, target_price_list=target_price_list, **model_data)


# #### Compute baseline given fixed permit price
# Calculate optimal emissions intensity baseline for a given permit price and average wholesale price target. Set the base-case wholesale price target to be the business-as-usual (BAU) price.

# In[12]:


def compute_bau_price():
    "Compute business as usual price"

    with open(os.path.join(output_dir, 'MPPDC_CalcFixed_tau_0_phi_0_tarp_30_FixHydro_True.pickle'), 'rb') as f:
        bau_scenario = pickle.load(f)

    # Initialise variables to sum total demand
    total_demand = 0
    total_revenue = 0

    # Loop through time stamps and nodes, computing revenues and demand at each node
    for t in range(1, 49):
        for i in range(1, 913):
            price = bau_scenario[0]['Solution'][0]['Variable']['LL_DUAL[{0}].lambda_var[{1}]'.format(t, i)]['Value']
            demand = bau_scenario[0]['Solution'][0]['Variable']['LL_PRIM[{0}].P_D[{1}]'.format(t, i)]['Value']

            revenue = price * demand

            total_demand += demand
            total_revenue += revenue

    # Compute average price
    average_price = total_revenue / total_demand

    return average_price

bau_p = compute_bau_price()
print('Average price = {0}'.format(bau_p))


# Define parameters for policy scenarios.

# In[13]:


# Policy parameter scenarios
target_price_list = [0.8*bau_p, 0.9*bau_p, bau_p, 1.1*bau_p, 1.2*bau_p]
tau_list = list(range(2, 71, 2))


# FOR TESTING - NEED TO CHANGE LATER
# ----------------------------------
# target_price_list = [1.2*bau_p]
# tau_list = list(range(16, 71, 20))


# Run model
# ---------
run_model(date_range_list, model_type='mppdc', mode='calc_phi', fix_hydro=True, stream_solver=True, tau_list=tau_list, target_price_list=target_price_list, **model_data)


# #### Compute baseline and permit price given emissions intensity and wholesale price targets (and optionally a revenue constraint)
# 
# For a given emissions intensity target and wholesale price target, compute the required permit price and emissions intensity baseline. (Adding a revenue constraint is also optional).

# In[14]:


# # Policy parameter scenarios
# target_price_list = [25.38]
# E_list = [0.85]

# # Run model
# run_model(date_range_list, model_type='mppdc', mode='calc_phi_tau', fix_hydro=True, target_price_list=target_price_list, E_list=E_list, **model_data)

