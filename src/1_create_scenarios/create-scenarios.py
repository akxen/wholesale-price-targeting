
# coding: utf-8

# # Scenario Construction
# Demand and dispatch data are obtained from the Australian Energy Market Operator's (AEMO's) Market Management System Database Model (MMSDM) [1], and a k-means clustering algorithm is implemented using the method outlined in [2] to create a reduced set of representative operating scenarios. The dataset described in [3,4] is used to identify specific generators, and assign historic power injections or withdrawals to individual nodes.
# 
# In this analysis data for 2017 is considered, with demand and dispatch time series re-sampled to 30min intervals, corresponding to the length of a trading interval within Australia's National Electricity Market (NEM). Using these data a reduced set of 48 operating conditions are constructed. These operating scenarios are comprised on demand, intermittent renewable injections, and fixed power injections from hydro generators.
# 
# ## Import packages

# In[1]:


import os
import math
import pickle
import random
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialise random number generator
random.seed(1)

# Used to slice pandas DataFrames
idx = pd.IndexSlice


# ## Paths to files

# In[2]:


# Contains network information and generator parameters
data_dir = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data')

# MMSDM historic demand and dispatch signals
archive_dir = r'D:\nemweb\Reports\Data_Archive\MMSDM\zipped'

# Location for output files
output_dir = os.path.join(os.path.curdir, 'output')


# ## Import generator and network information

# In[3]:


# Generator information
df_g = pd.read_csv(os.path.join(data_dir, 'generators.csv'), index_col='DUID', dtype={'NODE': int})

# Network node information
df_n = pd.read_csv(os.path.join(data_dir, 'network_nodes.csv'), index_col='NODE_ID')


# ## Extract data
# Functions used to extract data from MMSDM tables.

# In[4]:


def dispatch_unit_scada(file):
    """Extract generator dispatch data
    
    Params
    ------
    file : bytes IO object
        Zipped CSV file of given MMSDM table
    
    Returns
    -------
    df : pandas DataFrame
        MMSDM table in formatted pandas DataFrame
    """
    
    # Columns to extract    
    cols = ['DUID', 'SCADAVALUE', 'SETTLEMENTDATE']

    # Read in data
    df = pd.read_csv(file, usecols=cols, parse_dates=['SETTLEMENTDATE'], skiprows=1)

    # Drop rows without DUIDs, apply pivot
    df = df.dropna(subset=['DUID']).pivot(index='SETTLEMENTDATE', columns='DUID', values='SCADAVALUE')
    
    return df


def tradingregionsum(file):
    """Extract half-hourly load data for each NEM region
    
    Params
    ------
    file : bytes IO object
        Zipped CSV file of given MMSDM table
    
    Returns
    -------
    df : pandas DataFrame
        MMSDM table in formatted pandas DataFrame
    """

    # Columns to extract    
    cols = ['REGIONID', 'TOTALDEMAND', 'SETTLEMENTDATE']

    # Read in data
    df = pd.read_csv(file, usecols=cols, parse_dates=['SETTLEMENTDATE'], skiprows=1)

    # Drop rows without DUIDs, apply pivot
    df = df.dropna(subset=['REGIONID']).pivot(index='SETTLEMENTDATE', columns='REGIONID', values='TOTALDEMAND')

    return df  


def get_data(archive_path, table_name, extractor_function):
    """Open CSV archive and extract data from zipped file
    
    Parameters
    ----------
    archive_path : str
        Path to MMSDM archive containing data for given year
    
    table_name : str
        Name of table in MMSDM archive from which data is to be extracted
        
    extractor_function : func
        Function that takes a bytes object of the unzipped table and returns a formatted DataFrame
        
    Returns
    -------
    df : pandas DataFrame
        Formatted DataFrame of the desired MMSDM table    
    """

    # Open MMSDM archive for a given year
    with zipfile.ZipFile(archive_path) as myzip:
        
        # All files of a particular type in archive (e.g. dispatch, quantity bids, price bands, load)
        zip_names = [f for f in myzip.filelist if (table_name in f.filename) and ('.zip' in f.filename)]

        # Check that only one zip file is returned, else raise exception
        if len(zip_names) != 1:
            raise Exception('Encounted {0} files in archive, should only encounter 1'.format(len(zip_names)))

        # Get name of csv in zipped folder
        csv_name = zip_names[0].filename.replace('.zip', '.CSV').split('/')[-1]

        # Convert zip files to BytesIO object
        zip_data = BytesIO(myzip.read(zip_names[0]))

        # Open inner zipfile and extract data using supplied function
        with zipfile.ZipFile(zip_data) as z:
            with z.open(csv_name) as f:
                df = extractor_function(f)      
    return df

# Historic demand and dispatch data
demand = []
dispatch = []

for i in range(1, 13):
    # Archive name and path from which data will be extracted
    archive_name = 'MMSDM_2017_{0:02}.zip'.format(i)
    archive_path = os.path.join(archive_dir, archive_name)

    # Extract data
    dispatch.append(get_data(archive_path, 'DISPATCH_UNIT_SCADA', dispatch_unit_scada))
    demand.append(get_data(archive_path, 'TRADINGREGIONSUM', tradingregionsum))

# Concatenate data from individual months into single DataFrames for load and dispatch
df_demand = pd.concat(demand, sort=True) # Demand
df_dispatch = pd.concat(dispatch, sort=True) # Dispatch

# Fill missing values
df_demand = df_demand.fillna(0)
    
# Resample to get average power output over 30min trading interval (instead of 5min) dispatch intervals
df_dispatch = df_dispatch.resample('30min', label='right', closed='right').mean()


# Re-index and format data:
# 
# 1. identify intermittent generators (wind and solar);
# 2. identify hydro generators;
# 3. compute nodal demand;
# 4. concatenate demand, hydro dispatch, and intermittent renewables dispatch into a single DataFrame.

# In[5]:


# Intermittent generators
mask_intermittent = df_g['FUEL_CAT'].isin(['Wind', 'Solar'])
df_g[mask_intermittent]

# Intermittent dispatch at each node
df_intermittent = (df_dispatch
                   .T
                   .join(df_g.loc[mask_intermittent, 'NODE'], how='left')
                   .groupby('NODE').sum()
                   .reindex(df_n.index, fill_value=0))
df_intermittent['level'] = 'intermittent'

# Hydro generators
mask_hydro = df_g['FUEL_CAT'].isin(['Hydro'])
df_g[mask_hydro]

# Hydro dispatch at each node
df_hydro = (df_dispatch
            .T
            .join(df_g.loc[mask_hydro, 'NODE'], how='left')
            .groupby('NODE').sum()
            .reindex(df_n.index, fill_value=0))
df_hydro['level'] = 'hydro'

# Demand at each node
def node_demand(row):
    return df_demand[row['NEM_REGION']] * row['PROP_REG_D']
df_node_demand = df_n.apply(node_demand, axis=1)
df_node_demand['level'] = 'demand'

# Concatenate intermittent, hydro, and demand series, add level to index
df_o = (pd.concat([df_node_demand, df_intermittent, df_hydro])
        .set_index('level', append=True)
        .reorder_levels(['level', 'NODE_ID']))


# ## K-nearest neighbours
# Construct clustering algorithm to transform the set of trading intervals into a reduced set of representative operating scenarios.

# In[6]:


def create_scenarios(df, k=1, max_iterations=100, stopping_tolerance=0):
    """Create representative demand and fixed power injection operating scenarios

    Parameters
    ----------
    df : pandas DataFrame
        Input DataFrame from which representative operating conditions 
        should be constructed

    k : int
        Number of clusters

    max_iterations : int
        Max number of iterations used to find centroid

    stopping_tolerance : float
        Max total difference between successive centroid iteration DataFrames

    Returns
    -------
    df_clustered : pandas DataFrame
        Operating scenario centroids and their associated duration.

    centroid_history : dict
        Dictionary where keys are the iteration number and values are a DataFrame describing
        the allocation of operating conditions to centroids, and the distance between these
        values. 
    """

    # Random time periods used to initialise centroids
    random_periods = random.sample(list(df.columns), k)
    df_centroids = df[random_periods]

    # Rename centroid DataFrame columns and keep track of initial labels
    timestamp_map = {timestamp: timestamp_key + 1 for timestamp_key, timestamp in enumerate(df_centroids.columns)}
    df_centroids = df_centroids.rename(columns=timestamp_map)

    def compute_distance(col):
        """Compute distance between each data associated with each trading interval, col, and all centroids.
        Return closest centroid.

        Params
        ------
        col : pandas Series
            Operating condition for trading interval

        Returns
        -------
        closest_centroid_ID : pandas Series
            Series with ID of closest centroid and the distance to that centroid
        """

        # Initialise minimum distance between data constituting a trading interval and all centroids to an 
        # arbitrarily large number
        min_distance = 9e9
        
        # Initially no centroid is defined as being closest to the given trading interval
        closest_centroid = None

        # Compute Euclidean (2-norm) distance between the data describing a trading interval, col, and
        # all centroids. Identify the closest centroid for the given trading interval.
        for centroid in df_centroids.columns:
            distance = math.sqrt(sum((df_centroids[centroid] - col) ** 2))

            # If present value less than minimum distance, update minimum distance and record centroid
            if distance <= min_distance:
                min_distance = distance
                closest_centroid = centroid

        # Return ID of closest centroid
        closest_centroid_ID = pd.Series(data={'closest_centroid': closest_centroid, 'distance': min_distance})

        return closest_centroid_ID


    def update_centroids(row):
        "Update centroids by taking element-wise mean value of all vectors in cluster"
        
        return df[row['SETTLEMENTDATE']].mean(axis=1)

    # History of computed centroids
    centroid_history = dict()

    for i in range(max_iterations):
        # Get closest centroids for each trading interval and save result to dictionary
        df_closest_centroids = df.apply(compute_distance)
        centroid_history[i] = df_closest_centroids

        # Timestamps belonging to each cluster
        clustered_timestamps = (df_closest_centroids.loc['closest_centroid']
                                .to_frame()
                                .reset_index()
                                .groupby('closest_centroid').agg(lambda x: list(x)))

        # Update centroids by computing average nodal values across series in each cluster
        df_centroids = clustered_timestamps.apply(update_centroids, axis=1).T

        # If first iteration, set total absolute distance to arbitrarily large number
        if i == 0:
            total_absolute_distance = 1e7

            # Lagged DataFrame in next iteration = DataFrame in current iteration
            df_centroids_lag = df_centroids
        
        else:
            # Element-wise absolute difference between current and previous centroid DataFrames
            df_centroids_update_distance = abs(df_centroids - df_centroids_lag)

            # Max total element-wise distance
            total_absolute_distance = df_centroids_update_distance.sum().sum()

            # Stopping condition
            if total_absolute_distance <= stopping_tolerance:
                print('Iteration number: {0} - Total absolute distance: {1}. Stopping criterion satisfied. Exiting loop.'.format(i+1, total_absolute_distance))
                break
            else:
                # Continue loop 
                df_centroids_lag = df_centroids   

        print('Iteration number: {0} - Difference between iterations: {1}'.format(i+1, total_absolute_distance))
        
        # Raise warning if loop terminates before stopping condition met
        if i == (max_iterations - 1):
            print('Max iteration limit exceeded before stopping tolerance satisfied.')


    # Get duration for each scenario
    # ------------------------------
    # Length of each trading interval (hours)
    interval_length = 0.5

    # Total number of hours for each scenario
    scenario_hours = (clustered_timestamps.apply(lambda x: len(x['SETTLEMENTDATE']), axis=1)
                      .to_frame().T
                      .mul(interval_length))

    # Renaming and setting duration index values
    scenario_hours = scenario_hours.rename(index={0: 'hours'})
    scenario_hours['level'] = 'duration'
    scenario_hours.set_index('level', append=True, inplace=True)

    # Final DataFrame with clustered values
    df_clustered = pd.concat([df_centroids, scenario_hours])

    # Convert column labels to type int
    df_clustered.columns = df_clustered.columns.astype(int)

    return df_clustered, centroid_history


# ## Create operating scenarios
# Create operating scenarios for different numbers of clusters and save to file.

# In[7]:


# Create operating scenarios for different numbers of clusters
for k in [48, 100]:
    # Create operating scenarios
    df_clustered, _ = create_scenarios(df=df_o, k=k, max_iterations=int(9e9), stopping_tolerance=0)
    
    # Save scenarios
    with open(os.path.join(output_dir, '{0}_scenarios.pickle'.format(k)), 'wb') as f:
        pickle.dump(df_clustered, f)


# ## References
# [1] - Australian Energy Markets Operator. Data Archive (2018). at [http://www.nemweb.com.au/#mms-data-model:download](http://www.nemweb.com.au/#mms-data-model:download)
# 
# [2] - Baringo L., Conejo, A. J., Correlated wind-power production and electric load scenarios for investment decisions. Applied Energy (2013).
# 
# [3] - Xenophon A. K., Hill D. J., Geospatial modelling of Australia's National Electricity Market allowing backtesting against historic data.  Scientific Data (2018).
# 
# [4] - Xenophon A. K., Hill D. J., Geospatial Modelling of Australia's National Electricity Market - Dataset (Version v1.3) [Data set]. Zenodo. [http://doi.org/10.5281/zenodo.1326942](http://doi.org/10.5281/zenodo.1326942)
