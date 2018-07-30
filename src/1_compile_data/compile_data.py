
# coding: utf-8

# # Emissions Intensity Scheme (EIS) Parameter Selection
# This notebook collates data from several sources which will then be used to test a mathematical framework for selecting EIS parameters.
# 
# Steps taken to compile data:
# 1. Import packages and declare declare paths to files
# 2. Load data:
#  * generator data;
#  * network edges;
#  * network HVDC interconnector data;
#  * network AC interconnector data;
#  * network nodes;
#  * NEM demand and dispatch data for 2017.
# 3. Combine network data:
#  * construct admittance matrix;
#  * compute demand signals for each node;
#  * construct signals for power injections from intermittent sources for each node;
#  * HVDC incidence matrix;
#  * define reference nodes.
# 6. Place model data in a dictionary and pickle.
# 
# ## Import packages

# In[1]:


import os
import pickle
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd


# ## Declare paths to files

# In[2]:


# Core data directory
data_dir = os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data'))

# Output directory
output_dir = os.path.abspath(os.path.join(os.path.curdir, 'output'))

# MMSDM data - SEE NOTE BELOW
archive_dir = 'D:\\nemweb\\Reports\\Data_Archive\\MMSDM\\zipped'


# `archive_dir` points to local directory containing MMSDM database files which were downloaded from [1]. If you  have already downloaded zipped monthly archives of MMSDM data and wish to replicate the results in their entirety, please update the archive directory path accordingly (so that it points to the directory containing MMSDM zipped archives on your machine). As will be shown below, I have created a class that extracts MMSDM data from zipped archives. Using this class I have collated unit dispatch and demand data, placed these data into Pandas DataFrames, and saved these DataFrames as pickled files. These pickled files are made available as part of the repository so you don't have to download the original MMSDM archives.

# ## Load data

# In[3]:


# Generator data
df_g = pd.read_csv(os.path.join(data_dir, 'generators.csv'), index_col='DUID', dtype={'NODE': np.int64})

# Network edges
df_e = pd.read_csv(os.path.join(data_dir, 'network_edges.csv'), index_col='LINE_ID')

# Network HVDC link information
df_hvdc = pd.read_csv(os.path.join(data_dir, 'network_hvdc_links.csv'), index_col='HVDC_LINK_ID')

# AC interconnector links
df_ac_i = pd.read_csv(os.path.join(data_dir, 'network_ac_interconnector_links.csv'), index_col='INTERCONNECTOR_ID')

# AC interconnector flow limits
df_ac_i_lim = pd.read_csv(os.path.join(data_dir, 'network_ac_interconnector_flow_limits.csv'), index_col='INTERCONNECTOR_ID')

# Network nodes
df_n = pd.read_csv(os.path.join(data_dir, 'network_nodes.csv'), index_col='NODE_ID')


# Class to extract data directly from MMSDM archives.

# In[4]:


class ExtractArchiveData:
    """Extract data from MMSDM archive files"""
    
    def _init_(self):
        pass
    
    @staticmethod
    def get_dispatch_unit_scada(file):
        """Extract dispatch data"""
    
        # Columns to extract    
        cols = ['DUID', 'SCADAVALUE', 'SETTLEMENTDATE']

        # Read in data
        df = pd.read_csv(file, usecols=cols, parse_dates=['SETTLEMENTDATE'], skiprows=1)

        # Drop rows without DUIDs, apply pivot
        df = df.dropna(subset=['DUID']).pivot(index='SETTLEMENTDATE', columns='DUID', values='SCADAVALUE')
        
        return df
    
    @staticmethod
    def get_tradingregionsum(file):
        """Extract half-hourly load data"""
    
        # Columns to extract    
        cols = ['REGIONID', 'TOTALDEMAND', 'SETTLEMENTDATE']

        # Read in data
        df = pd.read_csv(file, usecols=cols, parse_dates=['SETTLEMENTDATE'], skiprows=1)

        # Drop rows without DUIDs, apply pivot
        df = df.dropna(subset=['REGIONID']).pivot(index='SETTLEMENTDATE', columns='REGIONID', values='TOTALDEMAND')
        
        return df  
    
    @classmethod
    def extract_data(self, archive_name, file_name):
        """Given MMSDM monthly archive and file that must be extracted, open zips, transform data and return dataframe"""

        with zipfile.ZipFile(archive_name) as myzip:
            # All files of a particular type (e.g. dispatch, quantity bids, price bands, load)
            zip_names = [f for f in myzip.filelist if (file_name in f.filename) and ('.zip' in f.filename)]

            if len(zip_names) != 1:
                raise Exception('Encounted {0} files in archive, should only encounter 1'.format(len(zip_names)))

            # Get name of csv
            csv_name = zip_names[0].filename.replace('.zip', '.CSV').split('/')[-1]

            # Convert zip files to BytesIO object
            zip_data = BytesIO(myzip.read(zip_names[0]))

            with zipfile.ZipFile(zip_data) as z:
                with z.open(csv_name) as f:
                    if file_name is 'PUBLIC_DVD_DISPATCH_UNIT_SCADA':
                        df = self.get_dispatch_unit_scada(f)
                    elif file_name is 'PUBLIC_DVD_TRADINGREGIONSUM':
                        df = self.get_tradingregionsum(f)
                    else:
                        raise Exception('File type must be one in [PUBLIC_DVD_DISPATCH_UNIT_SCADA, PUBLIC_DVD_TRADINGREGIONSUM]')
        return df


# If pickled files already exists for dispatch and demand data load pickled files directly. Else, run class used to extract data from MMSDM archive.

# In[5]:


# If pickled dataframes already exist, extract load the dataframes directly
if os.path.isfile(os.path.join(output_dir, 'df_regd.pickle')) and os.path.isfile(os.path.join(output_dir, 'df_regd.pickle')):
    
    # Unit dispatch data
    with open(os.path.join(output_dir, 'df_dis.pickle'), 'rb') as f:
        df_dis = pickle.load(f)
        
    # Regional demand data
    with open(os.path.join(output_dir, 'df_regd.pickle'), 'rb') as f:
        df_regd = pickle.load(f)

# If either file doesn't exist, extract data from MMSDM archive
else:        
    # Names of individual files that should be extracted from each archived month
    file_names = ['PUBLIC_DVD_DISPATCH_UNIT_SCADA', 'PUBLIC_DVD_TRADINGREGIONSUM']

    # Lists to hold individual dataframes containing data for each archived month
    frames_dispatch = []
    frames_load = []

    # Loop over years
    for y in [2017]:
        # Loop over months (zero padded)
        for m in ['{0:02d}'.format(i) for i in range(1, 13)]:
            # Name of monthly archive from which data will be extracted
            archive_name = 'MMSDM_{0}_{1}.zip'.format(y, m)

            # Full path to archive
            archive_path = os.path.join(archive_dir, archive_name)

            # For each file type required to be extracted
            for file_name in file_names:
                # Extract data and return dataframe
                df = ExtractArchiveData.extract_data(archive_path, file_name)

                # Add dataframe to list of datframes
                if file_name is 'PUBLIC_DVD_DISPATCH_UNIT_SCADA':
                    frames_dispatch.append(df)
                elif file_name is 'PUBLIC_DVD_TRADINGREGIONSUM':
                    frames_load.append(df)

                print('Finished archive: {0}, filename: {1}'.format(archive_path, file_name))

    # Dispatch data for each DUID at 5 min dispatch intervals
    df_dis = pd.concat(frames_dispatch)

    # Regional demand (MW) at 30 min trading intervals
    df_regd = pd.concat(frames_load)
    
    # Save data
    with open(os.path.join(output_dir, 'df_dis.pickle'), 'wb') as f:
        pickle.dump(df_dis, f)
    
    with open(os.path.join(output_dir, 'df_regd.pickle'), 'wb') as f:
        pickle.dump(df_regd, f)


# ## Compile generator and network information
# ### Reindex nodes

# In[6]:


# Original node indices
df_index_map = df_n.index.to_frame().rename(columns={'NODE_ID': 'original'}).reset_index().drop('NODE_ID',axis=1)

# New node indices
df_index_map['new'] = df_index_map.apply(lambda x: x.name + 1, axis=1)

# Create dictionary mapping original node indices to new node indices
index_map = df_index_map.set_index('original')['new'].to_dict()


# Network nodes
# -------------
# Construct new index and assign to dataframe
new_index = pd.Index(df_n.apply(lambda x: index_map[x.name], axis=1), name=df_n.index.name)
df_n.index = new_index


# Network edges
# -------------
# Reindex 'from' and 'to' nodes in network edges dataframe
def reindex_from_and_to_nodes(row, order=False):
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
df_e[['FROM_NODE', 'TO_NODE']] = df_e.apply(reindex_from_and_to_nodes, args=(True,), axis=1)

# Sort lines by 'from' and 'to' node indices
df_e.sort_values(by=['FROM_NODE', 'TO_NODE'], inplace=True)


# Generators
# ----------
df_g['NODE'] = df_g['NODE'].map(lambda x: df_index_map.set_index('original')['new'].loc[x])


# Network HVDC links
# ------------------
df_hvdc[['FROM_NODE', 'TO_NODE']] = df_hvdc.apply(reindex_from_and_to_nodes, axis=1)


# Network interconnectors
# -----------------------
df_ac_i[['FROM_NODE', 'TO_NODE']] = df_ac_i.apply(reindex_from_and_to_nodes, axis=1)


# ### Regional demand signals

# In[7]:


# Resample to acheive a temporal resolution of 1hr
df_regd = df_regd.resample('H', closed='right', label='right').mean()


# ### Intermittent generation at each node

# In[8]:


# Resample to acheive a temporal resolution of 1hr
df_dis = df_dis.resample('H', closed='right', label='right').mean()

# Set NaN to 0
df_dis = df_dis.fillna(0)

# Set negative generation to zero
mask = df_dis < 0
df_dis[mask] = 0

# Semi-scheduled DUIDs (e.g. wind and solar)
ss_duids = df_g[df_g['SCHEDULE_TYPE'] == 'SEMI-SCHEDULED'].index

# Aggregate dispatch from intermittent generators
df_inter = df_dis[ss_duids].T.join(df_g['NODE']).groupby('NODE').sum().fillna(0).T

# Set negative dispatch to values to 0
mask = df_inter < 0
df_inter[mask] = 0

# Reindex such that each column represents a network node
df_inter = df_inter.reindex(df_n.index, axis=1, fill_value=0)


# ### Power injections from hydro plant at each node
# Can use this dataframe if it is decided to fix hydro output.

# In[9]:


# Find all hydro plant
mask_1 = df_g.FUEL_CAT == 'Hydro'
mask_2 = df_dis.T.index.isin(df_g[mask_1].index)

# Aggregate hydro output by the node to which the plant are connected
df_fx_hydro = df_dis.T[mask_2].join(df_g[['NODE']]).groupby('NODE').sum().T

# Reindex columns with node IDs
df_fx_hydro = df_fx_hydro.reindex(labels=df_n.index, axis='columns').fillna(0)


# ### HVDC incidence matrix

# In[10]:


# Incidence matrix for HVDC links
df_hvdc_c = pd.DataFrame(index=df_hvdc.index, columns=df_n.index, data=0)

for index, row in df_hvdc.iterrows():
    # From nodes assigned a value of 1 in the incidence matrix
    df_hvdc_c.loc[index, row['FROM_NODE']] = 1
    
    # To nodes assigned a value of -1 in the incidence matrix
    df_hvdc_c.loc[index, row['TO_NODE']] = -1
df_hvdc_c


# ### Construct network admittance matrix

# In[11]:


# Initialise dataframe
df_Y = pd.DataFrame(data=0j, index=df_n.index, columns=df_n.index)

# Off-diagonal elements
for index, row in df_e.iterrows():
    fn, tn = row['FROM_NODE'], row['TO_NODE']
    df_Y.loc[fn, tn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']
    df_Y.loc[tn, fn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']

# Diagonal elements
for i in df_n.index:
    df_Y.loc[i, i] = - df_Y.loc[i, :].sum()

# Add shunt susceptance to diagonal elements
for index, row in df_e.iterrows():
    fn, tn = row['FROM_NODE'], row['TO_NODE']
    df_Y.loc[fn, fn] += (row['B_PU'] / 2) * row['NUM_LINES']
    df_Y.loc[tn, tn] += (row['B_PU'] / 2) * row['NUM_LINES']

# Remove admittances for HVDC links
hvdc_links = [(df_hvdc_c.columns[i], df_hvdc_c.columns[j])  for i, j in zip(np.where(abs(df_hvdc_c) != 0)[0], np.where(abs(df_hvdc_c) != 0)[1])]
for link in hvdc_links:
    from_node, to_node = link
    df_Y.loc[from_node, to_node] = 0
    df_Y.loc[to_node, from_node] = 0

# Final admittance matrix
df_Y


# ### Reference nodes

# In[12]:


# Mainland reference node = Victoria's Regional Reference Node
def get_rrn(nem_region_id):
    """Return the Regional Reference Node (RRN) ID for a given NEM region"""
    
    mask = (df_n['NEM_REGION'] == nem_region_id) & (df_n['RRN'] == 1)
    return df_n.loc[mask].index.values[0]

# Mainland (zone 1) and Tasmanian (zone 2) reference node IDs
zones = {1: get_rrn('VIC1'), 2: get_rrn('TAS1')}
zones


# ### Summary of data at each node

# In[13]:


# DataFrame that will contain a summary of data used in the DCOPF model
df_m = df_n.copy()

# DUIDs assigned to each node
df_m['DUID'] = df_g.reset_index().groupby('NODE')[['DUID']].aggregate(lambda x: set(x)).reindex(df_m.index, fill_value=set())
df_m


# ### Model data summary
# Store data in dictionary and pickle. Will use this when constructing model.

# In[14]:


# Place model data in dictionary
model_data = {}
model_data['df_g'] = df_g
model_data['df_Y'] = df_Y
model_data['df_m'] = df_m
model_data['df_hvdc_c'] = df_hvdc_c
model_data['zones'] = zones
model_data['df_ac_i'] = df_ac_i
model_data['df_ac_i_lim'] = df_ac_i_lim
model_data['df_hvdc'] = df_hvdc
model_data['df_regd'] = df_regd
model_data['df_inter'] = df_inter
model_data['df_dis'] = df_dis
model_data['df_fx_hydro'] = df_fx_hydro

# Pickle model data
with open(os.path.join(output_dir, 'model_data.pickle'), 'wb') as f:
    pickle.dump(model_data, f)

