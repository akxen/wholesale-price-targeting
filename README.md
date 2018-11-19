# Emissions reduction and wholesale electricity price targeting using an output-based mechanism
This repository contains a series of Jupyter Notebooks which implement a mathematical program used to calibrate an output-based emissions abatement policy. Specifically, the program determines values for permit prices and the sectoral emissions intensity baseline that yields given environmental and economic objectives. A derivation of the mathematical program is given in `mppdc-derivation.pdf` within the `derviation` folder.

A summary of each Jupyter Notebook is as follows:

| Name | Description |
| ---- | ----------- |
| `create-scenarios.ipynb` | Uses a k-means clustering approach to generate a set of representative operating scenarios based on historic data |
| `parameter-selector.ipynb` | Implements a mathematical program used to calibrate scheme parameters |
| `process-results.ipynb` | Process model results |
| `plotting.ipynb` | Uses processed model results to construct plots illustrating scheme operation / impacts |


## Zenodo link
Network and generator datasets used in this analysis are obtained from the following repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326942.svg)](https://doi.org/10.5281/zenodo.1326942)


## Usage notes
A large number of output files will be generated when running the `parameter-selector.ipynb` notebook. Ensure there is enough space on disk (approximately 30GB) before running.

See `environment.yml` for specifications of the conda environment used to run the model.
