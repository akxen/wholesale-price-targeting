# Data Provenance

## Network data and generator technical parameters
Network and generator technical parameters are obtained from:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326942.svg)](https://doi.org/10.5281/zenodo.1326942)

The assays performed to construct these datasets are described in [1].

## Historic generator dispatch and demand signals
Historic generator dispatch and demand information for 2017 is obtained from the Australian Energy Market Operator's Market Management System Data Model (MMSDM) database [1]. Data for each month of 2017 were extracted from the following MMSDM tables:

| MMSDM Table Name | Description |
| ----------- | ----------- |
| DISPATCH_UNIT_SCADA | Generator dispatch signals |
| TRADINGREGIONSUM | Contains half-hourly demand signals for each region in Australia's National Electricity Market |


## References
[1] - Xenophon, A. & Hill, D. Open grid model of Australiaʼs National Electricity Market allowing backtesting against historic data. Sci. Data. 5:180203 doi: [10.1038/sdata.2018.203](https://doi.org/10.1038/sdata.2018.203) (2018).

[2] - Australian Energy Markets Operator. Data Archive (2018). at [http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/](http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/)
