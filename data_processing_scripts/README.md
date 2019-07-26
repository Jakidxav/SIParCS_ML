This directory creates the data subsets and corresponding labels for the training, development, and test sets. The development and testing sets are chosen to have a similar distribution of El Niño and La Niña years in them.

A suggested order to look at the Jupyter Notebooks is:
- sst_Import
- stationDict_Import
- ghcn_Import
- allStations_Labels

Some helper method files are included as .py files, and deep_heat_01_dense is an EDA Notebook that looks at how to implement a simple dense neural network for our data. Lastly, Cartopy_Colormapping is a simple tutorial on how to use Cartopy to plot the NOAA SST data.
