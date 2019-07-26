# SIParCS ML

Welcome to this repository! In it you will find the code we used as part of the Summer Internships in Parallel Computational Science or [SIParCS](https://www2.cisl.ucar.edu/siparcs) 2018 internship project entitled *Machine Learning For Long Term Weather Forecasts*.

# Motivation

In a stochastic regime, running deterministic models based on an initial state will naturally diverge as the initial uncertainty amplifies; two or more initial states become less and less similar as time progresses. In atmospheric phenomena, this uncertainty is unbound so it is almost impossible at any given time to fully map all future states. Because of this, forecasting the weather on the seasonal timescale (greater than 20 days) is inherently inaccurate. Traditionally, an ensemble of forecasts, based on slightly different initial conditions, is run to outline likely future states as a probability of occurrence. A correlation between abnormally hot sea surface temperatures (SST) in the Pacific and above average temperatures in the Eastern United states 20 - 50 days later was suggested by [McKinnon et al. 2016](https://www.nature.com/articles/ngeo2687). In that paper the authors predict above average ‘hot’ days using statistical models of the Pacific Extreme Pattern SST from 1982-2015. Here, we aim to replicate these results by taking a deep learning approach using neural networks. By training the networks on SST data, we have shown that it
is possible to predict whether an anomalously hot day will occur in 1613 stations
in the Eastern US 20, 30, 40, and 50 days in advance.

The data used for this project were the [NOAA OI SST V2 High Resolution Dataset](https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.highres.html) for sea surface temperature and the [Global Historical Climatology Network](https://www.ncdc.noaa.gov/ghcn-daily-description) - Daily (GHCND) dataset for land temperature data.

# Repository Organization

data_download_scripts is where we get the data

data_processing_scripts: EDA, create labels, process SST data to get the train, dev, and test sets

training_scripts is where we train the neural networks, and compute ROC curves and confusion matrices; we also save models to h5 files and output scores; modelTests.py is where we make our final predicitons

**Need to add some sort of figures showing end result**

