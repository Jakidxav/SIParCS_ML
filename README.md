# SIParCS ML

Welcome to this repository! In it you will find the code we used as part of the Summer Internships in Parallel Computational Science or [SIParCS](https://www2.cisl.ucar.edu/siparcs) 2018 internship project entitled *Machine Learning For Long Term Weather Forecasts*.

# Motivation

In a stochastic regime, running deterministic models based on an initial state will naturally diverge as the initial uncertainty amplifies; two or more initial states become less and less similar as time progresses. In atmospheric phenomena, this uncertainty is unbound so it is almost impossible at any given time to fully map all future states. Because of this, forecasting the weather on the seasonal timescale (greater than 20 days) is inherently inaccurate. Traditionally, an ensemble of forecasts, based on slightly different initial conditions, is run to outline likely future states as a probability of occurrence. A correlation between abnormally hot sea surface temperatures (SST) in the Pacific and above average temperatures in the Eastern United states 20 - 50 days later was suggested by [McKinnon et al. 2016](https://www.nature.com/articles/ngeo2687). In that paper the authors predict above average ‘hot’ days using statistical models of the Pacific Extreme Pattern SST from 1982-2015. Here, we aim to replicate these results by taking a deep learning approach using neural networks. By training the networks on SST data, we have shown that it
is possible to predict whether an anomalously hot day will occur in 1613 stations in the Eastern US 20, 30, 40, and 50 days in advance.

We show that with both a recurrent and convolutional neural network we can predict whether a heat event will occur up to 50 days in advance with greater than 60% accuracy.

# Repository Organization

[test_scripts/](https://github.com/NCAR/SIParCS_ML/tree/master/test_scripts): Programs to make sure that our data processing programs work.

[data_download_scripts/](https://github.com/NCAR/SIParCS_ML/tree/master/data_download_scripts): The data used for this project were the [NOAA OI SST V2 High Resolution Dataset](https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.highres.html) for sea surface temperature and the [Global Historical Climatology Network](https://www.ncdc.noaa.gov/ghcn-daily-description) - Daily (GHCND) dataset for land temperature data. This directory houses programs that you can use to download the data for yourself.

[data_processing_scripts/](https://github.com/NCAR/SIParCS_ML/tree/master/data_processing_scripts): This directory contains Jupyter Notebooks that process the SST data to create the training, development, and validation sets as well as generate our labels from the GHCN data.

[training_scripts/](https://github.com/NCAR/SIParCS_ML/tree/master/training_scripts): Here where we train the neural networks, perform a hyperparameter search, and compute ROC curves and confusion matrices. We demonstrate how we can save a neural network's architecture and then reload it for further training before predicting on the validation set.
