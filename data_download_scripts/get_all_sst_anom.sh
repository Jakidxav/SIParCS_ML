#! /bin/bash 
DATA_PATH=/glade/p/work/negins/sst/data
mkdir -p $DATA_PATH
cd $DATA_PATH 

#wget -N -v -a download_log ftp://ftp.cdc.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.anom.{1981..2017}.nc
wget ftp://ftp.cdc.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.anom.{1981..2017}.nc

