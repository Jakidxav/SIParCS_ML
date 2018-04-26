import subprocess
import os
import glob
import sys
import re
import csv
import netCDF4 as nc
import numpy as np
import datetime
import pandas as pd
import datetime as dt

def station_data (directory, stn_id, start_year = 1982, end_year = 2015, start_doy = 175, end_doy = 225):
            '''
                * Opens the csv file for the GHCND station as data frame
                * Add a pd.datetime column to the df
                * Add Julian Day (day of the year) jday to the df 
                * Selects data for only the trianing years
                * Selects data for only the selected days of a year ( e.g. 60 days of summer.) 
                
            ----------
            Parameters:
                directory --- path to GHCN processed csv files.
                stn_id --- GHCN station ID based on GHCN readme.txt
                    (ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt)
                
                start_year --- start year for the period to slice. 
                end_year 
                
                start_doy  --- start julian day (day of year) for the period
                                you are looking at.  
                end_doy
            -------
            Returns:
                stn_data --- 
            
            -------
            Example:
                stn_data = get_ghcnd_stn('./', 'USW00094911')
                
            '''
            
            stn_csv = os.path.join(directory, stn_id + '.csv')
            stn_data = pd.read_csv(stn_csv,na_values=-9999)
            
            stn_data['date'] = pd.to_datetime(stn_data['YYYY'].astype(str)+'-'+stn_data['MM'].astype(str)+'-'+stn_data['DD'].astype(str))
            stn_data['jday'] = stn_data['date'].dt.dayofyear.apply(lambda x: str(x).zfill(3)).astype(int)
            
            line = 'Data for '+ stn_id + ' is avialble for : ' \
                    + str(stn_data['YYYY'].min()) +' '\
                    + str(stn_data['YYYY'].max())
            print (line)

            ##processing based on year and days

            yrs_data = stn_data[(stn_data['YYYY']>=start_year) & (stn_data['YYYY']<=end_year)]
            stn_data__selected_time = yrs_data[(yrs_data['jday']>=start_doy) & (yrs_data['jday']<=end_doy)]
            
            return stn_data__selected_time
