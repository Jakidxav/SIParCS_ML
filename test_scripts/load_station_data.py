import os
import netCDF4 as nc
import numpy as np
import datetime
import pandas as pd
import datetime as dt

def all_stations_data (directory, stn_list, cut_off= 6.5, start_year = 1982, end_year = 2015, start_doy = 175, end_doy = 234, debug = False):
  ghcnd_list = [] 
  for counter, stn_id in enumerate(stn_list):
    if debug:
        print ('------------------------------------------------------------------')
        line = "Processing file number " + str(counter) +'\n'+ \
                        "station ID : " +stn_id
        print (line)
                
    stn_data, progress = station_data(directory, stn_id, start_year, end_year, start_doy, end_doy)

    stn_data = station_anomaly(stn_data, 'TMAX')
    stn_data = find_hot_days (stn_data, cut_off)
    ghcnd_list.append(stn_data)

  return ghcnd_list
                

def station_data (directory, stn_id, start_year = 1982, end_year = 2015, start_doy = 175, end_doy = 234):
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

            ##processing based on year and days

            yrs_data = stn_data[(stn_data['YYYY']>=start_year) & (stn_data['YYYY']<=end_year)]
            stn_data__selected_time = yrs_data[(yrs_data['jday']>=start_doy) & (yrs_data['jday']<=end_doy)]
            
            return stn_data__selected_time, line

def station_anomaly (stn_data, var):
            '''
            calc_stn_anom :
                *Calculates the anomalies of selected var for the station.
            ----------
            Parameters:
                stn_data ---
                var --- Name of the varibale to calculate anomalies on :
                        e.g. TMAX, TMIN, PRCP
            -------
            Returns:
                stn_data --- 
            
            -------
            Example:
                calc_stn_anom (stn_data, 'TMAX')
                
            '''
            var_anom = var + "_ANOM"
            means = stn_data.groupby(['MM','DD'])[var].transform('mean')
            stn_data[var_anom] = stn_data[var] - means
            return stn_data

def find_hot_days (stn_data, cut_off):
            '''
            * Find the hot days (or extreme events) based on a cut off value.
            * Store this flag ('HOT') as a column in the stn_data.
            ----------
            Parameters:
                stn_data ---
                cut_off --- cut off value for the extreme events
            -------
            Returns:
                stn_data --- 
            
            -------
            Example:
                find_hot_days (stn_data, cut_off)
                
            '''
            stn_data['HOT'] = np.where(stn_data['TMAX_ANOM']>= cut_off,1,0)
            return stn_data
