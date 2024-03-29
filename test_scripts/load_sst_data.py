import os
import netCDF4 as nc
import datetime
import pandas as pd
import datetime as dt

def all_sst_data(directory, lead_time=10, lat_lims = [20.,50.], lon_lims = [145.,230.],
                            start_year = 1982, end_year = 2015, start_doy = 175, end_doy = 234, debug = False):
  sst_list = []

  for yr in range(start_year, end_year+1):
    if debug:
      print ('------------------------------------------------------------------')
      print ("Processing SST data for Year : " + str(yr))
        
    sst_name = "sst.day.anom."+str(yr)+".nc"
    sst_year = process_sst_data (directory, yr, start_doy, end_doy, \
                      lead_time, lat_lims, lon_lims)
        
    sst_list.append(sst_year)

  return sst_list


def process_sst_data (sst_dir, yr, start_doy, end_doy, lead_time, lat_lims, lon_lims):
            '''
            * This functions import global daily sst data
            * Select the time period of interest in a year
            * Select the data between Lat and Lon range
            
            Parameters
            ----------
                sst_dir
                yr,
                start_doy,
                end_doy,
                lead_time,
                lat_lims ---  [lat_min lat_max]
                lon_lims ---  [lon_min lon_mas]

            Returns
            -------
            
            Example
            -------
            lat_lims = [20.,50.]
            lon_lims = [145.,230.]
            
            
            '''
            
            sst_name = os.path.join(sst_dir, "sst.day.anom." +str(yr)+".nc")
            f = nc.MFDataset(sst_name)
            anom = f.variables['anom'][:]
            lon  = f.variables['lon'][:]
            lat  = f.variables['lat'][:]
            dumb_time = f.variables['time'][:]

            time = pd.to_datetime(dumb_time, unit='D',
                       origin=pd.Timestamp('1800-01-01'))
            #print time.to_series()
            jday = time.dayofyear
            #jday = pd.to_datetime(dumb_time, unit='D',
            #           origin=pd.Timestamp('1800-01-01'))

            #print time.to_series().dayofyear.apply(lambda x: str(x).zfill(3)).astype(int)

            latidx1 = (lat >=lat_lims[0] ) & (lat <=lat_lims[1] ) 
            lonidx1 = (lon >=lon_lims[0] ) & (lon <=lon_lims[1] )

            timidx1  = (jday >= start_doy-lead_time)  & (jday <= end_doy-lead_time)

            ocean_anom = anom[:, latidx1][..., lonidx1]
            sst_year = ocean_anom[timidx1,:,:]
            
            return sst_year
