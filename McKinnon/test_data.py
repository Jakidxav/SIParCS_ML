import load_data
import pandas as pd
import itertools

def test_load_station():
   data, info = load_data.station_data("../McKinnon_data/ghcnd/ghcnd_all_csv", "USC00145063")
   
   assert 1982 in data['YYYY'].values
   assert 1999 in data['YYYY'].values

   assert 6 in data['MM'].values
   assert 7 in data['MM'].values
   assert 8 in data['MM'].values

   assert 200 in data['jday'].values

def test_station_anomaly():
  my_dates = [[1980, 1981, 1982], [7, 8], [1,2,3,4]]
  stn_data = pd.DataFrame(list(itertools.product(*my_dates)), columns = ["YYYY", "MM", "DD"])
  stuff = [10, 11, 12, 13, 14, 15, 16, 17]
  stn_data['TMAX'] = [i * 1.2 for i in stuff] + stuff + [i * .8 for i in stuff]
  print (stn_data)
  data_with_anomaly = load_data.station_anomaly (pd.DataFrame(stn_data), "TMAX")
  print (data_with_anomaly)
  tmax_in_1981 = data_with_anomaly[data_with_anomaly['YYYY'] == 1981]['TMAX_ANOM'] 
  print (tmax_in_1981)
  assert (tmax_in_1981 == 0).all()
  tmax_in_1980 = data_with_anomaly[data_with_anomaly['YYYY'] == 1980]['TMAX_ANOM'] 
  print (tmax_in_1980)
  assert tmax_in_1980[0] == 2
  tmax_in_1982 = data_with_anomaly[data_with_anomaly['YYYY'] == 1982]['TMAX_ANOM'] 
  assert tmax_in_1982[24 - 3] == - 15 * .2                                           # 24 elements, -3 is the third to last and its value is 15
