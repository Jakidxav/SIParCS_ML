import load_station_data
import pandas as pd
import itertools

def make_fake_data():
  my_dates = [[1980, 1981, 1982], [7, 8], [1,2,3,4]]
  stn_data = pd.DataFrame(list(itertools.product(*my_dates)), columns = ["YYYY", "MM", "DD"])
  stuff = [10, 11, 12, 13, 14, 15, 16, 17]
  stn_data['TMAX'] = [i * 1.2 for i in stuff] + stuff + [i * .8 for i in stuff]
  print (stn_data)
  data_with_anomaly = load_station_data.station_anomaly (pd.DataFrame(stn_data), "TMAX")
  print (data_with_anomaly)
  return data_with_anomaly

def test_load_many_stations():
  stn_list = []
  for line in open("10-stn-list.txt"):
    stn_list.append(line.strip())
  assert len(stn_list) > 1

  all_stations = load_station_data.all_stations_data("../McKinnon_data/ghcnd/ghcnd_all_csv", stn_list)
  assert len(all_stations) == len(stn_list)
  assert all_stations[-1].shape[0] > 100

def test_load_station():
   data, info = load_station_data.station_data("../McKinnon_data/ghcnd/ghcnd_all_csv", "USC00145063")
   
   assert 1982 in data['YYYY'].values
   assert 1999 in data['YYYY'].values

   assert 6 in data['MM'].values
   assert 7 in data['MM'].values
   assert 8 in data['MM'].values

   assert 200 in data['jday'].values

def test_station_anomaly():
  data_with_anomaly = make_fake_data()
  tmax_in_1981 = data_with_anomaly[data_with_anomaly['YYYY'] == 1981]['TMAX_ANOM'] 
  print (tmax_in_1981)
  assert (tmax_in_1981 == 0).all()
  tmax_in_1980 = data_with_anomaly[data_with_anomaly['YYYY'] == 1980]['TMAX_ANOM'] 
  print (tmax_in_1980)
  assert tmax_in_1980[0] == 2
  tmax_in_1982 = data_with_anomaly[data_with_anomaly['YYYY'] == 1982]['TMAX_ANOM'] 
  assert tmax_in_1982[24 - 3] == - 15 * .2                                           # 24 elements, -3 is the third to last and its value is 15

def test_hot_days():
  data_with_anomaly = make_fake_data()
  data_with_hot = load_station_data.find_hot_days(data_with_anomaly, 3.0)
  tmax_in_1980 = data_with_anomaly[data_with_anomaly['YYYY'] == 1980]['HOT']
  assert tmax_in_1980[0] == 0                                                        # hot year, but not day
  assert tmax_in_1980[5] == 1                                                        # hot year, hot day
  assert tmax_in_1980[7] == 1                                                        # hot year, hot day
  tmax_in_1981 = data_with_anomaly[data_with_anomaly['YYYY'] == 1981]['HOT']
  assert (tmax_in_1981 == 0).all()                                                   # average
  tmax_in_1982 = data_with_anomaly[data_with_anomaly['YYYY'] == 1982]['HOT']
  assert (tmax_in_1982 == 0).all()                                                   # cold
