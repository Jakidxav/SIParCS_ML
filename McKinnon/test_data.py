import load_data

def test_load_station():
   data, info = load_data.station_data("../McKinnon_data/ghcnd/ghcnd_all_csv", "USC00145063")
   
   assert 1982 in data['YYYY'].values
   assert 1999 in data['YYYY'].values

   assert 6 in data['MM'].values
   assert 7 in data['MM'].values
   assert 8 in data['MM'].values

   assert 200 in data['jday'].values
