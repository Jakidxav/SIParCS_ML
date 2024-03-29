import load_sst_data

def test_process_sst_data():
  yr = 2000
        
  start_doy  = 175
  end_doy    = 234

  lead_time = 10

  lat_lims = [20.,50.]
  lon_lims = [145.,230.]
 
  sst_data = load_sst_data.process_sst_data('../McKinnon_data/sst/data/', yr, start_doy, end_doy, lead_time, lat_lims, lon_lims)
  print (sst_data)
  assert sst_data.shape == (end_doy - start_doy + 1, 120, 340)

def test_process_many_years_sst_data():
  start_doy  = 175
  end_doy    = 234

  sst_data = load_sst_data.all_sst_data('../McKinnon_data/sst/data/', start_year = 1985, end_year = 1987, 
                                         start_doy = start_doy, end_doy = end_doy)
  assert len(sst_data) == 3
  for x in sst_data:
    assert x.shape == (end_doy - start_doy + 1, 120, 340)
