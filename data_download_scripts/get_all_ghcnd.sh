#! /bin/bash 
DATA_PATH=/glade/p/work/negins/ghcnd/data
mkdir -p $DATA_PATH
cd $DATA_PATH 
if wget -N -nv -a download_log ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-version.txt
  then sleep 2
  wget -N -nv -a download_log ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/status.txt
  sleep 2
  wget -N -nv -a download_log ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt
  sleep 2 
  wget -N -nv -a download_log ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt
  sleep 2
  wget -N -nv -a download_log ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
  sleep 2
  if wget -N -nv -a download_log ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz
    then tar -xvf ghcnd_all.tar.gz
  fi
fi
