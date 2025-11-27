import netCDF4 as nc
import numpy as np
import pandas as pd

def load_pm25_nc(file_path="../data/raw/pm25_raw.nc"):
    dataset = nc.Dataset(file_path)

    # check variables
    print("Variables:", dataset.variables.keys())

    # adjust variable names
    pm25 = dataset.variables["pm2p5_conc"][:]
    time = dataset.variables["time"][:]
    lat = dataset.variables["latitude"][:]
    lon = dataset.variables["longitude"][:]

    # flatten 3D array (time, lat, lon) -> rows of data
    times, lats, lons = np.meshgrid(time, lat, lon, indexing="ij")
    df = pd.DataFrame({
        "time": times.flatten(),
        "latitude": lats.flatten(),
        "longitude": lons.flatten(),
        "pm25": pm25.flatten()
    })

    print("Data loaded, shape:", df.shape)
    return df
