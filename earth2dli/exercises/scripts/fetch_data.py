from cartopy.feature import NaturalEarthFeature
from windpowerlib.data import store_turbine_data_from_oedb

if __name__ == "__main__":
    _ = store_turbine_data_from_oedb()

    for resolution in ["10m", "50m", "110m"]:
        _ = NaturalEarthFeature("physical", "coastline", "10m").geometries()
