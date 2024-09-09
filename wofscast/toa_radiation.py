import numpy as np
import xarray as xr
import pandas as pd

class TOARadiationFlux:
    """ 
    Compute the instanteous solar radiation flux 
    
    Computation based on longitude ranges of [-180, 180]. 
    If you have longitude ranges of [0,360], then set the 
    range to [0, 360] and longitude will be converted to
    [-180, 180], but the appended toa_radiation will have 
    the original longitude to be consistent with the input dataset. 
    
    """
    
    EARTHS_TILT = 23.45

    def __init__(self, longitude_range="[-180, 180]"):
        # Solar constant in W/m^2
        self.S0 = 1361
        self.longitude_range = longitude_range
        
    def solar_declination_in_radians(self, day_of_year):
        """Calculate solar declination as a function of day of the year."""
        angle = self.EARTHS_TILT * np.sin(np.radians((360 / 365) * (day_of_year - 81)))
        return np.radians(angle)
    
    def equation_of_time(self, day_of_year):
        """Calculate the equation of time in minutes."""
        B = 2 * np.pi * (day_of_year - 81) / 364
        eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
        return eot  # in minutes
    
    def convert_longitude(self, longitude):
        """Convert longitude from [0, 360] to [-180, 180] if needed, compatible with xarray datasets."""
        if '360' in self.longitude_range and (longitude > 0).all():
            # Use xarray's apply_ufunc to handle the conversion, keeping metadata intact
            longitude = xr.apply_ufunc(
                lambda x: np.mod((x + 180), 360) - 180,  # Conversion function
                longitude,
                vectorize=True,  # Allow element-wise operation
                dask="allowed"  # Enable Dask support if applicable
            )
        elif '360' in self.longitude_range and (longitude < 0).any():
            raise ValueError(
                'Some longitude values are negative, but longitude_range is set to [0, 360]'
            )
        return longitude
    
    def calculate_solar_time(self, utc_time, longitude, day_of_year):
        """Calculate solar time by applying longitude and equation of time corrections."""
        # Convert longitude to equivalent time (4 minutes per degree of longitude)
        longitude_correction = 4 * longitude
        
        # Equation of time correction
        eot = self.equation_of_time(day_of_year)
        
        # Convert UTC time to fractional hours
        utc_hour = utc_time.hour + utc_time.minute / 60
        
        # Solar time is UTC time corrected by longitude and EoT (in hours)
        solar_time = utc_hour + longitude_correction / 60 + eot / 60
        
        return solar_time

    def hour_angle_in_radians(self, utc_time, longitude, day_of_year):
        """Calculate solar hour angle based on UTC time and longitude."""
        # Calculate solar time
        solar_time = self.calculate_solar_time(utc_time, longitude, day_of_year)
        
        # Hour angle: 15 degrees per hour, relative to solar noon
        hour_angle = 15 * (solar_time - 12)
        
        return np.radians(hour_angle)

    def add_toa_radiation(self, ds):
        """Calculate TOA solar radiation flux and add it to the input xarray Dataset."""
        # Assume the xarray dataset (ds) has time, lat, lon dimensions.
        lat_grid = ds['lat']
        lon_grid = ds['lon']
        times = ds['datetime']
        
        # Convert longitude if necessary
        lon_grid_converted = self.convert_longitude(lon_grid)
        
        # Ensure time is in datetime format
        times = pd.to_datetime(times.values)

        NT = len(times)
        NY, NX = len(lat_grid), len(lon_grid)
        toa_radiation = xr.DataArray(np.zeros((NT, NY, NX), dtype=np.float32), 
                                     dims=['time', 'lat', 'lon'],
                                     coords={'time': ds['time'], 'lat': lat_grid, 'lon': lon_grid})

        lat_grid_rad = np.radians(lat_grid)

        for i, datetime_obj in enumerate(times):
            # Get the day of the year (Julian day)
            day_of_year = datetime_obj.timetuple().tm_yday
            
            # Calculate solar declination
            declination_rad = self.solar_declination_in_radians(day_of_year)
            
            # Vectorized hour angle calculation for the entire longitude grid
            hour_angle_rad = self.hour_angle_in_radians(datetime_obj, 
                                                        lon_grid_converted, 
                                                        day_of_year)

            # Vectorized zenith angle calculation
            cos_zenith = np.sin(lat_grid_rad) * np.sin(declination_rad) + \
                         np.cos(lat_grid_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)

            cos_zenith = np.clip(cos_zenith, -1.0, 1.0)  # Avoid invalid values for arccos
            
            zenith_angle = np.degrees(np.arccos(cos_zenith))

            # Replace NaNs in cos_zenith with 0 (avoids propagation of NaNs)
            cos_zenith = np.nan_to_num(cos_zenith)
            
            # Calculate TOA radiation flux, mask out where the sun is below the horizon
            toa_radiation[i, :, :] = xr.where(zenith_angle < 90, self.S0 * cos_zenith, 0)
        
        # Add the calculated TOA radiation to the dataset
        ds['toa_radiation'] = toa_radiation

        return ds