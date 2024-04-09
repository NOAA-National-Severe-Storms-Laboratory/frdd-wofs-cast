from . import my_graphcast as graphcast
import numpy as np

# the number of gridpoints in one direction; square domain.
DOMAIN_SIZE = 150

VARS_3D = ['U', 'V', 'W', 'T', 'GEOPOT', 'QVAPOR']
VARS_2D = ['T2', 'COMPOSITE_REFL_10CM', 'UP_HELI_MAX', 'RAIN_AMOUNT']
STATIC_VARS = ['XLAND', 'HGT']

INPUT_VARS = VARS_3D + VARS_2D + STATIC_VARS
TARGET_VARS = VARS_3D + VARS_2D

# I compute this myself rather than using the GraphCast code. 
FORCING_VARS = (
            'local_solar_time_sin',
            'local_solar_time_cos'
        )
# Not pressure levels, but just vertical array indices at the moment. 
# When I created the wrfwof files, I pre-sampled every 3 levels. 
PRESSURE_LEVELS = np.arange(17)

# Loads data from the past 20 minutes (2 steps) and 
# creates a target over the next 10-60 min. 
INPUT_DURATION = '20min'
# 110 min (13*10 - (2*10)) is the max, but keeping it lower for testing the workflow
train_lead_times = '10min'

WOFS_TASK_CONFIG = graphcast.TaskConfig(
      input_variables=INPUT_VARS,
      target_variables=TARGET_VARS,
      forcing_variables=FORCING_VARS,
      pressure_levels=PRESSURE_LEVELS,
      input_duration=INPUT_DURATION,
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE
 )