from . import graphcast_lam as graphcast
import numpy as np

# the number of gridpoints in one direction; square domain.
DOMAIN_SIZE = 150

VARS_3D = ['U', 
           'V', 
           'W', 
           'T', 
           'GEOPOT', 
           'QVAPOR']

VARS_2D = ['T2', 
           'COMPOSITE_REFL_10CM', 
           'RAIN_AMOUNT'] #'UP_HELI_MAX',

STATIC_VARS = ['XLAND', 'HGT']

INPUT_VARS = VARS_3D + VARS_2D + STATIC_VARS
TARGET_VARS = VARS_3D + VARS_2D

# I compute this myself rather than using the GraphCast code. 
FORCING_VARS = (
            'local_solar_time_sin',
            'local_solar_time_cos',
        )

GC_FORCINGS_VARS = (
    'toa_radiation',
    #'day_progress', 
    'day_progress_cos', 
    'day_progress_sin', 
    #'year_progress', 
    'year_progress_cos', 
    'year_progress_sin', 
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
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = '20min', #train_lead_times,
     loss_callable=None
 )


WOFS_TASK_CONFIG_GC = graphcast.TaskConfig(
      input_variables=INPUT_VARS,
      target_variables=TARGET_VARS,
      forcing_variables=GC_FORCINGS_VARS,
      pressure_levels=PRESSURE_LEVELS,
      input_duration=INPUT_DURATION,
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = train_lead_times,
    loss_callable=None
 )


WOFS_TASK_CONFIG_NO_FORCE = graphcast.TaskConfig(
      input_variables=INPUT_VARS,
      target_variables=TARGET_VARS,
      forcing_variables=None,
      pressure_levels=PRESSURE_LEVELS,
      input_duration=INPUT_DURATION,
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = train_lead_times,
      loss_callable=None
 )


WOFS_TASK_CONFIG_30MIN_GC = graphcast.TaskConfig(
      input_variables=INPUT_VARS,
      target_variables=TARGET_VARS,
      forcing_variables=GC_FORCINGS_VARS,
      pressure_levels=PRESSURE_LEVELS,
      input_duration='60min',
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = '30min',
     loss_callable=None
 )

DA_VARS = ['U', 'V', 'W', 'COMPOSITE_REFL_10CM', 'T', 'GEOPOT']

WOFS_TASK_CONFIG_DA = graphcast.TaskConfig(
      input_variables=DA_VARS + STATIC_VARS,
      target_variables=DA_VARS,
      forcing_variables=None,
      pressure_levels=PRESSURE_LEVELS,
      input_duration='10min', # Only 1 time! 
      n_vars_2D = 1, # Composite refl
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = '10min',
      loss_callable=None
 )


LEVELS_EVERY_3 = list(np.arange(0,50,3))
LEVELS_EVERY_2 = list(np.arange(0,50,2))

ALL_LEVELS = list(np.arange(0,50))

WOFS_TASK_CONFIG_ALL_LEVELS = graphcast.TaskConfig(
      input_variables=INPUT_VARS,
      target_variables=TARGET_VARS,
      forcing_variables=FORCING_VARS,
      pressure_levels=ALL_LEVELS,
      input_duration=INPUT_DURATION,
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = train_lead_times,
      loss_callable=None
 )


WOFS_TASK_CONFIG_1HR = graphcast.TaskConfig(
      input_variables=INPUT_VARS,
      target_variables=TARGET_VARS,
      forcing_variables=FORCING_VARS,
      pressure_levels=LEVELS_EVERY_3,
      input_duration='2hr',
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = '1hr',
      loss_callable=None
 )

WOFS_TASK_CONFIG_5MIN = graphcast.TaskConfig(
      input_variables=INPUT_VARS,
      target_variables=TARGET_VARS,
      forcing_variables=FORCING_VARS,
      pressure_levels=PRESSURE_LEVELS,
      input_duration='10min',
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = '5min',
      loss_callable=None
 )



##############

VARS_2D = ['COMPOSITE_REFL_10CM']
STATIC_VARS = ['XLAND', 'HGT']

INPUT_VARS_DBZ = VARS_2D + STATIC_VARS
TARGET_VARS_DBZ =  VARS_2D

DBZ_TASK_CONFIG = graphcast.TaskConfig(
      input_variables=INPUT_VARS_DBZ,
      target_variables=TARGET_VARS_DBZ,
      forcing_variables=FORCING_VARS,
      pressure_levels=PRESSURE_LEVELS,
      input_duration=INPUT_DURATION,
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE,
      tiling=None,
      train_lead_times = train_lead_times,
      loss_callable=None
 )


DBZ_TASK_CONFIG_1HR = graphcast.TaskConfig(
      input_variables=INPUT_VARS_DBZ,
      target_variables=TARGET_VARS_DBZ,
      forcing_variables=FORCING_VARS,
      pressure_levels=PRESSURE_LEVELS,
      input_duration='120min',
      n_vars_2D = len(VARS_2D),
      domain_size = 150,
      tiling=None,
      train_lead_times = '60min',
      loss_callable=None
 )


DBZ_TASK_CONFIG_FULL = graphcast.TaskConfig(
      input_variables=INPUT_VARS_DBZ,
      target_variables=TARGET_VARS_DBZ,
      forcing_variables=FORCING_VARS,
      pressure_levels=np.arange(50),
      input_duration='20min',
      n_vars_2D = len(VARS_2D),
      domain_size = 300,
      tiling=None,
      train_lead_times = '10min',
      loss_callable=None
 )

