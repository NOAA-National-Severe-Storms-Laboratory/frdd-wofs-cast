## Dataset Builder 

Before creating training dataset files, create a dataset config yaml in `/dataset_gen_configs`. Options 
include the timestep size, offset from forecast initialization, number of timesteps to save, ensemble members to keep, 
among others. In the `train_dataset_builder.py`, change the `config_fname` and any `tags` (options can be found in `format_wofs_wrfouts.py`). Then run the following: 

```bash
stdbuf -oL python -u train_dataset_builder.py > & logs/log_dataset_builder &
```

The script will generate the training dataset files and the corresponding normalization statistics. 


Similarly, for testing dataset files, create another config yaml (usually with a larger number of timesteps). Then run the following: 

```bash
stdbuf -oL python -u test_dataset_builder.py > & logs/log_test_dataset &
```

Test notebooks for checking the output can be found in `/tests` 

### File Descriptions 

* `wrfout_file_formatter.py` : Handles formatting the raw WRFOUTS into an AI-ready format. Includes converting perturbation fields into full fields, destaggering relevant grids, dropping variables, adding a datetime and leadtime dimension, renaming coordiantes, 
subsetting vertical levels, unaccumulating rainfall, converting longitude to positive values. Data saved in compressed Zarr format. 

* `format_wofs_wofsouts.py`: Wrapper script for running `wrfout_file_formatter.py` 

* `compute_norm_stats.py` : Script for generating the normalization statistics files. Includes mean, standard deviation, and time-difference standard deviation by vertical level. 

* `save_wofscast_to_torch_tensors.py`: Script for generating WoFSCast forecast, computing residual fields, and saving the output as PyTorch tensors for training the corrective diffusion model. 

