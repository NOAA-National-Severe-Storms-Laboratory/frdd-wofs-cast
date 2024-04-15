# WoFSCast: AI-NWP For Storm-Scale Applications 

This repository has code for running, training, and visualizing the preliminary "WoFSCast" model. 
The repository is derived from the original [GraphCast](https://github.com/google-deepmind/graphcast). Primary changes include 
reconfiguring the meshing and autoregressive rollout for a limited area domain rather than a full global. This is an actively developed project, file paths and configurations are subject to change.


### Reformating WRFOUTs for GraphCast code 
Rather than refactor the graphcast code, I've opted to reformat the WoFS WRFOUT files. The files are reformated with 
[`training_pipeline/build_dataset/wrfout_file_formatter.py`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/blob/master/training_pipeline/build_dataset/wrfout_file_formatter.py) and [`training_pipeline/build_dataset/format_wofs_wrfouts`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/blob/master/training_pipeline/build_dataset/format_wofs_wrfouts.py).
The reformatting includes: 
1. Combining multiple times into a single file 
2. Convert perturbation variables into the full variables
3. Renaming coordinates 
4. Destaggering the grid 
5. Reducing the number of vertical levels 

This list is not exhaustive. 


### Training WoFSCast 
Prior to training WoFSCast requires creating normalization statistics. The normalization statistics are computed 
with [`training_pipeline/build_dataset/compute_norm_stats.ipynb`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/blob/master/training_pipeline/build_dataset/compute_norm_stats.ipynb). 
]
Training the WoFSCast model is found in [`training_pipeline/train_model/train_wofscast.py`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/blob/master/training_pipeline/train_model/train_wofscast.py). The current codebase supports multiple GPU training. 


### Running and Evaluating WoFSCast 
Visualizing the WoFSCast output is found in [`training_pipeline/evaluate_model/visualize_wofscast.ipynb`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/blob/master/training_pipeline/evaluate_model/visualize_wofscast.ipynb). Evaluation code is found in [`training_pipeline/evaluate_model/visualize_wofscast.ipynb`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/blob/master/training_pipeline/evaluate_model/evaluate_wofscast.ipynb). The evaluation includes time-series of RMSE and storm-based metrics such as object matching. Object identification, matching, and verification is provided in [frdd-monte-python](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-monte-python). 


### Brief description of library files (distinct from the original GraphCast library files):

*   `autoregressive_lam.py`: Wrapper used to run (and train) the one-step GraphCast
    to produce a sequence of predictions by auto-regressively feeding the
    outputs back as inputs at each step, in JAX a differentiable way. Modified from the original 
    GraphCast code to allow for updating boundary conditions for limited area modelling. 
*   `my_graphcast.py`: A modified version of the original GraphCast model architecture for one-step of
    predictions.
*   `square_mesh.py`: Definitionof a square multi-mesh and tools for converting between regular grids 
     and the mesh grid. 
*   `rollout.py`: Similar to `autoregressive_lam.py` but used only at inference time
    using a python loop to produce longer, but non-differentiable trajectories.
*   `wofscast_task_config.py` : Definition of the WoFS configuration 
    (including domain size, variables, pressure levels, etc)
*   `data_generator.py`: Handles data and batch loading and Top-of-the-atmosphere (TOA) radiation computation

### Dependencies.

[Chex](https://github.com/deepmind/chex),
[Dask](https://github.com/dask/dask),
[Haiku](https://github.com/deepmind/dm-haiku),
[JAX](https://github.com/google/jax),
[JAXline](https://github.com/deepmind/jaxline),
[Jraph](https://github.com/deepmind/jraph),
[Numpy](https://numpy.org/),
[Pandas](https://pandas.pydata.org/),
[Python](https://www.python.org/),
[SciPy](https://scipy.org/),
[Tree](https://github.com/deepmind/tree),
[Trimesh](https://github.com/mikedh/trimesh) and
[XArray](https://github.com/pydata/xarray).
