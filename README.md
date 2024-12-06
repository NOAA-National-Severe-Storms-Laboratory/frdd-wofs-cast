## WoFSCast: AI-NWP For Storm-Scale Applications 

Repository for training and evaluating the National Severe Storms Laboratory (NSSL) limited-area modeling version of 
[GraphCast](https://github.com/google-deepmind/graphcast) known as ["WoFSCast"](https://essopenarchive.org/users/829074/articles/1223249-wofscast-a-machine-learning-model-for-predicting-thunderstorms-at-watch-to-warning-scales). This is an actively developed project, file paths and configurations are subject to change.


### Installation 

To install from source from the develop branch

```bash 
git clone git@github.com:NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast.git -b develop 
cd conda 
conda env create -f env-cuda118.yaml or [env-cuda12.yaml] 

# Optionally to train and evaluate the diffusion model 
pip install -r diffusion_requirements.txt

```
The conda environment depends on the CUDA driver available on your GPUs. For different version, you'll need to select the appropriate JAX and PyTorch versions. The code is not thoroughly vetted on different versions, so user beware. 


## Branches & Pull Requests 

For development, we recommend creating a branch off of `develop` following the below naming conventions:
- `documentation/user_branch_name`: Documenation additions and/or corrections
- `feature/user_branch_name`: Enhancements/upgrades
- `fix/user_branch_name`: Bug-type fixes
- `hotfix/user_branch_name`: Bug-type fixes which require immediate attention and are required to fix significant issues that compromise the integrity of the software

Once the desired contributions are complete in your branch, submit a pull request (PR) to merge your branch into `develop`.
Also note that all python coding additions should follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide.

When prototypes are completed (or other significant milestones worthy of a release are reached), we will open a PR to `main` and tag the release.


## Training and Evaluating WoFSCast 

Scripts for training and evaluating WoFSCast are found in [`workflow/model_training`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/tree/master/workflow/model_training) and [`workflow/model_evaluation`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/tree/master/workflow/model_evaluation), respectively. Other related scripts for generating data, plotting the latent space triangular mesh, etc are also in `workflow`. 


### Brief description of library files (distinct from the original GraphCast library files):

*   `autoregressive_lam.py`: Wrapper used to run (and train) the one-step GraphCast
    to produce a sequence of predictions by auto-regressively feeding the
    outputs back as inputs at each step, in JAX a differentiable way. Modified from the original 
    GraphCast code to allow for updating boundary conditions for limited area modelling. 
*   `graphcast_lam.py`: A modified version of the original GraphCast model architecture for one-step of
    predictions.
*   `square_mesh.py`: Definition of a square multi-mesh and tools for converting between regular grids 
     and the mesh grid. 
*   `rollout.py`: Similar to `autoregressive_lam.py` but used only at inference time
    using a python loop to produce longer, but non-differentiable trajectories.
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

## Disclaimer

The United States Department of Commerce (DOC) GitHub project code is
provided on an "as is" basis and the user assumes responsibility for
its use. The DOC has relinquished control of the information and no longer
has responsibility to protect the integrity, confidentiality, or
availability of the information.  Any claims against the Department of
Commerce stemming from the use of its GitHub project will be governed
by all applicable Federal law.  Any reference to specific commercial
products, processes, or services by service mark, trademark,
manufacturer, or otherwise, does not constitute or imply their
endorsement, recommendation or favoring by the Department of
Commerce.  The Department of Commerce seal and logo, or the seal and
logo of a DOC bureau, shall not be used in any manner to imply
endorsement of any commercial product or activity by DOC or the United
States Government.


