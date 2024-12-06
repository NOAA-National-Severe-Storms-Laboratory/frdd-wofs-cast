## Model Evaluation 

To evaluate WoFSCast, first create a config yaml in `eval_configs`. Options include data paths, number of samples, model path, and the variables to evaluate. Run the following (with the appropriate config yaml): 

```bash 
stdbuf -oL python -u evaluate.py --config eval_config.yaml > & logs/log_eval &
```

The evaluation code can be found in [`frdd-wofs-cast/wofscast/evaluate`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/tree/master/wofscast/evaluate) 

Notebooks for visualizing WoFSCast output and the evaluation results can be found in `nbs`. 



