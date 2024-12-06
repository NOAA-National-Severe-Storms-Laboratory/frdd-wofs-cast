## Model Training 

To train WoFSCast, first create a config yaml in `training_configs`. Then run the following: 

```bash 
stdbuf -oL python -u train_wofscast.py --config config.yaml > & logs/log_train &
```

The actual model code can be found in [`frdd-wofs-cast/wofscast/model.py`](https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-cast/blob/master/wofscast/model.py) 

