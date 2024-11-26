import os
import subprocess
import time

# A RUNNER SCRIPT FOR GENERATING A WOFSCAST-STYLE TESTING DATASET
# FROM EITHER INDIVIDUAL RAW WRFOUT FILES OR CURATED ZARR FILES

""" usage: stdbuf -oL python -u test_dataset_builder.py > & logs/log_test_dataset_builder & """

if __name__ == "__main__":
    
    config_name = "dataset_10min_test_config.yaml"
    
    tags = "--overwrite --legacy " #--do_drop_vars"
    
    # Command to reformat the WoFS wrfout files or Zarr files
    format_cmd = f"stdbuf -oL python -u format_wofs_wrfouts.py --config {config_name} {tags} >  logs/log_test_dataset 2>&1"
    
    print("Running format command:")
    print(format_cmd)
    
    # Execute the reformatting command
    process = subprocess.Popen(format_cmd, shell=True)
    
    # Wait for the dataset reformatting process to finish
    print("Waiting for dataset creation to complete...")
    process.wait()
    
    print("Dataset creation completed!")

