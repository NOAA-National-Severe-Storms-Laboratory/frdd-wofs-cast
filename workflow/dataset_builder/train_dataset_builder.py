import os
import subprocess
import time

# A RUNNER SCRIPT FOR GENERATING A WOFSCAST-STYLE TRAINING DATASET
# FROM EITHER INDIVIDUAL RAW WRFOUT FILES OR CURATED ZARR FILES
# ADDITIONALLY, COMPUTE THE NORMALIZATION STATISTICS FOR THE NEW TRAINING DATASET.

""" usage: stdbuf -oL python -u train_dataset_builder.py > & log_dataset_builder & """

if __name__ == "__main__":
    
    config_name = "dataset_10min_train_config.yaml"

    # Command to reformat the WoFS wrfout files or Zarr files
    format_cmd = f"stdbuf -oL python -u format_wofs_wrfouts.py --config {config_name} --overwrite >  logs/log_formatter 2>&1"
    
    print("Running format command:")
    print(format_cmd)
    
    # Execute the reformatting command
    process = subprocess.Popen(format_cmd, shell=True)
    
    # Wait for the dataset reformatting process to finish
    print("Waiting for dataset creation to complete...")
    process.wait()
    
    print("Dataset creation completed!")

    # Command to compute normalization statistics
    norm_stats_cmd = f"stdbuf -oL python -u compute_norm_stats.py --config {config_name} > logs/log_compute_norm_stats 2>&1"
    
    print("Running normalization statistics command:")
    print(norm_stats_cmd)
    
    # Execute the normalization statistics command
    process_norm = subprocess.Popen(norm_stats_cmd, shell=True)

    # Wait for normalization statistics computation to finish
    print("Waiting for normalization statistics computation to complete...")
    process_norm.wait()

    print("Normalization statistics computation completed!")


