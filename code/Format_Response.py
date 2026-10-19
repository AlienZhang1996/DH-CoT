"""
Format the H-CoT and DH-CoT response files, remove redundant content, and add the original promot questions..
"""

import os, argparse
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from Tools.Base import save_format_descriptions_HCoT

if __name__ == "__main__":
    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/cfg_Format_Response.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    # Specify folder path
    dataset_csv = config.dataset_csv # the dataset of promot questions 
    description_folder = config.description_folder # Folder path of files to be processed
    
    # Recursive matching to find all .csv files
    folder_path = Path(description_folder)
    csv_files = list(folder_path.rglob('*.csv'))
    
    # Iterate over the found CSV files and read the contents
    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        try:
            # save_format_descriptions(dataset_csv, file_path)
            save_format_descriptions_HCoT(dataset_csv, file_path)
        except Exception as e:
            print(f"Handle exceptions: {e}")