[This repository is to open and share an example codes for the research program supported by Korean Ministry of Health and Welfare.]

# Segmentation model on obfuscated medical image
This repository is about learning a segmentation model based on obfuscated medical images 
and making inferences based on it. 

## Requirements
Run `pip install requirements.txt` command to set your own virtual environment.

## Config
Check `scr/config/config.py` and adapt the data paths according to your environment.

## Usage
Run `bash experiments/train_f_dev.sh` and `bash experiments/validate_f_dev.sh` to 
train a segmentation model for anonymization and make inferences based on it.
