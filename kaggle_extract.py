import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# Path to the kaggle.json file
kaggle_json_path = r"C:\Users\piotr\.kaggle\kaggle.json"

# Load the credentials from the kaggle.json file
with open(kaggle_json_path, 'r') as file:
    kaggle_credentials = json.load(file)

    # Extract username and key
    kaggle_username = kaggle_credentials['username']
    kaggle_key = kaggle_credentials['key']

# Set up Kaggle API credentials in environment variables
os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

# Authenticate with the Kaggle API
api = KaggleApi()
api.authenticate()

# Download a dataset
dataset = r'julian3833/jigsaw-unintended-bias-in-toxicity-classification' 
download_path = os.getcwd()
api.dataset_download_files(dataset, path=download_path, unzip=True)