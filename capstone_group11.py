import import pandas as pd
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mahatiratusher/flight-price-dataset-of-bangladesh")

print("Path to dataset files:", path)