import os
from aac_datasets.datasets.functional.clotho import download_clotho_dataset

os.makedirs("./data", exist_ok=True)

subset = "eval"

print(f"downloading Clotho {subset} dataset files...")
download_clotho_dataset(root="./data", subset=subset, verbose=1)