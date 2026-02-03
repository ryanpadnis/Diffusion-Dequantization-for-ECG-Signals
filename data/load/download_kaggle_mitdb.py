import kagglehub
from pathlib import Path
import shutil

# Download latest version of the dataset
path = kagglehub.dataset_download("protobioengineering/mit-bih-arrhythmia-database-modern-2023")

print("Path to dataset files:", path)

# Move all CSVs to your project data directory
dest_dir = Path("data/data/raw/mitdb_kaggle")
dest_dir.mkdir(parents=True, exist_ok=True)

for csv_file in Path(path).rglob("*.csv"):
    shutil.copy(csv_file, dest_dir / csv_file.name)
    print(f"Copied {csv_file} to {dest_dir / csv_file.name}")
