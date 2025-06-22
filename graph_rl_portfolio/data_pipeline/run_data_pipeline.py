import subprocess
import os

pipeline_dir = os.path.dirname(__file__)

print("[1/3] Running data_loader.py (fetch raw OHLCV if needed)...")
subprocess.run(["python3", os.path.join(pipeline_dir, "data_loader.py")], check=True)

print("[2/3] Running preprocessor.py (compute features)...")
subprocess.run(["python3", os.path.join(pipeline_dir, "preprocessor.py")], check=True)

print("[3/3] Running splitter.py (train/val/test split)...")
subprocess.run(["python3", os.path.join(pipeline_dir, "splitter.py")], check=True)

print("\nData pipeline complete. Check the data_cache directory for outputs.") 