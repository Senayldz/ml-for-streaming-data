import os
from pathlib import Path

path = Path('c:/Users/90530/Desktop/barış/dataset')
for f in path.glob('*.csv'):
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"{f.name}: {size_mb:.2f} MB")
