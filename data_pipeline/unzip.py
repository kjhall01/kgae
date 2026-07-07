import zipfile 
from pathlib import Path 

# Run from the target extraction directory.
for z in sorted([i for i in Path('../era5/dev').glob('*wind*.zip')]):
    print(z)
    with zipfile.ZipFile(z, 'r') as zip_ref:
        zip_ref.extractall(Path('.'.join(str(z).split('.')[:-1])).absolute())
