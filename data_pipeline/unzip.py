import zipfile 
from pathlib import Path 

# run in directory you want unzipped
for z in sorted([i for i in Path('../era5/dev').glob('*wind*.zip')]):
    print(z)
    with zipfile.ZipFile(z, 'r') as zip_ref:
        zip_ref.extractall(Path('.'.join(str(z).split('.')[:-1])).absolute())