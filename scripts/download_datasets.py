
import os
import gdown
import zipfile


LICENSE = """
These are either re-distribution of the original datasets or derivatives (through simple processing) of the original datasets. 
Please read and respect their licenses and terms before use. 
You should cite the original papers if you use any of the datasets.

Links:

YouTubeVOS: https://youtube-vos.org
DAVIS: https://davischallenge.org/
"""

print(LICENSE)
print('Datasets will be downloaded and extracted to ../YouTube2018, ../DAVIS16, ../DAVIS17')
reply = input('[y] to confirm, others to exit: ')
if reply != 'y':
    exit()


"""
DAVIS dataset
"""
# Google drive mirror: https://drive.google.com/drive/folders/1hEczGHw7qcMScbCJukZsoOW4Q9byx16A?usp=sharing
os.makedirs('../DAVIS/2017', exist_ok=True)

print('Downloading DAVIS 2016...')
gdown.download('https://drive.google.com/uc?id=198aRlh5CpAoFz0hfRgYbiNenn_K8DxWD', output='../DAVIS/DAVIS-data.zip', quiet=False)

print('Downloading DAVIS 2017 trainval...')
gdown.download('https://drive.google.com/uc?id=1kiaxrX_4GuW6NmiVuKGSGVoKGWjOdp6d', output='../DAVIS/2017/DAVIS-2017-trainval-480p.zip', quiet=False)

print('Downloading DAVIS 2017 testdev...')
gdown.download('https://drive.google.com/uc?id=1fmkxU2v9cQwyb62Tj1xFDdh2p4kDsUzD', output='../DAVIS/2017/DAVIS-2017-test-dev-480p.zip', quiet=False)

print('Downloading DAVIS 2017 scribbles...')
gdown.download('https://drive.google.com/uc?id=1JzIQSu36h7dVM8q0VoE4oZJwBXvrZlkl', output='../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip', quiet=False)

print('Extracting DAVIS datasets...')
with zipfile.ZipFile('../DAVIS/DAVIS-data.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/')
os.rename('../DAVIS/DAVIS', '../DAVIS/2016')

with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-trainval-480p.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
os.rename('../DAVIS/2017/DAVIS', '../DAVIS/2017/trainval')

with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-test-dev-480p.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
os.rename('../DAVIS/2017/DAVIS', '../DAVIS/2017/test-dev')

print('Cleaning up DAVIS datasets...')
os.remove('../DAVIS/2017/DAVIS-2017-trainval-480p.zip')
os.remove('../DAVIS/2017/DAVIS-2017-test-dev-480p.zip')
os.remove('../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip')
os.remove('../DAVIS/DAVIS-data.zip')

print('Done.')
