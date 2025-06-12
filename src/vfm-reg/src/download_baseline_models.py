# This script downloads the trained weights of the baseline models and puts them in the corresponding paths.

import shutil
import urllib.request
import zipfile
from pathlib import Path

import gdown

download_url = {
    'dip':
    'https://raw.githubusercontent.com/fabiopoiesi/dip/master/model/final_chkpt.pth',
    'gedi':
    'https://drive.google.com/file/d/1Lpep5QigALjk60h8bNJAUH3DnxtnGcZX/view?usp=sharing',
    'spinnet':
    'https://raw.githubusercontent.com/QingyongHu/SpinNet/main/pre-trained_models/KITTI_best.pkl',
    'gcl':
    'https://drive.google.com/file/d/17rt_eNBiLdOr5WxxYz8rOuUDwGsnDTXZ/view?usp=sharing',
    'fcgf':
    'https://node1.chrischoy.org/data/publications/fcgf/2019-07-31_19-30-19.pth',
    'pointdsc':
    'https://raw.githubusercontent.com/XuyangBai/PointDSC/master/snapshot/PointDSC_KITTI_release/models/model_best.pkl'
}

root = Path(__file__).parent
weight_path = {
    'dip': root / 'dip' / 'final_chkpt.pth',
    'gedi': root / 'gedi' / 'chkpt.tar',
    'spinnet': root / 'spinnet' / 'KITTI_best.pkl',
    'gcl': root / 'gcl' / 'kitti_chkpt.pth',
    'fcgf': root / 'fcgf' / '2019-07-31_19-37-00.pth',
    'pointdsc': root / 'pointdsc' / 'model_best.pkl'
}

for model, url in download_url.items():
    path = weight_path[model]
    if url is None:
        continue
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    print(f'Downloading {model} weights...')
    if 'drive.google.com' in url:
        gdown.download(url, f'{str(path)}.zip', fuzzy=True, quiet=True)
        with zipfile.ZipFile(f'{str(path)}.zip', 'r') as zip_ref:
            zip_ref.extractall(str(path.parent))

        if model == 'gedi':
            extracted_path = path.parent / 'chkpts' / '3dmatch' / 'chkpt.tar'
            extracted_path.rename(path)
        elif model == 'gcl':
            extracted_path = path.parent / 'checkpoints' / 'KITTI' / 'best_val_checkpoint.pth'
            extracted_path.rename(path)
        else:
            assert False, 'Unknown model'
        Path(f'{str(path)}.zip').unlink()
        shutil.rmtree(extracted_path.parent.parent, ignore_errors=True)

    else:
        urllib.request.urlretrieve(url, str(path))
