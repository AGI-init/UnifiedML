"""Simple CelebA dataset utility class for pytorch. Depends on Kaggle.
https://github.com/sayantanauddy/vae_lightning/blob/main/data.py"""
import os
from functools import partial
import glob
import getpass
from cryptography.fernet import Fernet

import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import verify_str_arg

import PIL
import pandas as pd


class CelebA(Dataset):
    """Dataset for CelebA"""
    def __init__(self,
                 root,
                 split: str = "train",
                 target_type="attr",
                 transform=None,
                 target_transform=None,
                 download=False,
                 username=None,
                 key=None
                 ):

        # Get password, encrypt, and save for reuse
        for credential, value in {'KAGGLE_USERNAME': username, 'KAGGLE_KEY': key}.items():
            if value is None:
                if os.path.exists(credential):
                    with open(credential, 'r') as file:
                        encryption_key, encryption = file.readlines()
                        value = Fernet(encryption_key).decrypt(bytes(encryption, 'utf-8')).decode()
                else:
                    print('Please enter your Kaggle credentials (see Kaggle Account -> API -> New API Token).')
                    value, encryption_key = getpass.getpass(credential + ':'), Fernet.generate_key()
                    encryption = Fernet(encryption_key).encrypt(bytes(value, 'utf-8'))
                    with open(credential, 'w') as file:
                        file.writelines([encryption_key.decode('utf-8') + '\n', encryption.decode('utf-8')])

            os.environ[credential] = value

        self.root = root
        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download_from_kaggle()

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }

        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pd.read_csv(fn("list_eval_partition.csv"), delim_whitespace=False, header=0, index_col=0)
        # This file is not available in Kaggle
        # identity = pd.read_csv(fn("identity_CelebA.csv"), delim_whitespace=True, header=None, index_col=0)
        bbox = pd.read_csv(fn("list_bbox_celeba.csv"), delim_whitespace=False, header=0, index_col=0)
        landmarks_align = pd.read_csv(fn("list_landmarks_align_celeba.csv"), delim_whitespace=False, header=0, index_col=0)
        attr = pd.read_csv(fn("list_attr_celeba.csv"), delim_whitespace=False, header=0, index_col=0)

        mask = slice(None) if split_ is None else (splits['partition'] == split_)

        self.filename = splits[mask].index.values
        # self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = list(attr.columns)

    def download_from_kaggle(self):

        # Annotation files will be downloaded at the end
        label_files = ['list_attr_celeba.csv', 'list_bbox_celeba.csv', 'list_eval_partition.csv', 'list_landmarks_align_celeba.csv']

        # Check if files have been downloaded already
        files_exist = False
        for label_file in label_files:
            if os.path.isfile(os.path.join(self.root, label_file)):
                files_exist = True
            else:
                files_exist = False

        if files_exist:
            print("Files exist already")
        else:
            print("Downloading dataset. Please while while the download and extraction processes complete")
            # Download files from Kaggle using its API as per
            # https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python

            # Kaggle authentication
            # Remember to place the API token from Kaggle in $HOME/.kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            # Download all files of a dataset
            # Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
            api.dataset_download_files(dataset='jessicali9530/celeba-dataset',
                                       path=self.root,
                                       unzip=True,
                                       force=False,
                                       quiet=False)

            # Downoad the label files
            # Signature: dataset_download_file(dataset, file_name, path=None, force=False, quiet=True)
            for label_file in label_files:
                api.dataset_download_file(dataset='jessicali9530/celeba-dataset',
                                          file_name=label_file,
                                          path=self.root,
                                          force=False,
                                          quiet=False)

            # Clear any remaining *.csv.zip files
            files_to_delete = glob.glob(os.path.join(self.root, "*.csv.zip"))
            for f in files_to_delete:
                os.remove(f)

            print("Done!")

    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root,
                                        "img_align_celeba",
                                        "img_align_celeba",
                                        self.filename[index]))

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            # elif t == "identity":
            #     target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(f"Target type {t} is not recognized")

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

