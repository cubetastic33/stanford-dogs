import os
import shutil
import tarfile
import random

import requests
from scipy.io import loadmat
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class StanfordDogsDataset(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.

    Args:
        root (string): Directory where the data is stored
        set_type (string, optional): Specify `train`, `validation`, or `test`. If
            unspecified, it is taken as `test`.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed tensor.
    """

    def __init__(self, root, set_type="test", transform=T.ToTensor):
        self.root = root
        self.transform = transform
        self.file_paths = []
        self.labels = []
        label_names = self.get_labels()
        if not os.path.isdir(os.path.join(root, "images")):
            self.download()
        for dirpath, _, files in os.walk(os.path.join(root, "images", set_type)):
            for file in files:
                self.file_paths.append(os.path.join(dirpath, file))
                self.labels.append(label_names[os.path.split(dirpath)[-1]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = Image.open(self.file_paths[item])
        image = self.transform(image)
        image = torch.from_numpy(np.asarray(image))

        return image, torch.tensor(self.labels[item])

    def download(self):
        """Download the dataset"""
        downloads_dir = os.path.join(self.root, "downloads")
        data_dir = os.path.join(self.root, "images")
        try:
            shutil.rmtree(self.root)
        except FileNotFoundError:
            pass
        finally:
            os.mkdir(self.root)
            os.mkdir(downloads_dir)
        for url in [
            "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
        ]:
            get_response = requests.get(url, stream=True)
            file_name = os.path.join(downloads_dir, url.split("/")[-1])
            with open(file_name, "wb") as f:
                print(f"Downloading {url}")
                for chunk in get_response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
        for file in [
            os.path.join(downloads_dir, "images.tar"),
            os.path.join(downloads_dir, "lists.tar"),
        ]:
            with tarfile.open(file) as f:
                print(f"Extracting {file}")
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner) 
                    
                
                safe_extract(f, downloads_dir)
                os.remove(file)
        # Split images into train, validation, and test sets
        print("Splitting dataset")
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, "train"))
        os.mkdir(os.path.join(data_dir, "validation"))
        train_list = [f[0][0] for f in loadmat(os.path.join(downloads_dir, "train_list.mat"))["file_list"]]
        # Shuffle the training images
        random.shuffle(train_list)
        for (i, file) in enumerate(train_list):
            if i < 200:
                # The first 200 training images get put into the validation directory
                target_dir = os.path.join(data_dir, "validation")
            else:
                # The rest go into the train directory
                target_dir = os.path.join(data_dir, "train")
            try:
                # Create the directory for the breed if it doesn't exist
                os.mkdir(os.path.join(target_dir, os.path.split(file)[0]))
            except FileExistsError:
                # The directory was already there
                pass
            finally:
                # Move the image
                shutil.move(os.path.join(downloads_dir, "Images", file), os.path.join(target_dir, file))
        # Move the test images
        os.mkdir(os.path.join(data_dir, "test"))
        test_list = loadmat(os.path.join(downloads_dir, "test_list.mat"))["file_list"]
        for file in test_list:
            if not os.path.isdir(os.path.join(data_dir, "test", os.path.split(file[0][0])[0])):
                # Create the directory for the breed if it doesn't exist
                os.mkdir(os.path.join(data_dir, "test", os.path.split(file[0][0])[0]))
            # Move the image
            shutil.move(os.path.join(downloads_dir, "Images", file[0][0]), os.path.join(data_dir, "test", file[0][0]))
        shutil.move(os.path.join(downloads_dir, "file_list.mat"))
        shutil.rmtree(downloads_dir)
        print("Splitting complete")

    def get_labels(self):
        subdirs = set()
        labels = {}
        for subdir, _, _ in os.walk(os.path.join(self.root, "images/test")):
            if (label := os.path.split(subdir)[-1]) != "test":
                subdirs |= {label}
                labels[label] = len(subdirs) - 1
        return labels
