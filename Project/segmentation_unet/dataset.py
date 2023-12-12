import os
import numpy as np
import nibabel as nib
import torch
from monai.data import ArrayDataset


def extract_dataset(dataset_path='BraTS2021_Training_Data/'):
    dataset_path = 'BraTS2021_Training_Data/'
    # pour chaque dossier dans le dataset
    for folder in os.listdir(dataset_path):

        # on récupère les fichiers "nom_du_dossier_flair.nii.gz" et "nom_du_dossier_seg.nii.gz"
        flair_path = os.path.join(dataset_path, folder, folder+'_flair.nii.gz')
        seg_path = os.path.join(dataset_path, folder, folder+'_seg.nii.gz')
        # on décompresse les fichiers et on les convertit en array numpy masks et images
        flair = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # on les sauvegarde dans un dossier "data"
        with open(os.path.join('data', folder+'_flair.npy'), 'wb') as f:
            np.save(f, flair)
        with open(os.path.join('data', folder+'_seg.npy'), 'wb') as f:
            np.save(f, seg)


def load_npy_files(folder='data'):
    # on récupère les fichiers .npy dans le dossier "data"
    files = os.listdir(folder)
    # gather in one array all the images and in another array all the masks
    images = []
    masks = []
    for file in files:
        if 'flair' in file:
            images.append(np.load(os.path.join(folder, file)))
        elif 'seg' in file:
            masks.append(np.load(os.path.join(folder, file)))
    return images, masks


def get_dataset(folder='data'):
    images, masks = load_npy_files(folder)
    dataset = ArrayDataset(images, masks, transform=None)
    return dataset


def split_dataset(dataset, train_ratio=0.8):
    """ Split dataset in train and test sets. """
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    return train_dataset, test_dataset
