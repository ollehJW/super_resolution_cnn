import torch
import numpy as np
import glob as glob
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class SRCNNDataset(Dataset):
    """
    The SRCNN dataset module.

    - Output
        image: Low resolution image.
        label: High resolution image.
    """
    def __init__(self, image_paths, label_paths):
        self.all_image_paths = glob.glob(f"{image_paths}/*")
        self.all_label_paths = glob.glob(f"{label_paths}/*") 
    def __len__(self):
        return (len(self.all_image_paths))
    def __getitem__(self, index):
        image = Image.open(self.all_image_paths[index]).convert('RGB')
        label = Image.open(self.all_label_paths[index]).convert('RGB')
        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        image /= 255.
        label /= 255.
        image = image.transpose([2, 0, 1])
        label = label.transpose([2, 0, 1])
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )

# Prepare the data loaders
def get_dataloaders(train_image_paths, train_label_paths, valid_image_path, valid_label_paths, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE):
    """
    Make torch dataloaders.

    - Parameters
        train_image_paths: Path of train images.
        valid_image_path: Path of valid images.
        train_label_paths: Path of train labels.
        valid_label_path: Path of valid labels.
        TRAIN_BATCH_SIZE: Batch size of train dataset.
        TEST_BATCH_SIZE: Batch size of test dataset.
    """
    dataset_train = SRCNNDataset(
        train_image_paths, train_label_paths
    )
    dataset_valid = SRCNNDataset(
        valid_image_path, valid_label_paths
    )
    train_loader = DataLoader(
        dataset_train, 
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )

    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_valid)}")
    
    return train_loader, valid_loader
