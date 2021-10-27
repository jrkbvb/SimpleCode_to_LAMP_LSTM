import torch
from torch.utils.data import Dataset
# We'll create a Dataset class to use with PyTorch's Built-In Dataloaders
class S2LDataset(Dataset):
    '''
    A custom dataset class to use with PyTorch's built-in dataloaders.
    This will make feeding data to our models much easier downstream.

    data: np.arrays
    '''
    def __init__(self, data, labels, vectorize=False):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        sequence_data = self.data[idx]
        sequence_label = self.labels[idx]
        return sequence_data, sequence_label

    def __len__(self):
        return self.data.shape[0]