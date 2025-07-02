from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, data, targets, transform=None, target_transform=None):
        """
        Custom dataset for loading data and targets.
        
        Args:
            data (list or array-like): Input data.
            targets (list or array-like): Corresponding targets for the data.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample and its corresponding target from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (sample, target) where sample is the input data and target is the corresponding label.
        """
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)
        
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target