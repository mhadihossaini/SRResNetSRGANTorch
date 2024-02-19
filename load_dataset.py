import torch
import os
from PIL import Image
from torchvision.transforms.functional import to_tensor

# Define a dataset class for loading local flower images in high and low resolution
class LocalFlowersDataset(torch.utils.data.Dataset):
    """
    A custom dataset for loading flower images in both high and low resolution.
    
    Parameters:
    - directory: A string indicating the path to the directory containing the images.
    """
    
    def __init__(self, directory):
        """
        Initializes the dataset object, listing all relevant image files.
        """
        super(LocalFlowersDataset, self).__init__()
        self.directory = directory
        # List all high-resolution images, ignoring low-resolution counterparts
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.jpg') and '_lr' not in f]
        # Sort filenames numerically for consistent ordering
        self.filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    def __len__(self):
        """
        Returns the total number of image pairs in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Retrieves the low-resolution and high-resolution image pair by index.
        
        Parameters:
        - idx: An integer index for the data item.
        
        Returns:
        - A tuple of tensors representing the low-resolution and high-resolution images.
        """
        # Construct file paths for the high-resolution and corresponding low-resolution images
        hr_img_path = os.path.join(self.directory, self.filenames[idx])
        lr_img_path = hr_img_path.replace('.jpg', '_lr.jpg')
        
        # Load images and convert them to tensors
        lr_img = Image.open(lr_img_path)
        hr_img = Image.open(hr_img_path)

        return to_tensor(lr_img), to_tensor(hr_img)


# Use the code bellow in train_model to load dataset
    
# train_dir = 'train_dataset'
# test_dir = 'test_dataset'

# # Creating dataset instances
# train_dataset = LocalFlowersDataset(train_dir)
# test_dataset = LocalFlowersDataset(test_dir)

# # Creating DataLoader instances

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
