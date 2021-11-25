from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import torch
class FruitsDataset(Dataset):
    printSize = True
    def __init__(self, pd_dataframe):
        """
        Args:
            pd_dataframe (dataframe): pandas dataframe
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd_dataframe
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = torch.tensor(list(self.data_info.iloc[:, 1]))
        # Third column is for an operation indicator
        self.operation_arr = np.asarray([False]*self.data_info.count())
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Transform image to tensor
        #normalized = torch.from_numpy(img / 255.).unsqueeze(0)
        img_as_tensor = self.to_tensor(img_as_img)
        if self.printSize:
            print("img tensor shape:")
            print(img_as_tensor.shape)
            self.printSize = False
        # Get label of the image based on the cropped pandas column
        single_image_label = torch.tensor(self.label_arr[index])
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len