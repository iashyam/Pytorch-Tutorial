from PIL import Image
import os
import pandas as pd
import numpy as np


image_dtype = np.float32
abel_dtype = np.uint8

class MNIST:
    def __init__(self):

        self.root_dir = 'data'
        self.mnist_dir = os.path.join(self.root_dir, 'mnist')
        self.training_data_dir = os.path.join(self.mnist_dir, 'training_data')
        self.test_data_dir = os.path.join(self.mnist_dir, 'testing_data')

    @property
    def training_data(self) -> tuple:
        """
        Returns the training data as a tuple of images and labels.
        """
        return self.get_data(self.training_data_dir)
    @property
    def test_data(self) -> tuple:
        """
        Returns the test data as a tuple of images and labels.
        """
        return self.get_data(self.test_data_dir)

    def get_data(self, directory: str) -> tuple:
        """
        convert any data into lists. 
        """
        images_path = os.path.join(directory, 'data')
        labels_path = os.path.join(directory, 'targets.csv')
        labels_csv = pd.read_csv(labels_path, index_col=0)

        data = []
        labels = []

        #assert os.path.exists(images_path), f"Images directory {images_path} does not exist."
        #assert os.path.exists(labels_path), f"Labels file {labels_path} does not exist."

        for filename in os.listdir(images_path):
            img_path =  os.path.join(images_path, filename)
            data.append(np.array(Image.open(img_path).convert('L'), dtype=image_dtype))
            labels.append(labels_csv.loc[filename, 'label'])

        return np.array(data), np.array(labels)
