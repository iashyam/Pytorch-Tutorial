import requests
import os
import gzip
import shutil
from zipfile import ZipFile
import struct
import numpy as np
from PIL import Image
import csv
    

# making the data directory 
def check_data_directory(path: str) -> None:
    """Create the directory if it does not exist.
    Args:
        path (str): The directory path to check or create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url: str, save_path: str) -> None:
    """Download a file from a URL and save it to the specified path.
    Args:
        url (str): The URL of the file to download.
        save_path (str): The path where the file will be saved.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {save_path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

def extract_dir(zip_path: str, extract_to: str) -> None:
    """Extract a zip file to the specified directory.
    Args:
        zip_path (str): The path to the zip file.
        extract_to (str): The directory where the contents will be extracted.
    """
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def unzip_file(zip_path: str, extract_to: str) -> None:
    """Unzip a file to the specified directory.
    Args:
        zip_path (str): The path to the zip file.
        extract_to (str): The directory where the contents will be extracted.
    """
    with gzip.open(zip_path, 'rb') as f_in:
        with open(extract_to, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Unzipped {zip_path} to {extract_to}")

def parse_idx_images(filename: str) -> np.ndarray:
    """Parse IDX image file and return images as a numpy array.
    Args:
        filename (str): Path to the gzipped IDX image file.
    Returns:
        np.ndarray: Array of images with shape (num, rows, cols).
    """
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def parse_idx_labels(filename: str) -> np.ndarray:
    """Parse IDX label file and return labels as a numpy array.
    Args:
        filename (str): Path to the gzipped IDX label file.
    Returns:
        np.ndarray: Array of labels.
    """
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_images_and_targets(images: np.ndarray, labels: np.ndarray, out_dir: str) -> None:
    """Save images as PNG files and create a CSV linking filenames to labels.
    Args:
        images (np.ndarray): Array of images.
        labels (np.ndarray): Array of labels.
        out_dir (str): Output directory to save images and CSV.
    """
    data_dir = os.path.join(out_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    targets_path = os.path.join(out_dir, 'targets.csv')
    with open(targets_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for idx, (img, label) in enumerate(zip(images, labels)):
            img_filename = f"img_{idx:05d}.png"
            img_path = os.path.join(data_dir, img_filename)
            Image.fromarray(img).save(img_path)
            writer.writerow([img_filename, label])

# Downloading the MNIST dataset
request_url = 'http://yann.lecun.com/exdb/mnist/'

# make and MNIST data directoryk
mnist_data_path = './data/mnist'
check_data_directory(mnist_data_path)

# List of files to download
train_images = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
train_labels = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
test_images = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
test_labels = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
files = {
    "train-images": train_images,
    "train-labels": train_labels,
    "test-images": test_images,
    "test-labels": test_labels
}
# Download and extract MNIST files only if not already present
for file_name, url in files.items():
    save_path = os.path.join(mnist_data_path, f"{file_name}.gz")
    if not os.path.exists(save_path):
        download_file(url, save_path)
    else:
        print(f"File {save_path} already exists. Skipping download.")

# Parse and save training data only if not already present
train_data_dir = os.path.join(mnist_data_path, 'training_data', 'data')
train_targets_csv = os.path.join(mnist_data_path, 'training_data', 'targets.csv')
if not (os.path.exists(train_data_dir) and os.path.exists(train_targets_csv)):
    train_images_path = os.path.join(mnist_data_path, 'train-images.gz')
    train_labels_path = os.path.join(mnist_data_path, 'train-labels.gz')
    train_images = parse_idx_images(train_images_path)
    train_labels = parse_idx_labels(train_labels_path)
    save_images_and_targets(train_images, train_labels, os.path.join(mnist_data_path, 'training_data'))
else:
    print(f"Training data already exists. Skipping extraction.")

# Parse and save testing data only if not already present
test_data_dir = os.path.join(mnist_data_path, 'testing_data', 'data')
test_targets_csv = os.path.join(mnist_data_path, 'testing_data', 'targets.csv')
if not (os.path.exists(test_data_dir) and os.path.exists(test_targets_csv)):
    test_images_path = os.path.join(mnist_data_path, 'test-images.gz')
    test_labels_path = os.path.join(mnist_data_path, 'test-labels.gz')
    test_images = parse_idx_images(test_images_path)
    test_labels = parse_idx_labels(test_labels_path)
    save_images_and_targets(test_images, test_labels, os.path.join(mnist_data_path, 'testing_data'))
else:
    print(f"Testing data already exists. Skipping extraction.")
