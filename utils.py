import numpy as np
import torch
from torch import Tensor
from typing import Tuple

def text_to_bytes(texts: list[str]) -> torch.Tensor:
    '''converts text to bytes and returns [batch, seq_len] tensor 
    and pads to max_seq_len in batch with zeros'''
    return torch.nn.utils.rnn.pad_sequence([torch.Tensor([ord(c) for c in text]).to(dtype=torch.long) for text in texts], batch_first=True)

def bytes_to_text(bytes: torch.Tensor) -> list[str]:
    '''converts bytes in torch.Tensor to text'''
    texts = []
    bytes = bytes.to(dtype=torch.uint8)
    for i in range(bytes.size(0)): # iter over batch
        texts.append(''.join([chr(b) for b in bytes[i].tolist()]))
    return texts

def img_to_bytes(images: np.ndarray) -> Tensor:
    '''converts a list of images to bytes, images are shape [batch, image]'''
    return torch.from_numpy(images.astype(np.uint8)).reshape(images.shape[0], -1).to(dtype=torch.long) # batch first

def bytes_to_img(bytes: torch.Tensor, image_shape: Tuple[int, int]) -> np.ndarray:
    '''converts a list of bytes to images, images are shape [batch, image_sequence]'''
    bytes = bytes.to(dtype=torch.uint8)
    return np.frombuffer(bytes.numpy(), dtype=np.uint8).reshape((-1,) + image_shape).squeeze(0) # batch first


# loading mnist

import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self):
        input_path = 'MNIST Dataset'
        training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: # (x_train, y_train), (x_test, y_test
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        x_train = np.stack(x_train)
        x_test = np.stack(x_test)
        y_train = np.stack(y_train)
        y_test = np.stack(y_test)
        return (x_train, y_train),(x_test, y_test)   

import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#
from torch import Tensor


def patch_images(images: torch.Tensor, patch_size: Tuple[int, int] = (4, 4)) -> torch.Tensor:
    # images are shape [batch, height, width]
    # returns images of shape [batch, height, width, patch_height, patch_width]
    patch_height, patch_width = patch_size
    patches = images.unfold(1, patch_height, 1).unfold(2, patch_width, 1)
    return patches
    

