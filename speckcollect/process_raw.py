import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

# Define the neural network
snn = nn.Sequential(
    nn.AvgPool2d(kernel_size=(2, 2)),
    # the input of the 1st DynapCNN Core will be (1, 64, 64)
    nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    nn.ReLU()
    
)

# Function to process a single image
def process_image(image_path):
    # Load image, assuming gray scale image
    image = read_image(image_path).float()  # Convert to float to match expected NN input
    if image.shape[0] > 1:
        image = image.mean(0, keepdim=True)  # Convert to grayscale if not already
    
    # Apply the neural network
    with torch.no_grad():
        output = snn(image.unsqueeze(0))  # Add batch dimension
    
    # Convert back to PIL Image to save
    output_image = to_pil_image(output.squeeze(0))  # Remove batch dimension
    return output_image

# Load images from a directory
source_directory = '/home/adam/Downloads/test002'
destination_directory = '/home/adam/Documents/test002-conv'

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Get sorted list of image file paths
image_files = sorted([os.path.join(source_directory, f) for f in os.listdir(source_directory) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Process each image and save the result
for img_path in tqdm(image_files, desc="Processing Images"):
    output_image = process_image(img_path)
    # Define a new file name for the processed image
    base_name = os.path.basename(img_path)
    save_path = os.path.join(destination_directory, base_name)
    output_image.save(save_path)
