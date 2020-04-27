# Creating a dataset for PyTorch trainig. Files need to be stored without a dot in name. The ground truth file must be named the same as the image file with a "_gt" extension.

# Imports
import sys
import torch
import numpy as np
import scipy.io
import h5py
from torch.utils.data import random_split, TensorDataset, DataLoader


# Loading the data
def mat_dataset(file_name, var_name):             # If the data file is a matlab file
    directory = "labeled_data/"         # Standard folder where the data is stored
    try:
        f_i = h5py.File(directory + file_name + ".mat", "r")
        f_gt = h5py.File(directory + file_name + "_gt.mat", "r")
        print("Loading with h5py")
    except:                             # If matlab file version <7.3
        try:
            f_i = scipy.io.loadmat(directory + file_name + ".mat")
            f_gt = scipy.io.loadmat(directory + file_name + "_gt.mat")
            print("Loading with SciPy")
        except:                         # Raise exception if neither file assumption is valid
            sys.exit("Cannot load data!")
    image = np.array(f_i[var_name+'_corrected']).astype('uint8')              # Converting imagedata to numpy arrays
    # print(image)
    # print(image.shape)
    gt = np.array(f_gt[var_name+'_gt'])                 # Converting gt data to numpy arrays
    return (image,gt)

# If the data is stored in images
# N/A (yet)


# Splitting a one-image dataset into a dataset of square patches
def split_image(image,gt,size=5):
    patches_horizontal = (image.shape[0])//size         # Number of patches on horizontal axis
    patches_vert = patches_horizontal                   # Same for vertical because of square dimensions
    images = np.zeros((patches_horizontal*patches_vert,5,5,200), dtype=int) # Empty dataframe for image patches
    gts = np.zeros((patches_horizontal*patches_vert,5,5), dtype=int)        # Empty dataframe for ground-truth patches
    i = 0
    for hor in range(patches_horizontal):                           # Iterate over horizontal patches
        for vert in range(patches_vert):                            # Iterate over vertical patches
            images[i] = image[hor*5:(hor+1)*5,vert*5:(vert+1)*5,:]  # Fill image-patches dataframe
            gts[i] = gt[hor*5:(hor+1)*5,vert*5:(vert+1)*5]          # Fill gt-patches dataframe
            i += 1
    print(f"Resizing image of size {image.shape} to patches {images.shape} and grund-truth of size {gt.shape} to ground-truth patches {gts.shape}")
    return images,gts


# Creating the PyTorch dataset
def create_dataset(config):
    file_name = (config.get("file_name")).split(".")
    if file_name[-1] == "mat":
        img,gt = mat_dataset(file_name[0],var_name=config.get("variable_name"))      # Load data
    elif file_name[-1] == "jpg":
        sys.exit("jpg files not implemented yet")
    else:
        sys.exit("This file type is not implemented (yet).")

    if img.ndim == 3:           # Split single image dataset up into patches
        images,gts = split_image(img,gt)
    elif img.ndim == 4:         # Correct image dataframe dimensions
        images,gts = img,gt
    else:                           # Incorrect image dataframe dimensions
        print(f"Cannot work with images of dimensions {images.shape}.")
        sys.exit()
    images,gts = torch.from_numpy(images), torch.from_numpy(gts)    # Transform dataframes to tensors
    data = TensorDataset(images, gts)                               # Create default dataset from tensors

    training_set_length = int(len(data) * config.get("tt-split", 0.8))      # Data split ratio
    print(f"Splitting the dataset of {len(data)} images into train/test datasets of sizes {[training_set_length,len(data)-training_set_length]}.")

    trainset, testset = random_split(data, [training_set_length, len(data)-training_set_length])           # Split data

    train_loader = DataLoader(trainset,batch_size=config.get("batch_size",5),shuffle=True)  # Create PyTorch dataloaders
    test_loader = DataLoader(testset,batch_size=config.get("batch_size",5),shuffle=True)
    return train_loader, test_loader
