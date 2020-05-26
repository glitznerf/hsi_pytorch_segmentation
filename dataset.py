# Creating a dataset for PyTorch trainig. Files need to be stored without a dot in name. The ground truth file must be named the same as the image file with a "_gt" extension.

# Imports
import sys
import torch
import numpy as np
import scipy.io
from PIL import Image
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
    gt = np.array(f_gt[var_name+'_gt'])                 # Converting gt data to numpy arrays
    return (image,gt)

# If the data is stored in images
# N/A (yet)


# Splitting a one-image dataset into a dataset of square patches
def split_image(image,gt,size=5):
    patches_horizontal = (image.shape[0])//size         # Number of patches on horizontal axis
    patches_vert = patches_horizontal                   # Same for vertical because of square dimensions
    images = np.zeros((patches_horizontal*patches_vert,200,5,5), dtype=int) # Empty dataframe for image patches
    gts = np.zeros((patches_horizontal*patches_vert,16,5,5), dtype=int)     # Empty dataframe for ground-truth patches
    i = 0
    filled = 0
    for hor in range(patches_horizontal):                           # Iterate over horizontal patches
        for vert in range(patches_vert):                            # Iterate over vertical patches
            for layer in range(200):
                images[i][layer] = image[hor*5:(hor+1)*5,vert*5:(vert+1)*5,layer]  # Fill image-patches dataframe
                if layer < 16:                                      # Introduce layering to gts
                    gt_temp = gt[hor*5:(hor+1)*5,vert*5:(vert+1)*5]
                    gt_temp[gt_temp!=layer+1] = 0                   # Make all pixels zero that aren't of current class
                    gt_temp[gt_temp>0] = 1                          # Make current class pixels activated (one)
                    gts[i][layer] = gt_temp                         # Fill gt-patches dataframe
                    if np.sum(gts[i])>0:                            # Count layers with positive activations
                        filled += 1
            i += 1
    print(f"Resizing image of size {image.shape} to patches {images.shape} and grund-truth of size {gt.shape} to ground-truth patches {gts.shape}")
    print(filled, " patches contain some class relation.")
    return images,gts

def augment_image(image,gt,method="orig"):                          # Augment images by intuitively named methods
    assert method in ["orig","rot","horflip","verflip"]
    if method == "orig":
        pass
    elif method == "rot":
        image,gt = np.array(image), np.array(gt)
        print(image.shape, gt.shape)
        image,gt = Image.fromarray(image),Image.fromarray(gt)
        image,gt = image.rotate(45), gt.rotate(45)
        image,gt = np.array(image), np.array(gt)
    elif method == "horflip":
        image,gt = np.flipud(image), np.flipud(gt)
    elif method == "verflip":
        image,gt = np.fliplr(image), np.fliplr(gt)
    return image,gt

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

    # for method in config.get("augmentations", []):
    #     imgs_aug, gts_aug = augment_image(images,gts,method)
    #     images = np.append(images, imgs_aug, axis=0)
    #     gts = np.append(gts, gts_aug, axis=0)
    print(images.shape, gts.shape)

    images,gts = torch.from_numpy(images), torch.from_numpy(gts)    # Transform dataframes to tensors
    data = TensorDataset(images, gts)                               # Create default dataset from tensors

    training_set_length = int(len(data) * config.get("tt-split", 0.8))      # Data split ratio
    print(f"Splitting the dataset of {len(data)} images into train/test datasets of sizes {[training_set_length,len(data)-training_set_length]}.")

    trainset, testset = random_split(data, [training_set_length, len(data)-training_set_length])           # Split data

    train_loader = DataLoader(trainset,batch_size=config.get("batch_size",5),shuffle=True)  # Create PyTorch dataloaders
    test_loader = DataLoader(testset,batch_size=config.get("batch_size",5),shuffle=True)
    return train_loader, test_loader
