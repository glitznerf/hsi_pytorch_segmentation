# Creating a dataset for PyTorch trainig

# Imports
import numpy as np
import scipy.io
import h5py
from torch.utils.data import random_split, DataLoader


# Loading the data
def mat_dataset(file_name):             # If the data file is a matlab file
    directory = "labeled_data/"         # Standard folder where the data is stored
    try:
        f_i = h5py.File(directory + file_name + ".mat", "r")
        f_gt = h5py.File(directory + file_name + "_gt.mat", "r")
    except:                             # If matlab file version <7.3
        try:
            f_i = scipy.io.loadmat(directory + file_name + ".mat")
            f_gt = scipy.io.loadmat(directory + file_name + "_gt.mat")
        except:                         # Raise exception if neither file assumption is valid
            print("Cannot load data!")
            assert 1==2
    images = np.array(f_i)              # Converting data to numpy arrays
    gt = np.array(f_gt)                 # Converting data to numpy arrays
    return (images,gt)

# If the data is stored in images
# N/A (yet)


# Creating the PyTorch dataset
def create_dataset(config):
    file_name = (config.get("file_name")).split(".")
    if file_name[-1] == ".mat":
        data = mat_dataset(file_name[0])
    elif file_name[-1] == ".jpg":
        print("jpg files not implemented yet")
        assert 1==2
    else:
        print("This file type is not implemented (yet).")
        assert 1==2
    training_set_length = int(len(data) * 0.8)
    trainset, testset = random_split(training_set_length,[1-training_set_length])
    train_loader = DataLoader(trainset,batch_size=config.get("batch_size",5),shuffle=True)
    test_loader = DataLoader(testset,batch_size=config.get("batch_size",5),shuffle=True)
    return train_loader, test_loader
