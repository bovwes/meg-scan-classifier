import h5py
import numpy as np
from scipy.stats import zscore
import os
from skimage.transform import rescale, resize

def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def downsample(X, y, downsample_factor):

    X_train = []
    y_train = []

    for (i,data) in enumerate(X):
        X_train.extend([data[:,j::downsample_factor] for j in range(downsample_factor)])
        y_train.extend(np.repeat(y[i], downsample_factor))

    return X_train, y_train

def get_matrix(filename):
    with h5py.File(filename, "r") as f:
        dataset_name = get_dataset_name(filename)
        matrix = f.get(dataset_name)[()]
        return matrix
    
def load_testing(directory, batch_size):
    test_files = os.listdir(f"./data/{directory}/test")

    for i in range(0, len(test_files), batch_size):
        x_test_batch = [get_matrix(f"./data/{directory}/test/{name}") for name in test_files[i:i+batch_size]]
        y_test_batch = ['_'.join(name.split('_')[:-2]) for name in test_files[i:i+batch_size]]
        yield x_test_batch, y_test_batch

def load_training(directory, batch_size):
    train_files = os.listdir(f"./data/{directory}/train")
    
    for i in range(0, len(train_files), batch_size):
        x_train_batch = [get_matrix(f"./data/{directory}/train/{name}") for name in train_files[i:i+batch_size]]
        y_train_batch = ['_'.join(name.split('_')[:-2]) for name in train_files[i:i+batch_size]]
        yield x_train_batch, y_train_batch
    
def relabel(s):
    """
    Map the fil
    """
    if s.startswith("rest"):
        return 0
    elif s.startswith("task_motor"):
        return 1
    elif s.startswith("task_story"):
        return 2
    elif s.startswith("task_working"):
        return 3
    
def relabel_all(labels):
    lab  = [relabel(s) for s in labels]
    return np.array(lab)

def build_val(X, y, val_split = 8):
    X_t = []
    y_t = []
    X_val = []  
    y_val = []
    for i in range(np.shape(X)[0]):
        if i % val_split == 0:
            X_val.append(X[i])
            y_val.append(y[i])
        else:
            X_t.append(X[i])
            y_t.append(y[i])
    return np.array(X_t), np.array(X_val), np.array(y_t), np.array(y_val)

def prep(X, y, downsample_factor, build_validation_set = False):
    X_ds, y_ds = downsample(X, y, downsample_factor)
    X_ds =  np.array(X_ds)
    if build_validation_set:
        X_train, X_val, y_train, y_val = build_val(X_ds, y_ds)
        return X_train, y_train, X_val, y_val
    else:
        return X_ds, y_ds
    
def append_to_h5(file_path, **arrays):
    with h5py.File(file_path, 'a') as f:
        for key, array in arrays.items():
            if key in f:
                dataset = f[key]
                current_len = dataset.shape[0]
                additional_len = array.shape[0]
                dataset.resize((current_len + additional_len,) + dataset.shape[1:])
                dataset[current_len:] = array
            else:
                maxshape = (None,) + array.shape[1:]
                f.create_dataset(key, data=array, maxshape=maxshape)

def collect_data(generator):
    data_list = []
    for X_batch, _ in generator:
        data_list.extend(X_batch)
    return np.vstack(data_list)

def get_mean_std(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std


def robust_scaler(data, p_lower, p_upper):
    scale = p_upper - p_lower
    scale[scale == 0] = 1
    scaled_data = (data - p_lower) / scale
    scaled_data = np.clip(scaled_data, 0, 1)
    return scaled_data