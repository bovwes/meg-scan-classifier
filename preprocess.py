import argparse
from utils import *
import os
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['Intra', 'Cross'], required=True)
parser.add_argument('--downsample_factor', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=8)

np.random.seed(42)

if __name__ == '__main__':
    args = parser.parse_args()

    out_file = f"data/{args.data}/processed.h5"

    if not os.path.exists(out_file):

        print(f"Loading {args.data} data...")
        training_generator = load_training(args.data, args.batch_size) # Generator to process files sequentially
        X_all_train = collect_data(training_generator)

        print(f"Calculating scaling parameters...")
        p_lower = np.percentile(X_all_train, 5, axis=0)
        p_upper = np.percentile(X_all_train, 95, axis=0)

        # Process data in batches
        print(f"Processing data in batches...")
        training_generator = load_training(args.data, args.batch_size) # Generator to process files sequentially
        test_generator = load_testing(args.data, args.batch_size)

        i = 1
        for X_train, y_train in training_generator:
            X_train = robust_scaler(np.array(X_train), p_lower, p_upper) # Scale data 
            X_train, y_train, X_val, y_val = prep(X_train, y_train, args.downsample_factor, build_validation_set=True)
            y_train = relabel_all(y_train)
            y_val = relabel_all(y_val)
            
            # Append batch to file
            append_to_h5(out_file,
                         X_train=X_train, y_train=y_train,
                         X_val=X_val, y_val=y_val)
            print("Processed training batch", i)
            i += 1

        i = 1
        for X_test, y_test in test_generator:
            X_test = robust_scaler(np.array(X_test), p_lower, p_upper)
            X_test, y_test = prep(X_test, y_test, args.downsample_factor)
            y_test = relabel_all(y_test)

            # Append batch to file
            append_to_h5(out_file,
                         X_test=X_test, y_test=y_test)
            print("Processed test batch", i)
            i += 1

        # Shuffle data
        print("Shuffling data...")
        with h5py.File(out_file, 'a') as f:
            dataset_pairs = [('X_train', 'y_train'), ('X_val', 'y_val'), ('X_test', 'y_test')]
            for data_key, label_key in dataset_pairs:
                data, labels = f[data_key][:], f[label_key][:]
                combined = list(zip(data, labels))
                np.random.shuffle(combined)
                data_shuffled, labels_shuffled = zip(*combined)

                del f[data_key], f[label_key]
                f.create_dataset(data_key, data=np.array(data_shuffled))
                f.create_dataset(label_key, data=np.array(labels_shuffled))

        with h5py.File(out_file, 'r') as f:
            print(f"Finished\nX_train shape: {f['X_train'].shape}\nX_val shape: {f['X_val'].shape}\nX_test shape: {f['X_test'].shape}")

    else:
        print(f"A processed file already exists for {args.data}")
