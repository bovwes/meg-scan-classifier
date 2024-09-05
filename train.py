import argparse
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import utils as p
from models import RNNModel
from tqdm import tqdm
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['Intra', 'Cross'], required=False)
parser.add_argument('--num_train_epochs', type=int, default=5)

np.random.seed(42)
tf.random.set_seed(42)

if __name__ == '__main__':     
    args = parser.parse_args()

    # Load data
    with h5py.File(f'data/{args.data}/processed.h5', 'r') as f:
        X_train = f['X_train']
        y_train = f['y_train']
        X_val = f['X_val']
        y_val = f['y_val']
        X_test = f['X_test']
        y_test = f['y_test']

        # Tranpose
        X_train = np.transpose(X_train,axes=(0,2,1))
        X_val = np.transpose(X_val,axes=(0,2,1))
        X_test = np.transpose(X_test,axes=(0,2,1))

        # Model
        model_instance = RNNModel(input_shape=(X_train.shape[1], X_train.shape[2]))
        model = model_instance.build_model()
        model.compile(
                    optimizer=tf.keras.optimizers.Adam(clipvalue=0.5),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
    
        model.fit(X_train, y_train, epochs=args.num_train_epochs,
                        validation_data=(X_val, y_val), verbose=1)
            
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

        print(f'Test accuracy: {accuracy}, test loss: {loss}')

        # Save
        save_path = os.path.join('models', f'RNN_model_{args.data}.keras')
        os.makedirs('models', exist_ok=True)
        model.save(save_path)
        print(f'Model saved to {save_path}')
