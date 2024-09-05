import argparse
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['Intra', 'Cross'], required=False)
parser.add_argument('--data', choices=['Intra', 'Cross'], required=False)

if __name__ == '__main__':
    args = parser.parse_args()

    model = tf.keras.models.load_model(f'models/RNN_model_{args.model}.keras')

    with h5py.File(f'data/{args.data}/processed.h5', 'r') as f:
        X_test = f['X_test']
        y_test = f['y_test']

        X_test = np.transpose(X_test, axes=(0, 2, 1))

        loss, accuracy = model.evaluate(X_test, y_test)

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_test, y_pred_classes)

        f1_scores = f1_score(y_test, y_pred_classes, average=None)
        f1_average = f1_score(y_test, y_pred_classes, average='macro')

        print(f'Test accuracy: {accuracy}')
        print(f'Test loss: {loss}')
        print(f'F1 Scores: {f1_scores}')
        print(f'Average F1 Score: {f1_average}')
        
        label_names = ['Rest', 't_Motor', 't_Story', 't_Working']
        cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        fig, ax = plt.subplots(figsize=(6,4))
        cmp.plot(ax=ax, cmap=plt.cm.Blues)

        plt.savefig(os.path.join('results', f'cm_RNN_{args.model}_model_{args.data}_data.png'))
        plt.close()
        
        print(f'Saved confusion matrix to results dir')
