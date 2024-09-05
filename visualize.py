import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['Intra', 'Cross'], default='Intra', required=True)
parser.add_argument('--label', choices=['rest', 'task_motor', 'task_story_math', 'task_working_memory'], default='rest')
parser.add_argument('--index', type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()

    with h5py.File(f'data/{args.data}/processed.h5', 'r') as file:
        X_train = file['X_train'][:]
        y_train = file['y_train'][:]

    label_dict = {
        0: "rest",
        1: "task_motor",
        2: "task_story_math",
        3: "task_working_memory"
    }

    num_label = list(label_dict.keys())[list(label_dict.values()).index(args.label)]
    obs_idx = np.where(y_train==num_label)[0][args.index]

    num_timesteps = X_train.shape[2]
    selected_timesteps = list(range(-1, num_timesteps, 100))  # Get every 100th timestep
    label = label_dict[y_train[obs_idx]]

    global_min = np.min(X_train[:, :, :])
    global_max = np.max(X_train[:, :, :])

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(np.zeros((16, 16)), vmin=global_min, vmax=global_max, cmap='magma')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04) 

    def animate(i):
        # Observation
        obs = X_train[obs_idx, :, selected_timesteps[i]]
        
        # Padding
        pad_size = 16**2 - 248 # Create padding for 248 sensors distributed over a grid size of 16 by 16. 
        padded_obs = np.pad(obs, (0, pad_size), mode='constant', constant_values=0)
        reshaped_obs = padded_obs.reshape(16, 16) # Reshape into 16x16 grid
        
        # Update the image
        cax.set_data(reshaped_obs)
        ax.set_title(f'{label} (t = {selected_timesteps[i] + 1})', fontsize='16')
        return cax,

    # Animation
    ani = FuncAnimation(fig, animate, frames=len(selected_timesteps), interval=500, blit=True)

    # Save GIF
    ani.save(f'visualization/vis_{args.type}_{args.label}_{args.index}.gif', writer='pillow', fps=8)

    print("Finished")

    plt.close()
