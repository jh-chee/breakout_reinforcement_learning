import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from datetime import datetime


def check_dirs_exist(folder_saved_plots, folder_saved_models):
    run_path = "runs"
    if not os.path.exists(run_path):
        os.mkdir(run_path)

    dt = datetime.now().strftime("%Y%m%d-%H%M%S")

    plot_path = os.path.join(run_path, dt, folder_saved_plots)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    model_path = os.path.join(run_path, dt, folder_saved_models)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return plot_path, model_path


def save_plot(episodes, scores, plot_path):
    plt.plot(episodes, scores, 'b')
    plt.savefig(os.path.join(plot_path, "breakout_dqn.png"))


def preprocess(X, height, width):
    x = np.uint8(resize(rgb2gray(X), (height, width), mode='reflect') * 255)
    return x


def get_init_state(history, s, history_size, height, width):
    for i in range(history_size):
        history[i, :, :] = preprocess(s, height, width)
    return history
