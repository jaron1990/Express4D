import logging
import torch
import os
from tqdm import tqdm
import numpy as np

from data_loaders.data_loader_utils import get_data_by_key
from utils.data_loaders_utils import convert_to_6d, smooth_filter, get_identity, process_lmks, process_bsps


logger = logging.getLogger(__name__)


def calculate_mean_std(add_velocities=False, add_landmarks_diffs=False, filter_length=7, debug=False):
    data_dir = 'dataset/Express4D'

    # means = []
    # stds = []
    tensors = []

    # data_mode_for_paths = data_mode if not '_full' in data_mode else data_mode.replace('_full','')
    for root, dirs, files in os.walk(data_dir):
        for fl in files:
            if '.npy' in fl:
                data = np.load(os.path.join(root, fl))
                tensors.append(data)
    all_frames = np.vstack(tensors)

    overall_mean = all_frames.mean(axis=0)
    overall_std = all_frames.std(axis=0)
    return overall_mean, overall_std

if __name__ == "__main__":
    calculate_mean_std('landmarks_70_centralized', filter_length=7)


