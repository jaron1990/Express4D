import logging
import torch
import os
from tqdm import tqdm

from data_loaders.data_loader_utils import get_data_by_key
from utils.data_loaders_utils import convert_to_6d, smooth_filter, get_identity, process_lmks, process_bsps


logger = logging.getLogger(__name__)


def calculate_mean_std(data_mode, add_velocities=False, add_landmarks_diffs=False, filter_length=7, debug=False):
    dataset = 'FAMOS'
    base_dir = 'dataset/'
    """data mode is one of the following:
    landmarks, landmarks_centralized, landmarks_68, landmarks_68_centralized, landmarks_70, landmarks_70_centralized, blendshapes, blendmarks_70_centralized
    """
    # data_mode = 'landmarks_70_centralized'
    # add_velocities = False
    # add_landmarks_diffs = True


    datapath = os.path.join(base_dir, f'FAMOS_data_train_test_split.json' if debug==False else f'FAMOS_data_train_test_split_debug.json')
    # means = []
    # stds = []
    tensors = []

    # data_mode_for_paths = data_mode if not '_full' in data_mode else data_mode.replace('_full','')

    list_of_relevent_paths = get_data_by_key(datapath, "train")
    list_of_relevent_paths += get_data_by_key(datapath, "test")

    for ii, f in enumerate(tqdm(list_of_relevent_paths)):
        try:
            if data_mode == 'blendshapes':
                data_tensor_bsps = torch.load(os.path.join(f, f'{data_mode}.pt'))
                data_tensor_lmks = None
                data_len = data_tensor_bsps.shape[0]
            elif data_mode.startswith('landmarks'):
                data_tensor_bsps = None
                data_tensor_lmks = torch.load(os.path.join(f, f'{data_mode}.pt'))
                data_len = data_tensor_lmks.shape[0]
            elif data_mode.startswith('blendmarks'):
                data_tensor_bsps = torch.load(os.path.join(f, f'blendshapes.pt'))
                data_tensor_lmks = torch.load(os.path.join(f, f'{data_mode.replace("blendmarks", "landmarks")}.pt'))
                assert data_tensor_bsps.shape[0]==data_tensor_lmks.shape[0]
                data_len = data_tensor_bsps.shape[0]

        except:
            print(f'BAD!!!\n{f}\n')
            continue

        if data_len < 60:
            continue

        if data_tensor_bsps != None:
            data_tensor_bsps, _ = process_bsps(data_tensor_bsps, filter_length=filter_length)

        if data_tensor_lmks != None:
            data_tensor_lmks = process_lmks(data_tensor_lmks, filter_length=filter_length, add_landmarks_diffs=add_landmarks_diffs).detach()

        if data_mode=='blendshapes':
            data_tensor_final = data_tensor_bsps
        elif data_mode.startswith('landmarks'):
            data_tensor_final = data_tensor_lmks
        elif data_mode.startswith('blendmarks'):
            data_tensor_lmks = data_tensor_lmks.reshape((data_tensor_lmks.shape[0],-1))
            data_tensor_final = torch.cat([data_tensor_bsps, data_tensor_lmks], axis=1).detach()
        else:
            raise NotImplementedError(f'data_mode(={data_mode}) must be blendshapes, landmarks or blendmarks')

        if add_velocities:
            velocities = data_tensor_final[1:] - data_tensor_final[:-1]
            velocities = torch.cat([velocities, velocities[-1].unsqueeze(0)])
            data_tensor_final = torch.cat([data_tensor_final, velocities], dim=1)

        tensors.append(data_tensor_final)


    tensors = torch.concatenate(tensors, dim=0)
    overall_mean = tensors.mean(dim=0)
    overall_std = tensors.std(dim=0)
    return overall_mean, overall_std

if __name__ == "__main__":
    calculate_mean_std('landmarks_70_centralized', filter_length=7)


