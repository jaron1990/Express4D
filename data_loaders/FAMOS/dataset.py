
import logging
import torch
from torch.utils.data import Dataset
import os
from scipy.stats import norm
from scipy.signal import convolve2d
from tqdm import tqdm

from data_loaders.data_loader_utils import FAMOS_EXPRESSION_LIST, get_data_by_key
from utils.data_loaders_utils import convert_to_6d, smooth_filter, get_identity, process_lmks, process_bsps
from random import randint
from data_loaders.FAMOS.calc_mean_std import calculate_mean_std

logger = logging.getLogger(__name__)




class FAMOS(Dataset):
    def __init__(self, datapath: str = './dataset',
                 split: str = "train",
                 tiny: bool = False,
                 mode: str = 'generator',
                 data_mode: str = 'landmarks',
                 normalize_data = False,
                 minimum_frames=60,
                 classifier_step=15,
                 debug=False,
                 smoothing_filter_length = 7,
                 add_velocities = False,
                 add_landmarks_diffs = False,
                 max_len = 246,
                 **kwargs):
        assert mode in ['classifier', 'train_classifier', 'generator']
        assert split in ['train', 'test']
        assert not add_landmarks_diffs or (data_mode.startswith('landmarks') or data_mode.startswith('blendmarks'))

        datapath = os.path.join(datapath, f'FAMOS_data_train_test_split.json' if debug==False else f'FAMOS_data_train_test_split_debug.json')
        self.epsilon = 1E-10
        self.max_len = max_len
        self.split = split
        self.mode = mode
        self.data_mode = data_mode
        self.expression = []  # used for text expression label
        self.classifier_step=classifier_step
        self.minimum_frames = minimum_frames


        list_of_relevent_paths = get_data_by_key(datapath, split)
        # list_of_relevent_paths = [os.path.join(path,f'{data_mode_for_paths}.pt') for path in list_of_relevent_paths]

        self.expression_list = FAMOS_EXPRESSION_LIST
        self.remove_expression_list = [] #['disgust', 'fear', 'happiness', 'surprise', 'lip_corners_down', 'mouth_side', 'sadness', 'rolling_lips']#, 'disgust', 'anger', 'sadness', 'contempt']
        self.num_actions = len(self.expression_list)

        self.data = []

        self.mean, self.std = calculate_mean_std(self.data_mode, add_velocities=add_velocities, add_landmarks_diffs=add_landmarks_diffs, filter_length=smoothing_filter_length, debug=debug)

        bad_path = []
        jj = 0
        identity = None
        for ii, f in enumerate(tqdm(list_of_relevent_paths, disable=not debug)):
            label = f.split("/")[-1]
            if label in self.remove_expression_list:
                continue
            action = self.expression_list.index(label)
            label = label.replace("_", " ") if "_" in label else label
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
                
                # data_tensor = torch.load(f)  # item is frames x blendshapes
                # if(type(data_tensor) is dict):
                #     data_tensor_lmks = data_tensor['landmarks_70_centralized']
                #     data_tensor_bsps = data_tensor['blendshapes']
                #     data_len = data_tensor_lmks.shape[0]
                # else: 
                #     data_len = data_tensor.shape[0]

            except:
                bad_path.append(f)
                print(f'BAD!!!\n{f}\n')
                continue

            if data_len < minimum_frames:
                print(f"Less Than {minimum_frames}!!")
                continue
            # if data_tensor.shape[0] > 300:
            #     data_tensor = data_tensor[:300, ...]
            if data_len > self.max_len:
                if data_tensor_lmks != None:
                    data_tensor_lmks = data_tensor_lmks[:self.max_len, ...]
                if data_tensor_bsps != None:
                    data_tensor_bsps = data_tensor_bsps[:self.max_len, ...]            

            if data_tensor_bsps != None:
                data_tensor_bsps, identity = process_bsps(data_tensor_bsps, filter_length=smoothing_filter_length)

            if data_tensor_lmks != None:
                data_tensor_lmks = process_lmks(data_tensor_lmks, filter_length=1, add_landmarks_diffs=add_landmarks_diffs).detach()

            if data_mode=='blendshapes':
                data_tensor_final = data_tensor_bsps
            elif data_mode.startswith('landmarks'):
                data_tensor_final = data_tensor_lmks
            elif data_mode.startswith('blendmarks'):
                data_tensor_lmks = data_tensor_lmks.reshape((data_tensor_lmks.shape[0],-1))
                data_tensor_final = torch.cat([data_tensor_bsps, data_tensor_lmks], axis=1).detach()
            else:
                raise NotImplementedError('must be blendshapes, landmarks or blendmarks')
                
            if add_velocities:
                velocities = data_tensor_final[1:] - data_tensor_final[:-1]
                velocities = torch.cat([velocities, velocities[-1].unsqueeze(0)])
                data_tensor_final = torch.cat([data_tensor_final, velocities], dim=1)

            if normalize_data:
                data_tensor_final = (data_tensor_final - self.mean) / (self.std + self.epsilon)

            tmp_data_dict = {"label": label,
                             "inp": data_tensor_final,
                             "length": data_tensor_final.shape[0],
                             "action": action,
                             "file": f}
            
            if identity != None:
                tmp_data_dict['identity'] = identity.detach()

            self.data.append(tmp_data_dict)


    def __getitem__(self, idx):
        raw_data = self.data[idx]
        bias = randint(0, 10) 
        
        if self.mode in ['train_classifier', 'classifier', 'generator'] or self.split != 'test':
            bias=0

        raw_data['inp'] = raw_data['inp'][bias:, ...]
        raw_data['length'] -= bias

        if raw_data['length'] > self.max_len:
            raw_data['inp'] = raw_data['inp'][:self.max_len, ...]
            if not 'full' in self.data_mode:
                raw_data['identity'] = raw_data['identity'][:self.max_len]
            raw_data['length'] = self.max_len

        if self.mode == 'train_classifier':
            raw_data['full_inp'] = raw_data['inp']
            raw_data['inp'] = raw_data['inp'][::self.classifier_step, ...]
        # else:
            # print(6)
        return raw_data #self.data[idx]

    def __len__(self):
        return len(self.data)

    def _num_actions(self):
        return len(self.expression_list)

    def _get_mean(self):
        overall_mean = torch.load(f'./dataset/dataset_means_stds/FAMOS_{self.data_mode}_mean.pkl')
        return overall_mean

    def _get_std(self):
        overall_std = torch.load(f'./dataset/dataset_means_stds/FAMOS_{self.data_mode}_std.pkl')
        return overall_std




