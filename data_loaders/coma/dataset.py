
import logging
import torch
from torch.utils.data import Dataset
import os
from scipy.stats import norm
from scipy.signal import convolve2d
from tqdm import tqdm

from data_loaders.data_loader_utils import COMA_EXPRESSION_LIST, get_data_by_key
from utils.data_loaders_utils import convert_to_6d, smooth_filter
from random import randint

logger = logging.getLogger(__name__)




class COMA(Dataset):
    def __init__(self, datapath: str = './dataset',
                 split: str = "train",
                 tiny: bool = False,
                 mode: str = 'train',
                 data_mode: str = 'landmarks',
                 normalize_data = False,
                 classifier_step=15,
                 minimum_frames=60,
                 **kwargs):
        datapath = os.path.join(datapath, f'COMA_data_train_test_split.json')
        self.split = split
        self.mode = mode
        self.data_mode = data_mode
        self.expression = []  # used for text expression label
        self.classifier_step=classifier_step
        self.minimum_frames = minimum_frames
        list_of_relevent_paths = get_data_by_key(datapath, split)
        list_of_relevent_paths = [os.path.join(path,f'{data_mode}.pt') for path in list_of_relevent_paths]

        self.expression_list = COMA_EXPRESSION_LIST
        self.num_actions = len(self.expression_list)

        self.data = {}

        bad_path = []
        jj = 0
        all_len = []
        for ii, f in enumerate(tqdm(list_of_relevent_paths)):
            label = f.split("/")[-2]
            if label in self.remove_expression_list:
                continue
            action = self.expression_list.index(label)
            label = label.replace("_", " ") if "_" in label else label
            try:
                data_tensor = torch.load(f)  # item is frames x blendshapes
                if(type(data_tensor) is dict):
                    data_tensor_lmks = data_tensor['landmarks_70_centralized']
                    data_tensor_bsps = data_tensor['blendshapes']
                    data_len = data_tensor_lmks.shape[0]
                else: 
                    data_len = data_tensor.shape[0]

            except:
                bad_path.append(f)
                print(f'BAD!!!\n{f}\n')
                continue

            if data_len < minimum_frames:
                print(f"Less Than {minimum_frames}!!")
                continue
            # if data_tensor.shape[0] > 300:
            #     data_tensor = data_tensor[:300, ...]
            if data_len > 246:
                if(type(data_tensor) is dict):
                    data_tensor_lmks = data_tensor_lmks[:246, ...]
                    data_tensor_bsps = data_tensor_bsps[:246, ...]
                else:
                    data_tensor = data_tensor[:246, ...]

            if data_mode == 'blendshapes':
                # here need to convert 3d to 6d rotations
                data_tensor_enhanced = convert_to_6d(data_tensor) #6d

                # here also use smoother
                data_tensor_enhanced = smooth_filter(data_tensor_enhanced)

            elif data_mode in ['landmarks', 'landmarks_68', 'landmarks_70', 'landmarks_centralized', 'landmarks_68_centralized', 'landmarks_70_centralized']:
                data_tensor_enhanced = self.process_lnmks(data_tensor) # TODO[Yaron] check if smoothing is needed
            elif data_mode == 'blendmarks_70_centralized':
                data_tensor_enhanced_bsps = convert_to_6d(data_tensor_bsps) #6d

                # here also use smoother
                data_tensor_enhanced_bsps = smooth_filter(data_tensor_enhanced_bsps)

                data_tensor_enhanced_lmks = self.process_lnmks(data_tensor_lmks)
                data_tensor_enhanced_lmks = data_tensor_enhanced_lmks.reshape((data_tensor_enhanced_lmks.shape[0],-1))


                data_tensor_enhanced = torch.cat([data_tensor_enhanced_bsps, data_tensor_enhanced_lmks], axis=1)

            tmp_data_dict = {"label": label,
                             "inp": data_tensor_enhanced,
                             "length": data_tensor_enhanced.shape[0],
                             "action": action}
            all_len.append(data_tensor_enhanced.shape[0])
            self.data[jj] = tmp_data_dict
            jj += 1
        self.mean = self._get_mean()
        self.std = self._get_std()

        if normalize_data:
            for ii in range(len(self.data)):
                self.data[ii]['inp'] = (self.data[ii]['inp'] - self.mean) / self.std

    def __getitem__(self, idx):
        raw_data = self.data[idx]
        bias = randint(0, self.minimum_frames) #9

        raw_data['inp'] = raw_data['inp'][bias:, ...]
        raw_data['length'] -= bias

        if raw_data['length'] > 245:
            raw_data['inp'] = raw_data['inp'][:245, ...]
            raw_data['length'] = 245

        if self.mode == 'train_classifier':
            raw_data['inp'] = raw_data['inp'][::self.classifier_step, :, :]
        # else:
            # print(6)
        return raw_data #self.data[idx]

    def __len__(self):
        return len(self.data)

    def _num_actions(self):
        return len(self.expression_list)

    def _get_mean(self):
        overall_mean = torch.load(f'./dataset/dataset_means_stds/COMA_{self.data_mode}_mean.pkl')
        return overall_mean

    def _get_std(self):
        overall_std = torch.load(f'./dataset/dataset_means_stds/COMA_{self.data_mode}_std.pkl')
        return overall_std

    def process_lnmks(self, data_tensor):
        return data_tensor


