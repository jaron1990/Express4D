
import logging
import torch
from argparse import Namespace
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from data_loaders.data_loader_utils import get_data_by_key
from utils.arkit_utils import livelink_csv_to_sequence
from random import randint
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.Express4D.calc_mean_std import calculate_mean_std
# from scipy.stats import norm

# from utils.data_loaders_utils import convert_to_6d, smooth_filter, get_identity, process_lmks, process_bsps
# from data_loaders.FAMOS.calc_mean_std import calculate_mean_std

logger = logging.getLogger(__name__)

class Express4D(Dataset):
    def __init__(self, datapath: str = './dataset',
                 split: str = "train",
                #  tiny: bool = False,
                 mode: str = 'generator',
                 data_mode: str = 'arkit',
                #  normalize_data = False,
                 minimum_frames=60,
                #  classifier_step=15,
                 debug=False,
                #  smoothing_filter_length = 7,
                #  add_velocities = False,
                 flip_face_on = False,
                 fps = 20,
                 max_motions = -1,
                 **kwargs):
        assert mode in ['evaluator_train', 'eval', 'train_classifier', 'generator', 'gt']
        assert split in ['train', 'test']
        # assert not add_landmarks_diffs or (data_mode.startswith('landmarks') or data_mode.startswith('blendmarks'))

        datapath = os.path.join(datapath, f'Express4D_data_train_test_split.json' if debug==False else f'Express4D_data_train_test_split_debug.json')
        
        self.opt = Namespace()
        self.opt.data_root = 'dataset/Express4D'
        self.opt.dataset_name = 'express4d'
        self.opt.motion_dir = os.path.join(self.opt.data_root, 'data')
        if data_mode=='arkit_labels':
            self.opt.text_dir = os.path.join(self.opt.data_root, 'labels')
        else:
            self.opt.text_dir = os.path.join(self.opt.data_root, 'texts')
        # self.opt.joints_num = 61
        # self.opt.dim_pose = 263
        # self.opt.unit_length = 4
        self.opt.max_motion_length = 196
        self.opt.max_text_len = 20
        self.opt.max_motions = max_motions
        self.minimum_frames = minimum_frames

        self.opt.meta_dir ='dataset'
        self.opt.data_rep ='hml_vec'
        self.opt.use_cache = kwargs.get('use_cache', False)
        
        self.split_file = os.path.join(self.opt.data_root, f'{split}.txt')

        self.epsilon = 1E-10
        self.split = split
        self.mode = mode
        self.opt.fps = fps
        self.opt.flip_face_on = flip_face_on
        # self.data_mode = data_mode
        # self.expression = []  # used for text expression label
        # self.classifier_step=classifier_step

        self.mean, self.std = calculate_mean_std()

        self.w_vectorizer = WordVectorizer('glove', 'our_vab')
        self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, self.opt.max_motions)
        self.num_actions = 1  # dummy placeholder


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
    
    def inv_transform(self, data):
        return self.t2m_dataset.inv_transform(data)

    # def _get_mean(self):
    #     overall_mean = torch.load(f'./dataset/dataset_means_stds/FAMOS_{self.data_mode}_mean.pkl')
    #     return overall_mean

    # def _get_std(self):
    #     overall_std = torch.load(f'./dataset/dataset_means_stds/FAMOS_{self.data_mode}_std.pkl')
    #     return overall_std




