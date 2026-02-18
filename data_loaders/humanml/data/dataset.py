import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
from utils import dist_util
from random import sample

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from utils.misc import get_project_root_path
from utils.arkit_utils import blendshapes

# import spacy

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, max_motions=-1):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.origin_fps = 60
        self.fps = opt.fps
        self.split_file = split_file
        self.downsample_factor = self.origin_fps/opt.fps
        
        assert self.downsample_factor.is_integer()
        self.downsample_factor = int(self.downsample_factor)
        
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 20

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # id_list = id_list[:200]
        if 'train' in split_file and max_motions!=-1:
            random.seed(1)
            shuffled_list = id_list.copy()
            random.shuffle(shuffled_list)
            id_list = shuffled_list[:max_motions]
            id_list_ints = sorted([int(i.split('_')[1]) for i in id_list])
            print(f"sampled {max_motions} samples")
            print(f"indices are: {id_list_ints}")
            

        new_name_list = []
        length_list = []

        _split = os.path.basename(split_file).replace('.txt', '')
        cache_path = os.path.join(self.opt.meta_dir, self.opt.dataset_name + '_' + _split + '.npy')
        if opt.use_cache and os.path.exists(cache_path):
            _cache = np.load(cache_path, allow_pickle=True)[None][0]
            name_list, length_list, data_dict = _cache['name_list'], _cache['length_list'], _cache['data_dict']
        else:
            for name in tqdm(id_list):
                try:
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    motion = motion[::self.downsample_factor]
                    motion = motion[:self.max_motion_length]
                    
                    # #TODO: remove this!!!!!
                    # print("ZEROING ALL BUT ANGLES!!!")
                    # motion[:,:-9]=0

                    if (motion.shape[0]) < min_motion_len:
                        continue
                
                    text_data = []
                    # flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            # f_tag = float(line_split[2])
                            # to_tag = float(line_split[3])
                            # f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            # to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            # if f_tag == 0.0 and to_tag == 0.0:
                                # flag = True
                            text_data.append(text_dict)
                            # else:
                            #     try:
                            #         n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                            #         if (len(n_motion)) < min_motion_len or (len(n_motion) >= 500):
                            #             continue
                            #         new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            #         while new_name in data_dict:
                            #             new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            #         data_dict[new_name] = {'motion': n_motion,
                            #                                'length': len(n_motion),
                            #                                'text':[text_dict]}
                            #         new_name_list.append(new_name)
                            #         length_list.append(len(n_motion))
                            #     except:
                            #         print(line_split)
                            #         print(line_split[2], line_split[3], f_tag, to_tag, name)
                            #         # break

                    # if flag:
                    
                    data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                except:
                    pass

            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            np.save(cache_path, {
                'name_list': name_list,
                'length_list': length_list,
                'data_dict': data_dict})

        self.mean = mean
        self.std = std
        self.std_tensor = torch.tensor(self.std, device=dist_util.dev())
        self.mean_tensor = torch.tensor(self.mean, device=dist_util.dev())
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        # self.reset_max_len(self.max_length)

        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3)
        # rot_data (B, seq_len, (joint_num - 1)*6)
        # local_velocity (B, seq_len, joint_num*3)
        # foot contact (B, seq_len, 4)
        # self.vec_offsets = [4,
        #                     4+(self.opt.joints_num-1)*3,
        #                     4+(self.opt.joints_num-1)*(3+6),
        #                     4+(self.opt.joints_num-1)*(3+6)+self.opt.joints_num*3]
        # self.vec_sizes = [3,6,3,1]
        # self.mat_offsets = [0, 3, 3+6, 3+6+3]
        # if self.opt.dataset_name == 't2m':  # humanml
        #     self.foot_idx = [7, 10, 8, 11]
        # elif self.opt.dataset_name == 'kit':
        #     self.foot_idx = [19, 20, 14, 15]

    # def reset_max_len(self, length):
    #     assert length <= self.max_motion_length
    #     self.pointer = np.searchsorted(self.length_arr, length)
    #     print("Pointer Pointing at %d"%self.pointer)
    #     self.max_length = length

    def flip_relevant_entries(self, motion):
        motion = np.hstack([motion[:,blendshapes.index('eyeBlinkRight'):blendshapes.index('jawForward')], 
                              motion[:,blendshapes.index('eyeBlinkLeft'):blendshapes.index('eyeBlinkRight')],
                              motion[:,blendshapes.index('jawForward')].reshape(-1,1),
                              motion[:,blendshapes.index('jawLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('jawRight')].reshape(-1,1),
                              motion[:,blendshapes.index('jawOpen'):blendshapes.index('mouthRight')],
                              motion[:,blendshapes.index('mouthLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthRight')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthSmileRight')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthSmileLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthFrownRight')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthFrownLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthDimpleRight')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthDimpleLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthStretchRight')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthStretchLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthRollLower'):blendshapes.index('mouthPressLeft')],
                              motion[:,blendshapes.index('mouthPressRight')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthPressLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthLowerDownRight')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthLowerDownLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthUpperUpRight')].reshape(-1,1),
                              motion[:,blendshapes.index('mouthUpperUpLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('browDownRight')].reshape(-1,1),
                              motion[:,blendshapes.index('browDownLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('browInnerUp')].reshape(-1,1),
                              motion[:,blendshapes.index('browOuterUpRight')].reshape(-1,1),
                              motion[:,blendshapes.index('browOuterUpLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('cheekPuff')].reshape(-1,1),
                              motion[:,blendshapes.index('cheekSquintRight')].reshape(-1,1),
                              motion[:,blendshapes.index('cheekSquintLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('noseSneerRight')].reshape(-1,1),
                              motion[:,blendshapes.index('noseSneerLeft')].reshape(-1,1),
                              motion[:,blendshapes.index('TongueOut'):blendshapes.index('LeftEyeYaw')],
                              motion[:,blendshapes.index('RightEyeYaw'):blendshapes.index('RightEyeRoll') + 1],
                              motion[:,blendshapes.index('LeftEyeYaw'):blendshapes.index('RightEyeYaw')]
                              ])
        for bs in blendshapes:
            if bs in ['HeadYaw', 'HeadRoll', 'LeftEyeYaw', 'LeftEyeRoll', 'RightEyeYaw', 'RightEyeRoll']:
                motion[:,blendshapes.index(bs)] *=-1
        return motion

    def inv_transform(self, data):
        std_tensor = self.std_tensor.view(61, 1, 1)
        mean_tensor = self.mean_tensor.view(61, 1, 1)
        return data * std_tensor + mean_tensor

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def _set(self,a , b, inverse=False):
        if inverse: b[:] = a
        else: a[:] = b

    # def map_vec_mat(self, vec_motion, mat_motion, is_mat2vec):
    #     # Convert to mat/vec representation
    #     # vec_motion[196, 263] <-> mat_motion[196, n_joints(22), n_feat(13)]
    #     # root_rot_velocity (B, seq_len, 1)
    #     # root_linear_velocity (B, seq_len, 2)
    #     # root_y (B, seq_len, 1)
    #     # ric_data (B, seq_len, (joint_num - 1)*3)
    #     # rot_data (B, seq_len, (joint_num - 1)*6)
    #     # local_velocity (B, seq_len, joint_num*3)
    #     # foot contact (B, seq_len, 4)

    #     # root
    #     self._set(mat_motion[:, :, 0, :4], vec_motion[:, :, :4], is_mat2vec)
    #     self._set(mat_motion[:, :, 0, 4:4+3], vec_motion[:, :, self.vec_offsets[2]:self.vec_offsets[2]+self.vec_sizes[2]], is_mat2vec)

    #     # all other joints
    #     for joint_i in range(1,self.opt.joints_num):
    #         self._set(mat_motion[:, :, joint_i, self.mat_offsets[0]:self.mat_offsets[0]+self.vec_sizes[0]],
    #                   vec_motion[:, :, self.vec_offsets[0]+self.vec_sizes[0]*(joint_i-1):self.vec_offsets[0]+self.vec_sizes[0]*(joint_i)]
    #                   , is_mat2vec)# ric_data
    #         self._set(mat_motion[:, :, joint_i, self.mat_offsets[1]:self.mat_offsets[1]+self.vec_sizes[1]],
    #                   vec_motion[:, :, self.vec_offsets[1]+self.vec_sizes[1]*(joint_i-1):self.vec_offsets[1]+self.vec_sizes[1]*(joint_i)]
    #                   , is_mat2vec)# rot_data
    #         self._set(mat_motion[:, :, joint_i, self.mat_offsets[2]:self.mat_offsets[2]+self.vec_sizes[2]],
    #                   vec_motion[:, :, self.vec_offsets[2]+self.vec_sizes[2]*(joint_i):self.vec_offsets[2]+self.vec_sizes[2]*(joint_i+1)]
    #                   , is_mat2vec)# local_velocity
    #         if joint_i in self.foot_idx:
    #             _idx = self.foot_idx.index(joint_i)
    #             self._set(mat_motion[:, :, joint_i, self.mat_offsets[3]:self.mat_offsets[3] + self.vec_sizes[3]],
    #                       vec_motion[:, :, self.vec_offsets[3]+self.vec_sizes[3]*(_idx):self.vec_offsets[3]+self.vec_sizes[3]*(_idx+1)],
    #                       is_mat2vec)

    # def vec2mat(self, vec_motion):
    #     if len(vec_motion.shape) == 2:  # dataset mode
    #         n_frames = vec_motion.shape[0]
    #         bs = 1
    #         _input = np.expand_dims(vec_motion, 0)
    #     elif len(vec_motion.shape) == 4:  # external mode
    #         bs, n_feats, _, n_frames = vec_motion.shape
    #         _input = vec_motion.squeeze(2).transpose(0, 2, 1)
    #     else:
    #         ValueError()
    #     mat_motion = np.zeros([bs, n_frames, self.opt.joints_num, 13], dtype=vec_motion.dtype)
    #     self.map_vec_mat(_input, mat_motion, is_mat2vec=False)
    #     if len(vec_motion.shape) == 2:  # dataset mode
    #         mat_motion = mat_motion.squeeze(0)
    #     elif len(vec_motion.shape) == 4:  # external mode
    #         mat_motion = mat_motion.transpose(0, 2, 3, 1)
    #     return mat_motion

    # def mat2vec(self, mat_motion):
    #     if len(mat_motion.shape) == 3:  # dataset mode
    #         n_frames = mat_motion.shape[0]
    #         bs = 1
    #         _input = np.expand_dims(mat_motion, 0)
    #     elif len(mat_motion.shape) == 4:  # external mode
    #         bs, n_feats, _, n_frames = mat_motion.shape
    #         _input = mat_motion.transpose(0, 3, 1, 2)
    #     else:
    #         ValueError()
    #     vec_motion = np.zeros([bs, n_frames, self.opt.dim_pose], dtype=mat_motion.dtype)
    #     self.map_vec_mat(vec_motion, _input, is_mat2vec=True)
    #     if len(mat_motion.shape) == 3:  # dataset mode
    #         vec_motion = vec_motion.squeeze(0)
    #     elif len(mat_motion.shape) == 4:  # external mode
    #         vec_motion = np.expand_dims(vec_motion.transpose(0, 2, 1), 2)
    #     return vec_motion

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        orig_capption = text_data['caption']
        caption= text_data['caption']
        tokens= text_data['tokens']

        #TODO: check if needed in evaluation
        # left-right flip
        if self.opt.flip_face_on:
            flipped=False
            temp_string='fdkjhvbasldh' #in case we have both left and right - we need a temp string
            if random.random()>0.5:
                if 'left/NOUN' in tokens:
                    tokens[tokens.index('left/NOUN')] = 'right/NOUN'
                    caption = caption.replace('left', temp_string)
                if 'right/NOUN' in tokens:
                    tokens[tokens.index('right/NOUN')] = 'left/NOUN'
                    caption = caption.replace('right', 'left')
                if temp_string in tokens:
                    caption = caption.replace(temp_string, 'right')

                flipped=True
                motion = self.flip_relevant_entries(motion)

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        # if self.opt.unit_length < 10:
        #     coin2 = np.random.choice(['single', 'single', 'double'])
        # else:
        #     coin2 = 'single'

        # if coin2 == 'double':
        #     m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        # elif coin2 == 'single':
        #     m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        #TODO: check if needed in evaluation
        if 'train' in self.split_file:
            offset = random.randint(0, 10)
        else:
            offset = 0
        motion = motion[offset:offset+m_length]
        m_length -= offset
        motion_unnormalized = motion

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
            
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        # if self.opt.data_rep == 'mat':
        #     # assert (motion == self.mat2vec(self.vec2mat(motion))).all()  # TEST
        #     motion = self.vec2mat(motion)
        
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

        # visualize_data = True
        # if visualize_data:
        #     from visualize.arkit_visualization import save_scatter_animation
        #     save_scatter_animation(torch.tensor(motion_unnormalized[:m_length]), f'{self.name_list[idx]}.mp4', fps=20, title=f'caption:{caption}\norig:{caption}\nflipped={flipped}')

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


'''For use of training baseline'''
class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx: s_idx + m_length]
        tgt_motion = motion[s_idx: s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([src_motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption':line.strip(), "tokens":tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))


    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len

class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None
        # fixed_length can be set from outside before sampling


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, datapath='dataset/humanml_opt.txt', split="train", **kwargs):
        self.mode = mode
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = get_project_root_path()
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)  # TODO(?) DELETE
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = pjoin(abs_base_path, 'dataset')
        opt.data_rep = kwargs.get('data_rep', 'vec')
        # opt.use_cache = kwargs.get('use_cache', True)
        opt.use_cache = False
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1  # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)