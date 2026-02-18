
import json
from operator import itemgetter
import os
from glob import glob
from platform import architecture
from re import A
from typing import Dict, List, Optional, Tuple
import logging
import joblib

import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import random


logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test", "all", "subset"]


def get_split(path: str, split: str, subset: Optional[str] = ''):
    assert split in SPLITS
    filepath = Path(path) / f'{split}{subset}.pth.tar'
    split_data = joblib.load(filepath)
    return split_data


def get_keys(path: str):
    filepath = Path(path) / f'../babel-teach/id2fname/amass-path2babel.json'
    from pandas import read_json
    amass2babel = read_json(filepath)
    return amass2babel


def segments_sorted(segs_fr: List[List], acts: List) -> Tuple[List[List], List]:
    assert len(segs_fr) == len(acts)
    if len(segs_fr) == 1: return segs_fr, acts
    L = [(segs_fr[i], i) for i in range(len(segs_fr))]
    L.sort()
    sorted_segs_fr, permutation = zip(*L)
    sort_acts = [acts[i] for i in permutation]

    return list(sorted_segs_fr), sort_acts


def get_all_seq_durs(duration, max_len, min_len, crop_samples):
    if crop_samples:
        dur1, dur_tr, dur2 = duration
        dur1_t = dur1 + dur_tr if dur1 > min_len else None
        dur2_t = dur2 + dur_tr if dur2 > min_len else None

        durations = {"dur1": dur1 if dur1 > min_len else None,
                     "dur2": dur2 if dur2 > min_len else None,
                     "dur1_t": dur1_t,
                     "dur2_t": dur2_t,
                     "bias_dur1": 0,
                     "bias_dur2": 0,
                     "bias_dur1_t": 0,
                     "bias_dur2_t": 0}

        for dur_key, dur in durations.items():
            if "bias" in dur_key:
                continue
            if dur != None:
                if dur > max_len:
                    durations[f'bias_{dur_key}'] = random.randint(0, dur - max_len)
                    durations[dur_key] = max_len
    else:
        dur1, dur_tr, dur2 = duration
        dur1_t = dur1 + dur_tr if ((dur1 >= min_len) and ((dur1 + dur_tr) <= (max_len + 4))) else None
        dur2_t = dur2 + dur_tr if ((dur2 >= min_len) and ((dur2 + dur_tr) <= (max_len + 4))) else None
        durations = {"dur1": dur1 if ((dur1 >= min_len) and (dur1 <= (max_len + 4))) else None,
                     "dur2": dur2 if ((dur2 >= min_len) and (dur2 <= (max_len + 4))) else None,
                     "dur1_t": dur1_t,
                     "dur2_t": dur2_t,
                     "bias_dur1": 0,
                     "bias_dur2": 0,
                     "bias_dur1_t": 0,
                     "bias_dur2_t": 0}
        for dur_key, dur in durations.items():
            if "bias" in dur_key:
                continue
            if dur != None:
                if dur > max_len:
                    bias = random.randint(dur - max_len, 4)
                    durations[f'bias_{dur_key}'] = bias
                    durations[dur_key] = dur - bias
                else:
                    bias = random.randint(0, 4)
                    durations[f'bias_{dur_key}'] = bias
                    durations[dur_key] = dur - bias

    return durations


class VOCA(Dataset):
    dataname = "VOCA"
    # data_path = '../dataset/COMA_data'
    # dataset -> ID -> Expression -> data
    def __init__(self, datapath: str = '../dataset/COMA_data',
                 split: str = "train",
                 downsample=True,
                 tiny: bool = False,
                 mode: str = 'train',
                 parse_tokens: bool = False,
                 **kwargs):

        self.split = split
        self.parse_tokens = parse_tokens
        self.downsample = downsample

        self.expression = [] # used for text expression label
        self.sequence = [] # seq.


    def _load_datastruct(self, keyid, frame_ix=None):
        features = self.motion_data[keyid][frame_ix]
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        return sequences

    def _load_tokens(self, keyid):
        sequences = self.tokens_data[keyid]
        return sequences

    def _load_actions(self, keyid):
        actions_all = self.action_datas[keyid]
        return actions_all

    def load_keyid(self, keyid, mode='train'):
        num_frames = self._num_frames_in_sequence[keyid]

        text = self._load_text(keyid)
        if self.parse_tokens:
            tokens = self._load_tokens(keyid)

        if mode == 'train':
            frame_ix = self.sampler(num_frames)
            datastruct = self._load_datastruct(keyid, frame_ix)
            element = {'datastruct': datastruct, 'text': text,
                       'length': len(datastruct), 'keyid': keyid}
        else:
            raise ValueError("mdm project - you should never use mode other than train in our scope")
        return element

    def load_seqid(self, seqid):
        segs_keyids = [keyid for keyid in self._split_index if keyid.split('-')[0] == seqid]
        segs_keyids = sorted([(e.split('-')[0], int(e.split('-')[1])) for e in segs_keyids], key=lambda x: x[1])
        segs_keyids = ['-'.join([seq, str(id)]) for seq, id in segs_keyids]
        keyids_to_return = []
        current = segs_keyids[0]
        texts = []
        lens = []
        ov = False
        if len(segs_keyids) == 1:
            t0, t1 = self._load_text(current)
            l0, lt, l1 = self._num_frames_in_sequence[current]
            lens = [l0, l1 + lt]
            texts = [t0, t1]
        else:
            while True:
                t0, t1 = self._load_text(current)
                l0, lt, l1 = self._num_frames_in_sequence[current]
                if not ov:
                    texts.append(t0)
                    texts.append(t1)
                    l1t = lt + l1
                    lens.append(l0)
                    lens.append(l1t)
                else:
                    texts.append(t1)
                    l1t = lt + l1
                    lens.append(l1t)
                if current == segs_keyids[-1]:
                    break
                candidate_next = [i for i in segs_keyids[(segs_keyids.index(current) + 1):] if
                                  self._load_text(i)[0] == t1]

                if candidate_next:
                    ov = True
                    max_id = np.argmax(np.array([self._num_frames_in_sequence[cn][1] for cn in candidate_next]))
                    next_seg = candidate_next[max_id]
                    current = next_seg
                else:
                    ov = False
                    if current != segs_keyids[-1]:
                        current = segs_keyids[segs_keyids.index(current) + 1]
                    else:
                        continue

        element = {'length': lens,
                   'text': texts}
        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid, mode='train')

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"
