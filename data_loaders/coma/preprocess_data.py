import pandas

data_path = '/home/jaron1990/Express4D/results/COMA_data/FaceTalk_170725_00137_TA/bareteeth'

import torch
from os import listdir
from os.path import isfile, join
from scipy.stats import norm
from scipy.signal import convolve2d
from utils.indices import BlendShapeIDX_3d_angles, BlendShapeIDX_6d_angles



def stack_pkls(data_path):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    pkl_only = [f for f in onlyfiles if "pkl" in f]
    pkl_only.sort()
    for ii, f in enumerate(pkl_only):
        full_path = join(data_path, f)
        face_state = pandas.read_pickle(full_path)
        if ii == 0:
            shape_stacked = face_state.shape
            expression_stacked = face_state.expression
            rotation_stacked = face_state.rotation
            jaw_pose_stacked = face_state.jaw_pose
            eyes_pose_stacked = face_state.eyes_pose
            neck_pose_stacked = face_state.neck_pose
            translation_stacked = face_state.translation
            # pose_stacked = face_state.pose
            all_stacked = torch.concatenate([shape_stacked, expression_stacked, rotation_stacked, jaw_pose_stacked, eyes_pose_stacked, neck_pose_stacked, translation_stacked], dim=1)
        else:
            shape_stacked = torch.concatenate([shape_stacked, face_state.shape])
            expression_stacked = torch.concatenate([expression_stacked, face_state.expression])
            rotation_stacked = torch.concatenate([rotation_stacked, face_state.rotation])
            jaw_pose_stacked = torch.concatenate([jaw_pose_stacked, face_state.jaw_pose])
            eyes_pose_stacked = torch.concatenate([eyes_pose_stacked, face_state.eyes_pose])
            neck_pose_stacked = torch.concatenate([neck_pose_stacked, face_state.neck_pose])
            translation_stacked = torch.concatenate([translation_stacked, face_state.translation])
            # pose_stacked = torch.concatenate([pose_stacked, face_state.pose])
            all_stacked = torch.concatenate([shape_stacked, expression_stacked, rotation_stacked, jaw_pose_stacked, eyes_pose_stacked, neck_pose_stacked, translation_stacked], dim=1)

    label = f.split(".")[0]

    filter_length = 7

    filter_type = "step"
    # filter_type = "PDF"
    
    if filter_type == "delta":
        filter_kernel = torch.zeros((filter_length, 1))
        filter_kernel[(filter_length-1)//2] = 1
    if filter_type == "step":
        filter_kernel = torch.ones((filter_length, 1))
    if filter_type == "PDF":
        x = torch.linspace(-2, 2, filter_length)
        filter_kernel = torch.tensor(norm.pdf(x))

    filter_kernel /= filter_kernel.sum()

    convolution_result = convolve2d(all_stacked.detach(), filter_kernel, mode='same', boundary="symm")

    return {label: convolution_result}

stack_pkls(data_path)