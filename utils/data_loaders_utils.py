import logging
import torch
from torch.utils.data import Dataset
import os
from scipy.stats import norm
from scipy.signal import convolve2d

from utils.indices import BlendShapeIDX_3d_angles, BlendShapeIDX_6d_angles
import utils.rotation_conversions as geometry

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test", "all", "subset"]


def convert_to_6d(x):
    rotation = x[:, BlendShapeIDX_3d_angles.rotation.start: BlendShapeIDX_3d_angles.rotation.stop]
    jaw_pose = x[:, BlendShapeIDX_3d_angles.jaw_pose.start: BlendShapeIDX_3d_angles.jaw_pose.stop]
    eyes_pose_l = x[:, BlendShapeIDX_3d_angles.eyes_pose_l.start: BlendShapeIDX_3d_angles.eyes_pose_l.stop]
    eyes_pose_r = x[:, BlendShapeIDX_3d_angles.eyes_pose_r.start: BlendShapeIDX_3d_angles.eyes_pose_r.stop]
    neck_pose = x[:, BlendShapeIDX_3d_angles.neck_pose.start: BlendShapeIDX_3d_angles.neck_pose.stop]

    rotation_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(rotation))
    jaw_pose_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(jaw_pose))
    eyes_pose_l_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(eyes_pose_l))
    eyes_pose_r_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(eyes_pose_r))
    neck_pose_6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(neck_pose))

    # shape_expression = x[:, BlendShapeIDX_3d_angles.shape.start: BlendShapeIDX_3d_angles.expression.stop]
    expression = x[:, BlendShapeIDX_3d_angles.expression.start: BlendShapeIDX_3d_angles.expression.stop]
    translation = x[:, BlendShapeIDX_3d_angles.translation.start: BlendShapeIDX_3d_angles.translation.stop]
    x_6d = torch.cat([expression, rotation_6d, jaw_pose_6d,
                              eyes_pose_l_6d, eyes_pose_r_6d, neck_pose_6d, translation], dim=1)

    return x_6d

def get_identity(x):
    return x[:, BlendShapeIDX_3d_angles.shape.start : BlendShapeIDX_3d_angles.shape.stop]

def convert_to_3d(x, keep_identity = False):
    shape = x[:, BlendShapeIDX_6d_angles.shape.start: BlendShapeIDX_6d_angles.shape.stop, ...]
    rotation = x[:, BlendShapeIDX_6d_angles.rotation.start: BlendShapeIDX_6d_angles.rotation.stop, ...]
    jaw_pose = x[:, BlendShapeIDX_6d_angles.jaw_pose.start: BlendShapeIDX_6d_angles.jaw_pose.stop, ...]
    eyes_pose_l = x[:, BlendShapeIDX_6d_angles.eyes_pose_l.start: BlendShapeIDX_6d_angles.eyes_pose_l.stop, ...]
    eyes_pose_r = x[:, BlendShapeIDX_6d_angles.eyes_pose_r.start: BlendShapeIDX_6d_angles.eyes_pose_r.stop, ...]
    neck_pose = x[:, BlendShapeIDX_6d_angles.neck_pose.start: BlendShapeIDX_6d_angles.neck_pose.stop, ...]

    rotation_3d = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(rotation.squeeze(2).permute(0, 2, 1))).permute(0,2,1).unsqueeze(2)
    jaw_pose_3d = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(jaw_pose.squeeze(2).permute(0, 2, 1))).permute(0,2,1).unsqueeze(2)
    eyes_pose_l_3d = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(eyes_pose_l.squeeze(2).permute(0, 2, 1))).permute(0,2,1).unsqueeze(2)
    eyes_pose_r_3d = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(eyes_pose_r.squeeze(2).permute(0, 2, 1))).permute(0,2,1).unsqueeze(2)
    neck_pose_3d = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(neck_pose.squeeze(2).permute(0, 2, 1))).permute(0,2,1).unsqueeze(2)

    # shape_expression = x[:, BlendShapeIDX_6d_angles.shape.start: BlendShapeIDX_6d_angles.expression.stop, ...]
    expression = x[:, BlendShapeIDX_6d_angles.expression.start: BlendShapeIDX_6d_angles.expression.stop, ...]
    translation = x[:, BlendShapeIDX_6d_angles.translation.start: BlendShapeIDX_6d_angles.translation.stop, ...]
    if keep_identity:
        x_6d = torch.cat([shape, expression, rotation_3d, jaw_pose_3d,
                            eyes_pose_l_3d, eyes_pose_r_3d, neck_pose_3d, translation], dim=1)

    else:
        x_6d = torch.cat([expression, rotation_3d, jaw_pose_3d,
                              eyes_pose_l_3d, eyes_pose_r_3d, neck_pose_3d, translation], dim=1)

    return x_6d

def smooth_filter(x, filter_type="step", filter_length=7):
    if filter_type == "delta":
        filter_kernel = torch.zeros((filter_length, 1))
        filter_kernel[(filter_length - 1) // 2] = 1
    elif filter_type == "step":
        filter_kernel = torch.ones((filter_length, 1))
    elif filter_type == "PDF":
        x = torch.linspace(-2, 2, filter_length)
        filter_kernel = torch.tensor(norm.pdf(x))
    else:
        raise TypeError("Smooth filter not defined!! \n")

    filter_kernel /= filter_kernel.sum()

    convolution_result = torch.tensor(convolve2d(x.detach(), filter_kernel, mode='same', boundary="symm"))

    return convolution_result

def process_bsps(data_tensor, filter_length=7):
    data_tensor = smooth_filter(data_tensor, filter_length=filter_length)
    identity = get_identity(data_tensor)
    # here need to convert 3d to 6d rotations
    data_tensor = convert_to_6d(data_tensor)
    return data_tensor, identity



def process_lmks(data_tensor, add_landmarks_diffs = False, filter_length=7):
    data_reshaped = data_tensor.reshape(data_tensor.shape[0], -1)
    data_reshaped = smooth_filter(data_reshaped, filter_length=filter_length)
    data_tensor = data_reshaped.reshape(data_tensor.shape)

    if add_landmarks_diffs:
        jaw_right = data_tensor[:,0:8]
        jaw_left = data_tensor[:,9:17]
        jaw_left_reversed = jaw_left[:, torch.arange(jaw_left.size(1) - 1, -1, -1), :]
        jaw_diff = jaw_left_reversed-jaw_right

        eyebrow_right   = data_tensor[:,17:22]
        eyebrow_left = data_tensor[:,22:27]
        eyebrow_left_reversed = eyebrow_left[:, torch.arange(eyebrow_left.size(1) - 1, -1, -1), :]
        eyebrow_diff = eyebrow_left_reversed-eyebrow_right
        
        nose_right      = data_tensor[:,31:33]
        nose_left = data_tensor[:,34:36]
        nose_left_reversed = nose_left[:, torch.arange(nose_left.size(1) - 1, -1, -1), :]
        nose_diff = nose_left_reversed-nose_right
        
        eye_top_right   = data_tensor[:,36:40]
        eye_top_left = data_tensor[:,42:46]
        eye_top_left_reversed = eye_top_left[:, torch.arange(eye_top_left.size(1) - 1, -1, -1), :]
        eye_top_diff = eye_top_left_reversed-eye_top_right
        
        eye_bot_right   = data_tensor[:,40:42]
        eye_bot_left = data_tensor[:,46:48]
        eye_bot_left_reversed = eye_bot_left[:, torch.arange(eye_bot_left.size(1) - 1, -1, -1), :]
        eye_bot_diff = eye_bot_left_reversed-eye_bot_right
        
        mouth_outer_top_right   = data_tensor[:,48:51]
        mouth_outer_top_left = data_tensor[:,52:55]
        mouth_outer_top_left_reversed = mouth_outer_top_left[:, torch.arange(mouth_outer_top_left.size(1) - 1, -1, -1), :]
        mouth_outer_top_diff = mouth_outer_top_left_reversed-mouth_outer_top_right

        mouth_outer_bot_right   = data_tensor[:,55:57]
        mouth_outer_bot_left = data_tensor[:,58:60]
        mouth_outer_bot_left_reversed = mouth_outer_bot_left[:, torch.arange(mouth_outer_bot_left.size(1) - 1, -1, -1), :]
        mouth_outer_bot_diff = mouth_outer_bot_left_reversed-mouth_outer_bot_right

        mouth_inner_top_right   = data_tensor[:,60:62]
        mouth_inner_top_left = data_tensor[:,63:65]
        mouth_inner_top_left_reversed = mouth_inner_top_left[:, torch.arange(mouth_inner_top_left.size(1) - 1, -1, -1), :]
        mouth_inner_top_diff = mouth_inner_top_left_reversed-mouth_inner_top_right

        mouth_inner_bot_right   = data_tensor[:,65].unsqueeze(1)
        mouth_inner_bot_left = data_tensor[:,67].unsqueeze(1)
        mouth_inner_bot_diff = mouth_inner_bot_left-mouth_inner_bot_right

        cheek_right   = data_tensor[:,68].unsqueeze(1)
        cheek_left = data_tensor[:,69].unsqueeze(1)
        cheek_diff = cheek_left-cheek_right

        diffs = torch.cat([jaw_diff, eyebrow_diff, nose_diff, eye_top_diff, eye_bot_diff, mouth_outer_top_diff, 
                            mouth_outer_bot_diff, mouth_inner_top_diff, mouth_inner_bot_diff, cheek_diff], dim=1)
        
        data_tensor = torch.cat([data_tensor, diffs], dim=1)
    return data_tensor
