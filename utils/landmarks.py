'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''

import sys
import pickle
import numpy as np

class landmarks_locations:
    r_jaw       = range(0,8)
    center_jaw  = range(8,9)
    l_jaw       = range(9,17)
    r_eyebrow   = range(17,22)
    l_eyebrow   = range(22,27)
    nose_bridge = range(27,31)
    r_nosetip   = range(31,33)
    mid_nosetip = range(33,34)
    l_nosetip   = range(34,36)
    r_eye       = range(36,42)
    l_eye       = range(42,48)
    out_up_lip  = range(48,55) #from left corner to right corner
    out_low_lip = range(55,60)
    in_up_lip   = range(60,65)
    in_low_lip  = range(65,68)
    r_cheek     = range(68,69)
    l_cheek     = range(69,70)

    jaw         = range(r_jaw.start, l_jaw.stop)
    eyebrows    = range(r_eyebrow.start, l_eyebrow.stop)
    nose        = range(nose_bridge.start, l_nosetip.stop)
    eyes        = range(r_eye.start, l_eye.stop)
    lips        = range(out_up_lip.start, in_low_lip.stop)
    cheeks      = range(r_cheek.start, l_cheek.stop)
    
    jaw_flatten         = range(jaw.start*3, jaw.stop*3)
    eyebrows_flatten    = range(eyebrows.start*3, eyebrows.stop*3)
    nose_flatten        = range(nose.start*3, nose.stop*3)
    eyes_flatten        = range(eyes.start*3, eyes.stop*3)
    lips_flatten        = range(lips.start*3, lips.stop*3)
    cheeks_flatten      = range(cheeks.start*3, cheeks.stop*3)
    
    out_up_lips_flatten = range(out_up_lip.start*3, out_up_lip.stop*3)


def load_binary_pickle(filepath):
    with open(filepath, 'rb') as f:
        if sys.version_info >= (3, 0):
            data = pickle.load(f, encoding='latin1')
        else:
            data = pickle.load(f)
    return data

def load_embedding( file_path ):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = load_binary_pickle( file_path )
    lmk_face_idx = lmk_indexes_dict[ 'lmk_face_idx' ].astype( np.uint32 )
    lmk_b_coords = lmk_indexes_dict[ 'lmk_b_coords' ]
    return lmk_face_idx, lmk_b_coords

def load_picked_points(filename):
    """
    Load a picked points file (.pp) containing 3D points exported from MeshLab.
    Returns a Numpy array of size Nx3
    """

    f = open(filename, 'r')

    def get_num(string):
        pos1 = string.find('\"')
        pos2 = string.find('\"', pos1 + 1)
        return float(string[pos1 + 1:pos2])

    def get_point(str_array):
        if 'x=' in str_array[0] and 'y=' in str_array[1] and 'z=' in str_array[2]:
            return [get_num(str_array[0]), get_num(str_array[1]), get_num(str_array[2])]
        else:
            return []

    pickedPoints = []
    for line in f:
        if 'point' in line:
            str = line.split()
            if len(str) < 4:
                continue
            ix = [i for i, s in enumerate(str) if 'x=' in s][0]
            iy = [i for i, s in enumerate(str) if 'y=' in s][0]
            iz = [i for i, s in enumerate(str) if 'z=' in s][0]
            pickedPoints.append(get_point([str[ix], str[iy], str[iz]]))
    f.close()
    return np.array(pickedPoints)