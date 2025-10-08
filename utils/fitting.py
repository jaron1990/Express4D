import numpy as np
import pickle
from utils.indices import BlendShapeIDX_3d_angles
import trimesh
import torch
from utils import dist_util


def get_lmks_from_blendshapes(args, blendshapes, mesh_faces, lmk_face_idx, lmk_b_coords, centralize_blendshapes=False, flame_model = None):
    blendshapes = blendshapes.squeeze(1)
    if flame_model == None:
        flame_model = FLAME(args, batch_size=blendshapes.shape[0]).to(dist_util.dev()).double()
    shape       = torch.nn.Parameter(blendshapes[:, BlendShapeIDX_3d_angles.shape])
    expression  = torch.nn.Parameter(blendshapes[:, BlendShapeIDX_3d_angles.expression])
    rotation    = torch.nn.Parameter(blendshapes[:, BlendShapeIDX_3d_angles.rotation])
    jaw_pose    = torch.nn.Parameter(blendshapes[:, BlendShapeIDX_3d_angles.jaw_pose])
    eyes_pose   = torch.nn.Parameter(blendshapes[:, BlendShapeIDX_3d_angles.eyes_pose])
    neck_pose   = torch.nn.Parameter(blendshapes[:, BlendShapeIDX_3d_angles.neck_pose])
    translation = torch.nn.Parameter(blendshapes[:, BlendShapeIDX_3d_angles.translation])

    model = flame_model(shape_params=shape, expression_params=expression, pose_params=torch.cat([rotation, jaw_pose], dim=1), 
                        eye_pose=eyes_pose, neck_pose=neck_pose, transl=translation)
    # mesh_points = model[0]
    # if centralize_blendshapes:
    #     global_transformation = (mesh_points[0].max(axis=0)[0] + mesh_points[0].min(axis=0)[0])/2
    #     mesh_points = mesh_points - global_transformation
    #     # mesh_points = mesh_points
    # v_selected = mesh_points_by_barycentric_coordinates(mesh_points, mesh_faces, lmk_face_idx, lmk_b_coords)
    return model[1]



def mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):

    v_selected = torch.stack([(mesh_verts[:, mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=2),
                    (mesh_verts[:, mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=2),
                    (mesh_verts[:, mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=2)], dim=2)
    return v_selected

def blendshape_to_trimesh(blendshape, flame_model, target_faces):
    best_model = flame_model(shape_params=blendshape[:, BlendShapeIDX_3d_angles.shape], expression_params=blendshape[:, BlendShapeIDX_3d_angles.expression], pose_params=blendshape[:, BlendShapeIDX_3d_angles.FLAME_pose], 
                        eye_pose=blendshape[:, BlendShapeIDX_3d_angles.eyes_pose], neck_pose=blendshape[:, BlendShapeIDX_3d_angles.neck_pose], transl=blendshape[:, BlendShapeIDX_3d_angles.translation])

    mesh = trimesh.Trimesh(best_model[0][0].to('cpu').detach(), target_faces)
    return mesh

def load_binary_pickle( filepath ):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return 

def get_flame_faces():
    return np.load('./dataset/flame_mesh_faces.npy')

def load_embedding( file_path ):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = load_binary_pickle( file_path )
    lmk_face_idx = lmk_indexes_dict[ 'lmk_face_idx' ].astype( np.uint32 )
    lmk_b_coords = lmk_indexes_dict[ 'lmk_b_coords' ]
    return lmk_face_idx, lmk_b_coords


def landmark_error_3d( mesh_verts, mesh_faces, lmk_3d, lmk_face_idx, lmk_b_coords, weight=1.0 ):
    """ function: 3d landmark error objective
    """

    # select corresponding vertices
    v_selected = mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords )
    lmk_num  = lmk_face_idx.shape[0]

    # an index to select which landmark to use
    lmk_selection = np.arange(0,lmk_num).ravel() # use all

    # residual vectors
    lmk3d_obj = weight * ( v_selected[lmk_selection] - lmk_3d[lmk_selection] )

    return lmk3d_obj