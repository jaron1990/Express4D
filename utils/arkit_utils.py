import os
import trimesh
import torch
import numpy as np
import pandas as pd
import random
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
from utils import dist_util#, landmarks


blendshapes = ['eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeSquintLeft',
               'eyeWideLeft', 'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight',
               'eyeSquintRight', 'eyeWideRight', 'jawForward', 'jawRight', 'jawLeft', 'jawOpen', 'mouthClose', 'mouthFunnel',
               'mouthPucker', 'mouthRight', 'mouthLeft', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
               'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper',
               'mouthShrugLower', 'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft',
               'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft', 'browDownRight', 'browInnerUp',
               'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft',
               'noseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch', 'HeadRoll', 'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw',
               'RightEyePitch', 'RightEyeRoll']

blendshapes_indices = {'eyeLeft': range(blendshapes.index('eyeLookDownLeft'),blendshapes.index('eyeLookUpLeft')+1),
                       'eyeRight': range(blendshapes.index('eyeLookDownRight'),blendshapes.index('eyeLookUpRight')+1),
                       'head_lod0': range(0,blendshapes.index('noseSneerRight')+1),
                       'teeth': range(blendshapes.index('jawForward'),blendshapes.index('mouthClose')+1),
                       'head_rotation': range(blendshapes.index('HeadYaw'),blendshapes.index('HeadRoll')+1),
                       'left_eye_rotation': range(blendshapes.index('LeftEyeYaw'),blendshapes.index('LeftEyeRoll')+1),
                       'right_eye_rotation': range(blendshapes.index('RightEyeYaw'),blendshapes.index('RightEyeRoll')+1),
                       }

head_origin = np.array([0, 3, 149])

objects = ['eyeLeft', 'eyeRight', 'teeth', 'head_lod0']

all_meshes_diffs = {}
meshes_base = {}
for i, ob in enumerate(objects):
    mesh_base = trimesh.load(os.path.join('obj_exports', f'{ob}_ORIGINAL_Basis.obj'))
    meshes_base[ob] = mesh_base
    base_vertices = torch.tensor(mesh_base.vertices)
    all_meshes_diffs[ob] = {}
    
    for bs in blendshapes:
        try:
            mesh = trimesh.load(os.path.join('obj_exports', f'{ob}_ORIGINAL_{bs}_1.0.obj'))
            all_meshes_diffs[ob][bs] = torch.tensor(mesh.vertices) - base_vertices
        except:
            continue

landmarks_mesh_face_idxs = torch.tensor([18014, 8905, 12218, 17025, 17054, 19170, 19054, 18862, 42898, 42709, 43160, 41114, 
                                    41050, 40990, 32928, 32907, 42025 , 15108, 15065, 15026, 15034, 15009, 39044, 39036, 
                                    39026, 39066, 39108	, 13236, 5150, 19437, 21477, 20486, 4225, 11730, 35598, 28214,
                                      9686, 6170, 6059, 6006, 5668, 5612, 30342, 30115, 30170, 33691, 29596, 29671, 3876, 
                                      32, 168, 24216, 27966, 27766, 27874, 24051, 24043, 24114, 110, 52, 124, 1178, 25304, 
                                      25121, 24424, 26231, 26361, 2298, 16955, 40954])

mesh = trimesh.load('obj_exports/head_lod0_ORIGINAL_Basis.obj')
landmarks_vertices = mesh.faces[landmarks_mesh_face_idxs][:,0]

def euler_zxy_to_matrix(zxy, degrees=True):
    """
    zxy: Tensor of shape (T, 3) – rotation angles in degrees (z, x, y)
    returns: Tensor of shape (T, 3, 3) – rotation matrices
    """
    if degrees:
        zxy = torch.deg2rad(zxy)

    z, x, y = zxy[:, 0], zxy[:, 1], zxy[:, 2]

    cz, sz = torch.cos(z), torch.sin(z)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)

    # Rotation matrices for each axis
    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
        torch.stack([sz,  cz, torch.zeros_like(cz)], dim=-1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1)
    ], dim=-2)  # (T, 3, 3)

    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
        torch.stack([torch.zeros_like(cx), sx,  cx], dim=-1)
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1)
    ], dim=-2)

    return Rz @ Rx @ Ry  # (T, 3, 3)

def get_mesh_faces(object_name = 'head_lod0'):
    assert object_name in objects
    
    mesh_base = trimesh.load(os.path.join('obj_exports', f'{object_name}_ORIGINAL_Basis.obj'))
    mesh_faces = np.array(mesh_base.faces)
    return mesh_faces

def create_blendshapes_recursive(path):
    for root, dirs, files in os.walk(path):
        csv_files = [fl for fl in files if '.csv' in fl]
        for fl in csv_files:
            new_filename = os.path.join(root, fl.replace('.csv', '.npy'))
            if os.path.exists(new_filename):
                print(f'{new_filename} already exists')
                continue
            vec = livelink_csv_to_sequence(os.path.join(root,fl))
            np.save(new_filename, vec)

#read livelink csv file and return tensor of shape [length, blendshapes] 
def livelink_csv_to_sequence(path): 
    # Load the CSV
    df = pd.read_csv(path)
    df = df.iloc[:, 2:]  # Skip the first two columns

    # Convert the DataFrame to a numpy array
    vec = np.array(df.values, dtype=np.float32)
    return vec

def tensor_to_livelink(tensor, out_path):
    ref_df = pd.read_csv('dataset/Express4D/data/MySlate_35_iPhone_cal.csv')
    tensor_df = pd.DataFrame(tensor.cpu().numpy())
    ref_df.iloc[:tensor.shape[0], 2:63] = tensor_df.values
    ref_df = ref_df.iloc[:tensor.shape[0]]
    ref_df.to_csv(out_path, index=False)


# single frame. takes [bsps] and returns [vertices,3]
def create_new_vertices_from_blendshapes(blendshapes, obj = 'head_lod0'):
    new_vertices = torch.tensor(meshes_base[obj].vertices)
    # if obj=='head_lod0':
    if obj=='eyeLeft':
        eye_rotation = blendshapes[blendshapes_indices['left_eye_rotation']]
    elif obj=='eyeRight':
        eye_rotation = blendshapes[blendshapes_indices['right_eye_rotation']]
    
    #TODO: apply eye rotation!

    rotation = blendshapes[blendshapes_indices['head_rotation']]
    rotation[0]*=-1
    rotation[1]*=-1

    # rotation = 
    # rotation[2] *= 90/1.5
    rotation*=(90*0.8)
    # else:
    #     rotation=[0,0,0]
        
    valid_blendshape_values = blendshapes[blendshapes_indices[obj]]
    for bs, bs_value in zip(all_meshes_diffs[obj].keys(), valid_blendshape_values):
        new_vertices += all_meshes_diffs[obj][bs] * bs_value
    
    
    r = R.from_euler('zxy', rotation, degrees=True) #order is zyx because the data is yaw->pitch->roll instead of roll->pitch->yaw
    mask = new_vertices[:, 2] > head_origin[2]
    new_vertices = new_vertices - head_origin
    new_vertices[mask] = torch.tensor(r.apply(new_vertices[mask]))
    new_vertices = new_vertices + head_origin
    new_vertices[:, 0] = -new_vertices[:, 0]
    return new_vertices

# full sequence. takes [t, bsps] and returns [t, vertices, 3]
def create_vertices_sequence(blendshapes, obj = 'head_lod0'):
    vertices = []
    blendshapes_local = blendshapes.copy()
    for bs in blendshapes_local:
        vertices.append(create_new_vertices_from_blendshapes(bs, obj))
    vertices = torch.stack(vertices)
    return vertices

def split_train_test(dataset_dir = 'dataset/Express4D', train_part=0.8):
    with open(os.path.join(dataset_dir, "texts_plain.txt"), "r") as f:
        captions = f.readlines()
    fl_list = os.listdir(os.path.join(dataset_dir, 'data'))
    fl_list = [fl.replace('.csv','') for fl in fl_list if '.csv' in fl]
    length = len(fl_list)
    cutoff = int(length*train_part)
    random.shuffle(fl_list)
    train_data = fl_list[:cutoff]
    test_data = fl_list[cutoff:]
    with open(os.path.join(dataset_dir, "test.txt"), "w") as f:
        f.write('\n'.join(test_data))
    with open(os.path.join(dataset_dir, "test_texts.txt"), "w") as f:
        for item in test_data:
            idx = int(item.split('_')[1])
            f.write(captions[idx-1]) #as the indices in the file lists are 1-based
    with open(os.path.join(dataset_dir, "train.txt"), "w") as f:
        f.write('\n'.join(train_data))
    with open(os.path.join(dataset_dir, "train_texts.txt"), "w") as f:
        for item in train_data:
            idx = int(item.split('_')[1])
            f.write(captions[idx-1]) #as the indices in the file lists are 1-based



def create_text_files_from_single_file(orig_text_file = 'dataset/Express4D/texts.txt', output_dir = 'dataset/Express4D/texts'):
    with open(orig_text_file, 'r') as file:
        for i, line in enumerate(file):
            new_name = f'MySlate_{(i+1):0d}_iPhone_cal.txt'
            with open(f'{output_dir}/{new_name}', 'w') as file:
                file.write(line.replace('.\n',''))

def convert_pt_dir_to_csv_recursively(root_dir, upscale=-1):
    for root, dirs, files in os.walk(root_dir):
        for fl in files:
            if '.pt' in fl:
                tensor = torch.tensor(torch.load(os.path.join(root, fl)))
                if upscale!=-1:
                    x,y = tensor.shape
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                    upscaled_tensor = F.interpolate(tensor, size=(x*upscale, y), mode='bilinear', align_corners=False)
                    tensor = upscaled_tensor.squeeze()
                tensor_to_livelink(tensor, os.path.join(root, fl.replace('.pt','.csv')))

def blendshape_batch_to_vertices(blendshapes_tensor):
    head_meshes_diffs = all_meshes_diffs['head_lod0']
    device = dist_util.dev()
    base_vertices = torch.tensor(meshes_base['head_lod0'].vertices, device=device, dtype=torch.float32)

    blendshape_deltas = torch.stack([
        torch.tensor(head_meshes_diffs[bs], device=device, dtype=torch.float32)
        for bs in head_meshes_diffs
    ])  # shape: (S, V, 3)

    B, S, _, T = blendshapes_tensor.shape
    blendshapes_tensor = blendshapes_tensor.squeeze(2)  # (B, S, T)

    outputs = []
    for b in range(B):
        # ---- linear deformation
        weights = blendshapes_tensor[b, blendshapes_indices['head_lod0']]  # (S, T)
        deformation = (weights[:, None, None, :] * blendshape_deltas[:, :, :, None]).sum(dim=0)  # (V, 3, T)
        verts = base_vertices[:, :, None] + deformation  # (V, 3, T)

        # Get head rotation blendshapes: (3, T)
        rotation_weights = blendshapes_tensor[b, blendshapes_indices['head_rotation']]  # (3, T)
        rotation_weights = rotation_weights.T  # (T, 3)

        # Apply scaling and flip
        rotation_weights[:, 0] *= -1
        rotation_weights[:, 1] *= -1
        rotation_weights *= (90 * 0.8)

        # Compute rotation matrices: (T, 3, 3)
        rot_mats = euler_zxy_to_matrix(rotation_weights)  # (T, 3, 3)

        # Reshape verts: (V, 3, T) → (T, V, 3)
        verts_TVT = verts.permute(2, 0, 1)  # (T, V, 3)

        # Apply rotation using bmm
        rotated_TVT = torch.bmm(verts_TVT, rot_mats)  # (T, V, 3)

        # Reshape back: (B, V, 3, T)
        rotated = rotated_TVT.permute(1, 2, 0)  # (V, 3, T)
        outputs.append(rotated)

        
    outputs = torch.stack(outputs)
    return outputs  # (B, V, 3, T)

if __name__ == '__main__':
    # create_blendshapes_recursive('/home/dcor/yaronaloni/Express4D/dataset/Express4D/data')
    # create_text_files_from_single_file()
    # split_train_test()
    # print('finished')


    # # histogram for length
    # # Directory containing the .npy files
    # directory = "dataset/Express4D/data"

    # # List to store the first-dimension values from all files
    # first_dim_sizes = []

    # # Iterate through the directory and process .npy files
    # for file in os.listdir(directory):
    #     if file.endswith(".npy"):
    #         filepath = os.path.join(directory, file)
    #         array = np.load(filepath)  # Load the array
    #         if array.ndim == 2 and array.shape[1] == 61:  # Ensure correct shape
    #             first_dim_sizes.append(array.shape[0])  # Collect the first dimension size

    # plt.hist(first_dim_sizes, bins=50, color='green', alpha=0.7)
    # plt.title("Histogram of First-Dimension Sizes")
    # plt.xlabel("Size of First Dimension (x)")
    # plt.ylabel("Frequency")
    # plt.grid(True)
    # plt.savefig('dataset_lengths_hist.png')


    # save_scatter_animation(tensor, video_path, ['eyeLeft', 'eyeRight'], title='try')


    # rt = '/home/dcor/yaronaloni/Express4D/save/20250105_train_arkit'
    # for root, dirs, files in os.walk(rt):
    #     for fl in files:
    #         if 'sample' in fl and '.pt' in fl:
    #             tensor_path = os.path.join(root, fl)
    #             video_path = tensor_path.replace('.pt', '.mp4')
    #             if os.path.exists(video_path):
    #                 print(f'{video_path} already exist')
    #                 continue
    #             tensor = torch.load(tensor_path)
    #             save_scatter_animation(tensor, video_path)

    # tensor1 = tensor1[::8]
    # tensor1 = tensor1[:60]
    # for i in range(tensor1.shape[0]):
    #     tensor1[i] = tensor1[0]
    # tensor1[:,-9:-6]=0
    # tensor1[:20,-9] = torch.linspace(-45, 45, 20)
    # tensor1[20:40,-8] = torch.linspace(-45, 45, 20)
    # tensor1[40:60,-7] = torch.linspace(-45, 45, 20)
    # vertices = create_vertices_sequence(tensor1.to('cpu'))
    # save_scatter_animation(tensor1, 'test_real.gif', valid_objects=['head_lod0'])

    # tensor2 = torch.load('save/20250105_train_arkit/step_000046000_samples/sample_4.pt')
    # two_sequences_on_one_animation(tensor1, tensor2, 'save/20250105_train_arkit/step_000046000_samples/1vs4.mp4')

    # get_mesh_faces()

    convert_pt_dir_to_csv_recursively('temporary', upscale=3)

    # visualize_single_frame(torch.tensor(list(blendshape_values.values()), dtype=torch.float32), 'try.png')


