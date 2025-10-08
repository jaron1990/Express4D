# from FLAME import FLAME
from utils.fitting import get_flame_faces, blendshape_to_trimesh, get_lmks_from_blendshapes
from fit_3D_landmarks import fit_single_4d_lmk_sequence
from utils.landmarks import load_embedding
from utils import dist_util
import torch
import os
from visualize.save_animations import save_animation_from_lmks, save_animation_from_meshes

# sample size is [frame_count x lmks_count x 3] for landmarks
# or [frame_count x 1 x blendshape_len] for blendshapes
def visualize_sample(args, sample_lmks, sample_bsps, path, inpainting_mask = None): 
    '''
        inpainting_mask is a torch.tensor of either size [T], or [T, 1, lmks_count] (for partial lmks inpainting)
            inpainting_mask is True for GT and False for generated. None will be regarded as generated
    '''
    if args.data_mode in ['landmarks', 'landmarks_centralized']:
        args.static_landmark_embedding_path = './model/flame_static_embedding.pkl'
    elif '_68' in args.data_mode:
        args.static_landmark_embedding_path = './model/flame_static_embedding_68.pkl'
    elif '_70' in args.data_mode or args.data_mode in ['blendshapes', 'blendshapes_full']:
        args.static_landmark_embedding_path = './model/flame_static_embedding_70.pkl'
    else:
        raise Exception("no relevant landmark file")

    lmk_face_idx, lmk_b_coords = load_embedding(args.static_landmark_embedding_path)
    lmk_b_coords = torch.tensor(lmk_b_coords).to(dist_util.dev())

    if 'blendshapes' == args.data_mode:
        sample_lmks = get_lmks_from_blendshapes(args, sample_bsps, get_flame_faces(), lmk_face_idx, lmk_b_coords)

    if (not args.no_animation_output) or (not args.no_meshes_output):
        target_faces = get_flame_faces()
        flame_model = FLAME(args, batch_size=1).to(args.device)
        if 'blendshapes' == args.data_mode:
            #blendshapes_to_meshes
            out_meshes = [blendshape_to_trimesh(blendshape, flame_model, target_faces) for blendshape in sample_bsps]
        elif 'landmarks' in args.data_mode:
            out_meshes = fit_single_4d_lmk_sequence(args, sample_lmks, lmk_face_idx, lmk_b_coords, freeze_identity=True)
        elif 'blendmarks' in args.data_mode:
            out_meshes = [blendshape_to_trimesh(blendshape, flame_model, target_faces) for blendshape in sample_bsps]
        else: 
            raise Exception("not implemented")

        if not args.no_animation_output:
            out_path_animation = path + '.gif'
            save_animation_from_meshes(out_meshes, out_path_animation)
        if not args.no_meshes_output:
            meshes_output_dir = path + '_ply_files'
            os.makedirs(meshes_output_dir, exist_ok=True)
            for ii, mesh in enumerate(out_meshes):
                mesh.export(os.path.join(meshes_output_dir, f'frame_{ii:03d}.ply'))

    # no need to generate meshes for this
    if not args.no_landmarks_animation:
        save_animation_from_lmks(sample_lmks, path + '_lmks_scatter.gif', inpainting_mask)
