# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from visualize.arkit_visualization import save_scatter_animation
from utils.arkit_utils import blendshape_batch_to_vertices, create_vertices_sequence

def cond_fn(predicted_blendshapes, timesteps, **model_kwargs):
    device = dist_util.dev()
    mean = torch.tensor(model_kwargs['y']['dataset_mean'], device=device).view(1, 61, 1, 1)
    std = torch.tensor(model_kwargs['y']['dataset_std'], device=device).view(1, 61, 1, 1)
    predicted_blendshapes_unnormalized = predicted_blendshapes * std + mean
    predicted_vertices = blendshape_batch_to_vertices(predicted_blendshapes_unnormalized)
    target_vertices_idx = model_kwargs['y']['target_vertices_idx']
    target_vertices_location = model_kwargs['y']['target_vertices_location']

    loss = 0.0
    for b in range(predicted_vertices.shape[0]):
        pred = predicted_vertices[b]  # (V, 3, T)
        idxs = target_vertices_idx[b].to(device)  # (K,)
        target = target_vertices_location[b].to(device).squeeze(0).permute(0,2,1)  # (K, 3, T)
        pred_selected = pred[idxs]  # (K, 3, T)
        print(pred_selected.shape)
        print(target.shape)
        curr_loss = torch.nn.functional.mse_loss(pred_selected, target)
        print(f'b={b}, loss={curr_loss}')
        loss += curr_loss

    loss = loss / predicted_vertices.shape[0]

    print(f'batch_loss={loss}')
    return loss

def main():
    args = edit_args()
    fixseed(args.seed)
    out_path = args.output_dir
    max_frames = 196
    dist_util.setup_dist(args.device)
    args.edit_mode = 'control_vertex_traj_guidance_fixed'
    folder_prefix = args.model_path.split('/')[-2]
    timestep = args.model_path.split('/')[-1][:-3]
    save_folder = os.path.join(out_path, args.edit_mode, folder_prefix, timestep) #'_'.join(args.model_path.split('/')[-2:])[:-3])

    args.save_folder_ext ='drag_multi_right'# 'cfg values'
    if args.save_folder_ext != '':
        save_folder = save_folder + "_" + args.save_folder_ext
    # if not os.path.isdir(save_folder):
    os.makedirs(save_folder, exist_ok=True)

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset,
                        batch_size=args.batch_size,
                        num_frames=max_frames,
                        split='test',
                        hml_mode='generator',
                        data_mode=args.data_mode, #data_mode doesn't really matter because the data is thrown away. here just for good measure
                        normalize_data=not args.normalize_data_off, minimum_frames=args.minimum_frames, 
                        add_velocities=args.add_velocities, add_landmarks_diffs=args.add_landmarks_diffs) 

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    fixseed(args.seed) #model creation harms seeding. reseed

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'model_avg' in state_dict:
        print('loading only model_avg')
        state_dict, state_dict_avg = state_dict['model'], state_dict[
            'model_avg']
        load_model_wo_clip(model, state_dict_avg)
    else:
        load_model_wo_clip(model, state_dict)

    model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking


    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())
    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
    model_kwargs['y']['recon_steps'] = list(range(0,1000))
    model_kwargs['y']['dataset_std'] = data.dataset.std
    model_kwargs['y']['dataset_mean'] = data.dataset.mean
    
    nose_indices = (list(range(515, 585)) + list(range(688, 695)) + list(range(760, 773 )) + list(range(1489, 1504)) + 
                    list(range(1605, 1628)) + list(range(1630, 1635)) + list(range(1821, 1832)) + list(range(1886, 1889)) + 
                    list(range(2512, 2598)) + list(range(2918, 2940)) + list(range(3586, 3660)) + list(range(3760, 3768)) + 
                    list(range(3831, 3844)) + list(range(4560, 4575)) + list(range(4676, 4699)) + list(range(4700, 4706)) + 
                    list(range(4743, 4749)) + list(range(4892, 4904)) + list(range(5583, 5669)) + list(range(7131, 7229)) + 
                    list(range(11033, 11242)) + list(range(17056, 17278)))
    nose_indices = torch.tensor(nose_indices)
    edit_idxs = nose_indices
    model_kwargs['y']['target_vertices_idx'] = [edit_idxs]*input_motions.shape[0]

    input_motions_unnormalized = data.dataset.inv_transform(input_motions)
    
    input_motion_vertices = [create_vertices_sequence(sequence.squeeze().T.cpu().detach().numpy()) for sequence in input_motions_unnormalized]
    input_motion_vertices = torch.stack(input_motion_vertices).permute(0,2,3,1)
    input_target_start = input_motion_vertices[:,edit_idxs,:,0]

    target_location_batch = input_target_start.unsqueeze(2).repeat([1,1,max_frames,1]).to(dist_util.dev())
    target_location_batch = target_location_batch.float()
    
    model_kwargs['y']['tokens'] = None
    model_kwargs['y']['text'] = ['' for _ in model_kwargs['y']['text']]

    motion='drag'
    if motion=='circle':
        radius=1
        angles = torch.linspace(0, 2 * np.pi, steps=max_frames)
        
        x_diff = radius * torch.cos(angles)  # Row 0
        y_diff = torch.zeros(max_frames)     # Row 1
        z_diff = radius * torch.sin(angles)  # Row 2

        # Stack into shape [3, 196]
        circle_tensor = torch.stack([x_diff, y_diff, z_diff], dim=0)  # Shape: [3, 196]
        assert 1==0 # need to change the next line. shape is not good. should be [batch_size, num_of_vertices, max_frames, 3]
        target_diff = circle_tensor.unsqueeze(0).repeat([input_motions.shape[0],1,1]).to(dist_util.dev())
    elif motion=='drag':
        target_diff = torch.tensor([2.5,0,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([input_motions.shape[0], target_location_batch.shape[1], max_frames, 1]).to(dist_util.dev())

    target_location_batch += target_diff
    model_kwargs['y']['target_vertices_location'] = [target_location_single.unsqueeze(0)  for target_location_single in target_location_batch]
    
    for recon_param in [0, 10, 20, 25, 30]:
        model_kwargs['y']['recon_param'] = recon_param

        args.guidance_param = 2.5
        for rep_i in range(args.num_repetitions):
            print(f'### Start sampling [repetitions #{rep_i}]')
            # add CFG scale to batch
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
                recon_guidance=True,
                cond_fn=cond_fn,
            )

            if type(sample) == list:
                pass
            else:
                dump_steps = [999]
                sample = [sample]

            for step, smpl in zip(dump_steps, sample):
                smpl = data.dataset.inv_transform(smpl)
                input_motions_inv = data.dataset.inv_transform(input_motions)
                #visualize
                for i, (cur_smpl, inpt) in enumerate(zip(smpl, input_motions_inv)):
                    sample_length = model_kwargs['y']['lengths'][i]
                    cur_smpl_inv = cur_smpl.permute((2,0,1)).squeeze()[:sample_length]

                    path = os.path.join(save_folder, f'rep_{rep_i:02}_sample_{i:05}_recon_param_{recon_param}.mp4')
                    
                    target_trajectory = model_kwargs['y']['target_vertices_location'][i]
                    save_scatter_animation(cur_smpl_inv, path, fps=args.fps, title=model_kwargs['y']['text'][i], 
                                            inpainting_idxs=model_kwargs['y']['target_vertices_idx'][i], target_trajectory=target_trajectory, valid_objects=['head_lod0'])

                    torch.save(cur_smpl_inv, path.replace('.mp4', '.pt'))
                    print(f"saved - path={path}, {model_kwargs['y']['text'][i]}")
                

if __name__ == "__main__":
    main()

