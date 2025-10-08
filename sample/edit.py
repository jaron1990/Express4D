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
from utils.arkit_utils import blendshapes
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util#, landmarks
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from visualize.arkit_visualization import save_scatter_animation, two_sequences_on_one_animation
from data_loaders.Express4D.tokenize import process_text

def main():
    args = edit_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196
    fps = 20
    n_frames = max_frames # min(max_frames, int(args.motion_length*fps))
    dist_util.setup_dist(args.device)

    folder_prefix = args.model_path.split('/')[-2]
    timestep = args.model_path.split('/')[-1][:-3]
    save_folder = os.path.join(out_path, args.edit_mode, folder_prefix, timestep) #'_'.join(args.model_path.split('/')[-2:])[:-3])

    args.save_folder_ext ='only_relevant_motions'# 'cfg values'
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

    # data = load_dataset(args, max_frames, n_frames)
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions
    
    # if not args.normalize_data_off:
    #     std = data.dataset.std.flatten().to(dist_util.dev()).to(torch.float32)
    #     mean = data.dataset.mean.flatten().to(dist_util.dev()).to(torch.float32)

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


    gt_frames_per_sample = []
    title = []
    if args.edit_mode == 'in_between':
        model_kwargs['y']['inpainted_motion'] = input_motions.detach()

        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.float,
                                                            device=input_motions.device)  # True means use gt motion
        
        # model_kwargs['y']['inpainting_frames'] = []
        
        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
            gt_frames_per_sample.append(list(range(0, start_idx)) + list(range(end_idx, input_motions.shape[3])))
            ramp = torch.linspace(0, 1, steps=(end_idx-start_idx)//4)
            frames_to_fill = end_idx-start_idx-2*ramp.shape[0]

            down_up = torch.cat([1-ramp, torch.zeros(frames_to_fill), ramp])
            down_up = down_up.unsqueeze(0).unsqueeze(0).repeat(61,1,1)
            model_kwargs['y']['inpainting_mask'][i, :, :,start_idx: end_idx] = down_up  # do inpainting in those frames
            title.append(model_kwargs['y']['text'][i])
            # model_kwargs['y']['inpainting_frames'].append(list(range(start_idx, end_idx)))
    
    elif args.edit_mode == 'mouth_inpainting':
        model_kwargs['y']['inpainted_motion'] = input_motions.detach()
        new_title = args.save_folder_ext
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.float,
                                                            device=input_motions.device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, range(blendshapes.index('mouthClose'), blendshapes.index('mouthUpperUpRight') + 1),:,:] = 0
        title = [f'original: "{orig_text}"\nnew: "{new_title}"' for orig_text in model_kwargs['y']['text']]
        model_kwargs['y']['text'] = [new_title]*len(model_kwargs['y']['text'])
        new_tokens = [process_text(txt).split('#')[1].split(' ') for txt in model_kwargs['y']['text']]

        for i, tokens in enumerate(new_tokens):
            if len(tokens) < data.dataset.opt.max_text_len:
            # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (data.dataset.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:data.dataset.opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            new_tokens[i] = '_'.join(tokens)

    elif args.edit_mode == 'mouth_n_jaw_inpainting':
        model_kwargs['y']['inpainted_motion'] = input_motions.detach()
        new_title = args.save_folder_ext
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.float,
                                                            device=input_motions.device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, range(blendshapes.index('jawForward'), blendshapes.index('mouthUpperUpRight') + 1),:,:] = 0
        title = [f'original: "{orig_text}"\nnew: "{new_title}"' for orig_text in model_kwargs['y']['text']]
        model_kwargs['y']['text'] = [new_title]*len(model_kwargs['y']['text'])

    # for cfg in [1,2.5,5,7]:
    for cfg in [2.5]:
        args.guidance_param = cfg
        for rep_i in range(args.num_repetitions):
            print(f'### Start sampling [repetitions #{rep_i}]')
            # add CFG scale to batch
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            # dump_steps = [0,250,500,750,999]
            if 'inpainted_motion' in model_kwargs['y'].keys():
                sample = sample_fn(
                    model,
                    (args.batch_size, model.njoints, model.nfeats, max_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=model_kwargs['y']['inpainted_motion'],
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
            else:
                sample = sample_fn(
                    model,
                    (args.batch_size, model.njoints, model.nfeats, max_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
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
                    cur_input_inv = inpt.permute((2,0,1)).squeeze()[:sample_length]

                    path = os.path.join(save_folder, f'rep_{rep_i:02}_sample_{i:05}_cfg_{cfg}.mp4')
                    if 'inpainting_mask' in model_kwargs['y'].keys():
                        save_scatter_animation(cur_smpl_inv, path, fps=args.fps, title=title[i], 
                                            inpainting_frames=model_kwargs['y']['inpainting_mask'][i,0,0])
                    else:
                        save_scatter_animation(cur_smpl_inv, path, fps=args.fps, title=title[i])
                    # two_sequences_on_one_animation(cur_smpl_inv, cur_input_inv, path.replace('.mp4', '_with_input.mp4'), 
                    #                             title=title[i], fps=args.fps, 
                    #                             inpainting_frames=model_kwargs['y']['inpainting_mask'][i,0,0])

                    torch.save(cur_smpl_inv, path.replace('.mp4', '.pt'))
                    print(f"saved - path={path}, {title[i]}")


if __name__ == "__main__":
    main()

