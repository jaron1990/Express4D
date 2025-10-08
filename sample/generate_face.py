# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from visualize.arkit_visualization import save_scatter_animation
from data_loaders.tensors import collate


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    max_frames = 196
    dist_util.setup_dist(args.device)

    save_folder = os.path.join(out_path, '_'.join(args.model_path.split('/')[-2:])[:-3])
    
    is_using_data = not any([args.input_text, args.text_prompt])
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
        save_folder = os.path.join(save_folder, args.text_prompt.replace(' ','_'))
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
        save_folder = os.path.join(save_folder, args.input_text.split('/')[-1].split('.')[0])
    else:
        save_folder = os.path.join(save_folder, 'data')
        
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

    for rep_i in range(args.num_repetitions):
        if is_using_data:
            iterator = iter(data)
            _, model_kwargs = next(iterator)
        else:
            collate_args = [{'inp': torch.zeros(max_frames), 'tokens': None, 'lengths': max_frames}] * args.num_samples
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
            _, model_kwargs = collate(collate_args)
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
        )

        sample = data.dataset.inv_transform(sample)

        #visualize
        for i, cur_smpl in enumerate(sample):
            sample_length = model_kwargs['y']['lengths'][i]
            cur_smpl_inv = cur_smpl.permute((2,0,1)).squeeze()[:sample_length]

            path = os.path.join(save_folder, f'rep_{rep_i:02}_sample_{i:05}.mp4')
            
            save_scatter_animation(cur_smpl_inv, path, fps=args.fps, title=model_kwargs['y']['text'][i])
            torch.save(cur_smpl_inv, path.replace('.mp4', '.pt'))
            print(f"saved - path={path}, {model_kwargs['y']['text'][i]}")


if __name__ == "__main__":
    main()

