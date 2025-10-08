# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import csv

from utils.fixseed import fixseed
import os
import numpy as np
import torch
# from utils.parser_util import inject_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
import pandas as pd


# def main():
#     args = inject_args()
#     fixseed(args.seed)
#     enrich_args(args)
#     dist_util.setup_dist(args.device)


#     print('(1) Collecting features with the first text prompt.')
#     dataloader = get_dataloader(args)

#     print("Creating model and diffusion...")
#     model, diffusion = create_model_and_diffusion(args, dataloader)

#     print(f"Loading checkpoints from [{args.model_path}]...")
#     model = load_weights_and_prepare_model(args, model)

#     print("Define sampling function")
#     sample_fn = diffusion.ddim_sample_loop

#     input_motions, model_kwargs = get_batch(args, dataloader)

#     df = get_gen_edit_instructions(args, model_kwargs)

#     if args.input_mode == 'invert':
#         init_noise = torch.randn_like(input_motions)
#         input_motions = ddim_sample(init_noise, model, model_kwargs, sample_fn)
#         init_noise = diffusion.ddim_reverse_sample_loop(model, input_motions, clip_denoised=False,
#                                                         model_kwargs=model_kwargs, progress=True, )['sample']
#     elif args.input_mode == 'gen':
#         init_noise = torch.randn_like(input_motions)
#     else:
#         raise ValueError('Invalid input_mod')

#     print('(1) Collecting features with the first text prompt.')
#     model.pnp_mode = 'get'
#     input_motions = ddim_sample(init_noise, model, model_kwargs, sample_fn)  # store all features in get_dict
#     print('(2) Collecting features with the first text prompt.')
#     collect_features(args, diffusion, model)  # collect relevant features from get_dict to inject_dict

#     print('(3) Generate with injection.')
#     model.pnp_mode = 'inject'
#     # switch text
#     model_kwargs['y']['text'] = df.edit_text.to_list()[:args.num_samples]

#     all_lengths, all_motions, all_text = generate_injected_motions(args, init_noise, model, model_kwargs, sample_fn,
#                                                                    dataloader)

#     ResultsSaver.store_results(input_motions,
#                                all_lengths, all_motions, all_text,
#                                df, args, dataloader, model, model_kwargs)


def enrich_args(args):
    args.maximum_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    args.fps = 12.5 if args.dataset == 'kit' else 20
    args.n_frames = min(args.maximum_frames, int(args.motion_length * args.fps))
    # if args.output_dir == '':
    #     name = os.path.basename(os.path.dirname(args.model_path))
    #     niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    #     args.out_path = os.path.join(os.path.dirname(args.model_path), 'pnp_edit_{}_{}'.format(name, niter),
    #                             get_dir_name(args))
    args.total_num_samples = args.num_samples * args.num_repetitions

def get_dataloader(args):
    # assert args.num_samples <= args.batch_size, \
    #     f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger than default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    # data = get_dataset_loader(name=args.dataset,
    #                         batch_size=args.batch_size,
    #                         num_frames=args.maximum_frames,
    #                         split='test',
    #                         hml_mode='train',
    #                         data_rep=args.data_rep  # in train mode, you get both text and motion.)
    #                         )

    dataloader = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=args.maximum_frames,
                              split='test',
                              hml_mode='generator',
                              data_mode=args.data_mode, #data_mode doesn't really matter because the data is thrown away. here just for good measureד
                              normalize_data=not args.normalize_data_off,
                              debug=args.debug, 
                              smoothing_filter_length=args.smoothing_filter_length, add_velocities=args.add_velocities, 
                              add_landmarks_diffs=args.add_landmarks_diffs) 


    # dataloader = get_dataset_loader(name=args.dataset,
    #                                 batch_size=args.batch_size,
    #                                 num_frames=args.maximum_frames,
    #                                 split='test',
    #                                 hml_mode='train',
    #                                 data_rep=args.data_rep  # in train mode, you get both text and motion.)
    #                                 )
    # dataloader.fixed_length = n_frames
    return dataloader


def get_dir_name(args):
    t_inreval = args.inject_timesteps_interval.split('_')
    t_print = '{}_t{}-{},{}'.format(args.input_mode, t_inreval[0], t_inreval[1], args.inject_timesteps_increment)
    components = args.inject_components.split(',')
    used_components = [e for e in ['sa', 'ff', 'mha'] if e in args.inject_components]
    collected_components = {e: [d for d in components if e in d] for e in used_components}
    printed_components = [''.join([k] + [c.replace(k, '') for c in comps]).replace('_', '') for k, comps in collected_components.items()]
    printed_components_w_params = ['{}{},{}'.format(print_name, getattr(args, f'{name}_layers_interval').replace('_','-'), getattr(args, f'{name}_layers_increment')) for print_name, name in zip(printed_components, used_components)]
    name = '_'.join([t_print] + printed_components_w_params + ['seed{}'.format(args.seed)])
    if args.guidance_param != 1.:
        name += f'_gscale{args.guidance_param}'
    return name

def prepare_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


class ResultsSaver:
    @staticmethod
    def store_results(input_motions,
                      all_lengths, all_motions, all_text,
                      df, args,
                      dataloader, model, model_kwargs):
        prepare_dir(args.out_path)

        input_motions = to_plottable(input_motions, dataloader, model)

        ResultsSaver._store_npy_files(all_lengths, all_motions, all_text, args)
        ResultsSaver._store_mp4_files(all_lengths, all_motions, args, df, input_motions, model_kwargs)
        abs_path = os.path.abspath(args.out_path)
        print(f'[Done] Results are at [{abs_path}]')

    @staticmethod
    def _store_mp4_files(all_lengths, all_motions, args, df, input_motions, model_kwargs):
        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
        for sample_i in range(args.num_samples):
            _input_txt = df.input_text.to_list()[sample_i] if args.text_condition == '' else args.text_condition
            _category = df.category.to_list()[sample_i] if args.text_condition == '' else args.text_condition
            caption = 'Input Motion: {}'.format(_input_txt)
            length = model_kwargs['y']['lengths'][sample_i]
            input_motions[0][0][0][10]
            motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
            save_file = 'input_motion{:02d}.mp4'.format(sample_i)
            animation_save_path = os.path.join(args.out_path, save_file)
            rep_files = [animation_save_path]
            print(f'[({sample_i}) "{caption}" | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption, dataset=args.dataset, fps=args.fps,
                           vis_mode='gt', figsize=(5, 5))
            for rep_i in range(args.num_repetitions):
                _edit_txt = df.edit_text.to_list()[sample_i] if args.edit_text == '' else args.edit_text
                caption = '[Edit][{}]: {}'.format(_category, _edit_txt)
                length = all_lengths[rep_i * args.batch_size + sample_i]
                motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
                animation_save_path = os.path.join(args.out_path, save_file)
                rep_files.append(animation_save_path)
                print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
                plot_3d_motion(animation_save_path, skeleton, motion, title=caption, dataset=args.dataset, fps=args.fps,
                               figsize=(5, 5))
                # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

            all_rep_save_file = ResultsSaver.stack_mp4s(_category, args, rep_files, sample_i)
            for f in rep_files:
                os.remove(f)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    @staticmethod
    def stack_mp4s(_category, args, rep_files, sample_i):
        all_rep_save_file = os.path.join(args.out_path, '{}_sample{:02d}.mp4'.format(_category, sample_i))
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + 1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'

        os.system(ffmpeg_rep_cmd)
        print(ffmpeg_rep_cmd)
        return all_rep_save_file

    @staticmethod
    def _store_npy_files(all_lengths, all_motions, all_text, args):
        npy_path = os.path.join(args.out_path, 'results.npy')
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path,
                {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
                 'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
        with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
            fw.write('\n'.join(all_text))
        with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
            fw.write('\n'.join([str(l) for l in all_lengths]))
        print(f"saving visualizations to [{args.out_path}]...")


def switch_text_in_model_kwargs(args, csv_edit_texts, model_kwargs):
    if args.text_condition != '':
        edit_texts = [args.edit_text] * args.num_samples
        model_kwargs['y']['text'] = edit_texts
    elif args.input_mode == 'gen' or args.input_mode == 'invert':  # from benchmark file
        model_kwargs['y']['text'] = csv_edit_texts[:args.num_samples]
        edit_texts = csv_edit_texts




def generate_injected_motions(args, init_noise, model, model_kwargs, sample_fn, dataloader):
    all_motions = []
    all_lengths = []
    all_text = []
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, args.maximum_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_noise,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        all_text += model_kwargs['y']['text']
        # all_motions.append(to_plottable(dataloader, model, sample))
        all_motions.append(to_plottable(sample, dataloader, model))
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)[:args.total_num_samples] # [bs, njoints, 6, seqlen]
    all_text = all_text[:args.total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:args.total_num_samples]
    return all_lengths, all_motions, all_text

def to_plottable(motions, dataloader, model):
    # this function works on batches of motions
    # [21, 263, 1, 196] --> [21, 22, 3, 196]
    if model.data_rep == 'hml_mat':
        motions = motions.cpu().numpy()
        motions = torch.tensor(dataloader.dataset.t2m_dataset.mat2vec(motions))
    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep in ['hml_vec', 'hml_mat']:
        n_joints = 22 if motions.shape[1] == 263 else 21
        motions = dataloader.dataset.t2m_dataset.inv_transform(motions.cpu().permute(0, 2, 3, 1)).float()
        motions = recover_from_ric(motions, n_joints)
        motions = motions.view(-1, *motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    return motions


def collect_features(args, diffusion, model):
    # Set time steps
    _timestep_interval = [int(float(a) * diffusion.num_timesteps) for a in args.inject_timesteps_interval.split('_')]
    assert len(_timestep_interval) == 2
    _timesteps = list(range(_timestep_interval[0], _timestep_interval[1]))[::args.inject_timesteps_increment]
    print('Injecting [{}] timesteps: {}'.format(len(_timesteps), _timesteps))
    # Set layers
    _layers = {}
    for _feat in ['ff', 'sa', 'mha']:  # fead-forward, self-attention, multi-head-attention
        __interval = [int(a) for a in getattr(args, f'{_feat}_layers_interval').split('_')]
        assert len(_timestep_interval) <= 2
        if len(__interval) == 2:
            __interval = list(range(__interval[0], __interval[1] + 1))[::getattr(args, f'{_feat}_layers_increment')]
        _layers.update({_feat: __interval})
    print(f'Injecting layers:\n{_layers}')
    # Build all features
    used_components = [] if args.inject_components == '' else args.inject_components.split(',')
    _features = {_comonent: {'steps': _timesteps, 'layers': _layers[
        _comonent.replace('_k', '').replace('_q', '').replace('_v', '').replace('_joint', '')]}
                 for _comonent in used_components}
    print(f'Features to be injected:\n{_features}')
    model.inject_dict = {}
    inject_dict_debug_delete_me = {}
    for s in range(diffusion.num_timesteps):
        model.inject_dict[f'step{s:03d}'] = {}
        for l in range(model.num_layers):
            model.inject_dict[f'step{s:03d}'][f'layer{l:02d}'] = {}
            for f_name, f_data in _features.items():
                if l in f_data['layers'] and s in f_data['steps']:
                    model.inject_dict[f'step{s:03d}'][f'layer{l:02d}'][f_name] = model.get_dict[f'step{s:03d}'][f'layer{l:02d}'][f_name].to(dist_util.dev())
                    inject_dict_debug_delete_me[(f'step{s:03d}', f'layer{l:02d}', f_name)] = model.get_dict[f'step{s:03d}'][f'layer{l:02d}'][f_name].to(dist_util.dev())


def ddim_sample(init_noise, model, model_kwargs, sample_fn):
    motion = sample_fn(
        model,
        init_noise.shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=init_noise,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    return motion

def get_gen_edit_instructions(args, model_kwargs) -> pd.DataFrame:
    df = pd.read_csv(args.csv_edit)
    assert {'input_text', 'edit_text', 'category'} <= set(df.columns)
    df.input_text = df.input_text if args.text_condition == '' else [args.text_condition] * args.num_samples
    model_kwargs['y']['text'] = df.input_text.to_list()[:args.num_samples]
    return df



def explain_batch_content():
    # input_motions.shape = [21, 263, 1, 196] = [sample, joints, ?, maximum_frames]
    # model_kwargs = {'mask': [True, True, True, False, False],
    #                'lengths': 3,
    #                'text': 'a man walks in a straight line.',
    #                'tokens': 'sos/OTHER_a/DET_man/NOUN_walk/VERB_in/ADP_a/DET_straight/ADJ_line/NOUN_eos,...}
    return None


def get_batch(args, dataloader):
    explain_batch_content()
    iterator = iter(dataloader)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())
    assert args.maximum_frames == input_motions.shape[-1]

    # add CFG scale to batch
    model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    model_kwargs['y']['lengths'] = torch.ones_like(model_kwargs['y']['lengths']) * args.n_frames  # fixed length
    model_kwargs['y']['mask'] = torch.zeros_like(model_kwargs['y']['mask'])  #
    model_kwargs['y']['mask'][..., :args.n_frames] = True  # full batch
    return input_motions, model_kwargs


def load_weights_and_prepare_model(args, model) -> ClassifierFreeSampleModel:
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'model_avg' in state_dict.keys():
        load_model_wo_clip(model, state_dict)
    else:
        load_model_wo_clip(model, state_dict)
    # if args.guidance_param != 1:
    #     model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    # assert args.guidance_param == 1, 'FIXME: currently supporting no CFG only!'
    # experimental: always use CF sampler
    model = ClassifierFreeSampleModel(model)  
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    return model


# if __name__ == "__main__":
#     main()
