import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

import re
from os.path import join as pjoin
from typing import Optional
from diffusion import logger

from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
# from eval import eval_famos
from data_loaders.get_data import get_dataset_loader
from utils.model_util import load_model_wo_clip
# from visualize.visualize_sample import visualize_sample
from utils.data_loaders_utils import convert_to_3d
from utils.landmarks import load_embedding
from data_loaders.data_loader_utils import FAMOS_EXPRESSION_LIST, COMA_EXPRESSION_LIST
from utils.parse_sample import parse_sample
from visualize.arkit_visualization import save_scatter_animation
from eval import eval_humanml_double_take#, eval_multi


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.model_avg = None
        if self.args.use_ema:
            self.model_avg = copy.deepcopy(self.model)
        self.model_for_eval = self.model_avg if self.args.use_ema else self.model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        if self.args.use_ema:
            self.opt = AdamW(
                # with amp, we don't need to use the mp_trainer's master_params
                (self.model.parameters()
                 if self.use_fp16 else self.mp_trainer.master_params),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
            )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        if args.dataset == 'express4d' and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=args.num_frames, data_mode=args.data_mode, 
                            smoothing_filter_length=args.smoothing_filter_length, add_velocities=args.add_velocities, 
                            add_landmarks_diffs=args.add_landmarks_diffs, max_len=args.maximum_frames, flip_face_on=False, fps=args.fps,
                            split='test', hml_mode='generator')
            
            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=args.num_frames, data_mode=args.data_mode, 
                            smoothing_filter_length=args.smoothing_filter_length, add_velocities=args.add_velocities, 
                            add_landmarks_diffs=args.add_landmarks_diffs, max_len=args.maximum_frames, flip_face_on=False, fps=args.fps,
                            split='test', hml_mode='gt', shuffle=False)
            
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, args.eval_model_name, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml_double_take.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }


        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model_wo_clip(self.model, state_dict)
                load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.use_ema:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict())

    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd

    def run_loop(self):
        print(self.args)
        print('train steps:', self.num_steps)

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data, disable=not self.args.debug):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.visualize()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for k, v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step+self.resume_step, group_name='Loss')

                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def visualize(self):
        if not self.args.visualize_during_training:
            return
        
        visualization_save_dir = os.path.join(self.save_dir, f'step_{(self.step+self.resume_step):09d}_samples')
        if os.path.exists(visualization_save_dir) and len([pt_file for pt_file in os.listdir(visualization_save_dir) if '.pt' in pt_file])==self.args.visualize_num:
            return
        os.makedirs(visualization_save_dir, exist_ok=True)

        
        print('starting visualization...')
        visualize_num=self.args.visualize_num
        visualization_batch_size = min(self.args.batch_size, visualize_num)
        sample_fn = self.diffusion.p_sample_loop
        max_frames = 196

        data = get_dataset_loader(name=self.args.dataset,
                                batch_size=visualization_batch_size,
                                num_frames=max_frames,
                                split='test',
                                hml_mode='generator',
                                data_mode=self.args.data_mode, #data_mode doesn't really matter because the data is thrown away. here just for good measureד
                                normalize_data=not self.args.normalize_data_off, minimum_frames=self.args.minimum_frames, 
                                debug = self.args.debug, smoothing_filter_length=self.args.smoothing_filter_length, 
                                add_velocities=self.args.add_velocities, add_landmarks_diffs=self.args.add_landmarks_diffs) 
        

        iterator = iter(data)
        _, model_kwargs = next(iterator)
        visualized = 0
        if self.dataset == 'express4d':
            fl = open(os.path.join(visualization_save_dir, 'texts.txt'), "w")

        while visualized < visualize_num:
            iterator = iter(data)
            _, model_kwargs = next(iterator)

            max_frames = model_kwargs['y']['lengths'].max()
            n_frames = model_kwargs['y']['lengths'].max()

            # add CFG scale to batch
            if self.args.guidance_param_for_visualization != 1:
                model_kwargs['y']['scale'] = torch.ones(visualization_batch_size, device=dist_util.dev()) * self.args.guidance_param_for_visualization

            sample_fn = self.diffusion.p_sample_loop

            sample = sample_fn(
                self.model,
                (visualization_batch_size, self.model.njoints, self.model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            if self.model.data_rep == 'hml_vec':
                if self.dataset == 'express4d':
                    for i, smpl in enumerate(sample):
                        if visualized >= visualize_num:
                            break

                        smpl_inv = data.dataset.inv_transform(smpl)
                        sample_length = model_kwargs['y']['lengths'][i]
                        cur_smpl_inv = smpl_inv.permute((2,0,1)).squeeze()[:sample_length]
                        path = os.path.join(visualization_save_dir, f'sample_{visualized}.mp4')
                        save_scatter_animation(cur_smpl_inv, path, fps=self.args.fps, title=model_kwargs['y']['text'][i])

                        fl.write(f"sample_{visualized}: {model_kwargs['y']['text'][i]}\n")
                        torch.save(cur_smpl_inv, path.replace('.mp4', '.pt'))
                        print(f"saved - path={path}, sample_{visualized}: {model_kwargs['y']['text'][i]}")
                        visualized += 1

                else:
                    sample = self.dataset

                    if self.args.add_velocities:
                        half_shape = sample.shape[1]//2
                        sample = sample[:,:half_shape]

                    if self.args.add_landmarks_diffs:
                        if not (self.args.data_mode.startswith('blendmarks') or self.args.data_mode.startswith('landmarks')):
                            raise Exception("add landmarks without landmarks to add to")
                        #remove landmarks diffs
                        sample = sample[:,:-30*3]


                    sample_lmks, sample_bsps = parse_sample(self.args, sample, model_kwargs)

                    for i in range(visualization_batch_size):
                        if visualized >= visualize_num:
                            break
                        text = FAMOS_EXPRESSION_LIST[model_kwargs['y']['action'][i]] if self.dataset=='famos' else COMA_EXPRESSION_LIST[model_kwargs['y']['action'][i]]

                        sample_path = os.path.join(visualization_save_dir, f'sample_{visualized}_{("uncond" if self.args.cond_mode == "uncond" else text)}.pt')
                        torch.save(sample[:,i].squeeze(), sample_path)
                        if (sample_lmks is not None) and len(sample_lmks.shape)==3: #no batch
                            sample_lmks = sample_lmks.unsqueeze(1)
                        sample_i_lmks = sample_lmks[:,i] if sample_lmks is not None else None
                        sample_bsps_i = sample_bsps[:,i].unsqueeze(1) if sample_bsps is not None else None
                        # visualize_sample(self.args, sample_i_lmks, sample_bsps_i, sample_path.replace('.pt', ''))
                        visualized += 1
        if self.dataset == 'express4d':
            fl.close()


    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.step + self.resume_step):09d}.log')
            diversity_times = 48
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml_double_take.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.step + self.resume_step,
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
                                                      group_name='Eval')


        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]

            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)
            del_clip(state_dict)

            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            if self.use_fp16:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt': opt_state,
                    'scaler': self.scaler.state_dict(),
                }

            torch.save(opt_state, f)

    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.
        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        matches = {file: re.match(r'model(\d+).pt', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
