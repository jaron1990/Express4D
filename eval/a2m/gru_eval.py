import copy
import random

import pickle as pkl
from tqdm import tqdm
import torch
import functools
from torch.utils.data import DataLoader

from utils.data_loaders_utils import convert_to_3d
from utils.fixseed import fixseed
from data_loaders.tensors import coma_collate_classifier as collate
from eval.a2m.action2motion.evaluate import A2MEvaluation
from .tools import save_metrics, format_metrics
from utils import dist_util
from utils.landmarks import load_embedding
from data_loaders.data_loader_utils import FAMOS_EXPRESSION_LIST


num_samples_unconstrained = 1000

class NewDataloader:
    def __init__(self, mode, model, diffusion, dataiterator, device, normalize_data, args, num_samples: int=-1, centralize_blendshapes=False, idx_to_test=None):
        assert mode in ["gen", "gt"]
        self.batches = []
        data_from_file = False
        sample_fn = diffusion.p_sample_loop

        with torch.no_grad():
             # if gt - load data only once. else - create num_samples samples
            rng = range(0,num_samples, dataiterator.batch_size) if mode=='gen' else range(1) 
            for _ in tqdm(rng, f"Construct dataloader: {mode}.."):
                for motions, model_kwargs in dataiterator:
                    if mode == "gen":
                        if 'data_path' in vars(args) and args.data_path is not None:
                            data_from_file = True
                            normalize_data = False #TODO - remove after implementation of normalization in ACTOR
                            with open(args.data_path, 'rb') as handle:
                                data = pkl.load(handle)
                            if args.data_mode == 'landmarks_70_centralized':
                                motions = data['output'].reshape((data['output'].shape[0], 210, 1, -1))
                            elif args.data_mode in ['blendshapes_full']:
                                motions = data['output']
                            elif args.data_mode == 'blendmarks_70_centralized':
                                motions = data['output']
                            else:
                                raise Exception()
                            model_kwargs['y']['action'] = data['y'].unsqueeze(1)
                            model_kwargs['y']['text'] = [FAMOS_EXPRESSION_LIST[idx] for idx in model_kwargs['y']['action']]

                            model_kwargs['y']['lengths'] *=0
                            model_kwargs['y']['lengths'] = data['lengths']
                            model_kwargs['y']['mask'] = data['mask'].unsqueeze(1).unsqueeze(2)
                            model_kwargs['y']['file'] = [model_kwargs['y']['file'][0]]*data['output'].shape[0]

                        else:
                            motions = sample_fn(model, motions.shape, clip_denoised=False, model_kwargs=model_kwargs, skip_timesteps =0)
                            
                        #right padding - same as in the collate classifier
                        new_motions = torch.zeros_like(motions)
                        for i in range(motions.shape[0]):
                            current_len = model_kwargs['y']['lengths'][i]
                            new_motions[i,:,:,-current_len:] = motions[i,:,:,:current_len]
                            pass
                        motions = new_motions

                        if normalize_data:  # classifier was trained with unnormalized data, in case model was trained with normalize_data - the data needs to be denormed
                            motions = motions * dataiterator.dataset.std.view(1, -1, 1, 1).to(device) + dataiterator.dataset.mean.view(1, -1, 1, 1).to(device)
                            motions = motions.to(torch.float32)
                                
                    skip=args.classifier_step
                    model_kwargs['y']['lengths'] //= skip
                    model_kwargs['y']['lengths'] += 1


                    model_kwargs['y']['full_motions'] = motions.squeeze(2).permute(0,2,1)
                    motions = motions[..., ::skip]

                    motions = motions.to(device)

                    if num_samples != -1 and len(self.batches) * dataiterator.batch_size > num_samples:
                        continue  # do not break because it confuses the multiple loaders
                    batch = dict()
                    batch["output"] = motions.squeeze(2).permute(0,2,1)
                    batch["full_output"] =model_kwargs['y']['full_motions']
                    # batch["full_output"] = full_motions.squeeze(2).permute(0,2,1)

                    bs, n_feat, _, n_frames = motions.shape
                    # batch["output_xyz"] = motions.view(bs, n_feat//3, 3, n_frames)
                    batch["lengths"] = model_kwargs['y']['lengths'].to(device)
                    batch["file"] = model_kwargs['y']['file']
                    batch["y"] = model_kwargs['y']['action'].squeeze().long().cpu()  # using torch.long so lengths/action will be used as indices
                    self.batches.append(batch)

                    if data_from_file:
                        break
                if data_from_file:
                     break

            if data_from_file:
                all_batches = self.batches
                self.batches = []
                start = 0
                while start < all_batches[0]['output'].shape[0]:
                    stop = min(all_batches[0]['output'].shape[0], start+dataiterator.batch_size)
                    new_batch = {}
                    for k,v in all_batches[0].items():
                        new_batch[k] = v[start:stop]
                    self.batches.append(new_batch)
                    start = stop
            else:
                num_samples_last_batch = num_samples % dataiterator.batch_size
                if num_samples_last_batch > 0:
                    for k, v in self.batches[-1].items():
                        self.batches[-1][k] = v[:num_samples_last_batch]

    def __iter__(self):
        return iter(self.batches)

def evaluate(args, model, diffusion, data, idx_to_test=None):
    # FIXME: here is the code for evaluate MDM
    num_frames = 196 #FIXME: this is the number of frames for the dataset

    # fix parameters for action2motion evaluation
    args.num_frames = num_frames
    # args.jointstype = "smpl"
    # args.vertstrans = True
    args.num_seeds = 1  # num repetitions

    device = dist_util.dev()

    model.eval()

    a2mevaluation = A2MEvaluation(device=device, eval_path=args.eval_path, dataset=args.dataset, classifier_hidden_size = args.hidden_size)
    a2mmetrics = {}

    datasetGT1 = copy.deepcopy(data)
    datasetGT2 = copy.deepcopy(data)

    allseeds = list(range(args.num_seeds))

    try:
        for index, seed in enumerate(allseeds):
            print(f"Evaluation number: {index+1}/{args.num_seeds}")
            fixseed(seed)

            random.shuffle(datasetGT1.data)
            random.shuffle(datasetGT2.data)

            dataiterator = DataLoader(datasetGT1, batch_size=args.batch_size,
                                      shuffle=False, num_workers=8, collate_fn=collate)
            dataiterator2 = DataLoader(datasetGT2, batch_size=args.batch_size,
                                       shuffle=False, num_workers=8, collate_fn=collate)

            if args.debug:
                args.num_samples = 50
            new_data_loader = functools.partial(NewDataloader,
            
                                                args=args,
                                                model=model, diffusion=diffusion, device=device,
                                                normalize_data=not args.normalize_data_off,
                                                num_samples=args.num_samples, centralize_blendshapes=args.centralize_blendshapes)
            
            if idx_to_test is None:
                motionloader = new_data_loader(mode="gen", dataiterator=dataiterator, idx_to_test=idx_to_test)
            gt_motionloader = new_data_loader("gt", dataiterator=dataiterator)
            gt_motionloader2 = new_data_loader("gt", dataiterator=dataiterator2, idx_to_test=idx_to_test)

            # Action2motionEvaluation
            loaders = {"gt": gt_motionloader,
                       "gt2": gt_motionloader2}
            
            if idx_to_test is None:
                loaders["gen"]= motionloader

            a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)

            del loaders


    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in a2mmetrics[allseeds[0]]}}
    # if args.unconstrained:
    #     metrics["feats"] = {**metrics["feats"], **unconstrained_metrics}

    return metrics
