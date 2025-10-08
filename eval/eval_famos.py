"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import torch
import re

from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from eval.a2m.tools import save_metrics
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from eval.a2m.gru_eval import evaluate as evaluate_famos
from eval.classifier.lstm_classifier_adapted import LSTMClassifier


def evaluate(args, model, diffusion, data, idx_to_test=None):
    # print(args)

    assert args.data_mode in ['arkit', 'arkit_labels']

    # print(args)
    # exit()

    scale = None
    # args.guidance_param = 1 # FIXME: this is a hack to make the code run
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
        scale = {
            'action': torch.ones(args.batch_size) * args.guidance_param,
        }
    model.to(dist_util.dev())
    model.eval()  # disable random masking


    folder, ckpt_name = os.path.split(args.model_path)
    if args.dataset in ["famos", "coma"]:
        eval_results = evaluate_famos(args, model, diffusion, data, idx_to_test=idx_to_test)
    else:
        raise NotImplementedError("This dataset is not supported.")

    # save results
    iter = int(re.findall('\d+', ckpt_name)[0])
    scale = 1 if scale is None else scale['action'][0].item()
    scale = str(scale).replace('.', 'p')
    metricname = "evaluation_results_iter{:09}_samp{}_scale{}_a2m.yaml".format(iter, args.num_samples, scale)
    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, eval_results)

    return eval_results


def main():
    args = evaluation_parser()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    print(f'Eval mode [{args.eval_mode}]')
    assert args.eval_mode in ['debug', 'full'], f'eval_mode {args.eval_mode} is not supported for dataset {args.dataset}'
    if args.eval_mode == 'debug':
        args.num_samples = 10
        args.num_seeds = 2
    else:
        args.num_samples = 1000
        args.num_seeds = 10
        
    num_frames = 196
    data_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=num_frames, data_mode=args.data_mode, 
                                     hml_mode='classifier', split='test', classifier_step=args.classifier_step, minimum_frames=args.minimum_frames, 
                                     add_velocities=args.add_velocities, add_landmarks_diffs=args.add_landmarks_diffs)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data_loader)
    model.to(dist_util.dev())

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'model_avg' in state_dict:
        print('only average model')
        state_dict = state_dict['model_avg']
    load_model_wo_clip(model, state_dict)

    # for idx_to_test in range(70):
    #     print(f'testing only lmk {idx_to_test}')
    #     eval_results = evaluate(args, model, diffusion, data_loader.dataset, idx_to_test)
    #     print(eval_results)

    eval_results = evaluate(args, model, diffusion, data_loader.dataset)

    fid_to_print = {k : sum([float(vv) for vv in v])/len(v) for k, v in eval_results['feats'].items() if 'fid' in k and 'gen' in k}
    print(fid_to_print)

if __name__ == '__main__':
    main()