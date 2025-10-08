import torch
from utils.data_loaders_utils import convert_to_3d

def parse_sample(args, sample, model_kwargs):
    if 'blendshapes' == args.data_mode:
        create_with_identity = True
        if 'full' in args.data_mode:
            motions_expression = sample[:,300:,:,:]
            motions_identity = sample[:,:300,:,:]
        else:
            motions_expression = sample
            motions_identity = model_kwargs['y']['identity']

        motions_expression = convert_to_3d(motions_expression)  # sample shape [bs x blendshapes x 1 x nframes]
        motions_expression = motions_expression.squeeze(2).permute(2, 0, 1)

        id = motions_identity.squeeze(2).permute(2, 0, 1)
        if create_with_identity:
            sample = torch.cat([id.to(args.device), motions_expression.to(args.device)], axis=2)
        else:
            sample = torch.cat([torch.zeros(sample.shape[0], sample.shape[1], 300).to(args.device), motions_expression.to(args.device)], axis=2)  # add zero identity to all frames

        sample_bsps = sample
        sample_lmks = None
    elif args.data_mode.startswith('landmarks'):
        sample = sample.squeeze(2).reshape((sample.shape[0],-1,3,sample.shape[-1])).permute(3,0,1,2)
        sample = sample.squeeze(1)
        sample_lmks = sample
        sample_bsps = None
    elif args.data_mode.startswith('blendmarks'):
        sample_bsps = sample[:,:133]
        sample_lmks = sample[:,133:]
        sample_bsps = convert_to_3d(sample_bsps) #sample shape [bs x blendshapes x 1 x nframes]
        sample_bsps = sample_bsps.squeeze(2).permute(2, 0, 1)
        sample_bsps = torch.cat([torch.zeros(sample_bsps.shape[0], sample_bsps.shape[1], 300).to(args.device), sample_bsps], axis=2) # add zero identity to all frames
        sample_bsps = sample_bsps.to(torch.float32)
        sample_lmks = sample_lmks.squeeze(2).reshape((sample_lmks.shape[0],-1,3,sample_lmks.shape[-1])).permute(3,0,1,2)
        sample_lmks = sample_lmks.squeeze(1)
    else:
        raise NotImplementedError(f"data_mode={args.data_mode}. not in options")
    
    return sample_lmks, sample_bsps
