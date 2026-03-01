from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate, coma_collate, coma_collate_classifier
from data_loaders.tensors import t2m_collate
from data_loaders.voca.dataset import VOCA
from data_loaders.coma.dataset import COMA
from data_loaders.FAMOS.dataset import FAMOS
from data_loaders.Express4D.dataset import Express4D
from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.data.dataset import collate_fn as sorted_collate


def get_dataset_class(name):
    if name == "humanml": # TODO:delete once done - used as a reference
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "voca":
        return VOCA
    elif name == "coma":
        return COMA
    elif name == "famos":
        return FAMOS
    elif name == "express4d":
        return Express4D
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='generator'):
    if name in ["humanml", "kit"]:
        return t2m_collate
    elif name == 'express4d':
        if hml_mode in ['gt', 'evaluator_train']:
            return sorted_collate
        else:
            return t2m_collate

    elif name in ["coma", "voca", "famos"]:
        if hml_mode in ['classifier', 'train_classifier']:
            return coma_collate_classifier
        else:
            return coma_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='generator', data_mode='landmarks', normalize_data=False, classifier_step=15, minimum_frames = 60, debug=False, smoothing_filter_length = 7, add_velocities=False, add_landmarks_diffs = False, max_len=196, flip_face_on=False, fps=20, max_motions=-1):
    DATA = get_dataset_class(name)
    if name in ["famos", "voca", "coma"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, data_mode=data_mode, normalize_data=normalize_data,classifier_step=classifier_step, minimum_frames=minimum_frames, debug=debug, smoothing_filter_length= smoothing_filter_length, add_velocities=add_velocities, add_landmarks_diffs=add_landmarks_diffs, max_len=max_len)
                       # datapath=f'dataset/FAMOS_data_train_test_split.json')
    elif name in ["express4d"]:
        dataset = DATA(split=split, mode=hml_mode, data_mode=data_mode, minimum_frames=minimum_frames, debug=debug, max_len=max_len, flip_face_on = flip_face_on, fps=fps, max_motions=max_motions)
    # elif name in ["coma"]:
    #     dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, data_mode=data_mode, datapath='dataset/COMA_data')
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='generator', data_mode='landmarks', normalize_data=False, classifier_step=15, minimum_frames=60, debug=False, smoothing_filter_length = 7, add_velocities=False, add_landmarks_diffs = False, max_len=196, flip_face_on=False, fps=20, max_motions=-1, shuffle=True):
    # dataset = get_dataset(name, num_frames, split, hml_mode, 'landmarks_68', normalize_data) #FIXME: !!! not good
    try:
        dataset = get_dataset(name, num_frames, split, hml_mode, data_mode, normalize_data, classifier_step=classifier_step, minimum_frames=minimum_frames, debug=debug, smoothing_filter_length=smoothing_filter_length, add_velocities=add_velocities, add_landmarks_diffs=add_landmarks_diffs, max_len=max_len, flip_face_on=flip_face_on, fps=fps, max_motions=max_motions) #FIXME: !!! not good
    except:
        error = 'Unable to load dataset [{}] with data_mode [{}]\n'.format(name, data_mode)
        error += 'check if dataset is under dataset/Express4D'
        raise ValueError(error)

    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader