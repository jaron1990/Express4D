import torch

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_tensors_classifier(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size).to(batch[0].device)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, max_size[d] - b.size(d), b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch, train_classifier=False):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'identity' in notnone_batches[0]:
        idbatch = [b['identity'] for b in notnone_batches]
    
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    if train_classifier:
        databatchTensor = collate_tensors_classifier(databatch)
        if 'identity' in notnone_batches[0]:
            idbatchTensor = collate_tensors_classifier(idbatch)
    else:
        databatchTensor = collate_tensors(databatch)
        if 'identity' in notnone_batches[0]:
            idbatchTensor = collate_tensors(idbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'file' in notnone_batches[0]:
        filebatch = [b['file'] for b in notnone_batches]
        cond['y'].update({'file': filebatch})

    if 'identity' in notnone_batches[0]:
        cond['y'].update({'identity': idbatchTensor})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2],
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

def coma_collate(batch):
    adapted_batch = [{
        'inp': (b["inp"].clone().detach().permute(1, 0).float().unsqueeze(1)  # [seqlen, J] -> [J, 1, seqlen]
                if len(b["inp"].shape) == 2
                else b["inp"].clone().detach().reshape(b['inp'].shape[0], -1).permute(1, 0).float().unsqueeze(1)),  # for lmks. [seqlen, landmarks, 3] -> [landmarks*3, 1, seqlen]
        'text': b["label"],
        'action': b["action"],
        'lengths': b["length"],
        'file': b["file"],
    } for b in batch]
    if 'identity' in batch[0].keys():
        for i, b in enumerate(batch):
            adapted_batch[i]['identity'] = (b["identity"].clone().detach().permute(1, 0).float().unsqueeze(1)  # [seqlen, J] -> [J, 1, seqlen]
                if len(b["identity"].shape) == 2
                else b["identity"].clone().detach().reshape(b['identity'].shape[0], -1).permute(1, 0).float().unsqueeze(1))
    return collate(adapted_batch)


def coma_collate_classifier(batch):
    adapted_batch = [{
        'inp': (b["inp"].clone().detach().permute(1, 0).float().unsqueeze(1)  # [seqlen, J] -> [J, 1, seqlen]
                if len(b["inp"].shape) == 2
                else b["inp"].clone().detach().reshape(b['inp'].shape[0], -1).permute(1, 0).float().unsqueeze(1)),  # for lmks. [seqlen, landmarks, 3] -> [landmarks*3, 1, seqlen]
        'text': b["label"],
        'action': b["action"],
        'lengths': b["length"],
        # 'orig_lengths': b["orig_length"],
        'file': b["file"],
    } for b in batch]
    if 'identity' in batch[0].keys():
        for i, b in enumerate(batch):
            adapted_batch[i]['identity'] = (b["identity"].clone().detach().permute(1, 0).float().unsqueeze(1)  # [seqlen, J] -> [J, 1, seqlen]
                if len(b["identity"].shape) == 2
                else b["identity"].clone().detach().reshape(b['identity'].shape[0], -1).permute(1, 0).float().unsqueeze(1))

    return collate(adapted_batch, train_classifier=True)