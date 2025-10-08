import torch


def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        for batch in motion_loader:
            # batch_prob = classifier(batch["inp"], lengths=batch["lengths"])
            batch_prob = classifier(batch["output"])[0]
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch["y"], batch_pred):
                confusion[label][pred] += 1

    accuracy = torch.trace(confusion)/torch.sum(confusion)
    return accuracy.item(), confusion

def calculate_top_k(model, motion_loader, num_labels, classifier, device, k=3):
    hits = 0
    total = 0
    with torch.no_grad():
        for batch in motion_loader:
            # batch_prob = classifier(batch["inp"], lengths=batch["lengths"])
            batch_prob = classifier(batch["output"])[0]
            top_k_preds = torch.argsort(batch_prob)[:, -k:].cpu()
            hits += int((top_k_preds == batch["y"].unsqueeze(1)).sum())
            total += len(batch["y"])

    top_k = hits/total
    return top_k
