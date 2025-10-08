import os

import re
import matplotlib.pyplot as plt
def plot_loss_from_log(file_path, save_path="loss_plot.png", log_scale=False, epochs_range=[0,float("inf")], ylim=None, xlim=None):
    """
    Reads a log file, extracts loss and val_loss from lines starting with 'epoch:',
    and plots them on a specified scale for epochs between 0 and 200.
    The plot is saved to the specified file.
    :param file_path: Path to the log file
    :param save_path: Path to save the generated plot (default: "loss_plot.png")
    :param log_scale: If True, use log scale; otherwise, use linear scale (default: False)
    """
    epochs, val_loss, loss = [], [], []
    pattern = re.compile(r"epoch:\s*(\d+).*?val_loss:\s*([\d.]+).*?loss:\s*([\d.]+)", re.S)
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                if epochs_range[0] <= epoch <= epochs_range[1]:
                    epochs.append(epoch)
                    val_loss.append(float(match.group(2)))
                    loss.append(float(match.group(3)))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_loss, label="val_loss", marker='o')
    plt.plot(epochs, loss, label="loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss Values")
    plt.yscale("log" if log_scale else "linear")
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.title("Loss and Val Loss over Epochs (0-200)")
    plt.grid(True, which="both" if log_scale else "major", linestyle="--", linewidth=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as {save_path}")
    if len(val_loss) != 0:
        min_loss = min(val_loss)
    else:
        min_loss = float("inf")
    return min_loss


if __name__ == '__main__':
    # root = 'slurm_files/evaluator_train/logs/20250319_1129'
    root = 'slurm_files/evaluator_train/logs/20250323_1428'
    files = os.listdir(root)
    files = [f for f in files if '.out' in f]
    min_losses = []
    training_sizes = []
    for fl in files:
        min_losses.append(plot_loss_from_log(os.path.join(root, fl), os.path.join(root, fl.replace('.out', '.png')), ylim=[0,70], xlim=[0,100]))
        training_sizes.append(int(fl.split('_')[4].split('.')[0]))


    plt.figure(figsize=(10, 6))
    sorted_pairs = sorted(zip(training_sizes, min_losses))
    sorted_training_sizes, sorted_losses = zip(*sorted_pairs)

    plt.plot(sorted_training_sizes, sorted_losses, label="losses", marker='o')
    plt.title("losses vs training_sizes")
    plt.savefig(os.path.join(root,'all_losses.png'), dpi=300, bbox_inches="tight")

    