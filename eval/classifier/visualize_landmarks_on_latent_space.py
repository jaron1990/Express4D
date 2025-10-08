import torch
import matplotlib.pyplot as plt
import os
from eval.classifier.lstm_classifier_adapted import LSTMClassifier
from utils import dist_util
from data_loaders.data_loader_utils import FAMOS_EXPRESSION_LIST, COMA_EXPRESSION_LIST
from utils.fixseed import fixseed
from utils.parser_util import classifier_args, train_args
from data_loaders.get_data import get_dataset_loader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE




if __name__ == '__main__':
    args = classifier_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)


    device = dist_util.dev()
    hidden_size = 64
    num_points = 70 #83
    args.classifier_step = 15
    normalize_data = True
    data_list = FAMOS_EXPRESSION_LIST

    classifier_path = 'save/20250613_classifier_from_lmks_centralized/train_cls_lr_00001_wd_1e-05_step_15_hid_64_classes/classfier_180.pt'
    c = len(data_list)

    net = LSTMClassifier(3*num_points, hidden_size, 1, c).to(device)
    net.load_state_dict(torch.load(classifier_path, map_location=device))

    hiddens_famos = []
    hiddens_express4d = []
    with torch.no_grad():
        args.dataset = 'famos'
        # if args.dataset == 'famos':
        train_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
                    hml_mode='train_classifier', data_mode=args.data_mode, split='train', classifier_step=args.classifier_step, minimum_frames=args.minimum_frames, debug=args.debug, normalize_data=normalize_data)
        valid_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
                    data_mode=args.data_mode, hml_mode='train_classifier', split='test', classifier_step=args.classifier_step, minimum_frames=args.minimum_frames, debug=args.debug, normalize_data=normalize_data)

        for data, _ in train_loader:
            pts = data.squeeze(2).permute(0,2,1)
            pts = pts.to(device)
            _, hidden = net(pts)
            hiddens_famos.append(hidden)
        for data, _ in valid_loader:
            pts = data.squeeze(2).permute(0,2,1)
            pts = pts.to(device)
            _, hidden = net(pts)
            hiddens_famos.append(hidden)

        args.dataset = 'express4d'            
    # elif args.dataset == 'express4d':
        bs = args.batch_size
        filespath = 'dataset/Express4D/data'
        files = os.listdir(filespath)
        files = [f for f in files if 'lmks_to_compare.pt' in f]
        landmarks = [torch.load(os.path.join(filespath, f)) for f in files]
        landmarks = [lmk[::args.classifier_step] for lmk in landmarks]
        landmarks = [lmk.reshape(lmk.shape[0], -1) for lmk in landmarks]
        for i in range(len(landmarks)//bs):
            lmks_batch = landmarks[i*bs:(i+1)*bs]
            length = max([lmk.shape[0] for lmk in lmks_batch])
            batch = torch.zeros((bs, length, 210), device=device)
            for l in range(bs):
                batch[l][-lmks_batch[l].shape[0]:] = lmks_batch[l]
            _, hidden = net(batch)
            hiddens_express4d.append(hidden)

    
    hiddens_famos = torch.concat(hiddens_famos)
    hiddens_express4d = torch.concat(hiddens_express4d)

    express4d_np = hiddens_express4d.cpu().numpy()
    famos_np = hiddens_famos.cpu().numpy()

    # Combine embeddings
    all_np = np.vstack([express4d_np, famos_np])
    labels = np.array([0] * len(express4d_np) + [1] * len(famos_np))  # 0 = express4d, 1 = famos

    # =============================
    # PCA
    # =============================
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_np)
    express4d_pca = all_pca[labels == 0]
    famos_pca = all_pca[labels == 1]

    # =============================
    # t-SNE (after PCA to 50D for better quality/speed)
    # =============================
    tsne_input = PCA(n_components=50).fit_transform(all_np)  # optional pre-reduction
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    all_tsne = tsne.fit_transform(tsne_input)
    express4d_tsne = all_tsne[labels == 0]
    famos_tsne = all_tsne[labels == 1]

    # =============================
    # Plot both side-by-side
    # =============================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # PCA Plot
    ax1.scatter(express4d_pca[:, 0], express4d_pca[:, 1], c='blue', alpha=0.6, label='Express4D')
    ax1.scatter(famos_pca[:, 0], famos_pca[:, 1], c='red', alpha=0.6, label='FaMoS')
    ax1.set_title('PCA Projection')
    ax1.set_xlabel('PCA 1')
    ax1.set_ylabel('PCA 2')
    ax1.legend()
    ax1.grid(True)

    # t-SNE Plot
    ax2.scatter(express4d_tsne[:, 0], express4d_tsne[:, 1], c='blue', alpha=0.6, label='Express4D')
    ax2.scatter(famos_tsne[:, 0], famos_tsne[:, 1], c='red', alpha=0.6, label='FaMoS')
    ax2.set_title('t-SNE Projection')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend()
    ax2.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig("pca_tsne_embeddings.png", dpi=300)


    # Compute PCA separately
    pca_express4d = PCA(n_components=2).fit_transform(express4d_np)
    pca_famos = PCA(n_components=2).fit_transform(famos_np)

    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.scatter(pca_express4d[:, 0], pca_express4d[:, 1], color='blue', alpha=0.6)
    ax1.set_title("PCA — Express4D")
    ax1.set_xlabel("PCA 1")
    ax1.set_ylabel("PCA 2")
    ax1.grid(True)

    ax2.scatter(pca_famos[:, 0], pca_famos[:, 1], color='red', alpha=0.6)
    ax2.set_title("PCA — FaMoS")
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("independent_pca_embeddings.png", dpi=300)
    plt.close()


    # Fit PCA only on "express4d"
    pca = PCA(n_components=2)
    pca.fit(express4d_np)  # Only fit on baseline

    # Transform both groups using the same PCA basis
    express4d_pca = pca.transform(express4d_np)
    famos_pca = pca.transform(famos_np)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(express4d_pca[:, 0], express4d_pca[:, 1], c='blue', alpha=0.6, label='Express4D')
    plt.scatter(famos_pca[:, 0], famos_pca[:, 1], c='red', alpha=0.6, label='FaMoS')
    plt.title("PCA (fit on 'Express4D')")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("pca_fit_on_express4d.png", dpi=300)
    plt.close()

    # Fit PCA only on "famos"
    pca = PCA(n_components=2)
    pca.fit(famos_np)

    # Transform both groups using the same PCA basis
    famos_pca = pca.transform(famos_np)
    express4d_pca = pca.transform(express4d_np)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(famos_pca[:, 0], famos_pca[:, 1], c='red', alpha=0.6, label='FaMoS')
    plt.scatter(express4d_pca[:, 0], express4d_pca[:, 1], c='blue', alpha=0.6, label='Express4D')
    plt.title("PCA (fit on 'FaMoS')")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("pca_fit_on_famos.png", dpi=300)
    plt.close()

    # Combine both datasets
    all_np = np.vstack([express4d_np, famos_np])
    labels = np.array([0] * len(express4d_np) + [1] * len(famos_np))

    # Run t-SNE directly on 128D
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    tsne_2d = tsne.fit_transform(all_np)

    # Split back
    express4d_tsne = tsne_2d[labels == 0]
    famos_tsne = tsne_2d[labels == 1]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(express4d_tsne[:, 0], express4d_tsne[:, 1], color='blue', alpha=0.6, label='Express4D')
    plt.scatter(famos_tsne[:, 0], famos_tsne[:, 1], color='red', alpha=0.6, label='FaMoS')
    plt.title("t-SNE (without PCA)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("tsne_no_pca.png", dpi=300)
    plt.close()