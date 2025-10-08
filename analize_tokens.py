import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_and_freeze_clip(clip_version = 'ViT-B/32'):
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    # clip.model.convert_weights(
    #     clip_model)  # Actually this line is unnecessary since clip by default already on float16

    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encode_text(clip_model, raw_text):
    texts = clip.tokenize(raw_text, truncate=True) 
    return clip_model.encode_text(texts).float()


if __name__ == '__main__':
    clip_model = load_and_freeze_clip()
    captions_dict = torch.load('captions_dict.pt')
    for key,val in captions_dict.items():
        clip_embedding = encode_text(clip_model, key)
        clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)  # Normalize
        clip_embedding = clip_embedding.squeeze()
        captions_dict[key]['clip_embedding'] = clip_embedding

    clip_embeddings = np.stack([key['clip_embedding'] for key in captions_dict.values()])
    text_embeddings = np.stack([key['text_embedding'].cpu().numpy() for key in captions_dict.values()])
    captions = np.stack([key for key in captions_dict.keys()])
    top_1_correct = np.stack([key['top1_correct'] for key in captions_dict.values()])
    top_3_correct = np.stack([key['top3_correct'] for key in captions_dict.values()])
    tsne = TSNE(n_components=2, perplexity=80, random_state=42)
    clip_embeddings_2d = tsne.fit_transform(clip_embeddings)
    text_embeddings_2d = tsne.fit_transform(text_embeddings)



    colors = np.array(["red", "green"])[top_1_correct.astype(int)]
    with open('top1_false.txt', 'w') as f:
        for i, (val, caption) in enumerate(zip(top_1_correct, captions)):
            if not  val:
                f.write(f'{i:05}:\t{caption}\n')

    plt.figure(figsize=(10, 7))
    plt.scatter(clip_embeddings_2d[:, 0], clip_embeddings_2d[:, 1], c=colors, alpha=0.7)

    # Label incorrect samples (False points) with their index number
    for i, (x, y) in enumerate(clip_embeddings_2d):
        if not top_1_correct[i]:  # Only label the False (red) points
            plt.text(x, y, str(i), fontsize=9, color='black', ha='right', va='bottom')

    # Add legend
    plt.title("t-SNE Visualization of CLIP Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False')
    ])
    plt.savefig('top1_clip_vis_false.png')

    plt.figure(figsize=(10, 7))
    plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], c=colors, alpha=0.7)

    # Label incorrect samples (False points) with their index number
    for i, (x, y) in enumerate(text_embeddings_2d):
        if not top_1_correct[i]:  # Only label the False (red) points
            plt.text(x, y, str(i), fontsize=9, color='black', ha='right', va='bottom')

    # Add legend
    plt.title("t-SNE Visualization of CLIP Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False')
    ])
    plt.savefig('top1_text_vis_false.png')







    colors = np.array(["red", "green"])[top_3_correct.astype(int)]
    with open('top3_false.txt', 'w') as f:
        for i, (val, caption) in enumerate(zip(top_3_correct, captions)):
            if not  val:
                f.write(f'{i:05}:\t{caption}\n')

    plt.figure(figsize=(10, 7))
    plt.scatter(clip_embeddings_2d[:, 0], clip_embeddings_2d[:, 1], c=colors, alpha=0.7)

    # Label incorrect samples (False points) with their index number
    for i, (x, y) in enumerate(clip_embeddings_2d):
        if not top_3_correct[i]:  # Only label the False (red) points
            plt.text(x, y, str(i), fontsize=9, color='black', ha='right', va='bottom')

    # Add legend
    plt.title("t-SNE Visualization of CLIP Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False')
    ])
    plt.savefig('top3_clip_vis_false.png')

    plt.figure(figsize=(10, 7))
    plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], c=colors, alpha=0.7)

    # Label incorrect samples (False points) with their index number
    for i, (x, y) in enumerate(text_embeddings_2d):
        if not top_3_correct[i]:  # Only label the False (red) points
            plt.text(x, y, str(i), fontsize=9, color='black', ha='right', va='bottom')

    # Add legend
    plt.title("t-SNE Visualization of CLIP Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False')
    ])
    plt.savefig('top3_text_vis_false.png')

