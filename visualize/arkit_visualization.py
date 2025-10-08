import matplotlib.pyplot as plt
from utils.arkit_utils import create_new_vertices_from_blendshapes, create_vertices_sequence, livelink_csv_to_sequence
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip
import numpy as np
import torch
import os

#get livelink blendshapes_values as torch.tensor and save them as a png
def scatter_single_frame_blendshapes(blendshapes_values, out_path, valid_objects = ['eyeLeft', 'eyeRight', 'head_lod0', 'teeth']):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    colors = ['red', 'blue', 'green', 'orange']  # Different colors for each object
    colors = colors[:len(valid_objects)]

    for i, obj in enumerate(valid_objects):
        print(f'frame {i}, obj {obj}')
        new_vertices = create_new_vertices_from_blendshapes(blendshapes_values, obj)

        ax.scatter(
            new_vertices[:, 0],  # X-coordinates
            new_vertices[:, 2],  # Z-coordinates
            color=colors[i % len(colors)], alpha=0.7, s=1
        )

    ax.set_title("Blendshape Meshes (X-Z Plane)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend()
    plt.savefig(out_path, dpi=300)


#get vertices as torch.tensor and save them as a png
def scatter_single_frame_vertices(vertices, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2)
        
    ax1.scatter(
        vertices[:, 0],  # X-coordinates
        vertices[:, 2],  # Z-coordinates
        alpha=0.7, s=1
    )
    ax2.scatter(
        vertices[:, 1],  # X-coordinates
        vertices[:, 2],  # Z-coordinates
        alpha=0.7, s=1
    )

    ax1.set_title("Blendshape Meshes (X-Z Plane)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.legend()

    plt.savefig(out_path, dpi=300)


def two_sequences_on_one_animation(tensor1, tensor2, out_path, fps=60, inpainting_frames = None, inpainting_idxs = None, title=[]):
    # Prepare the figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    colors = ['red', 'blue']  # Colors for each sequence

    # Convert tensors to NumPy arrays
    tensor1 = tensor1.cpu().detach().numpy()
    tensor2 = tensor2.cpu().detach().numpy()
    tensors = [tensor1, tensor2]

    # Compute the maximum length
    length = max(tensor1.shape[0], tensor2.shape[0])

    # Compute axis limits
    xmax, zmax, xmin, zmin = -1E100, -1E100, 1E100, 1E100
    vertices = []
    for tens in tensors:
        obj_vertices = create_vertices_sequence(tens, 'head_lod0')
        xmax = max(obj_vertices[:, :, 0].max(), xmax)
        xmin = min(obj_vertices[:, :, 0].min(), xmin)
        zmax = max(obj_vertices[:, :, 2].max(), zmax)
        zmin = min(obj_vertices[:, :, 2].min(), zmin)
        vertices.append(obj_vertices)
    

    # Pre-create scatter objects
    scatter_objects = []
    for i, vert in enumerate(vertices):
        scatter = ax.scatter(vert[0, :, 0],  # First frame
                             vert[0, :, 2],
                             color=colors[i % len(colors)], alpha=0.7, s=1)
        scatter_objects.append(scatter)

    # Configure axis
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)

    # Update function for MoviePy
    def update(frame_i):
        frame_i = min(length - 1, int(frame_i * fps))
                
        # Determine background color based on inpainting_frames
        if inpainting_frames is not None:
            blend = inpainting_frames[frame_i].cpu()  # value between 0 and 1

            # Colors: white (GT) = [1, 1, 1], mistyrose (inpainting) ≈ [1, 0.894, 0.882]
            white_rgb = np.array([1.0, 1.0, 1.0])
            lightred_rgb = np.array([0.0, 1.0, 0.0])

            # Linearly interpolate between colors
            bg_color = (blend * white_rgb + (1 - blend) * lightred_rgb).cpu().numpy()
            bg_color = bg_color.clip(0,1)
            ax.set_facecolor(bg_color)
            
            if blend < 0.99:  # Slightly less than 1, to account for float errors
                ax.set_title(f'{title}\nframe={frame_i} [Inpainting: {1-blend:.2f}]', color='red')
            else:
                ax.set_title(f'{title}\nframe={frame_i}')
        else:
            ax.set_facecolor('white')
            ax.set_title(f'{title}\nframe={frame_i}')

        for i, vert in enumerate(vertices):
            if frame_i < vert.shape[0]:  # Avoid out-of-bound errors
                scatter_objects[i].set_offsets(
                    np.c_[vert[frame_i, :, 0], vert[frame_i, :, 2]]
                )
        return mplfig_to_npimage(fig)

    # Create animation
    duration = length / fps
    anim = VideoClip(update, duration=duration)

    # Write video file
    anim.write_videofile(out_path, fps=fps, threads=4)

    # Close plot to prevent memory leaks
    plt.close()

def save_scatter_animation(blendshapes_values, out_path, valid_objects=['eyeLeft', 'eyeRight', 'teeth', 'head_lod0'], fps=60, title='', inpainting_frames = None, inpainting_idxs = None):
    n_frames = blendshapes_values.shape[0]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Colors for each object
    colors = ['red', 'blue', 'orange', 'green']
    colors = colors[:len(valid_objects)]

    # Preprocess vertices
    blendshapes_values = blendshapes_values.cpu().detach().numpy()
    xmax, ymax, zmax, xmin, ymin, zmin = -1E100, -1E100, -1E100, 1E100, 1E100, 1E100
    vertices = {}
    for obj in valid_objects:
        obj_vertices = create_vertices_sequence(blendshapes_values, obj)
        xmax = max(obj_vertices[:, :, 0].max(), xmax)
        xmin = min(obj_vertices[:, :, 0].min(), xmin)
        zmax = max(obj_vertices[:, :, 2].max(), zmax)
        zmin = min(obj_vertices[:, :, 2].min(), zmin)
        vertices[obj] = obj_vertices

    # Initialize scatter plots for each object
    scatter_objects = []
    for i, obj in enumerate(valid_objects):
        scatter = ax.scatter(vertices[obj][0, :, 0],  # Use the first frame initially
                             vertices[obj][0, :, 2],
                             color=colors[i % len(colors)], label=obj, alpha=0.7, s=1)
        scatter_objects.append(scatter)

    # Configure axis
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)

    # Make sure inpainting_idxs is a NumPy array for safe indexing
    if inpainting_idxs is not None:
        inpainting_idxs = np.array(inpainting_idxs, dtype=int)

    yellow_overlays = []
    
    # Update function for each frame
    def update(frame_i):
        frame_i = min(n_frames - 1, int(frame_i * fps))

        # Background and title
        if inpainting_frames is not None:
            blend = inpainting_frames[frame_i].cpu()
            white_rgb = np.array([1.0, 1.0, 1.0])
            lightred_rgb = np.array([0.0, 1.0, 0.0])
            bg_color = (blend * white_rgb + (1 - blend) * lightred_rgb).cpu().numpy().clip(0, 1)
            ax.set_facecolor(bg_color)
            if blend < 0.99:
                ax.set_title(f'{title}\nframe={frame_i} [Inpainting: {1-blend:.2f}]', color='red')
            else:
                ax.set_title(f'{title}\nframe={frame_i}')
        else:
            ax.set_facecolor('white')
            ax.set_title(f'{title}\nframe={frame_i}')

        # Update base scatter plots
        for i, obj in enumerate(valid_objects):
            scatter_objects[i].set_offsets(
                np.c_[vertices[obj][frame_i, :, 0], vertices[obj][frame_i, :, 2]]
            )

        # Remove previous red overlays
        for overlay in yellow_overlays:
            overlay.remove()
        yellow_overlays.clear()

        # Draw red overlays at specific indices
        if inpainting_idxs is not None:
            for obj in valid_objects:
                verts = vertices[obj][frame_i]
                if inpainting_idxs.max() >= verts.shape[0]:
                    continue  # skip if out of bounds
                yellow_pts = verts[inpainting_idxs]
                yellow_scatter = ax.scatter(
                    yellow_pts[:, 0], yellow_pts[:, 2],
                    color='yellow',
                    s=20,             # bigger point
                    alpha=1.0,
                    zorder=100        # on top
                )
                yellow_overlays.append(yellow_scatter)

        return mplfig_to_npimage(fig)
        
    # Create animation
    duration = blendshapes_values.shape[0] / fps
    anim = VideoClip(update, duration=duration)

    # Write video file
    anim.write_videofile(out_path, fps=fps, threads=4)

def save_landmarks_animation(landmarks, out_path, fps=60, title=''):
    n_frames = landmarks.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Preprocess vertices
    landmarks = landmarks.cpu().detach().numpy()
    xmin, ymin, zmin = -1, -1, -1
    xmax, ymax, zmax = 1, 1, 1

    # Initialize scatter plots for X-Z and Y-Z projections
    scatter_xz = ax1.scatter(landmarks[0, :, 0], landmarks[0, :, 2], c='b')
    scatter_yz = ax2.scatter(landmarks[0, :, 1], landmarks[0, :, 2], c='b')

    # Configure axes
    ax1.set_title("X-Z view")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(zmin, zmax)

    ax2.set_title("Y-Z view")
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Z")
    ax2.set_xlim(ymin, ymax)
    ax2.set_ylim(zmin, zmax)

    # Update function for each frame
    def update(frame_i):
        frame_i = min(n_frames - 1, int(frame_i * fps))
        fig.suptitle(f'{title} — frame {frame_i}')

        # Update both scatter plots
        scatter_xz.set_offsets(np.c_[landmarks[frame_i, :, 0], landmarks[frame_i, :, 2]])  # X-Z
        scatter_yz.set_offsets(np.c_[landmarks[frame_i, :, 1], landmarks[frame_i, :, 2]])  # Y-Z

        return mplfig_to_npimage(fig)
        
    # Create animation
    duration = landmarks.shape[0] / fps
    anim = VideoClip(update, duration=duration)

    # Write video file
    anim.write_videofile(out_path, fps=fps, threads=4)

def save_incremental_scatter(blendshape_vector, out_path, object_name='head_lod0', title='', fps=30):
    assert blendshape_vector.shape == (61,), "Expected a single blendshape vector of shape (61,)"

    # Convert to NumPy with batch dim
    blendshape_vector = blendshape_vector.detach().cpu().numpy()  # shape: (61,)
    blendshape_batch = blendshape_vector[None, :]  # shape: (1, 61)

    # Generate vertices
    vertices = create_vertices_sequence(blendshape_batch, object_name)[0]  # shape: (V, 3)
    vertices = vertices.detach().cpu().numpy()  # Ensure it's NumPy

    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    z_vals = vertices[:, 2]
    num_vertices = vertices.shape[0]

    # Axis limits with margin
    def compute_limits(a, b):
        margin_a = (a.max() - a.min()) * 0.1
        margin_b = (b.max() - b.min()) * 0.1
        return (a.min() - margin_a, a.max() + margin_a), (b.min() - margin_b, b.max() + margin_b)

    xz_xlim, xz_ylim = compute_limits(x_vals, z_vals)
    yz_xlim, yz_ylim = compute_limits(y_vals, z_vals)

    # Setup plot with 2 subplots
    fig, (ax_xz, ax_yz) = plt.subplots(1, 2, figsize=(14, 6))

    # X-Z plot
    ax_xz.set_xlim(xz_xlim)
    ax_xz.set_ylim(xz_ylim)
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.set_title('X-Z View')

    scatter_old_xz = ax_xz.scatter([], [], color='blue', s=1)
    scatter_new_xz = ax_xz.scatter([], [], color='red', s=5)

    # Y-Z plot
    ax_yz.set_xlim(yz_xlim)
    ax_yz.set_ylim(yz_ylim)
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.set_title('Y-Z View')

    scatter_old_yz = ax_yz.scatter([], [], color='blue', s=1)
    scatter_new_yz = ax_yz.scatter([], [], color='red', s=5)

    # Update function receives time t (in seconds)
    def update(t):
        frame_i = min(int(t * fps), num_vertices - 1)

        # OLD vertices
        if frame_i > 0:
            scatter_old_xz.set_offsets(np.c_[x_vals[:frame_i], z_vals[:frame_i]])
            scatter_old_yz.set_offsets(np.c_[y_vals[:frame_i], z_vals[:frame_i]])
        else:
            scatter_old_xz.set_offsets(np.empty((0, 2)))
            scatter_old_yz.set_offsets(np.empty((0, 2)))

        # NEW vertex
        scatter_new_xz.set_offsets(np.array([[x_vals[frame_i], z_vals[frame_i]]]))
        scatter_new_yz.set_offsets(np.array([[y_vals[frame_i], z_vals[frame_i]]]))

        fig.suptitle(f'{title}\nVertex {frame_i+1}/{num_vertices}', fontsize=14)
        return mplfig_to_npimage(fig)

    # Generate and write video
    duration = num_vertices / fps
    anim = VideoClip(update, duration=duration)
    anim.write_videofile(out_path, fps=fps, threads=4)

    
def process_file(fl):
    root = 'dataset/Express4D/data'
    video_path = os.path.join(root, fl).replace('.csv', '.mp4')
    idx = int(fl.split('_')[1]) - 1

    # Read texts inside the function to avoid sharing issues across processes
    with open("/home/dcor/yaronaloni/Express4D/dataset/Express4D/texts_plain.txt", "r") as f:
        texts = f.readlines()
    title = texts[idx]

    if os.path.exists(video_path):
        print(f'{video_path} already exists')
        return

    n_array = livelink_csv_to_sequence(os.path.join(root, fl))
    tensor = torch.tensor(n_array)
    save_scatter_animation(tensor, video_path, title=title)


if __name__ == '__main__':
    root = 'dataset/Express4D/data'
    files = os.listdir(root)
    files = [fl for fl in files if '.csv' in fl and int(fl.split('_')[1]) == 794]
    
    # #animate all data:
    # root = 'dataset/Express4D/data'
    # files = os.listdir(root)
    # files = [fl for fl in files if '.csv' in fl]
    # f = open("/home/dcor/yaronaloni/Express4D/dataset/Express4D/texts_plain.txt", "r")
    # texts = f.readlines()
    # for fl in files:
    #     if int(fl.split('_')[1])>100:
    #         continue
    #     video_path = os.path.join(root,fl).replace('.csv','.mp4')
    #     idx = int(fl.split('_')[1]) - 1
    #     title = texts[idx]
    #     if os.path.exists(video_path):
    #         print(f'{video_path} already exist')
    #         continue
    #     n_array = livelink_csv_to_sequence(os.path.join(root,fl))
    #     tensor = torch.tensor(n_array)
    #     save_scatter_animation(tensor, video_path, title=title)
        # save_incremental_scatter(tensor[0], video_path)
