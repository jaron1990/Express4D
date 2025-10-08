import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import clips_array


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=None):
    gt_frames = gt_frames if gt_frames is not None else []
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        # ax.lines = []
        # ax.collections = []
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()


def plot_3d_motion_moviepy(kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                           vis_mode='default', gt_frames=None, save_path: str = None, plot_traj=False):
    gt_frames = gt_frames if gt_frames is not None else []
    matplotlib.use('Agg')
    title = '\n'.join(wrap(title, 20))


    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5  # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize) # , layout="constrained")
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        # ax.lines = []
        # ax.collections = []
        index = int(index*fps)
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 
                     0, 
                     MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)
        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent

        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)
    plt.close()
    return ani


###################################################### 

def plot_3d_motion_moviepy_two_motions(kinematic_tree, joints1, title, dataset, figsize=(3, 3), fps=120, radius=3,
                                       vis_mode='default', gt_frames=None, joints2=None):
    gt_frames = gt_frames if gt_frames is not None else []
    matplotlib.use('Agg')
    title = '\n'.join(wrap(title, 20))


    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data1 = joints1.copy().reshape(len(joints1), -1, 3)
    data2 = joints2.copy().reshape(len(joints2), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data1 *= 0.003  # scale for visualization
        data2 *= 0.003
    elif dataset == 'humanml':
        data1 *= 1.3  # scale for visualization
        data2 *= 1.3
    elif dataset in ['humanact12', 'uestc']:
        data1 *= -1.5  # reverse axes, scale for visualization
        data2 *= -1.5

    fig = plt.figure(figsize=figsize, layout="constrained")
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = np.minimum(data1.min(axis=0).min(axis=0), data2.min(axis=0).min(axis=0))
    MAXS = np.maximum(data1.max(axis=0).max(axis=0), data2.max(axis=0).max(axis=0))
    # MINS = np.concatenate((data1, data2)).min(axis=0).min(axis=0)
    # MAXS = np.concatenate((data1, data2)).max(axis=0).max(axis=0)
    # MINS = data1.min(axis=0).min(axis=0)
    # MAXS = data1.max(axis=0).max(axis=0)

    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data1.shape[0]
    height_offset = MINS[1]
    data1[:, :, 1] -= height_offset
    data2[:, :, 1] -= height_offset
    trajec1 = data1[:, 0, [0, 2]]  # root1 x and z (y is up)
    trajec2 = data2[:, 0, [0, 2]]  # root2 x and z (y is up) 
    
    
    if False:
        data1[..., 0] -= data1[:, 0:1, 0]
        data1[..., 2] -= data1[:, 0:1, 2]

        data2[..., 0] -= data2[:, 0:1, 0]
        data2[..., 2] -= data2[:, 0:1, 2]

    def update(index):
        # ax.lines = []
        # ax.collections = []
        index = int(index*fps)
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.view_init(elev=20, azim=-45, vertical_axis='y')
        ax.dist = 7.5
        ax.dist = 12
        # plot_xzPlane(minx=MINS[0] - trajec1[index, 0], 
        #              maxx=MAXS[0] - trajec1[index, 0], 
        #              miny=0, 
        #              minz=MINS[2] - trajec1[index, 1], 
        #              maxz=MAXS[2] - trajec1[index, 1])
        plot_xzPlane(minx=MINS[0], 
                     maxx=MAXS[0], 
                     miny=0, 
                     minz=MINS[2], 
                     maxz=MAXS[2])
        
        # plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 2], MAXS[2] - trajec[index, 2])

        
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)
        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth, color=colors_orange[i])
            ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=colors_blue[i])

        for sub_ind in range(0, index):
            ax.plot3D(
                [data1[sub_ind, 0, 0], data1[sub_ind+1, 0, 0]],
                [data1[sub_ind, 0, 1], data1[sub_ind+1, 0, 1]],
                [data1[sub_ind, 0, 2], data1[sub_ind+1, 0, 2]],
                'gray'
                        )
        
        main_joints = {
            'root': 0,
            'wirst_r': 21,
            'wrist_l': 20,
            'ankle_r': 11,
            'ankle_l': 7,
        }
        for data, c in zip([data1, data2], ['orange', 'blue']):
            for joint_name, joint_ind in main_joints.items():
                for sub_ind in range(0, index):
                    ax.plot3D(
                        [data[sub_ind, joint_ind, 0], data[sub_ind+1, joint_ind, 0]],
                        [data[sub_ind, joint_ind, 1], data[sub_ind+1, joint_ind, 1]],
                        [data[sub_ind, joint_ind, 2], data[sub_ind+1, joint_ind, 2]],
                        c
                                )



        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)
    plt.close()
    return ani

############################################ 






# Originally copied from https://github.com/sigal-raab/motion-diffusion-model-sigal
def plot_3d_motion_cross_attention(attention_dict, save_path, kinematic_tree, joints, title, dataset, figsize=(5, 5), fps=120, radius=3,
                                   vis_mode='default', gt_frames=None):
    gt_frames = gt_frames if gt_frames is not None else []
    matplotlib.use('Agg')

    no_color = np.array(mcolors.to_rgba('gray')[:3])
    full_color_name = list(mcolors.BASE_COLORS.keys())[attention_dict['color_idx']]
    full_color = np.array(mcolors.to_rgba(full_color_name)[:3])

    color_per_frame = np.concatenate([(((1.-e) * no_color) + (e * full_color))[None] for e in attention_dict['map_per_frame']], axis=0)

    title = '\n'.join(wrap(title, 40))
    n_frames = joints.shape[0]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # fig.suptitle(title, fontsize=10)  # , y=1+0.025*title.count('\n'))
        ax.set_title(title, fontsize=10)
        ax.grid(b=False)

        ax2.bar(range(n_frames), height=1, color=color_per_frame, width=1, align='edge', edgecolor="none")
        # ax2.set_xticks(range(0, n_frames, n_frames // 5))
        ax2.set_xticks([])
        # ax2.set_xlabel('time (frames)')
        ax2.set_xlabel(attention_dict['keyword'], fontweight='bold')
        ax2.yaxis.set_visible(False)
        ax2.xaxis.label.set_color(full_color_name)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        # ax2.axis('off')
        # ax2.tick_params(bottom=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5  # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(20)
    plt.tight_layout()
    # ax = fig.add_subplot(2, 1, 2, projection='3d')
    # ax2 = fig.add_subplot(2, 1, 1)
    ax = fig.add_subplot(gs[3:-1], projection='3d')  # animation
    ax2 = fig.add_subplot(gs[1])  # color bar
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        used_colors = np.tile(color_per_frame[index], (len(kinematic_tree), 1))
        # used_colors = colors_blue if index in gt_frames else feat_all_joints_colors[index] if feat_all_joints_pca is not None else colors

        #  plot all edges according to kinematic chains
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #  plot all vertices according to deep features

        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # indicate time marker on colorbar
        if index > 0:
            ax2.axvline(x=index - 1, color=color_per_frame[index], ymax=1)  # delete previous time marker
        ax2.axvline(x=index, color='black', ymax=1)  # draw time marker

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)  # , savefig_kwargs={'bbox_inches':'tight'})

    plt.close()


def test_plot3d_moviepy():
    skeleton = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    motion_path = '/home/dcor/roeyron/motion-latent-diffusion-pfork/results/mld/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2024-03-07-15-14-17/sample_00_rep_00_src_joints.npy'
    # motion_path = '/home/dcor/roeyron/tmp/motion_for_test_plot.npy'
    motion = np.load(motion_path)

    save_path = './animations_gallery.mp4'

    n_rows = 1
    n_cols = 2
    animations = np.empty(shape=(n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            animations[i, j] = plot_3d_motion_moviepy(kinematic_tree=skeleton, joints=motion, title='my title', dataset='humanml', fps=20)
    clips = clips_array(animations)
    clips.duration = 6
    clips.write_videofile(save_path, fps=20, threads=4, logger=None)
    for clip in clips.clips:
        clip.close()
    clips.close()



if __name__ == "__main__":
    test_plot3d_moviepy()
    # render_from_joints_numpy()
