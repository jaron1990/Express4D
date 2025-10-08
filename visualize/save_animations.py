import os
import numpy as np
import trimesh
from utils.indices import BlendShapeIDX_3d_angles
import  matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
from PIL import Image
from utils.fitting import get_flame_faces
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip

def save_animation(anim, save_path, fps):
    save_path = f"{save_path}"
    print(f"saving [{save_path}]")
    anim.save(save_path, fps=fps)
    # print("\033[A\033[A")
    print(f"done")

def get_meshes_from_states(states, args):
    meshes = []
    flame = FLAME(args)

    for sample in states:
        # flame = flame.to(device=sample.device)
        sample = sample.detach().cpu()
        model = flame(shape_params=sample[:, BlendShapeIDX_3d_angles.shape],
                      expression_params=sample[:, BlendShapeIDX_3d_angles.expression],
                      pose_params=sample[:, BlendShapeIDX_3d_angles.FLAME_pose],
                      neck_pose=sample[:, BlendShapeIDX_3d_angles.neck_pose],
                      eye_pose=sample[:, BlendShapeIDX_3d_angles.eyes_pose],
                      transl=sample[:, BlendShapeIDX_3d_angles.translation])
        mesh = trimesh.Trimesh(model[0][0].detach(), get_flame_faces())
        meshes.append(mesh)

    return meshes

def save_animation_from_meshes(meshes, out_path):
    scene = pyrender.Scene()
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, -0.03, 0.3])
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, 0, 2])
    pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0,  aspectRatio=1.414)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(pc, pose=camera_pose, name='pc-camera')
    scene.add(light, pose=light_pose)

    figsize = (1024, 1024)
    viewer = pyrender.OffscreenRenderer(*figsize)
    fig, ax = plt.subplots()

    def animate(frame_i):
        # ax.images = []
        if frame_i>0:
            for images in ax.images:
                images.remove()

        py_mesh = pyrender.Mesh.from_trimesh(meshes[frame_i])
        mesh_node = scene.add(py_mesh, name="mesh")
        color_img, _ = viewer.render(scene)
        scene.remove_node(mesh_node)
        im = Image.fromarray(color_img)
        ax.imshow(im)
        ax.set_title(f'frame {frame_i}')

    anim = FuncAnimation(fig, animate, interval=1000, frames=len(meshes))
    save_animation(anim, out_path, fps=30)
    plt.close()


def save_animation_from_lmks(lmks_4d, out_path, is_lmk_gt=None, fps=60):
    n_frames = lmks_4d.shape[0]
    lmks_4d = lmks_4d.cpu().detach().numpy()

    if is_lmk_gt is None:
        is_lmk_gt = np.zeros(lmks_4d.shape[:-1], dtype=np.bool)
    else:
        is_lmk_gt = is_lmk_gt.cpu().detach().numpy()
        if is_lmk_gt.ndim==1:
            is_lmk_gt = np.vstack([is_lmk_gt]*lmks_4d.shape[1]).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    xmax = lmks_4d[:,:,0].max()
    ymax = lmks_4d[:,:,1].max()
    zmax = lmks_4d[:,:,2].max()
    xmin = lmks_4d[:,:,0].min()
    ymin = lmks_4d[:,:,1].min()
    zmin = lmks_4d[:,:,2].min()

    scatter_xz = ax1.scatter(lmks_4d[0, :, 0], lmks_4d[0, :, 2], c='b')
    scatter_yz = ax2.scatter(lmks_4d[0, :, 1], lmks_4d[0, :, 2], c='b')


    # ax1.grid()
    # ax2.grid()
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

    def update(frame_i):
        frame_i = min(n_frames - 1, int(frame_i * fps))
        # fig.suptitle(f'{title} — frame {frame_i}')

        # Update both scatter plots
        scatter_xz.set_offsets(np.c_[lmks_4d[frame_i, :, 0], lmks_4d[frame_i, :, 2]])  # X-Z
        scatter_yz.set_offsets(np.c_[lmks_4d[frame_i, :, 1], lmks_4d[frame_i, :, 2]])  # Y-Z

        return mplfig_to_npimage(fig)
        
    # Create animation
    duration = lmks_4d.shape[0] / fps
    anim = VideoClip(update, duration=duration)

    # Write video file
    anim.write_videofile(out_path, fps=fps, threads=4)


    # def animate(frame_i):
    #     if frame_i>0:
    #         for collection in ax1.collections:
    #             collection.remove()
    #         for collection in ax2.collections:
    #             collection.remove()
                
    #     ax1.set_title(f'frame={frame_i}')
    #     ax2.set_title(f'frame={frame_i}')
    #     ax1.scatter(lmks_4d[frame_i,is_lmk_gt[frame_i],0], lmks_4d[frame_i,is_lmk_gt[frame_i],1], c='b')
    #     ax2.scatter(lmks_4d[frame_i,is_lmk_gt[frame_i],2], lmks_4d[frame_i,is_lmk_gt[frame_i],1], c='b')
    #     ax1.scatter(lmks_4d[frame_i,~is_lmk_gt[frame_i],0], lmks_4d[frame_i,~is_lmk_gt[frame_i],1], c='r')
    #     ax2.scatter(lmks_4d[frame_i,~is_lmk_gt[frame_i],2], lmks_4d[frame_i,~is_lmk_gt[frame_i],1], c='r')
        

    # anim = FuncAnimation(fig, animate, interval=1000, frames=lmks_4d.shape[0])
    # save_animation(anim, out_path, fps=30)
    # plt.close()