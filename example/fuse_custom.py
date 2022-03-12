
import argparse
import os

import numpy as np
import open3d as o3d
import py_vmapping
import pyrender
import trimesh
from glob import glob
import cv2
from associate import associate, read_file_list
from evaluate_rpe import read_trajectory
from utils import get_intrinsics, match_gt_pose
CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
)

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [
        0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)


def create_camera_actor(g, scale=0.05):
    """build open3d camera polydata"""
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES),
    )

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor




def main(args, map):
    train_cam = []
    test_cam = []
    total_num_imgs = len(glob(os.path.join(args.dataset, "color/*.png")))

    for i in range(total_num_imgs):
        if i % 100 == 0:
            print("registering prgress: {}/{}".format(i, total_num_imgs))

   
   
        rgb = os.path.join(args.dataset, f'color/{i}.png')
        depth = os.path.join(args.dataset, f'depth/{i}.png')
        pose = np.loadtxt(os.path.join(args.dataset, f'pose/{i}.txt'))
        map.load_and_fuse_depth(depth, pose)
        # depth = map.get_depth(pose)[..., 2] * 1000
        # cv2.imwrite(os.path.join(args.dataset, f'depth/{i}.png'), depth.astype(np.uint16))
        if i % 10 == 0:
            if i < 200:
                train_cam += [create_camera_actor(0.3, 0.02).transform(pose)]

            else:
                test_cam += [create_camera_actor(0.9, 0.02).transform(pose)]
       



    print("registration finished...extracting mesh...")


    verts, norms = map.get_polygon()
    verts = verts.reshape(verts.shape[0] // 3, 3)
    norms = norms.reshape(norms.shape[0] // 3, 3)
    faces = np.arange(verts.shape[0])
    faces = faces.reshape(faces.shape[0] // 3, 3)
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=norms).as_open3d
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh]+train_cam+test_cam)
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--sample_method', type=str, default='uniform')
    parser.add_argument('--show_pcd', action='store_true')
    parser.add_argument('--show_mesh', action='store_false')
    parser.add_argument('--est_traj', type=str, default='groundtruth.txt')
    args = parser.parse_args()

    w = 640
    h = 480
    K = np.eye(3)
    K[0,0] = K[1,1] = 585
    K[0,2] = 320
    K[1,2] = 240

    map = py_vmapping.map(w, h, K)
    map.set_depth_scale(1000)
    map.create_map(500000, 450000, 0.008)
    main(args, map)
