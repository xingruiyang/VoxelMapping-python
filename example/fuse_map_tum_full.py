
import argparse
import os

import numpy as np
import open3d as o3d
import py_vmapping
import pyrender
import trimesh

from associate import associate, read_file_list
from evaluate_rpe import read_trajectory
from utils import get_intrinsics, match_gt_pose


def main(args, map):
    dataset = args.dataset
    rgb_file_list = os.path.join(dataset, 'rgb.txt')
    depth_file_list = os.path.join(dataset, 'depth.txt')
    rgb_files = read_file_list(rgb_file_list)
    depth_files = read_file_list(depth_file_list)

    offset = 0
    max_difference = 0.02
    matches = associate(rgb_files, depth_files, offset, max_difference)
    total_gt_traj = read_trajectory(os.path.join(dataset, args.est_traj))

    rgb_ts, depth_ts = zip(*matches)
    camera_traj = match_gt_pose(rgb_ts, total_gt_traj)

    total_num_imgs = len(matches)

    for i in range(total_num_imgs):
        if i % 100 == 0:
            print("registering prgress: {}/{}".format(i, total_num_imgs))
        rgb_ts, depth_ts = matches[i]
        rgb = ' '.join(rgb_files[rgb_ts])
        depth = ' '.join(depth_files[depth_ts - offset])

        rgb = os.path.join(dataset, rgb)
        depth = os.path.join(dataset, depth)
        pose = camera_traj[i].astype(np.float32)

        map.load_and_fuse_depth(depth, pose)

    print("registration finished...extracting mesh...")

    if args.show_pcd:
        if args.sample_method == 'uniform':
            # uniform downsample
            points = map.get_surface_points()
            points = np.array(points)
            print(points.shape)
            indices = np.arange(points.shape[0])
            indices = np.random.choice(indices, 8192, replace=True)
            points = points[indices, :]
            trimesh.PointCloud(points).show()
        elif args.show_pcd:
            # voxel downsample
            points = map.get_surface_points()
            points = np.array(points)
            print(points.shape)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            points = pcd.voxel_down_sample(0.05)
            points = np.asarray(points.points)
            print(points.shape)
            trimesh.PointCloud(points).show()

    if args.show_mesh:
        verts, norms = map.get_polygon()
        verts = verts.reshape(verts.shape[0] // 3, 3)
        norms = norms.reshape(norms.shape[0] // 3, 3)
        faces = np.arange(verts.shape[0])
        faces = faces.reshape(faces.shape[0] // 3, 3)
        mesh = trimesh.Trimesh(verts, faces, vertex_normals=norms)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        light = pyrender.PointLight(color=np.ones(3), intensity=3.0)
        scene.add(light, np.eye(4))
        pyrender.Viewer(scene, viewer_flags={'use_direct_lighting': True})


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
    intrinsics = get_intrinsics(args.dataset)

    map = py_vmapping.map(w, h, intrinsics)
    map.set_depth_scale(5000)
    map.create_map(500000, 450000, 0.01)
    main(args, map)
