import re
import os
import numpy as np


def get_sensor_id(traj_name):
    traj_name = traj_name.split('/')
    if traj_name[-1] == '':
        traj_name = traj_name[-2]
    else:
        traj_name = traj_name[-1]
    sensor_id = re.findall(r'\d+', traj_name)[0]
    return int(sensor_id)


def load_settings(traj_name, root_dir):
    sensor_id = get_sensor_id(traj_name)
    settings_file = os.path.join(root_dir, "TUM{}.yaml".format(sensor_id))
    return settings_file


def get_intrinsics(traj_name):
    sensor_id = get_sensor_id(traj_name)

    intrinsics = np.zeros((3, 3), dtype=np.float32)

    if sensor_id == 1:
        intrinsics[0, 0] = 517.3
        intrinsics[1, 1] = 516.5
        intrinsics[0, 2] = 318.6
        intrinsics[1, 2] = 255.3
    elif sensor_id == 2:
        intrinsics[0, 0] = 520.9
        intrinsics[1, 1] = 521.0
        intrinsics[0, 2] = 325.1
        intrinsics[1, 2] = 249.7
    elif sensor_id == 3:
        intrinsics[0, 0] = 535.4
        intrinsics[1, 1] = 539.2
        intrinsics[0, 2] = 320.1
        intrinsics[1, 2] = 247.6
    else:
        print("Unkonwn sensor type")
        intrinsics = None

    return intrinsics


def load_groundtruth(dataset):
    filename = os.path.join(dataset, 'groundtruth.txt')
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    gt_list = [[float(line.split(" ")[0]), line]
               for line in lines if len(line) > 0 and line[0] != "#"]
    return dict([(l[0], l[1]) for l in gt_list])


def find_closest_index(L, t):
    """
    Find the index of the closest value in a list.

    Input:
    L -- the list
    t -- value to be found

    Output:
    index of the closest element
    """
    beginning = 0
    difference = abs(L[0] - t)
    best = 0
    end = len(L)
    while beginning < end:
        middle = int((end+beginning)/2)
        if abs(L[middle] - t) < difference:
            difference = abs(L[middle] - t)
            best = middle
        if t == L[middle]:
            return middle
        elif L[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best


def match_gt_pose(query, gt):
    query_time_stamps = list(query)
    gt_time_stamps = list(gt.keys())
    real_gt_traj = []
    for t_est in query_time_stamps:
        gt_ind = find_closest_index(gt_time_stamps, t_est)
        t_gt = gt[gt_time_stamps[gt_ind]]
        real_gt_traj.append(t_gt)
    return real_gt_traj
