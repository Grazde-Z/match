#-- coding: UTF-8 --
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import glob
import cv2
import open3d as o3d
import os

def read_trajectory_gt(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析数据并转换为位姿矩阵
    poses = []
    timestamps=[]
    for line in lines:
        elements = line.split()
        if elements[0]=='#':
            continue
        timestamp = float(elements[0])
        translation = np.array([float(elements[1]), float(elements[2]), float(elements[3])])
        rotation = np.array([float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])])
        pose = np.eye(4)
        pose[:3, :3] = quaternion_to_rotation_matrix(rotation)
        pose[:3, 3] = translation
        poses.append(pose)
        timestamps.append(timestamp)   

    return poses,timestamps

def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    rotation_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

    return rotation_matrix

#数据集类型
dataset='ICL' #ICL TUM

# 读取相机位姿真值
pose_file = '/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png/groundtruth.txt'
if dataset=='TUM':
    depth_file='/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop/depth.txt'
    rgb_file='/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop/rgb.txt'
    poses,timestamps = read_trajectory_gt(pose_file)
    poses,depth_file_name,rgb_file_name=select_pose(depth_file,rgb_file,poses,timestamps)
    
    # 获取相机内参
    fx = 481.2
    fy = -480.0
    cx = 319.5
    cy = 239.5
elif dataset=='ICL':
    poses,timestamps = read_trajectory_gt(pose_file)
elif dataset=='CMU':
    poses = read_trajectory_cmu_gt(pose_file)
num_frames = len(poses)

# 获取相机内参
fx = 481.2
fy = -480.0
cx = 319.5
cy = 239.5

# 定义降采样比例
downsample_ratio = 4

# 初始化拼接后的点云
point_cloud = o3d.geometry.PointCloud()
# 遍历所有深度图
for i in range(0,num_frames):

    # 读取深度图
    depth_files = os.path.join('/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png', f"depth/{i+1}.png")
    depth = cv2.imread(depth_files, -1)
    depth = depth /5000.0  # 深度值需要除以5000来得到真实的深度
    # depth = np.load(depth_files)

    # 根据相机内参计算相机坐标下的三维点
    rows, cols = depth.shape
    u, v = np.meshgrid(range(cols), range(rows))
    Z = depth  
    # Z=depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    # print(Z)
    pcd = o3d.geometry.PointCloud()      
    # 将三维点变换到第一帧坐标系下
    # ones = np.ones((1, len(X.flatten())))
    points_3d = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    # distance=np.linalg.norm(points_3d,axis=1)
    # points_3d=points_3d[distance<50]
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    # o3d.visualization.draw_geometries([pcd])
    # R_ENU2NED=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])

    # 获取当前帧相对于第一帧的位姿变换
    RT = poses[i]
    # 将三维点转换到第一帧坐标系下
    # pcd.transform(R_ENU2NED)
    pcd.transform(RT)
    
    # 拼接三维点云
    point_cloud+=pcd
    
    print(f"Processed frame {i+1}/{num_frames}")

# 降采样获取最佳三维点数据
point_cloud = point_cloud.voxel_down_sample(voxel_size=0.03)
best_points = point_cloud.points
# point_cloud = point_cloud[:, ::downsample_ratio]

# 打印最终点云数据
print("Final point cloud shape:", point_cloud)
o3d.visualization.draw_geometries([point_cloud])
o3d.io.write_point_cloud('/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png/Point_cloud.pcd',point_cloud,write_ascii=True,compressed=True)