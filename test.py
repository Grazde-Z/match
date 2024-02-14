import cv2
import numpy as np
import os
import open3d as o3d
import math
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def read_file(file_path):
    
    poses_path = os.path.join(file_path,f"camera_poses.txt")
    # 读取pose
    with open(poses_path, 'r') as f:
        pose_lines = f.readlines()
        
    points_path = os.path.join(file_path,f"point_landmarks.txt")
    # 读取points
    with open(points_path, 'r') as f:
        point_lines = f.readlines()
        
    lines_path = os.path.join(file_path,f"line_landmarks.txt")
    # 读取lines
    with open(lines_path, 'r') as f:
        line_lines = f.readlines()
        
    association_path = os.path.join(file_path,f"association.txt")
    # 读取association
    with open(association_path, 'r') as f:
        association_lines = f.readlines()
        
    # 解析数据并转换为位姿矩阵
    poses = []
    points = []
    lines=[]
    MappointFrameAsso=[]
    MaplineFrameAsso=[]
    k=0
    paralines={}
    for line in pose_lines:
        elements = line.split()
        translation = np.array([float(elements[1]), float(elements[2]), float(elements[3])])
        rotation = np.array([float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])])
        # translation = np.array([float(elements[0]), float(elements[1]), float(elements[2])])
        # rotation = np.array([float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])])
        pose = np.eye(4)
        pose[:3, :3] = quaternion_to_rotation_matrix(rotation)
        # print(-pose[:3, :3].T)
        # print(np.transpose(translation))
        pose[:3, 3] = translation
        poses.append(pose) 
    for point_line in point_lines:
        elements = point_line.split()
        point=np.array([float(elements[1]),float(elements[2]), float(elements[3]), float(elements[4])])
        points.append(point)
    for line_line in line_lines:
        elements = line_line.split()
        Mapline=np.array([float(elements[0]),float(elements[1]), float(elements[2]), float(elements[3]),float(elements[4]), float(elements[5]), float(elements[6])])
        lines.append(Mapline)
    for association_line in association_lines:
        k+=1
        elements = association_line.split()
        # if elements[0]=='MappointFrameAsso:':
        #     frame=np.array([float(elements[1]),float(elements[2]),float(elements[3]), float(elements[4])])
        #     # key=tuple(points[int(elements[1])].tolist())
        #     if len(MappointFrameAsso)==0:
        #         MappointFrameAsso=[frame]
        #     else:
        #         MappointFrameAsso=np.vstack((MappointFrameAsso,frame))

        if elements[0]=='MaplineFrameAsso:':
            frame=np.array([float(elements[1]),float(elements[2]),float(elements[3]), float(elements[4]),float(elements[5]), float(elements[6]), float(elements[7]),float(elements[8])])
            # key=tuple(lines[int(elements[1])].tolist())
            if len(MaplineFrameAsso)==0:
                MaplineFrameAsso=[frame]
            else:
                MaplineFrameAsso=np.vstack((MaplineFrameAsso,frame))

    return poses,points,lines,MappointFrameAsso,MaplineFrameAsso

def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    rotation_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

    return rotation_matrix

# 通过透视投影将三维点云投影到相机坐标系
def project_point_cloud(match_point_3d,pose):
    matches = []
    flag=True
    trans=np.dot(-pose[:3, :3].T,pose[:3, 3])

    one=np.ones(len(match_point_3d))
    point=np.hstack((match_point_3d,1))
    Tra=np.column_stack((pose[:3, :3].T,trans))
    
    Tra=np.vstack((Tra,np.array([0,0,0,1])))
    # projected_points = np.dot(pose[:3, :3].T, point_cloud.T) + pose[:3, 3]
    projected_points_d=np.dot(Tra,point.T)
    
    projected_points = projected_points_d[:2] / projected_points_d[2]
    pixel_u = fx * projected_points[0] + cx*np.ones(1)
    pixel_v = fy * projected_points[1] + cy*np.ones(1)
    
    pixel_coords=np.array((pixel_u,pixel_v))

    return pixel_coords, projected_points_d

  
# 读取三维点云数据
def read_point_cloud(point_cloud_path):
    return np.loadtxt(point_cloud_path)

# 读取RGB图像
def read_image(image_path):
    return cv2.imread(image_path,cv2.IMREAD_UNCHANGED)

#ICL
fx = 481.2
fy = -480.0
cx = 319.5
cy = 239.5

#cmu
# fx = 320.0  # focal length x
# fy = 320.0  # focal length y
# cx = 320.0  # optical center x
# cy = 240.0  # optical center y
      
# 主函数
def main():
    datafile = '/home/arondight/G.Z/Open-structure/output/datasets/office3'
    # cloud_file='/home/arondight/G.Z/Open-structure/examples/ICL-NUIM/lrkt2/Point_cloud.txt'
    
    poses,points,Line_3d,MappointFrameAsso,MaplineFrameAsso=read_file(datafile) 

    # keys_point=list(MappointFrameAsso.keys())
    
    error=0
    line_num=0
    k=0
    error_line=0
    # for k in range(len(points)):
    #     for i in range(len(MappointFrameAsso)):
    #         if MappointFrameAsso[i][0]==points[k][0]:
    #             pose=poses[int(MappointFrameAsso[i][1])]
    #             trans=np.dot(-pose[:3, :3].T,pose[:3, 3])
    #             point=np.hstack((points[k][1:4],1))
    #             Tra=np.column_stack((pose[:3, :3].T,trans))                
    #             Tra=np.vstack((Tra,np.array([0,0,0,1])))
    #             # projected_points = np.dot(pose[:3, :3].T, point_cloud.T) + pose[:3, 3]
    #             projected_points_d=np.dot(Tra,point.T)                
    #             projected_points = projected_points_d[:2] / projected_points_d[2]
    #             pixel_u = fx * projected_points[0] + cx*np.ones(1)
    #             pixel_v = fy * projected_points[1] + cy*np.ones(1) 
    #             pixel_coords=np.array((pixel_u,pixel_v))
    #             pix=np.array((MappointFrameAsso[i][2],MappointFrameAsso[i][3]))

    #             x=(pixel_coords.T)[0]-pix.T
    #             error+=(pixel_coords.T)[0]-pix.T
    #         else:
    #             break

    for k in range(len(Line_3d)):
        for i in range(len(MaplineFrameAsso)):
            if MaplineFrameAsso[i][0]==Line_3d[k][0]:
                pose=poses[int(MaplineFrameAsso[i][1])]
                pw1,d1=project_point_cloud(Line_3d[k][1:4],pose)
                pw2,d2=project_point_cloud(Line_3d[k][4:7],pose)
                direction=pw2-pw1
                direction/=np.linalg.norm(direction)
                pix1=MaplineFrameAsso[i][2:4]
                pix2=MaplineFrameAsso[i][4:6]
                direction_c=pix2-pix1
                direction_c/=np.linalg.norm(direction_c)
                cos=np.dot(direction_c,direction)
                pix1=np.array((pix1[0],pix1[1]))
                pw1=np.array((pw1[0][0],pw1[1][0]))
                direction=np.array((direction[0][0],direction[1][0]))               
                distance=np.linalg.norm(np.cross(direction, pix1-pw1))
                error_line+=distance
                print(distance)
                # error_line+=pw1-pix1
                # error_line+=pw2-pix2
    print(error)
    print(error_line)
    # points_num = np.zeros((len(poses)))
    # lines_num = np.zeros((len(poses)))
    # for i in range(len(MappointFrameAsso)):
    #     points_num[MappointFrameAsso[i][2]]+=1
    # for i in range(len(MaplineFrameAsso)):
    #     lines_num[MaplineFrameAsso[i][2]]+=1
    # print(points_num)
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(points)
    # cloud = cloud.voxel_down_sample(voxel_size=0.3)
    # draw_line_points=[]
    # line_indice=[]
    # k=0
    # for i in range(len(Line_3d)):
    #     draw_line_points.append(Line_3d[i][0:3])
    #     draw_line_points.append(Line_3d[i][3:6])
    #     line_indice.append([2*k,2*k+1])
    #     k+=1    

    # lines_set = o3d.geometry.LineSet()
    # lines_set.points = o3d.utility.Vector3dVector(draw_line_points)
    # lines_set.lines = o3d.utility.Vector2iVector(line_indice)

    # # 创建可视化窗口
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # 添加点云和线段到窗口
    # # vis.add_geometry(cloud)
    # vis.add_geometry(lines_set) 
    # # vis.add_geometry(rgbd_pcd)
    # # vis.add_geometry(cylinder)   
    # render=vis.get_render_option()    
    # # render.line_width=5.0
    # # render.point_size=3
    # render.show_coordinate_frame=True

    # # 显示可视化窗口
    # vis.update_renderer()
    
    # vis.run()
    # vis.destroy_window()
    
# 运行主函数
if __name__ == "__main__":
    main()