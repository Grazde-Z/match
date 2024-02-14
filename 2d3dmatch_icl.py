import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import os
import open3d as o3d
import math
import random
from scipy.spatial import KDTree
# from sklearn.linear_model import RANSACRegressor

Line_3d=[]      #三维地图线集
Point_3d=[]     #三维地图点集
Maplineasso={}  #三维地图线-观测帧关系集
Mappointasso={} #三维地图点-观测帧关系集

#select datasets
dataset_select=0 #0-ICL 1-cmu 2-tum

if dataset_select==0:
    # 获取icl/tum相机内参
    fx = 481.2
    fy = -480.0
    cx = 319.5
    cy = 239.5
    camera_matrix = np.array([[fx, 0, cx],
                [0, fy, cy],
            [0, 0, 1]])  # 内部参数矩阵
    dist_coeff = np.array([	0,0,0,0,0 ])
elif dataset_select==1:
    # 获取cmu相机内参
    fx = 320.0 
    fy = 320.0
    cx = 320.0
    cy = 240.0
elif dataset_select==2:
    # 获取tum相机内参
    # fx = 535.4 
    # fy = 539.2 
    # cx = 320.1 
    # cy = 247.6 
    #fr1
    fx=517.306408
    fy=516.469215
    cx=318.643040
    cy=255.313989
    camera_matrix = np.array([[fx, 0, cx],
                    [0, fy, cy],
                [0, 0, 1]])  # 内部参数矩阵
    dist_coeff = np.array([	0.2624,-0.9531,-0.0054,0.0026,1.1633 ])

def select_pose(depth_file,rgb_file,poses,pose_Q,timestamps):
    depth_file_name=[]
    rgb_file_name=[]
    depth_time=[]
    rgb_time=[]
    with open(depth_file, 'r') as file:
        depth_lines = file.readlines()
        for line in depth_lines:
            elements = line.split()
            if elements[0]=='#':
                continue
            depth_time.append(float(elements[0]))
            depth_file_name.append(elements[1])
    with open(rgb_file, 'r') as file:
        rgb_lines = file.readlines()
        for line in rgb_lines:
            elements = line.split()
            if elements[0]=='#':
                continue
            rgb_time.append(float(elements[0]))
            rgb_file_name.append(elements[1])

    posess=[]
    posess_Q=[]
    depth_file_names=[]
    time=[]   
    for i in range(len(depth_time)):
        for j in range(len(timestamps)):
            if abs(timestamps[j]-depth_time[i])<0.006:
                posess.append(poses[j])
                posess_Q.append(pose_Q[j])
                depth_file_names.append(depth_file_name[i])
                time.append(depth_time[i])
                break
    posess2=[]
    posess_Q2=[]
    depth_file_names2=[]
    rgb_file_names=[]
    output_time=[]
    for i in range(len(time)):
        for j in range(len(rgb_time)):
            if abs(rgb_time[j]-time[i])<0.006:
                posess2.append(posess[i])
                posess_Q2.append(posess_Q[i])
                depth_file_names2.append(depth_file_names[i])
                rgb_file_names.append(rgb_file_name[j])
                output_time.append(time[i])
                break

    return posess2,posess_Q2,depth_file_names2,rgb_file_names,output_time


def matrix2quaternion(m):
    #m:array
    w = ((np.trace(m) + 1) ** 0.5) / 2
    x = (m[2][1] - m[1][2]) / (4 * w)
    y = (m[0][2] - m[2][0]) / (4 * w)
    z = (m[1][0] - m[0][1]) / (4 * w)
    return x,y,z,w

#读取CMU位姿真值
def read_trajectory_cmu_gt(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析数据并转换为位姿矩阵
    poses = []
    pose_qua = []
    timestamps=[]
    for line in lines:
        elements = line.split()
        if elements[0]=='#':
            continue
        timestamp = float(elements[0])
        # translation = np.array([float(elements[1]), float(elements[2]), float(elements[3])])
        # rotation = np.array([float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])])
        translation = np.array([float(elements[0]), float(elements[1]), float(elements[2])])
        rotation = np.array([float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])])

        pose = np.zeros((3,4))
        R_ENU2NED=np.array([[0,0,1],[1,0,0],[0,1,0]])
        pose[:3, :3] = np.dot(quaternion_to_rotation_matrix(rotation),R_ENU2NED)
        pose[:3, 3] = translation
        # pose_Q=np.array([float(elements[0]), float(elements[1]), float(elements[2]),float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])])
        Q_x,Q_y,Q_z,Q_w=matrix2quaternion(pose[:3, :3])
        pose_Q=np.array([float(elements[0]), float(elements[1]), float(elements[2]),float(Q_x), float(Q_y), float(Q_z), float(Q_w)])
        poses.append(pose)   
        pose_qua.append(pose_Q)
        timestamps.append(timestamp)
    return poses,pose_qua

#读取ICL位姿真值
def read_trajectory_icl_gt(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析数据并转换为位姿矩阵
    poses = []
    pose_qua = []
    timestamps=[]
    for line in lines:
        elements = line.split()
        if elements[0]=='#':
            continue
        timestamp = float(elements[0])
        translation = np.array([float(elements[1]), float(elements[2]), float(elements[3])])
        rotation = np.array([float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])])
        pose = np.zeros((3,4))
        pose[:3, :3] = quaternion_to_rotation_matrix(rotation)
        pose[:3, 3] = translation
        pose_Q=np.array([float(elements[1]), float(elements[2]), float(elements[3]),float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])])
        poses.append(pose)   
        pose_qua.append(pose_Q)
        timestamps.append(timestamp)
    return poses,pose_qua,timestamps

def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    rotation_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

    return rotation_matrix

# 读取RGB图像
def read_image(image_path):
    return cv2.imread(image_path)

# 创建ORB特征点检测器
def create_detector():
    return cv2.ORB_create()

# 提取特征点和描述子
def extract_features(detector, image):
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

# 读取三维点云数据
def read_point_cloud(point_cloud_path):
    return np.loadtxt(point_cloud_path)

# 读取位姿真值
def read_poses(poses_path):
    return np.load(poses_path)

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
    # distance =pixel_coords.T-match_point_2d
    # if abs(distance[0][0])<=2 and abs(distance[0][1])<=2 :
    #     flag=False
    # return flag, pixel_coords

# 定义一个函数来反投影特征点到三维空间
def backproject_2d_to_3d(keypoints,point_2d, pose,point_cloud,key_point_id,delete):

    point_2d=np.array(point_2d)
    one=np.ones(len(point_2d))
    point_2d = np.column_stack((point_2d, one))
    # 计算世界坐标系下的3D点
    points_world = np.dot(pose, point_2d.T).T[:, :3]
    
    #如果是直线匹配就直接投影回到图像平面
    if delete==False:
        matches_3d = []
        matches_2d = []
        matche_3d=[]
        matche_2d=[]
        projected_depth=[]
        for i in range(len(points_world)):
            matche_2d.append((np.append(keypoints[i][0],keypoints[i][1])))
            matche_3d.append(np.array([keypoints[i][0],keypoints[i][1],points_world[i][0],points_world[i][1],points_world[i][2]]))
            if (i+1)%11==0:
                matches_3d.append(matche_3d)
                matches_2d.append(matche_2d)
                matche_3d=[]
                matche_2d=[]
        return matches_3d,projected_depth,matches_2d

    # 使用KD树寻找最近邻的三维点
    kdtree = KDTree(point_cloud)
    _, indices = kdtree.query(points_world)

    matches_3d = []
    matches_2d = []
    state=np.ones(len(indices))
    k=0
    res=[]
    projected_depth=[]
    matche_3d=[]
    matche_2d=[]
    match=[]
    pix_origin_2d=[]
    for i in range(len(indices)):
        flag=True
        match_point_2d = keypoints[i]
        match_point_3d = point_cloud[indices[i], :]
        
        pixel_coords,projected_points_d=project_point_cloud(match_point_3d,pose)
        
        distance =pixel_coords.T-match_point_2d
        
        if abs(distance[0][0])<=2 and abs(distance[0][1])<=2 :
            flag=False
            
        if (flag or (indices[i] in res)):
            # if (i+1)%11==0:
            #     match=[]
            #     pix_origin_2d=[]
            #     k=0
            continue
    
        res.append((indices[i]))
        projected_depth.append((projected_points_d))
        matches_3d.append((np.append(pixel_coords,match_point_3d)))
        match.append(np.array([pixel_coords[0][0],pixel_coords[1][0],match_point_3d[0],match_point_3d[1],match_point_3d[2]]))
        pix_origin_2d.append(match_point_2d)
        matches_2d.append(match_point_2d)
        key_point_id.append((i,indices[i]))
        # if (i+1)%11==0:
        #     matche_3d.append(match)
        #     matche_2d.append(pix_origin_2d)
        #     match=[]
        #     pix_origin_2d=[]
        #     k=0
            
        # k=k+1
        
    # if delete==False:
    #     matches_3d=matche_3d
    #     matches_2d=matche_2d
        # print(matches_3d[1][1])
        # cv2.circle(img,(int(pixel_coords[0]),int(pixel_coords[1])),2,(0,255,0),-1)
    return matches_3d,projected_depth,matches_2d
#2D-3D线匹配
def Line_extractor2d(img,gray,depth):
    match_line=[]

    fld=cv2.ximgproc.createFastLineDetector()
    lines=fld.detect(gray)
    # print(lines.shape)
    if lines is None:
        return []
    for dline in lines:
        line=[]
        x0= int(round(dline[0][0]))
        y0= int(round(dline[0][1]))
        x1= int(round(dline[0][2]))
        y1= int(round(dline[0][3]))
        z0=depth[y0][x0]
        z1=depth[y1][x1]
        p2d=np.array([float(x0),float(y0),float(x1),float(y1)])
        normalized_p2d=cv2.undistortPoints(p2d.reshape(-1,1,2),camera_matrix,dist_coeff)
        x0=int(round(normalized_p2d[0][0][0]*fx+cx))
        y0=int(round(normalized_p2d[0][0][1]*fy+cy))
        x1=int(round(normalized_p2d[1][0][0]*fx+cx))
        y1=int(round(normalized_p2d[1][0][1]*fy+cy))
        dist=np.sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1))
        if dist > 100:
            match_line.append((x0,y0,x1,y1))
            cv2.line(img,(x0,y0),(x1,y1),(100,0,0),2,cv2.LINE_AA)
            # cv2.imshow("FLD",img)
    # cv2.imwrite("target_line_pre.png",img)
    # cv2.imshow("FLD",img)
    # cv2.waitKey(0)
    # print(len(match_line))
    return match_line

#RANSAC拟合直线
def ransac_lines(points, iterations, threshold):
    best_params = None
    best_inliers = None
    for _ in range(iterations):
        # 随机选择两个点作为模型参数
        sample_indices = np.random.choice(len(points), 2, replace=True)
        sample_points = points[sample_indices]

        # 拟合直线模型
        direction = sample_points[1] - sample_points[0]
        direction /= np.linalg.norm(direction)
        # normal = np.cross(direction, np.array([0, 0, 1]))

        # 计算距离
        # distances = np.abs(np.dot(points - sample_points[0], normal))
        distances = np.linalg.norm(np.cross(points - sample_points[0], direction),axis=1)
        # 根据阈值划分内点和外点

        inliers = points[distances < threshold]
        num_inliers = len(inliers)

        # 更新最好的模型参数和内点
        if best_inliers is None:
            best_inliers = inliers
            best_params = (sample_points[0], direction)
        else:
            if num_inliers > len(best_inliers):
                best_inliers = inliers
                best_params = (sample_points[0], direction)

    dis=np.dot(best_inliers - best_params[0], best_params[1])
    indices=np.argsort(dis)
    best_inliers=best_inliers[indices]
    
    return best_params, best_inliers

#判断是否为三维空间同一直线
def issameLine(point):
    
    global Line_3d
    
    #计算方向向量
    P0=np.array((point[0],point[1],point[2]))
    P1=np.array((point[3],point[4],point[5]))
    direction=P1-P0
    direction /= np.linalg.norm(direction)
    flag=False
    index=-1
    for i, Line in enumerate(Line_3d):
        P2=np.array((Line[0],Line[1],Line[2]))
        P3=np.array((Line[3],Line[4],Line[5]))
        direction_old= P3-P2

        direction_old /= np.linalg.norm(direction_old)
        L12=P2-P0
        normal_cross = np.cross(direction, direction_old)
        normal_cross/=np.linalg.norm(normal_cross)
        d02=np.linalg.norm(np.cross(direction, P0-P2))
        d03=np.linalg.norm(np.cross(direction, P0-P3))
        
        dis=[]
        dis.append(np.linalg.norm(P2-P0))
        dis.append(np.linalg.norm(P3-P0))
        dis.append(np.linalg.norm(P2-P1))
        dis.append(np.linalg.norm(P3-P1))
        dis=sorted(dis)
        
        dir1=P2-P0
        dir1/=np.linalg.norm(dir1)
        dir2=P3-P0
        dir2/=np.linalg.norm(dir2)
        cos23_0=np.dot(dir1,dir2)
        
        dir3=P2-P1
        dir3/=np.linalg.norm(dir3)
        dir4=P3-P1
        dir4/=np.linalg.norm(dir4)
        cos23_1=np.dot(dir3,dir4)
        
        if  np.linalg.norm(L12)==0 and np.linalg.norm(P3-P1)==0:
            flag=True
            index=i
            break
        
        min_distance=np.linalg.norm(np.cross(L12,direction))
        cos=np.dot(direction,direction_old)
        points=[]
        #判断两直线方向余弦与间距
        if abs(cos)>0.99:
            if d02>0.05 and d03>0.05:
                continue
            if cos23_0>0.9 and cos23_1>0.9:
                continue
            #设置迭代次数与阈值
            iterations=10
            threshold=0.05
            points=np.concatenate(([P0],[P1],[P2],[P3]))
            
            best_params, best_inliers=ransac_lines(points, iterations, threshold)
            
            #内点数大于3
            if len(best_inliers)>3 :
                flag=True
                index=i
                temp=np.concatenate((best_inliers[0],best_inliers[len(best_inliers)-1]))
                Line_3d[i][0]=temp[0]
                Line_3d[i][1]=temp[1]
                Line_3d[i][2]=temp[2]
                Line_3d[i][3]=temp[3]
                Line_3d[i][4]=temp[4]
                Line_3d[i][5]=temp[5]
                break
    return flag,index

#去除偏差线
def Delete_outlines(pix1,pix2,pwc1,pwc2):
   
    flag=True
    direction1=np.append(pix1[0],pix1[1])-np.append(pix2[0],pix2[1])
    direction2=np.append(pwc1[0],pwc1[1])-np.append(pwc2[0],pwc2[1])
    direction1/=np.linalg.norm(direction1)
    direction2/=np.linalg.norm(direction2)
    cos=np.dot(direction1,direction2)
    L12=np.append(pwc1[0],pwc1[1])-np.append(pix1[0],pix1[1])
    normal_cross = np.cross(direction1, L12)
    min_distance=np.linalg.norm(normal_cross)
    # print(min_distance)
    if abs(cos)<0.999 or min_distance>3:
        flag=False    
    return flag

#判断直线两端点是否超出图像边界
def is_inside_image(point):
    return 0<=point[0]<=639 and 0<=point[1]<=479

#截取超过边界的直线
def is_outside_image(point1_projected,direction_vector):
    # 计算线段与图像边界的交点
    intersection_points = []

    # 与左边界相交
    if point1_projected[0] < 0 and direction_vector[0] != 0:
        t = -point1_projected[0] / direction_vector[0]
        intersection_points.append(point1_projected + t * direction_vector)

    # 与右边界相交
    if point1_projected[0] >= 640 and direction_vector[0] != 0:
        t = (640 - 1 - point1_projected[0]) / direction_vector[0]
        intersection_points.append(point1_projected + t * direction_vector)

    # 与上边界相交
    if point1_projected[1] < 0 and direction_vector[1] != 0:
        t = -point1_projected[1] / direction_vector[1]
        intersection_points.append(point1_projected + t * direction_vector)

    # 与下边界相交
    if point1_projected[1] >= 480 and direction_vector[1] != 0:
        t = (480 - 1 - point1_projected[1]) / direction_vector[1]
        intersection_points.append(point1_projected + t * direction_vector)

    if len(intersection_points) > 0:
        # 找到最近的交点
        intersection_distances = [np.linalg.norm(point1_projected - p) for p in intersection_points]
        max_distance_index = np.argmax(intersection_distances)
        closest_intersection_point = intersection_points[max_distance_index]

        # 使用交点作为截断后的线段端点
        truncated_point1 = closest_intersection_point
        truncated_point2 = point1_projected + (closest_intersection_point - point1_projected)
    else:
        closest_intersection_point=[]
    return closest_intersection_point

def Line_projection(point,point_asso,frame_id,pose,depth,line_index):
    
    pc1=np.append(point[0][0],point[0][1])
    pc2=np.append(point[1][0],point[1][1])
    # print(point_asso)
    pw1,d1=project_point_cloud(point_asso[0:3],pose)
    pw2,d2=project_point_cloud(point_asso[3:6],pose)
    pw1=np.append(pw1[0],pw1[1])
    pw2=np.append(pw2[0],pw2[1])
    direction_line=pw2-pw1
    direction_line/=np.linalg.norm(direction_line)
    
    #排除投影后方向与距离相差过大的直线
    flag=Delete_outlines(pc1,pc2,pw1,pw2)
    frame=[]
    if flag==True:
        # direction_1=pc1-pw1
        # d_1=np.dot(direction_1,direction_line)
        # pix1=pw1+d_1*direction_line
        # direction_2=pc2-pw1
        # d_2=np.dot(direction_2,direction_line)
        # pix2=pw1+d_2*direction_line      
             
        # if is_inside_image(pix1)==False:
        #     closet_p1=is_outside_image(pix1,direction_line)
        #     if len(closet_p1)>0:
        #         pix1=closet_p1
        # if is_inside_image(pix2)==False:
        #     closet_p2=is_outside_image(pix2,direction_line)
        #     if len(closet_p2)>0:
        #         pix2=closet_p2
                
        # dir_line=d2-d1
        # distance_3d=np.linalg.norm(dir_line)
        # distance_2d=np.linalg.norm(pw2-pw1)
        # dir_line/=np.linalg.norm(dir_line)
        # p1c_3d=d1+dir_line*(distance_3d*(np.linalg.norm(pix1-pw1)/distance_2d))
        # p2c_3d=d1+dir_line*(distance_3d*(np.linalg.norm(pix2-pw1)/distance_2d))
        # d1=p1c_3d[2]
        # d2=p2c_3d[2]
        # d1=depth[int(pix1[1])][int(pix1[0])]
        # d2=depth[int(pix2[1])][int(pix2[0])]
        sample_dir=(d2-d1)/100
        sample_pt=[]
        for i in range(100):
            pt_i=d1+sample_dir*i
            u_i=fx*pt_i[0]/pt_i[2]+cx
            v_i=fy*pt_i[1]/pt_i[2]+cy
            if u_i<0 or v_i<0 or u_i>639 or v_i>479:
                continue
            pt=np.array([u_i,v_i,pt_i[2]])
            sample_pt.append(pt)
        sample_pt=np.array(sample_pt)
        if len(sample_pt)>5:
            dis_1 = np.linalg.norm(sample_pt[:,0:2] - pc1,axis=1)
            indices=np.argsort(dis_1)
            pix1=np.array([sample_pt[indices[0]][0],sample_pt[indices[0]][1]])
            dis_2 = np.linalg.norm(sample_pt[:,0:2] - pc2,axis=1)
            indices_2=np.argsort(dis_2)
            pix2=np.array([sample_pt[indices_2[0]][0],sample_pt[indices_2[0]][1]])
            d1=sample_pt[indices[0]][2]
            d2=sample_pt[indices_2[0]][2]
            # pix1=np.array([sample_pt[0][0],sample_pt[0][1]])
            # pix2=np.array([sample_pt[len(sample_pt)-1][0],sample_pt[len(sample_pt)-1][1]])
            # d1=sample_pt[0][2]
            # d2=sample_pt[len(sample_pt)-1][2]
            #排除深度为负的点
            if d1<0 or d2<0:
                flag=False
            frame=np.concatenate(([frame_id],[pix1[0]],[pix1[1]],[d1],[pix2[0]],[pix2[1]],[d2],[line_index]),axis=0)
        else:
            flag=False
    else:
        frame=[]
    return frame,flag
    

def Line_match(match_line,depth,pose,point_cloud,frame_id,img):
    
    global Line_3d
    
    #插值获取两端点间离散点
    match_line=np.array((match_line))
    l=((match_line[:,2]-match_line[:,0])**2+(match_line[:,3]-match_line[:,1])**2)**0.5
    n1=l/10
    theta=np.arcsin((match_line[:,3]-match_line[:,1])/l)
    temp=[]
    for i in range(len(match_line)):
        if (match_line[i,2]-match_line[i,0])==0:
            temp.append((0))
        else:
            temp.append(((match_line[i,2]-match_line[i,0])/abs((match_line[i,2]-match_line[i,0]))))
    dx=temp*n1*np.cos(theta)
    dy=n1*np.sin(theta)
    for i in range(10):
        if i>0:
            match_line=np.column_stack((match_line,match_line[:,0]+i*dx))
            match_line=np.column_stack((match_line,match_line[:,1]+i*dy))
    # points_2d=[]
    # Keys=[]
    
    #反投影三维空间寻找对应三维点
    point_2d=[]
    Key=[]
    numeber=[]
    for i in range(len(match_line)):
        line_points=match_line[i].reshape(11,2)
        for j in range(len(line_points)):
            x=int(round(line_points[j,0]))
            y=int(round(line_points[j,1]))
            Z=depth[y][x]
            p2d=np.array([float(x),float(y)])
            normalized_p2d=cv2.undistortPoints(p2d.reshape(-1,1,2),camera_matrix,dist_coeff)
            x=normalized_p2d[0][0][0]
            y=normalized_p2d[0][0][1]
            # X = (x - cx) * Z / fx
            # Y = (y - cy) * Z / fy
            X=x*Z
            Y=y*Z
            point_2d.append((X,Y,Z))
            # points_2d=np.array([point_2d],dtype=object)
            Key.append((line_points[j,0],line_points[j,1]))
        # numeber.append((len(line_points)))  
    key_line_id=[]
    matches_3d,projected_depth,matches_2d=backproject_2d_to_3d(Key,point_2d,pose,point_cloud,key_line_id,False)
    #RANSAC拟合直线
    iterations=50
    threshold=0.01    
    # matches_3d=np.array(matches_3d)
    # matches_points=matches_3d[:,2:5]
    matchs_line=[]
    keys_line_id=[]
    for i in range(len(matches_3d)):
        match_points=np.array(matches_3d[i])
        best_params, best_inliers=ransac_lines(match_points[:,2:5], iterations, threshold)
        
        if len(best_inliers)>6:
            pix1, d1=project_point_cloud(best_inliers[0],pose)
            pix2, d2=project_point_cloud(best_inliers[len(best_inliers)-1],pose)
            if math.isnan(pix1[0]) or math.isnan(pix2[0]):
                continue
            point=np.append(best_inliers[0],best_inliers[len(best_inliers)-1])
            flag,index=issameLine(point)
            if index in Maplineasso:
                point_asso=Line_3d[index]
                frame=np.concatenate(([frame_id],pix1[0],pix1[1],[d1[2]],pix2[0],pix2[1],[d2[2]],[index]),axis=0)  
                # frame=Line_projection([pix1,pix2],point_asso,frame_id,pose,depth)
                Maplineasso[index]=np.vstack((Maplineasso[index],frame))

            else:
                # print(len(Line_3d))
                if len(Line_3d) == 0:
                    Line_3d=np.array([point])
                else:
                    Line_3d=np.vstack((Line_3d,np.array(point)))
                frame=np.concatenate(([frame_id],pix1[0],pix1[1],[d1[2]],pix2[0],pix2[1],[d2[2]],[len(Line_3d)-1]),axis=0)  
                index=len(Maplineasso)
                Maplineasso[index]=[frame]  
   

# 主函数
def main():
    # 设置路径
    # point_cloud_path = "/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg1_360/Point_cloud.txt"
    # 读取三维点云数据
    # point_cloud = read_point_cloud(point_cloud_path)
    # 读取位姿真值
    # pose_file = '/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg1_360/groundtruth.txt'
    if dataset_select==0:
        # 设置路径
        point_cloud_path = "/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png/Point_cloud.txt"
        # 读取三维点云数据
        point_cloud = read_point_cloud(point_cloud_path)
        # 读取位姿真值
        pose_file = '/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png/groundtruth.txt'
        #ICL
        poses,pose_Q,timestamps = read_trajectory_icl_gt(pose_file)
    elif dataset_select==1:
        # 设置路径
        point_cloud_path = "/home/arondight/G.Z/Open-structure/examples/cmu/rgbd_dataset_freiburg1_360/Point_cloud.txt"
        # 读取三维点云数据
        point_cloud = read_point_cloud(point_cloud_path)
        # 读取位姿真值
        pose_file = '/home/arondight/G.Z/Open-structure/examples/cmu/rgbd_dataset_freiburg1_360/pose_left.txt'
        #CMU
        poses,pose_Q = read_trajectory_cmu_gt(pose_file)
    elif dataset_select==2:
        # 设置路径
        point_cloud_path = "/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg1_360/Point_cloud.txt"
        # 读取三维点云数据
        point_cloud = read_point_cloud(point_cloud_path)
        # 读取位姿真值
        pose_file = '/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg1_360/groundtruth.txt'

        #TUM
        depth_file='/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg1_360/depth.txt'
        rgb_file='/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg1_360/rgb.txt'
        poses,pose_Q,timestamps= read_trajectory_icl_gt(pose_file)
        poses,pose_Q,depth_file_name,rgb_file_name,output_time=select_pose(depth_file,rgb_file,poses,pose_Q,timestamps)  
    
    num_frames = len(poses)

    # 逐帧进行匹配
    all_matches = []
    last_frame_descriptors = []
    for i in range(num_frames):
        
        if dataset_select==0:
            #读取ICL RGB图像
            image_path = os.path.join('/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png', f"rgb/{i+1}.png")
            #读取ICL 深度图
            depth_files = os.path.join('/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png', f"depth/{i+1}.png")
            depth = cv2.imread(depth_files, -1)
            depth=depth/5000
        elif dataset_select==2:
            #读取TUM RGB图像
            image_path = os.path.join('/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg1_360', f"{rgb_file_name[i]}")
            #读取TUM 深度图
            depth_files = os.path.join('/home/arondight/G.Z/Open-structure/examples/TUM/rgbd_dataset_freiburg1_360', f"{depth_file_name[i]}")
            depth = cv2.imread(depth_files, -1)
            depth=depth/5000
        elif dataset_select==1:
            #读取CMU-RGB图像
            image_path = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/rgbd_dataset_freiburg1_360', f"image_left/{i:06}_left.png")
            #读取CMU-深度图
            depth_files = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/rgbd_dataset_freiburg1_360', f"depth_left/{i:06}_left_depth.npy")
            depth = np.load(depth_files)
        
        # 读取图像
        image = read_image(image_path)
        # image = cv2.undistort(image, camera_matrix, dist_coeff)
        
        # 转换为灰度图像
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        clipLimit=1
        while True:
            # 创建CLAHE对象  
            clahe = cv2.createCLAHE(clipLimit, tileGridSize=(5,5))  
            
            # 应用CLAHE到灰度图像  
            cla = clahe.apply(gray)
            
            # 应用直方图均衡化来增强对比度  
            # gray = cv2.equalizeHist(gray) 
            
            # 创建ORB特征点检测器
            # detector = create_detector()
            detector = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
            # detector = cv2.xfeatures2d.SIFT_create()
            
            # 提取特征点和描述子
            keypoints, descriptors = extract_features(detector, cla)
            
            # print(len(keypoints))
            # 投影三维点云到相机坐标系
            pose=poses[i]
            point_2d=[]
            Keys=[]
            for j, keypoint in enumerate(keypoints):
                x, y = keypoint.pt
                x = int(round(x))
                y = int(round(y))
                Keys.append((x,y))
                X=[]
                Y=[]
                Z=[]
                p2d=np.array([float(x),float(y)])
                normalized_p2d=cv2.undistortPoints(p2d.reshape(-1,1,2),camera_matrix,dist_coeff)
                if x>= 0 and x<= 640 and y>= 0 and y<= 480:
                    Z=depth[y][x]
                    x_=normalized_p2d[0][0][0]
                    y_=normalized_p2d[0][0][1]
                    # X = (x - cx) * Z / fx
                    # Y = (y - cy) * Z / fy
                    X=x_*Z
                    Y=y_*Z
                    # Point_cam=np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
                    point_2d.append([X,Y,Z])
            #     cv2.circle(image,(x,y),2,(0,0,255),-1)
            
            # cv2.imwrite(f"/media/grazy/DATA1/anaconda/PL_detection/lrkt0/{i+1}.png",image)
            # continue
            # cv2.imshow("keypoints",image)
            # cv2.waitKey(0)
            key_point_id=[]
            if len(point_2d)>0:   #当前图像检测到特征点数大于0
                #实现2d-3d点匹配
                matches_3d,projected_depth,matches_2d=backproject_2d_to_3d(Keys,point_2d,pose,point_cloud,key_point_id,True)
            if len(matches_3d)>3:
                break
            else:
                clipLimit=clipLimit+0.3
        
        # clipLimit=1    
        # clahe = cv2.createCLAHE(clipLimit, tileGridSize=(5,5))  
        # cla = clahe.apply(gray)                
        #提取２Ｄ线特征
        match_line=Line_extractor2d(image,gray,depth)
        if(len(match_line)>0):
            #2d-3d线匹配
            Line_match(match_line,depth,pose,point_cloud,i,image)
        
        #相邻帧特征匹配
        # key_point_id=np.array(key_point_id)
        # current_frame_descriptors=descriptors[key_point_id[:,0]]
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(current_frame_descriptors, last_frame_descriptors)
        
        # matched=[]
        # queryIdx=[]
        # trainIdx=[]
        # merged_points=[]
        # matches_3d=np.array(matches_3d)
        # CurrentKeypoints=matches_3d[:,2:5]
        # if len(matches)>0:
        #     for j,match in enumerate(matches):
        #         queryIdx.append(int(match.queryIdx))      
        #         trainIdx.append(int(match.trainIdx)) 
        #     matched=matches_3d[queryIdx]
        #     merged_points.append((last_frame_3d[trainIdx]+CurrentKeypoints[queryIdx])/2.)

        for j, match in enumerate(matches_3d):
            #选择部分特征点显示在图像中
            # if j%10==0:
            num_u = random.randint(-1, 1)
            u=int(match[0])+num_u
            num_v = random.randint(-1, 1)
            v=int(match[1])+num_v
            #显示3d投影点像素坐标
            cv2.circle(image,(u,v),2,(0,255,0),-1)
            cv2.putText(image, str(j), (u,v), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
            #显示orb特征点像素坐标
            cv2.circle(image,(int(matches_2d[j][0]),int(matches_2d[j][1])),2,(100,0,0),-1)
            cv2.putText(image, str(j), (int(match[0]),int(match[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,0,0), 1, cv2.LINE_AA)
            # if j in queryIdx:
            #     match_id=queryIdx.index(j)
            #     match[2:5]=merged_points[0][match_id]
            
            #将2d-3d匹配关系导入哈希表
            frame=np.append(i,match[0:2])
            key=tuple(match[2:5].tolist())
            if key in Mappointasso:
                Mappointasso[key]=np.vstack((Mappointasso[key],frame))
            else:
                Mappointasso[key]=[frame]
                Point_3d.append((key))
        #print(len(Line_3d))
        # last_frame_descriptors=current_frame_descriptors
        # last_frame_3d=matches_3d[:,2:5]
        print(f"Processed frame {i+1}/{num_frames}")
        cv2.imwrite(f"/media/grazy/DATA1/anaconda/PL_detection/lrkt0/{i}.png",image)
    
    # return
    #设置导出文本路径
    poses_path=os.path.join('/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png', f"camera_poses.txt")
    line_path=os.path.join('/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png', f"line_landmarks.txt")
    point_path=os.path.join('/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png', f"point_landmarks.txt")
    association_path=os.path.join('/media/grazy/DATA1/anaconda/PL_detection/living_room_traj0_frei_png', f"association.txt")

    with open(poses_path, 'w') as fs:
    #位姿数据导出
        for i in range(len(pose_Q)):
            fs.write('{} {} {} {} {} {} {} {} \n'.format(i, np.round(pose_Q[i][0],6),np.round(pose_Q[i][1],6),np.round(pose_Q[i][2],6),np.round(pose_Q[i][3],6),np.round(pose_Q[i][4],6),np.round(pose_Q[i][5],6),np.round(pose_Q[i][6],6)))
    #点数据导出
    with open(point_path, 'w') as fs:    
        keys=list(Mappointasso.keys())
        k=0
        for i in range(len(keys)):
            if len(Mappointasso[keys[i]])<2:
                continue
            fs.write('{} {} {} {} {}\n'.format("Mappoint:",k, np.round(keys[i][0],6),np.round(keys[i][1],6),np.round(keys[i][2],6)))
            k+=1

    with open(association_path, 'w') as fs:
        #点帧匹配关系导出
        k=0  
        for i in range(len(Mappointasso)):
            frame=Mappointasso[Point_3d[i]]
            if len(frame)<2:
                continue
            for j in range(len(frame)):
                fs.write('{} {} {} {} {}\n'.format("MappointFrameAsso:",k, int(round(frame[j][0])),round(frame[j][1],4),round(frame[j][2],4)))
            k+=1
        
        #线帧匹配关系导出
        outputline=[]
        k=0
        sub=0
        # print(Maplineasso)
        for i in range(len(Maplineasso)):
            frame=Maplineasso[i]
            if len(frame)<5:
               sub = sub+1
               continue
            ID=-1
            frame_num=0 #初始化该线的观测帧数量
            for j in range(len(frame)):
                fr,flag=Line_projection([(frame[j][1],frame[j][2]),(frame[j][4],frame[j][5])],Line_3d[i],frame[j][0],poses[int(round(frame[j][0]))],depth,int(round(frame[j][7])))
                if flag==False or frame[j][0]==ID:
                    continue
                ID=frame[j][0]
                frame_num+=1
                fs.write('{} {} {} {} {} {} {} {} {}\n'.format("MaplineFrameAsso:",k, int(round(fr[0])),round(fr[1],4),round(fr[2],4),round(fr[3],4),round(fr[4],4),round(fr[5],4),round(fr[6],4)))
                                    
                #投影线两端点加高斯噪声显示在图中
                img=cv2.imread(f"/media/grazy/DATA1/anaconda/PL_detection/lrkt0/{int(frame[j][0])}.png")
                num_u = random.randint(-1, 1)
                u_1=int(fr[1])+num_u
                num_v = random.randint(-1, 1)
                v_1=int(fr[2])+num_v
                num_u = random.randint(-1, 1)
                u_2=int(fr[4])+num_u
                num_v = random.randint(-1, 1)
                v_2=int(fr[5])+num_v
                cv2.line(img,(u_1,v_1),(u_2,v_2),(0,100,0),2,cv2.LINE_AA)            
                cv2.putText(img, str(k), (u_1,v_1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(100,100,0), 1, cv2.LINE_AA)
                cv2.imwrite(f"/media/grazy/DATA1/anaconda/PL_detection/lrkt0/{int(frame[j][0])}.png",img)
            if frame_num>0:
                k+=1
                outputline.append(i)
            # k+=1 

        #判断平行线
        k=0
        Paraline={}
        for l in range(len(outputline)):
            i=outputline[l]
            if len(Paraline)==0:
                Paraline[0]=[0]
                k+=1
                continue
            # if len(Maplineasso[i])<3:
            #     continue
            flag=False
            for j in range(len(Paraline)):
                P0=np.array((Line_3d[outputline[Paraline[j][0]]][0],Line_3d[outputline[Paraline[j][0]]][1],Line_3d[outputline[Paraline[j][0]]][2]))
                P1=np.array((Line_3d[outputline[Paraline[j][0]]][3],Line_3d[outputline[Paraline[j][0]]][4],Line_3d[outputline[Paraline[j][0]]][5]))
                direction1=P1-P0
                direction1 /= np.linalg.norm(direction1)
                
                P2=np.array((Line_3d[i][0],Line_3d[i][1],Line_3d[i][2]))
                P3=np.array((Line_3d[i][3],Line_3d[i][4],Line_3d[i][5]))
                direction2=P3-P2
                direction2 /= np.linalg.norm(direction2)
                
                cos=np.dot(direction1,direction2)
                
                if abs(cos)>0.999:
                    flag=True
                    Paraline[j]=np.hstack((Paraline[j],l))
                    break
            if flag==False:
                Paraline[len(Paraline)]=[l]
        # print(Paraline)
        for i in range(len(Paraline)):
            if(len(Paraline[i])>1):
                fs.write('{} {} {}'.format("ParalineMaplineAsso:",k,len(Paraline[i])))
                for j in range(len(Paraline[i])):
                    fs.write(' {}'.format(int(round(Paraline[i][j]))))
                fs.write('\n')
                k+=1
    #导出线数据
    with open(line_path, 'w') as fs:    
        k=0
        sub = 0
        for l in range(len(outputline)):
            i=outputline[l]
            # if len(Maplineasso[i])<3:
            #    sub = sub +1
            #    continue
            fs.write('{} {} {} {} {} {} {}\n'.format(l, round(Line_3d[i][0],6),round(Line_3d[i][1],6),round(Line_3d[i][2],6),round(Line_3d[i][3],6),round(Line_3d[i][4],6),round(Line_3d[i][5],6)))
            k+=1        
    
    #三维空间显示线 
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(keys)
    draw_line_points=[]
    line_indice=[]
    k=0
    for l in range(len(outputline)):
        i=outputline[l]
        # if len(Maplineasso[i])<5:
        #     continue

        draw_line_points.append(Line_3d[i][0:3])
        draw_line_points.append(Line_3d[i][3:6])
        line_indice.append([2*k,2*k+1])
        k+=1
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(draw_line_points)
    lines.lines = o3d.utility.Vector2iVector(line_indice)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云和线段到窗口
    # vis.add_geometry(cloud)
    vis.add_geometry(lines)

    # 显示可视化窗口
    vis.run()
    vis.destroy_window()
        
# 运行主函数
if __name__ == "__main__":
    main()
