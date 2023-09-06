import cv2
import numpy as np
import os
import open3d as o3d
import math
from scipy.spatial import KDTree
# from sklearn.linear_model import RANSACRegressor

Line_3d=[]      #三维地图线集
Point_3d=[]     #三维地图点集
Maplineasso={}  #三维地图线-观测帧关系集
Mappointasso={} #三维地图点-观测帧关系集
def read_trajectory_gt(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析数据并转换为位姿矩阵
    poses = []
    pose_qua = []
    for line in lines:
        elements = line.split()
        # timestamp = float(elements[0])
        translation = np.array([float(elements[1]), float(elements[2]), float(elements[3])])
        rotation = np.array([float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])])
        pose = np.zeros((3,4))
        pose[:3, :3] = quaternion_to_rotation_matrix(rotation)
        pose[:3, 3] = translation
        pose_Q=np.array([float(elements[1]), float(elements[2]), float(elements[3]),float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7])])
        poses.append(pose)   
        pose_qua.append(pose_Q)
    return poses,pose_qua

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

# 获取相机内参
fx = 481.2
fy = -480.0
cx = 319.5
cy = 239.5

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

    # 使用KD树寻找最近邻的三维点
    # pcd1 = o3d.geometry.PointCloud()   
    # pcd1.points = o3d.utility.Vector3dVector(points_world)
    # o3d.visualization.draw_geometries([pcd1])
    # kdt=o3d.geometry.KDTreeFlann(pcd1)
    # indices=[]
    # for i in range(len(points_world)):
    #     result=kdt.search_knn_vector_3d(points_world[i],1)
    #     if len(indices)==0:
    #         indices=np.array(result[1])
    #     else:
    #         indices=np.vstack((indices,np.array(result[1])))
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
            if (i+1)%11==0:
                match=[]
                pix_origin_2d=[]
                k=0
            continue
    
        res.append((indices[i]))
        projected_depth.append((projected_points_d))
        matches_3d.append((np.append(pixel_coords,match_point_3d)))
        match.append(np.array([pixel_coords[0][0],pixel_coords[1][0],match_point_3d[0],match_point_3d[1],match_point_3d[2]]))
        pix_origin_2d.append(match_point_2d)
        matches_2d.append(match_point_2d)
        key_point_id.append((i,indices[i]))
        if (i+1)%11==0:
            matche_3d.append(match)
            matche_2d.append(pix_origin_2d)
            match=[]
            pix_origin_2d=[]
            k=0
            
        k=k+1
        
    if delete==False:
        matches_3d=matche_3d
        matches_2d=matche_2d
        # print(matches_3d[1][1])
        # cv2.circle(img,(int(pixel_coords[0]),int(pixel_coords[1])),2,(0,255,0),-1)
    return matches_3d,projected_depth,matches_2d

#2D-3D线匹配
def Line_extractor2d(img,gray,depth):
    match_line=[]
    
    fld=cv2.ximgproc.createFastLineDetector()
    lines=fld.detect(gray)
    # print(lines.shape)
    for dline in lines:
        line=[]
        x0= int(round(dline[0][0]))
        y0= int(round(dline[0][1]))
        x1= int(round(dline[0][2]))
        y1= int(round(dline[0][3]))
        z0=depth[y0][x0]
        z1=depth[y1][x1]
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
        normal = np.cross(direction, np.array([0, 0, 1]))

        # 计算距离
        distances = np.abs(np.dot(points - sample_points[0], normal))

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
        normal_cross = np.cross(direction, direction_old)
        L12=P2-P0
        if  np.linalg.norm(L12)==0:
            flag=True
            index=i
            break
        min_distance=np.dot(L12,normal_cross)/np.linalg.norm(normal_cross)
        cos=np.dot(direction,direction_old)
        points=[]
        if abs(cos)>0.98 and abs(min_distance)<0.05:
            iterations=10
            threshold=0.03
            points=np.concatenate(([P0],[P1],[P2],[P3]))
            best_params, best_inliers=ransac_lines(points, iterations, threshold)
            if len(best_inliers)>2 :
                flag=True
                index=i
                temp=np.concatenate((best_inliers[0],best_inliers[len(best_inliers)-1]))
                # Line_3d[i][0]=temp[0]
                # Line_3d[i][1]=temp[1]
                # Line_3d[i][2]=temp[2]
                # Line_3d[i][3]=temp[3]
                # Line_3d[i][4]=temp[4]
                # Line_3d[i][5]=temp[5]
                break
    return flag,index

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
        # print(i)
        for j in range(len(line_points)):
            x=int(round(line_points[j,0]))
            y=int(round(line_points[j,1]))
            Z=depth[y][x]
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            point_2d.append((X,Y,Z))
            # points_2d=np.array([point_2d],dtype=object)
            Key.append((line_points[j,0],line_points[j,1]))
        numeber.append((len(line_points)))  
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
        if len(match_points)>11:
            print(match_points)
        best_params, best_inliers=ransac_lines(match_points[:,2:5], iterations, threshold)
        # best_inliers=best_inliers[np.argsort(best_inliers[:,0])]
        # best_inliers=best_inliers[np.argsort(best_inliers[:,1])]
        # best_inliers=best_inliers[np.argsort(best_inliers[:,2])]
        if len(best_inliers)>5:
            # pix=np.append(match_points[0,0:2],match_points[len(best_inliers)-1,0:2])
            point=np.append(best_inliers[0],best_inliers[len(best_inliers)-1])
            pix1, d1=project_point_cloud(best_inliers[0],pose)
            pix2, d2=project_point_cloud(best_inliers[len(best_inliers)-1],pose)
            
            #去除偏差线
            direction1=np.append(pix1[0],pix1[1])-np.append(pix2[0],pix2[1])
            direction2=np.append(matches_2d[i][0][0],matches_2d[i][0][1])-np.append(matches_2d[i][1][0],matches_2d[i][1][1])
            direction1/=np.linalg.norm(direction1)
            direction2/=np.linalg.norm(direction2)
            cos=np.dot(direction1,direction2)
            L12=np.append(matches_2d[i][0][0],matches_2d[i][0][1])-np.append(pix1[0],pix1[1])
            normal_cross = np.cross(direction1, L12)
            min_distance=np.linalg.norm(normal_cross)
            # print(min_distance)
            if abs(cos)<0.999 or min_distance>3:
                continue
            
            flag,index=issameLine(point)
  
            cv2.line(img,(pix1[0],pix1[1]),(pix2[0],pix2[1]),(0,100,0),2,cv2.LINE_AA)            

            if flag==False:
                if len(Line_3d) == 0:
                    Line_3d=np.array([point])
                else:
                    Line_3d=np.vstack((Line_3d,np.array(point)))
                    
            # print(len(Line_3d))
            frame=np.concatenate(([frame_id],pix1[0],pix1[1],[d1[2]],pix2[0],pix1[1],[d2[2]]),axis=0)
            if len(Maplineasso)==0:
                Maplineasso[0]=frame
            
            if index in Maplineasso:
                Maplineasso[index]=np.vstack((Maplineasso[index],frame))
            else:
                index=len(Maplineasso)
                Maplineasso[index]=[frame]

            cv2.putText(img, str(index), (int(pix1[0]),int(pix1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(100,100,0), 1, cv2.LINE_AA)

            # matchs_line.append((np.append(pix,point)))
            # keys_line_id.append((key_line_id[i*11:(11+i*11)]))
        # print(len(keys_line_id))
    # print(len(matchs_line))
    # return matchs_line,key_line_id
    # for i in range(len(Key)):
    #     cv2.circle(img,(int(Key[i][0]),int(Key[i][1])),2,(0,0,100),-1)
    # cv2.imwrite("target_line.png",img)
        
# 主函数
def main():
    # 设置路径
    # image_path = "/home/arondight/G.Z/Open-structure/rgb/*.png"
    point_cloud_path = "/home/arondight/G.Z/Open-structure/examples/ICL-NUIM/lrkt1/Point_cloud.txt"
    # poses_path = "/home/arondight/G.Z/Open-structure/groundtruth.txt"
    # output_path = "/home/arondight/G.Z/Open-structure/output.txt"

    # 读取三维点云数据
    point_cloud = read_point_cloud(point_cloud_path)
    # Line_extractor3D(point_cloud)
    # 读取位姿真值
    pose_file = '/home/arondight/G.Z/Open-structure/examples/ICL-NUIM/lrkt1/groundtruth.txt'
    poses,pose_Q = read_trajectory_gt(pose_file)
    num_frames = len(poses)
    # poses = read_poses(poses_path)

    # 逐帧进行匹配
    all_matches = []
    out_path_point = os.path.join('/home/arondight/G.Z/Open-structure/output', f"Frame_point.txt")
    out_path_line = os.path.join('/home/arondight/G.Z/Open-structure/output', f"Frame_line.txt")
    out_path=os.path.join('/home/arondight/G.Z/Open-structure/output', f"sequence.txt")
    # 将匹配关系导出到txt文本中
    with open(out_path_line, 'w') as fl:
        with open(out_path_point, 'w') as f:
            for i in range(num_frames):
                #读取RGB图像
                image_path = os.path.join('/home/arondight/G.Z/Open-structure/examples/ICL-NUIM/lrkt1', f"rgb/{i}.png")
                #读取深度图
                depth_files = os.path.join('/home/arondight/G.Z/Open-structure/examples/ICL-NUIM/lrkt1', f"depth/{i}.png")
                depth = cv2.imread(depth_files, -1)
                depth=depth/5000

                # 读取图像
                image = read_image(image_path)
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                # cv2.imwrite("target.png",image)
                # 创建ORB特征点检测器
                detector = create_detector()

                # 提取特征点和描述子
                keypoints, descriptors = extract_features(detector, image)
                match_line=Line_extractor2d(image,gray,depth)
                # print(descriptors)
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
                    if x>= 0 and x<= 640 and y>= 0 and y<= 480:
                        Z=depth[y][x]
                        X = (x - cx) * Z / fx
                        Y = (y - cy) * Z / fy
                        # Point_cam=np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
                        point_2d.append([X,Y,Z])
                
                key_point_id=[]
                if len(point_2d)>0:
                    matches_3d,projected_depth,matches_2d=backproject_2d_to_3d(Keys,point_2d,pose,point_cloud,key_point_id,True)
                if(len(match_line)>0):
                    Line_match(match_line,depth,pose,point_cloud,i,image)
                for j, match in enumerate(matches_3d):
                    cv2.circle(image,(int(match[0]),int(match[1])),2,(0,255,0),-1)
                    cv2.putText(image, str(j), (int(match[0]),int(match[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
                    cv2.circle(image,(int(matches_2d[j][0]),int(matches_2d[j][1])),2,(100,0,0),-1)
                    cv2.putText(image, str(j), (int(match[0]),int(match[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,0,0), 1, cv2.LINE_AA)
                 
                    frame=np.append(i,match[0:2])
                    key=tuple(match[2:5].tolist())
                    if key in Mappointasso:
                        Mappointasso[key]=np.vstack((Mappointasso[key],frame))
                    else:
                        Mappointasso[key]=[frame]
                    # print(Mappointasso)
                    Point_3d.append((key))
                    match=np.round(match,2)
 
                print(f"Processed frame {i}/{num_frames}")
                cv2.imwrite("target_line.png",image)
    #导出文本格式
    with open(out_path, 'w') as fs:
        for i in range(len(pose_Q)):
            fs.write('{} {} {} {} {} {} {} {} {} \n'.format("Vertex:",i, np.round(pose_Q[i][0],6),np.round(pose_Q[i][1],6),np.round(pose_Q[i][2],6),np.round(pose_Q[i][3],6),np.round(pose_Q[i][4],6),np.round(pose_Q[i][5],6),np.round(pose_Q[i][6],6)))
        
        keys=list(Mappointasso.keys())
        k=0
        for i in range(len(keys)):
            if len(Mappointasso[keys[i]])<5:
                    continue
            fs.write('{} {} {} {} {}\n'.format("Mappoint:",k, np.round(keys[i][0],6),np.round(keys[i][1],6),np.round(keys[i][2],6)))
            k+=1
        k=0
        for i in range(len(Line_3d)):
            if len(Maplineasso[i])<3:
                continue
            fs.write('{} {} {} {} {} {} {} {}\n'.format("Mapline:",k, round(Line_3d[i][0],6),round(Line_3d[i][1],6),round(Line_3d[i][2],6),round(Line_3d[i][3],6),round(Line_3d[i][4],6),round(Line_3d[i][5],6)))
            k+=1
        
        k=0  
        for i in range(len(Mappointasso)):
            frame=Mappointasso[Point_3d[i]]
            if len(frame)<5:
                continue
            for j in range(len(frame)):
                fs.write('{} {} {} {} {}\n'.format("MappointFrameAsso:",k, int(round(frame[j][0])),round(frame[j][1],4),round(frame[j][2],4)))
            k+=1
        
        k=0
        # print(Maplineasso)
        for i in range(len(Maplineasso)):
            frame=Maplineasso[i]
            if len(frame)<3:
                continue
            for j in range(len(frame)):
                fs.write('{} {} {} {} {} {} {} {} {}\n'.format("MaplineFrameAsso:",k, int(round(frame[j][0])),round(frame[j][1],4),round(frame[j][2],4),round(frame[j][3],4),round(frame[j][4],4),round(frame[j][5],4),round(frame[j][6],4)))
            k+=1
        
        k=0
        Paraline={}
        for i in range(len(Line_3d)):
            if len(Paraline)==0:
                Paraline[0]=[0]
                k+=1
                continue
            if len(Maplineasso[i])<3:
                continue
            flag=False
            for j in range(len(Paraline)):
                P0=np.array((Line_3d[Paraline[j][0]][0],Line_3d[Paraline[j][0]][1],Line_3d[Paraline[j][0]][2]))
                P1=np.array((Line_3d[Paraline[j][0]][3],Line_3d[Paraline[j][0]][4],Line_3d[Paraline[j][0]][5]))
                direction1=P1-P0
                direction1 /= np.linalg.norm(direction1)
                
                P2=np.array((Line_3d[i][0],Line_3d[i][1],Line_3d[i][2]))
                P3=np.array((Line_3d[i][3],Line_3d[i][4],Line_3d[i][5]))
                direction2=P3-P2
                direction2 /= np.linalg.norm(direction2)
                
                cos=np.dot(direction1,direction2)
                
                if abs(cos)>0.999:
                    flag=True
                    Paraline[j]=np.hstack((Paraline[j],k))
                    break
            if flag==False:
                Paraline[len(Paraline)]=[k]
            k+=1

        # print(Paraline)
        for i in range(len(Paraline)):
            if(len(Paraline[i])>1):
                fs.write('{}'.format("ParalineMaplineAsso:"))
                for j in range(len(Paraline[i])):
                    fs.write(' {}'.format(int(round(Paraline[i][j]))))
                fs.write('\n')
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(keys)
        draw_line_points=[]
        line_indice=[]
        for i in range(len(Line_3d)):
            if len(Maplineasso[i])<3:
                continue
            draw_line_points.append(Line_3d[i][0:3])
            draw_line_points.append(Line_3d[i][3:6])
            line_indice.append([2*i,2*i+1])
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(draw_line_points)
        lines.lines = o3d.utility.Vector2iVector(line_indice)
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 添加点云和线段到窗口
        vis.add_geometry(cloud)
        vis.add_geometry(lines)

        # 显示可视化窗口
        vis.run()
        vis.destroy_window()
        
# 运行主函数
if __name__ == "__main__":
    main()
