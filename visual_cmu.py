import cv2
import numpy as np
import os
import open3d as o3d
import math
import matplotlib.pyplot as plt
# import PyQt5.QtCore
# from traits.etsconfig.api import ETSConfig
# import mayavi.mlab as mlab
# ETSConfig.toolkit='pyside'
import open3d as o3d
import numpy as np

def create_cylinder(cylinder_coefficients, height=5, step=0.5, vis=False):
    """

    Args:
        cylinder_coefficients: A dictionary containing cylindrical coefficients:
                                (r,x0,y0,z0,a,b,c
                                r the radius of the cylinder
                                x0,y0,z0 the Starting center of the cylinder
                                a, b, c: axis coefficient of the cylinder)
        height: height_ of the cylinder
        step: Density of cylinder point cloud
        vis: whether to visualize the cylinder

    Returns:
        numpy form of the cylinder point cloud: n x 3
    References:
        https://blog.csdn.net/inerterYang/article/details/111998278
        https://blog.csdn.net/inerterYang/article/details/111304307

    @Author: Carlos_Lee 202111

    """
    r = cylinder_coefficients['r']
    x0 = cylinder_coefficients['x0']
    y0 = cylinder_coefficients['y0']
    z0 = cylinder_coefficients['z0']
    a = cylinder_coefficients['a']
    b = cylinder_coefficients['b']
    c = cylinder_coefficients['c']

    angle_ = np.arange(0, 2 * np.pi, step / 10).reshape(-1, 1)

    v = np.arange(0, height, step)
    npy = []
    for i in v:
        x = x0 + r * b / np.power(a * a + b * b, 0.5) * np.cos(angle_) + \
            r * a * c / np.power(a * a + b * b, 0.5) / np.power(a * a + b * b + c * c, 0.5) * \
            np.sin(angle_) + a / np.power(a * a + b * b + c * c, 0.5) * i

        y = y0 - r * a / np.power(a * a + b * b, 0.5) * np.cos(angle_) + \
            r * b * c / np.power(a * a + b * b, 0.5) / np.power(a * a + b * b + c * c, 0.5) * \
            np.sin(angle_) + b / np.power(a * a + b * b + c * c, 0.5) * i

        z = z0 - r * np.power(a * a + b * b, 0.5) / np.power(a * a + b * b + c * c, 0.5) * np.sin(angle_) + \
            c / np.power(a * a + b * b + c * c, 0.5) * i

        npy.append(np.concatenate([x, y, z], axis=-1))

    npy = np.concatenate(npy, axis=0)
    return npy

def read_file(file_path):
    # 读取文件内容
    with open(file_path, 'r') as f:
        file_lines = f.readlines()

    # 解析数据并转换为位姿矩阵
    poses = []
    points = []
    lines=[]
    MappointFrameAsso={}
    MaplineFrameAsso={}
    k=0
    paralines={}
    for line in file_lines:
        elements = line.split()
        if elements[0]=='Vertex:':
            translation = np.array([float(elements[2]), float(elements[3]), float(elements[4])])
            rotation = np.array([float(elements[5]), float(elements[6]), float(elements[7]), float(elements[8])])
            # translation = np.array([float(elements[0]), float(elements[1]), float(elements[2])])
            # rotation = np.array([float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])])

            pose = np.eye(4)
            pose[:3, :3] = quaternion_to_rotation_matrix(rotation)
            # print(-pose[:3, :3].T)
            # print(np.transpose(translation))
            pose[:3, 3] = translation
            poses.append(pose) 
        if elements[0]=='Mappoint:':
            point=np.array([float(elements[2]), float(elements[3]), float(elements[4])])
            points.append(point)
        if elements[0]=='Mapline:':
            Mapline=np.array([float(elements[2]), float(elements[3]), float(elements[4]),float(elements[5]), float(elements[6]), float(elements[7])])
            lines.append(Mapline)
        if elements[0]=='MappointFrameAsso:':
            frame=np.array([float(elements[2]),float(elements[3]), float(elements[4])])
            key=tuple(points[int(elements[1])].tolist())
            if key in MappointFrameAsso:
                MappointFrameAsso[key]=np.vstack((MappointFrameAsso[key],frame))
            else:
                MappointFrameAsso[key]=[frame]
        if elements[0]=='MaplineFrameAsso:':
            frame=np.array([float(elements[2]),float(elements[3]), float(elements[4]),float(elements[5]), float(elements[6]), float(elements[7]),float(elements[8])])
            key=tuple(lines[int(elements[1])].tolist())
            if key in MaplineFrameAsso:
                MaplineFrameAsso[key]=np.vstack((MaplineFrameAsso[key],frame))
            else:
                MaplineFrameAsso[key]=[frame]
        if elements[0]=='ParalineMaplineAsso:':
            for j in range(3,len(elements)):
                if k in paralines:
                    paralines[k]=np.vstack((paralines[k],float(elements[j])))
                else:
                    paralines[k]=[float(elements[j])]
            k+=1

    return poses,points,lines,MappointFrameAsso,MaplineFrameAsso,paralines

def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    rotation_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

    return rotation_matrix
    
# 读取三维点云数据
def read_point_cloud(point_cloud_path):
    return np.loadtxt(point_cloud_path)

# 读取RGB图像
def read_image(image_path):
    return cv2.imread(image_path)

#ICL
# fx = 481.2
# fy = -480.0
# cx = 319.5
# cy = 239.5

#cmu
fx = 320.0  # focal length x
fy = 320.0  # focal length y
cx = 320.0  # optical center x
cy = 240.0  # optical center y

def on_key(event):
    ax=event.inaxes
    if event.key=='x':
        ax.view_init(elev=ax.elev,azim=ax.azim+10)
    elif event.key=='y':
        ax.view_init(elev=ax.elev,azim=ax.azim-10)
    elif event.key=='z':
        ax.view_init(elev=ax.elev-10,azim=ax.azim)
    elif event.key=='c':
        ax.view_init(elev=ax.elev+10,azim=ax.azim)
    ax.figure.canvas.draw()        
# 主函数
def main():
    datafile = '/home/arondight/G.Z/Open-structure/output/datasets/cmu/cmu-hospital.txt'
    cloud_file='/home/arondight/G.Z/Open-structure/examples/cmu/P000/Point_cloud.txt'
    rgb_1 = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/P000', f"image_left/{0:06}_left.png")
    rgb_50 = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/P000', f"image_left/{49:06}_left.png")
    rgb_80 = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/P000', f"image_left/{79:06}_left.png")
    rgb_100 = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/P000', f"image_left/{99:06}_left.png")
    
    image_1 = read_image(rgb_1)
    image_50 = read_image(rgb_50)
    image_80 = read_image(rgb_80)
    image_100 = read_image(rgb_100)
    
    poses,points,Line_3d,MappointFrameAsso,MaplineFrameAsso,paralines=read_file(datafile) 
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(points)
    # cloud = cloud.voxel_down_sample(voxel_size=0.3)
    # points_after_voxel=np.array(cloud.points)
    # keys_point=points_after_voxel.tolist()
    # print(keys_point)
    keys_point=list(MappointFrameAsso.keys())
    frame_1=[]
    frame_50=[]
    frame_80=[]
    frame_100=[]
    Key_points={}
    for i in range(len(MappointFrameAsso)):
        Key_points[i]=0
    for i in range(len(MappointFrameAsso)):
        frame=MappointFrameAsso[keys_point[i]]
        for j in range(len(frame)):
            if frame[j][0]==0:
                Key_points[i]+=1
                frame_1=np.append(frame_1,i)
            if frame[j][0]==49:
                Key_points[i]+=1
                frame_50=np.append(frame_50,i)
            if frame[j][0]==79:
                Key_points[i]+=1
                frame_80=np.append(frame_80,i)
            if frame[j][0]==99:
                Key_points[i]+=1
                frame_100=np.append(frame_100,i)
    common_elements=[]
    for i in range(len(Key_points)):
        if Key_points[i]>=2:
            common_elements=np.append(common_elements,i)
            
            
            
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2BGRA)
    image_1[:, :, 3] = 100
    image_50 = cv2.cvtColor(image_50, cv2.COLOR_BGR2BGRA)
    image_50[:, :, 3] = 100
    image_80 = cv2.cvtColor(image_80, cv2.COLOR_BGR2BGRA)
    image_80[:, :, 3] = 100
    image_100 = cv2.cvtColor(image_100, cv2.COLOR_BGR2BGRA)
    image_100[:, :, 3] = 100
    # print(keys_point[0])
    for i in range(1,len(MappointFrameAsso)):
        if i not in common_elements:
            continue
        frame=MappointFrameAsso[keys_point[i]]
        for j in range(len(frame)):
            if frame[j][0]==0:
                cv2.circle(image_1,(int(frame[j][1]),int(frame[j][2])),2,(0,255,0,255),-1)
                cv2.putText(image_1, str(i), (int(frame[j][1]),int(frame[j][2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0,255), 1, cv2.LINE_AA)
            elif frame[j][0]==49:
                cv2.circle(image_50,(int(frame[j][1]),int(frame[j][2])),2,(0,255,0,255),-1)
                cv2.putText(image_50, str(i), (int(frame[j][1]),int(frame[j][2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0,255), 1, cv2.LINE_AA)
            elif frame[j][0]==79:
                cv2.circle(image_80,(int(frame[j][1]),int(frame[j][2])),2,(0,255,0,255),-1)
                cv2.putText(image_80, str(i), (int(frame[j][1]),int(frame[j][2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0,255), 1, cv2.LINE_AA)
            elif frame[j][0]==99:
                cv2.circle(image_100,(int(frame[j][1]),int(frame[j][2])),2,(0,255,0,255),-1)
                cv2.putText(image_100, str(i), (int(frame[j][1]),int(frame[j][2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0,255), 1, cv2.LINE_AA)
    
    keys_line=list(MaplineFrameAsso.keys())
    color=(150,0,50,255)

    for i in range(len(keys_line)):
        frame=MaplineFrameAsso[keys_line[i]]
        flag=False
        color=(150,0,50)
        for k in range(len(paralines)):
            if i in paralines[k]:
                color=(0,0+k*80,255-k*80,255)
                flag=True
        if flag==False:
            color=(150,0,50,255)
        for j in range(len(frame)):
            if frame[j][0]==0:
                # print((int(frame[j][1]),int(frame[j][2])))
                cv2.line(image_1,(int(frame[j][1]),int(frame[j][2])),(int(frame[j][4]),int(frame[j][5])),color,2,cv2.LINE_AA)            
                cv2.putText(image_1, str(i), (int(frame[j][1]),int(frame[j][2])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif frame[j][0]==49:
                cv2.line(image_50,(int(frame[j][1]),int(frame[j][2])),(int(frame[j][4]),int(frame[j][5])),color,2,cv2.LINE_AA)            
                cv2.putText(image_50, str(i), (int(frame[j][1]),int(frame[j][2])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 1, cv2.LINE_AA)
            elif frame[j][0]==79:
                cv2.line(image_80,(int(frame[j][1]),int(frame[j][2])),(int(frame[j][4]),int(frame[j][5])),color,2,cv2.LINE_AA)            
                cv2.putText(image_80, str(i), (int(frame[j][1]),int(frame[j][2])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 1, cv2.LINE_AA)
            elif frame[j][0]==99:
                cv2.line(image_100,(int(frame[j][1]),int(frame[j][2])),(int(frame[j][4]),int(frame[j][5])),color,2,cv2.LINE_AA)            
                cv2.putText(image_100, str(i), (int(frame[j][1]),int(frame[j][2])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,color, 1, cv2.LINE_AA)

    cv2.imwrite(f"{1}.png",image_1) 
    cv2.imwrite(f"{50}.png",image_50) 
    cv2.imwrite(f"{80}.png",image_80)    
    cv2.imwrite(f"{100}.png",image_100) 
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(points)
    # cloud = cloud.voxel_down_sample(voxel_size=0.3)
    
    print(poses[0][:3, 3])
    draw_line_points=[]
    line_indice=[]
    for i in range(len(poses)-1):
        draw_line_points.append(poses[i][:3, 3])
        draw_line_points.append(poses[i+1][:3, 3])
        line_indice.append([i,i+1])
    
    # draw_line_points=[]
    # line_indice=[]
    # k=0
    # for i in range(len(Line_3d)):
    #     draw_line_points.append(Line_3d[i][0:3])
    #     draw_line_points.append(Line_3d[i][3:6])
    #     line_indice.append([2*k,2*k+1])
    #     k+=1    

    # for i in range(len(Line_3d)):
        
    #     n=Line_3d[i][3:6]-Line_3d[i][0:3]
    #     l=np.linalg.norm(Line_3d[i][3:6]-Line_3d[i][0:3])    
    #     cylinder_coefficients = {'r': 0.005, 'x0': Line_3d[i][0], 'y0': Line_3d[i][1], 'z0': Line_3d[i][2], 'a': n[0], 'b': n[1], 'c': n[2]}
    #     cylinder_npy = create_cylinder(cylinder_coefficients=cylinder_coefficients, height=l, step=0.01, vis=True)
    #     draw_line_points.append(cylinder_npy)
        # draw_line_points.append((1-j/l)*Line_3d[i][0:3]+j/l*Line_3d[i][3:6])
            # draw_line_points.append(Line_3d[i][3:6])
            # line_indice.append([2*k,2*k+1])
            # k+=1
    # print(draw_line_points[0])
    lines_set = o3d.geometry.LineSet()
    lines_set.points = o3d.utility.Vector3dVector(draw_line_points)
    lines_set.lines = o3d.utility.Vector2iVector(line_indice)
    lines_set.paint_uniform_color([1,0,0])
    # lines_set = o3d.geometry.PointCloud()
    # lines_set.points = o3d.utility.Vector3dVector(draw_line_points)
    # # lines_set.lines = o3d.utility.Vector2iVector(line_indice)
    # lines_set.paint_uniform_color([1,0,0])
    # cloud.paint_uniform_color([0,1,0])

    # rgbd_pcd=o3d.geometry.PointCloud()
    # for i in range(len(poses)):
    #     depth_files = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/P000', f"depth_left/{i:06}_left_depth.npy")
    #     rgb_files = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/P000', f"image_left/{i:06}_left.png")
    #     # depth_files = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/P000', f"depth_left/{i:06}_left_depth.npy")
    #     # rgb_files = os.path.join('/home/arondight/G.Z/Open-structure/examples/cmu/P000', f"image_left/{i:06}_left.png")
    #     depth = np.load(depth_files)
    #     color=o3d.io.read_image(rgb_files)
    #     depth = o3d.geometry.Image(depth)
    #     # depth=o3d.io.read_image(depth_files)
    #     rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,depth_scale=1.0,depth_trunc=100,convert_rgb_to_intensity=False)
    #     inrinsic=o3d.camera.PinholeCameraIntrinsic(640,480,fx,fy,cx,cy)
    #     pcd=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,inrinsic,np.linalg.inv(poses[i]))
    #     rgbd_pcd+=pcd
    #     print(f"Processed frame {i+1}/{len(poses)}")
    
    # rgbd_pcd = rgbd_pcd.voxel_down_sample(voxel_size=0.06)
    # o3d.io.write_point_cloud('/home/arondight/G.Z/Open-structure/VENOM/rgbd_pcd_cmu.ply',rgbd_pcd,write_ascii=True,compressed=True)
    # pcd.transform([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    # o3d.visualization.draw_geometries([rgbd_pcd])

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # 添加点云和线段到窗口
    # vis.add_geometry(cloud)
    vis.add_geometry(lines_set) 
    # vis.add_geometry(rgbd_pcd)
    # vis.add_geometry(cylinder)   
    render=vis.get_render_option()    
    # render.line_width=5.0
    # render.point_size=3
    render.show_coordinate_frame=True

    # 显示可视化窗口
    vis.update_renderer()
    
    vis.run()
    vis.destroy_window()

    ##

    #matplotlib
    # fig=plt.figure()

    # ax=fig.add_subplot(111,projection='3d')
    
    # ax.scatter(np.asarray(rgbd_pcd.points)[:,0],np.asarray(rgbd_pcd.points)[:,1],np.asarray(rgbd_pcd.points)[:,2],c=np.asarray(rgbd_pcd.colors),alpha=0.5)

    # ax.scatter(np.asarray(cloud.points)[:,0],np.asarray(cloud.points)[:,1],np.asarray(cloud.points)[:,2],c='r',marker='.',s=5)
    # k=0
    # for line in lines_set.lines:
    #     ax.plot(np.asarray(lines_set.points)[line,0],np.asarray(lines_set.points)[line,1],np.asarray(lines_set.points)[line,2],c='g',linewidth=1)
    #     # ax.text()
    #     k+=1
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_ylim([-4,2])
    # # ax.set_box_aspect([1,1,1])
    # ax.axis('off')
    # f=ax.figure
    # f.canvas.mpl_connect('key_press_event',on_key)
    # # ax.view_init(azim=80,elev=-70)
    # plt.show()
    
# 运行主函数
if __name__ == "__main__":
    main()