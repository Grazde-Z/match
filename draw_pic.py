import cv2
import numpy as np
import os
import open3d as o3d
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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
            frame=np.array([float(elements[1]),float(elements[2]),float(elements[3]), float(elements[4])])
            # key=tuple(points[int(elements[1])].tolist())
            if len(MappointFrameAsso)==0:
                MappointFrameAsso=[frame]
            else:
                MappointFrameAsso=np.vstack((MappointFrameAsso,frame))
        if elements[0]=='MaplineFrameAsso:':
            frame=np.array([float(elements[1]),float(elements[2]),float(elements[3]), float(elements[4]),float(elements[5]), float(elements[6]), float(elements[7]),float(elements[8])])
            # key=tuple(lines[int(elements[1])].tolist())
            if len(MaplineFrameAsso)==0:
                MaplineFrameAsso=[frame]
            else:
                MaplineFrameAsso=np.vstack((MaplineFrameAsso,frame))
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
    datafile = '/home/arondight/G.Z/Open-structure/sphere2/input/venom_sequence.txt'

    poses,points,Line_3d,MappointFrameAsso,MaplineFrameAsso,paralines=read_file(datafile) 

    points_num = np.zeros(len(poses))
    lines_num = np.zeros(len(poses))

    # for i in range(len(MappointFrameAsso)):
    #     points_num[int(MappointFrameAsso[i][1])]+=1
    # for i in range(len(MaplineFrameAsso)):
    #     # print(MaplineFrameAsso[i][1])
    #     lines_num[int(MaplineFrameAsso[i][1])]+=1
    grid=np.zeros((64,48))
    all_grid=np.array(len(poses)*[grid])
    for i in range(len(MappointFrameAsso)):
        gr=all_grid[int(MappointFrameAsso[i][1])]
        posx=round(MappointFrameAsso[i][2]/10)
        posy=round(MappointFrameAsso[i][3]/10)
        if posx<0 or posx>63 or posy<0 or posy >47:
            continue
        gr[posx][posy]+=1

    grid_num=np.zeros((len(poses)))
    for i in range(len(all_grid)):
        gr=all_grid[i]
        grid_num[i]=len(gr[gr>0])
        
    for i in range(len(MaplineFrameAsso)):
        gr=all_grid[int(MaplineFrameAsso[i][1])]
        posx=round(MaplineFrameAsso[i][2]/10)
        posy=round(MaplineFrameAsso[i][3]/10)
        if posx<0 or posx>63 or posy<0 or posy >47:
            continue
        gr[posx][posy]+=1  
        posx_2=round(MaplineFrameAsso[i][5]/10)
        posy_2=round(MaplineFrameAsso[i][6]/10)
        if posx_2<0 or posx_2>63 or posy_2<0 or posy_2 >47:
            continue
        gr[posx_2][posy_2]+=1  
    
    PL_num=np.zeros((len(poses)))
    for i in range(len(all_grid)):
        gr=all_grid[i]
        PL_num[i]=len(gr[gr>0])    
    # print(lines_num)
    # print(points_num)
    # 添加标题和坐标轴标签
    
    plt.bar(range(len(PL_num)), PL_num, alpha=0.5, label='point&line')
    plt.bar(range(len(grid_num)), grid_num, alpha=0.5, label='point')
    # plt.bar(range(len(lines_num)), lines_num, alpha=0.5, label='Line')

    # plt.title('Feature Count')
    plt.xlabel('Frame Number')
    plt.ylabel('Grid Number')
    x_major_locator = MultipleLocator(10)
    x_major_formatter = FormatStrFormatter('%d')
    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().xaxis.set_major_formatter(x_major_formatter)
    # 设置x轴刻度标签
    # plt.xticks(range(len(points_num)), points_num)
    plt.xlim(0, len(points_num))
    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()
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