import math
import numpy as np
from numpy.core.defchararray import lower, split
from numpy.core.fromnumeric import mean
from utils import *
from plyfile import *
import cv2



def get_depth(extrinsics,structure):
    h,w = 512,1024
    size = max(w,h)
    depth_img = np.zeros((len(extrinsics_room),h,w))

    for i in range(len(structure)):
        X = np.array(structure[i]["X"])
        X_word = X.reshape(3,1)
        ob = structure[i]["ob"]
        for j in range(len(ob)):
            key = ob[j]["key"]
            ob_x = ob[j]["ob_x"]#获取2D观测值
            pose_C = extrinsics[str(key)]
            cam_o = pose_C[:,-1].reshape(3,1)
            X_cam = np.dot(pose_C[:,:-2],X_word) + pose_C[:,-2].reshape(3,1)#X转投相机坐标系下
            lon = math.atan2(X_cam[0,0],X_cam[2,0])
            lat = math.atan2(-X_cam[1][0],math.hypot(X_cam[0,0],X_cam[2,0]))
            img_w = lon/(2*math.pi)*size+w//2
            img_h =-lat/(2*math.pi)*size+h//2
            #print(img_w,img_h)
            #print(ob_x)
            #print(cam_o)
            depth_img[key][int(img_h)][int(img_w)] = np.linalg.norm(X-cam_o)
            #depth_img[key][int(ob_x[1])][int(ob_x[0])] = np.linalg.norm(X-cam_o)
            #print(depth_img[key][int(img_h)][int(img_w)])
    return depth_img

def scale(depth_mvg,depth_img,depth_name):
    scales =  depth_mvg / depth_img
    all_scales = []
    for i in range(len(scales)):
        for j in range(len(scales[0])):
            if scales[i][j] != 0:
                #print("mvg:",depth_mvg[i][j])
                #print("depth_img:",depth_img[i][j])
                all_scales.append(scales[i][j])
    inv_all_scales = np.divide(np.ones_like(all_scales)*1.0, all_scales) 
    scale_median = np.median(inv_all_scales)
    scale_mean = np.mean(inv_all_scales)
    print(f"Scale[{depth_name}]: scale={scale_median}, mean={scale_mean},std={np.std(inv_all_scales)}")
    return scale_median

'''
josn_238 = "out_238\\reconstruction_sequential\\sfm_data.json"
extrinsics_238 = get_extrinsics_from_json(josn_238)
print(extrinsics_238)
for i in range(1,len(extrinsics_238)):
    forward = extrinsics_238["{}".format(i-1)]
    backward = extrinsics_238["{}".format(i)]
    dist = np.linalg.norm(forward[:,-1]-backward[:,-1])
    print("distance[{}-{}]: {}".format(i-1,i,dist))
structure_238 = get_structure(josn_238)
depth_img_238 =  get_depth(extrinsics_238,structure_238)
np.save("depth_mvg_238",depth_img_238)
'''



json = "tt_sfm_data.json"
structure_me = get_structure(json)
extrinsics_me = get_extrinsics_from_json(json)
#相机轨迹
trace = []
centers = []
for i in range(len(extrinsics_me)):   
    pose_c = extrinsics_me[str(i)]
    centers.append(pose_c[:,-1])
    C = -np.dot(pose_c[:,:-2].T,pose_c[:,-2])
    trace.append(C)
trace = np.array(trace)
centers = np.array(centers)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(trace)
o3d.visualization.draw_geometries([
    pcd,
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
])
#o3d.io.write_point_cloud("room/centers.ply", pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(trace)
o3d.visualization.draw_geometries([
    pcd,
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
])
#o3d.io.write_point_cloud("room/rt.ply", pcd)



json_room = "room/sfm_data.json"
structure_room = get_structure(json_room)
extrinsics_room = get_extrinsics_from_json(json_room)
views = get_view_id(json_room)
depth_img_room =  get_depth(extrinsics_room,structure_room)
all_scale = []

for i in range(len(views)):
    depth_name = views[i].split(".")
    depth_name = depth_name[0]+"."+depth_name[1]+"."+"depth"+"."+depth_name[2]
    depth = cv2.imread(f"depth\\{depth_name}", -1)/1000.0
    out_scale = scale(depth_img_room[i],depth_img=depth,depth_name=views[i])
    all_scale.append(out_scale)
print(all_scale)
