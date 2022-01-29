import numpy as np
import json
import cv2 as  cv
import math
import open3d as o3d
import matplotlib.pyplot as plt

def read_json(src):
    with open(src,"r") as f:
        dict = json.load(f)
    return dict

def get_extrinsics_from_json(src):
    dict= read_json(src)
    extrinsics_data = dict["extrinsics"]
    extrinsics ={} #[R|t]=[R|−RC]
    for i in range(len(extrinsics_data)):
        key = extrinsics_data[i]["key"]
        value = extrinsics_data[i]["value"]
        R = np.array(value["rotation"]).reshape(3,3)
        C = np.array(value["center"]).reshape(3,1)
        t = -np.dot(R,C)
        P = np.concatenate((R,t),axis=1)
        extrinsics.update({str(key):P})
    return extrinsics

def dense2mvg(dense_path,pose,color_path):
    dense = cv.imread(dense_path,-1)
    color_img = cv.imread(color_path)
    h,w = dense.shape
    size = max(h,w)
    ##2d-->>3d
    all_points = []
    all_points_color = []
    for i in range(w):
        for j in range(h):
            p_x = (i-w//2)/size
            p_y = (j-h//2)/size
            lon = p_x*2*math.pi
            lat = -p_y*2*math.pi
            sphere_xyz = [math.cos(lat)*math.sin(lon),-math.sin(lat),math.cos(lat)*math.cos(lon)]
            sphere_xyz=np.array(sphere_xyz).reshape(3,1)
            word =np.dot(np.linalg.inv(pose[:,:-1]),sphere_xyz-pose[:,-1].reshape(3,1))*dense[j,i]
            all_points.append(word)
    
    for i in range(w):
        for j in range(h):  
            all_points_color.append(color_img[j][i]/255.0)

    return all_points,all_points_color
    

all_scale_room = [0.1159696669279481, 0.13301449660053605, 0.0710554260424773, 0.24726722491773295, 0.11836532852131192]
img_name = ["899_0026.JPG.depth.png","956_0028.JPG.depth.png","1018_0030.JPG.depth.png","1109_0032.JPG.depth.png","1165_0034.JPG.depth.png"]
json_room = "room/sfm_data.json"

extrinsics_room = get_extrinsics_from_json(json_room)
depth_index = [2,11,28,43,51]
camera_pose = []
for i in range(len(all_scale_room)):
    t = extrinsics_room[str(depth_index[i])][:,-1]
    t = t*all_scale_room[i]
    extrinsics_room[str(depth_index[i])][:,-1] = t
    camera_pose.append(extrinsics_room[str(depth_index[i])])

all_points_room1,all_points_color_room1 = dense2mvg("room\\depth\\899_0026.JPG.depth.png",camera_pose[0],"room\\img\\899_0026.JPG.png")
all_points_room2,all_points_color_room2 = dense2mvg("room\\depth\956_0028.JPG.depth.png",camera_pose[1],"room\\img\\956_0028.JPG.png")
all_points_room3,all_points_color_room3 = dense2mvg("room\\depth\\1018_0030.JPG.depth.png",camera_pose[2],"room\\img\\1018_0030.JPG.png")
all_points_room4,all_points_color_room4 = dense2mvg("room\\depth\\1109_0032.JPG.depth.png",camera_pose[3],"room\\img\\1109_0032.JPG.png")
all_points_room5,all_points_color_room5 = dense2mvg("room\\depth\\1165_0034.JPG.depth.png",camera_pose[4],"room\\img\\1165_0034.JPG.png")
all_points = [all_points_room1,all_points_room2,all_points_room3,all_points_room4,all_points_room5]
all_points_color = [all_points_color_room1,all_points_color_room2,all_points_color_room3,all_points_color_room4,all_points_color_room5]

'''
all_points_205,all_points_color_205 = dense2mvg("depth_238\\205_v.JPG.depth.png",camera_pose[0],"color_238\\205_v.JPG.jpg")
all_points_238,all_points_color_238 = dense2mvg("depth_238\\238_v.JPG.depth.png",camera_pose[1],"color_238\\238_v.JPG.jpg")
all_points_245,all_points_color_245 = dense2mvg("depth_238\\245_v.JPG.depth.png",camera_pose[2],"color_238\\245_v.JPG.jpg")
all_points_254,all_points_color_254 = dense2mvg("depth_238\\254_v.JPG.depth.png",camera_pose[3],"color_238\\254_v.JPG.jpg")
all_points_260,all_points_color_260 = dense2mvg("depth_238\\260_v.JPG.depth.png",camera_pose[4],"color_238\\260_v.JPG.jpg")
all_points = [all_points_205,all_points_238,all_points_245,all_points_254,all_points_260]
all_points_color = [all_points_color_205,all_points_color_238,all_points_color_245,all_points_color_254,all_points_color_260]
'''



pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(all_points).reshape(-1,3))
pcd.colors = o3d.utility.Vector3dVector(np.array(all_points_color).reshape(-1,3))
#o3d.io.write_point_cloud("205_v.ply", pcd)
o3d.visualization.draw_geometries([
    pcd,
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
])




'''
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
volume = o3d.pipelines.integration.UniformTSDFVolume(
    length=4.0,
    resolution=512,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

pic_list = [205,238,245,254,260]

for i in range(len(camera_pose)):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image("color_238/{}_v.JPG.jpg".format(pic_list[i]))
    depth = o3d.io.read_image("depth_238/{}_v.JPG.depth.png".format(pic_list[i]))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,depth_trunc=4.0, convert_rgb_to_intensity=False)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    volume.integrate(
        rgbd,
        camera_intrinsics,
        np.linalg.inv(camera_pose[i]),
    )


    print("Extract triangle mesh")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    print("Extract voxel-aligned debugging point cloud")
    voxel_pcd = volume.extract_voxel_point_cloud()
    o3d.visualization.draw_geometries([voxel_pcd])

    print("Extract voxel-aligned debugging voxel grid")
    voxel_grid = volume.extract_voxel_grid()
    o3d.visualization.draw_geometries([voxel_grid])

    print("Extract point cloud")
    pcd = volume.extract_point_cloud()
    o3d.visualization.draw_geometries([pcd])

'''