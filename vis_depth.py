import json
import numpy as np
import open3d as o3d
from imageio import imread

all_scale_room = [0.09532445588746995, 0.08617541948175494, 0.09499927370128586, 0.09652187447471763, 0.08979214933765373, 
0.10353885946450637, 0.09503845550833112, 0.10238575000988526, 0.11081184439387695, 0.12363840486751498, 0.15251448791494363, 
0.15661335067841387, 0.18713331489605997, 0.17783178760727475, 0.18284100851143337, 0.18206582372718494, 0.1867436692564775, 0.18382028220976307, 
0.18388434262187525, 0.16182820232645992, 0.15289835946161096, 0.15064338241149117, 0.1480538635391014, 0.13922165539280343, 0.1125603631632939, 
0.11841965085531636, 0.1221615842827267, 0.10694583362985773, 0.09487629464563535, 0.09579106692521552, 0.11587878186627598, 0.1407501908475467,
0.14337854291211688, 0.14652920330358565, 0.1655808971250752, 0.17712931424034334, 0.17126900410591026, 0.1784884175658306, 0.18018996238219467, 
0.17722564750707243, 0.173800713235398, 0.16413686683670958, 0.1484954478879686, 0.14593499794421527, 0.14988035908729347, 0.13547892151738194, 
0.12924405644487558, 0.12409107196878401, 0.11700119361137296, 0.11481601338937554, 0.09731324061760901, 0.09244269146086104]

def get_uni_sphere_xyz(H, W):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (i+0.5) / W * 2 * np.pi
    v = ((j+0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img',default="room\\images_room\\873_v.JPG.png",
                        help='Image texture in equirectangular format')
    parser.add_argument('--depth',default="depth\\873_v.JPG.depth.png",
                        help='Depth map')
    parser.add_argument('--scale', default=0.1221615842827267, type=float,
                        help='Rescale the depth map')
    parser.add_argument('--crop_ratio', default=80/512, type=float,
                        help='Crop ratio for upper and lower part of the image')
    parser.add_argument('--crop_z_above', default=1.2, type=float,
                        help='Filter 3D point with z coordinate above')
    args = parser.parse_args()

    # Reading rgb-d
    rgb = imread(args.img)
    depth = imread(args.depth)[...,None].astype(np.float32) #* args.scale

    # Project to 3d
    H, W = rgb.shape[:2]
    xyz = depth * get_uni_sphere_xyz(H, W)
    xyzrgb = np.concatenate([xyz, rgb/255.], 2)

    # Crop the image and flatten
    if args.crop_ratio > 0:
        assert args.crop_ratio < 1
        crop = int(H * args.crop_ratio)
        xyzrgb = xyzrgb[crop:-crop]
    xyzrgb = xyzrgb.reshape(-1, 6)

    # Crop in 3d
    #xyzrgb = xyzrgb[xyzrgb[:,2] <= args.crop_z_above]

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])

    o3d.visualization.draw_geometries([
        pcd,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    ])
    o3d.io.write_point_cloud("room/orign_ply/{}.ply".format('873_v'), pcd)
