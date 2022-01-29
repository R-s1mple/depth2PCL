import numpy as np
import json
import cv2 as  cv
import math
from plyfile import *
from numpy.core.records import array
import open3d as o3d
import matplotlib.pyplot as plt


def ply2arr(src):
    plydata = PlyData.read(src)
    x = plydata["vertex"]["x"]
    y = plydata["vertex"]["y"]
    z = plydata["vertex"]["z"]
    points = np.stack([x,y,z],-1)
    return points

def read_json(src):
    with open(src,"r") as f:
        dict = json.load(f)
    return dict

def get_view_id(src):
    dict = read_json(src)
    views = dict["views"]
    id_filename={}
    for i in range(len(views)):
        filename = views[i]["value"]["ptr_wrapper"]["data"]["filename"]
        id_pose = views[i]["value"]["ptr_wrapper"]["data"]["id_pose"]
        id_filename.update({id_pose:filename})
    return id_filename

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
        P = np.concatenate((R,t,C),axis=1)
        extrinsics.update({str(key):P})
    return extrinsics
    
def get_structure(src):
    dict_dir = read_json(src)
    structure = []

    for i in range(len(dict_dir["structure"])):
        value = dict_dir["structure"][i]["value"]
        X = value["X"]
        ob = value["observations"]
        observations = []
        for j in range(len(ob)):
            observations.append({"key":ob[j]["key"],"ob_x":ob[j]["value"]["x"]})
        structure.append({"X":X,"ob":observations})
    return structure

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
    