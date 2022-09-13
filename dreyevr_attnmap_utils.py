from pathlib import Path
import cv2
import pickle as pkl
import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R


################
# Geometric Fns
################

def ptsWorld2Cam(focus_hitpt, world2camMatrix, K):
    
    tick_focus_hitpt_homog = np.hstack((focus_hitpt,1))    
    sensor_points = np.dot(world2camMatrix, tick_focus_hitpt_homog)

    # Now we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    # points_2d = np.array([
    #     points_2d[0, :] / points_2d[2, :],
    #     points_2d[1, :] / points_2d[2, :],
    #     points_2d[2, :]])
#     print(points_2d)
    points_2d /= points_2d[2]
#     print(points_2d)

    # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
    # contains all the y values of our points. In order to properly
    # visualize everything on a screen, the points that are out of the screen
    # must be discarted, the same with points behind the camera projection plane.
    # points_2d = points_2d.T

    # Extract the screen coords (uv) as integers.
    u_coord = points_2d[0].astype(np.int)
    v_coord = points_2d[1].astype(np.int)
    
    return (u_coord, v_coord)

def world2pixels(focus_hitpt, vehicle_transform, K):
    '''
    takes in the dataframe row with all the information of where the world is currently 
    '''        
    vehicleP = vehicle_transform.get_matrix()
    
    # center image
    camera_loc_offset = carla.Location(x=1.3, y=0, z=1.3)    
    camera_rot_offset = carla.Rotation(pitch=0, yaw=0, roll=0)
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())    
    
    u,v = ptsWorld2Cam(focus_hitpt, world2cam, K)
#     u = np.clip(u, 0, cam_info['w']-1)
#     v = np.clip(v, 0, cam_info['h']-1)
    pts_mid = (u,v)
        
    # left image  
    camera_loc_offset = carla.Location(x=1.2, y=-0.25, z=1.3)
    camera_rot_offset = carla.Rotation(pitch=0, yaw=-45, roll=0)
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)    
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())
        
    u,v = ptsWorld2Cam(focus_hitpt, world2cam, K)
#     u = np.clip(u, 0, cam_info['w']-1)
#     v = np.clip(v, 0, cam_info['h']-1)
    pts_left = (u,v)
    
    # right image  
    camera_loc_offset = carla.Location(x=1.2, y=0.25, z=1.3)
    camera_rot_offset = carla.Rotation(pitch=0, yaw=45, roll=0)
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)    
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())
        
    u,v = ptsWorld2Cam(focus_hitpt, world2cam, K)
#     u = np.clip(u, 0, cam_info['w']-1)
#     v = np.clip(v, 0, cam_info['h']-1)
    pts_right = (u,v)    
    
    return pts_mid, pts_left, pts_right


#########################
# DReyeVR dataloader Fns
#########################

import sys
# sys.path.append("/scratch/abhijatb/Bosch22")
sys.path.append("../Bosch22")
from dreyevr_parser.parser import parse_file
from typing import Dict, List, Any
from utils import (
    check_for_periph_data,
    convert_to_df,
    split_along_subgroup,
    get_good_idxs,
)
from visualizer import plot_versus
from parser_utils import GetGazeDeviationFromHead
from ibmmpy.src.ibmmpy.ibmm import EyeClassifier


def load_dreyevr_data_and_gazeevent_annotate(recordingtxt_path):
    df_new = load_dreyevr_dataframe(recordingtxt_path)
    df_new = add_gaze_event2df(df_new)
    return df_new
    
    
def load_dreyevr_dataframe(recordingtxt_path):
    '''
    load _dreyevr_dataframe
    '''
    data: Dict[str, np.ndarray or dict] = parse_file(str(recordingtxt_path))
    """append/generate periph data if available"""
    # check for periph data
    PeriphData = check_for_periph_data(data)
    if PeriphData is not None:
        data["PeriphData"] = PeriphData

    """convert to pandas df"""
    # need to split along groups so all data lengths are the same
    data_groups = split_along_subgroup(data, ["CustomActor"])
    data_groups_df: List[pd.DataFrame] = [convert_to_df(x) for x in data_groups]
    df_new = data_groups_df[0] 
    return df_new


def add_gaze_event2df(df_new):    
    '''
    Given the data_df add a column for gaze events
    '''
    
    df2 = df_new.copy()
    # add approx head compensation
    df2['Cgaze_x'] = df_new.GazeDir_COMBINED.apply(lambda x: x[0])
    df2['Cgaze_y'] = df_new.GazeDir_COMBINED.apply(lambda x: x[1])
    df2['Cgaze_z'] = df_new.GazeDir_COMBINED.apply(lambda x: x[2])

    # gaze+head values
    gaze_pitches, gaze_yaws = GetGazeDeviationFromHead(df2.Cgaze_x, df2.Cgaze_y, df2.Cgaze_z)
    # head_rots = df2.CameraRot.values
    head_pitches =   df2.CameraRot.apply(lambda x: x[0])
    head_yaws = df2.CameraRot.apply(lambda x: x[2])
    gaze_head_pitches = gaze_pitches + head_pitches
    gaze_head_yaws = gaze_yaws + head_yaws       

    # Create the new pd
    gazeHeadDF = pd.DataFrame(df2[['TimeElapsed']])
    gazeHeadDF = gazeHeadDF.rename(columns={'TimeElapsed':'timestamp'})
    gazeHeadDF['confidence'] = (df2.EyeOpennessValid_LEFT*df2.EyeOpennessValid_RIGHT).astype(bool)
#     gazeHeadDF['x'] = gaze_head_pitches
#     gazeHeadDF['y'] = gaze_head_yaws
#     gazeHeadDF['z'] = np.zeros(len(gaze_head_pitches))
    gazeHeadDF['x'] = df2['Cgaze_x']
    gazeHeadDF['y'] = df2['Cgaze_y']
    gazeHeadDF['z'] = df2['Cgaze_z']

    vel_w = EyeClassifier.preprocess(gazeHeadDF, dist_method="vector")
#     vel_w = EyeClassifier.preprocess(gazeHeadDF, dist_method="euclidean")
    model = EyeClassifier()
    model.fit(world=vel_w)
    # raw_vel = vel_w[np.logical_not(vel_w.velocity.isna())].velocity.values
    # raw_vel[raw_vel > raw_vel.mean() + 3 * raw_vel.std()]
    # print("Velocity Means: ",model.world_model.means_)
    # 0- fix, 1- sacc, -1 ->noise
    labels, indiv_labels = model.predict(world=vel_w)
    labels_unique = labels
    
    df_new = df_new.join(labels_unique["label"])
    return df_new    