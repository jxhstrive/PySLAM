import os
import sys
import csv
import copy
import time
import random
import argparse
import glob
import logging

import numpy as np
np.set_printoptions(precision=4)

from tqdm import tqdm
import geohash

from utils.ScanContextManager import *
from utils.PoseGraphManager import *
from utils.UtilsMisc import *
import utils.UtilsPointcloud as Ptutils
import utils.ICP as ICP
import utils.UtilsIMU as IMU
import open3d as o3d

import imo_pcd_reader

# params
parser = argparse.ArgumentParser(description='PyICP SLAM arguments')

parser.add_argument('--num_icp_points', type=int, default=5000) # 5000 is enough for real time

parser.add_argument('--num_rings', type=int, default=20) # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60) # same as the original paper
parser.add_argument('--num_candidates', type=int, default=20) # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=10) # same as the original paper

parser.add_argument('--loop_threshold', type=float, default=0.11) # 0.11 is usually safe (for avoiding false loop closure)

parser.add_argument('--data_base_dir', type=str, 
                    default='/your/path/.../data_odometry_velodyne/dataset/sequences')
parser.add_argument('--sequence_idx', type=str, default='00')

parser.add_argument('--save_gap', type=int, default=300)

parser.add_argument('--IMU_dir', type=str, default='IMU.csv')

parser.add_argument('--indoor', action="store_true", default=False, help="will use lidar odometry if present (default: outdoor)")

args = parser.parse_args()

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    print(f"Logs are being saved to: {log_file}")

dataset_dir = os.path.abspath(os.path.join(args.data_base_dir, os.pardir))
log_dir = os.path.join("result", os.path.basename(dataset_dir))
if not os.path.exists(log_dir): os.makedirs(log_dir)
log_file = os.path.join(log_dir, "log.txt")
setup_logging(log_file)

# dataset 
folder_paths = glob.glob(os.path.join(args.data_base_dir, "scene*"))
folder_names = [os.path.basename(path) for path in folder_paths if os.path.isdir(path)]
scene_names = sorted(folder_names, key=lambda x: int(x.replace("scene", "")))

time_break_th = 1 # seconds

pcd_segments = []
tmp_seg = []
pc_extension = ".pcd"
good_ratio_th = 0.5
min_frame_th = 200
for index, scene_n in enumerate(scene_names):
    this_dir = os.path.join(args.data_base_dir, scene_n, 'lidarFusion_pcd')
    tmp_pcd_names = os.listdir(this_dir)
    pcd_names = [f for f in os.listdir(this_dir) if f.endswith(pc_extension)]
    if len(pcd_names) == 0:
        continue
    if(len(pcd_names) < len(tmp_pcd_names)):
        logging.warning(f"Transmission errors in {scene_n}, good ratio: {len(pcd_names)}/{len(tmp_pcd_names)}")
        good_ratio = float(len(pcd_names))/len(tmp_pcd_names)
        if good_ratio < good_ratio_th:
            if len(tmp_seg) > min_frame_th:
                pcd_segments.append(tmp_seg)
                logging.info(f"Segment {len(pcd_segments)}, length: {len(tmp_seg)}, time gap: at least 20 seconds")
            tmp_seg = []
            continue
    pcd_names.sort()
    pcd_paths = [os.path.join(this_dir, name) for name in pcd_names]
    tmp_seg.extend(pcd_paths)
    if index == (len(scene_names)-1):
        if len(tmp_seg) > min_frame_th:
            pcd_segments.append(tmp_seg)
            logging.info(f"Segment {len(pcd_segments)}, length: {len(tmp_seg)}")
    else:
        next_dir = os.path.join(args.data_base_dir, scene_names[index+1], 'lidarFusion_pcd')
        next_pcd_names = [f for f in os.listdir(next_dir) if f.endswith(pc_extension)]
        if len(next_pcd_names) == 0:
            if len(tmp_seg) > min_frame_th:
                pcd_segments.append(tmp_seg)
                logging.info(f"Segment {len(pcd_segments)}, length: {len(tmp_seg)}, time gap: at least 20 seconds")
            tmp_seg = []
            continue
        next_pcd_names.sort()
        next_pcd_paths = [os.path.join(next_dir, name) for name in next_pcd_names]
        this_end_dt = IMU.pcd_name_2_dt(pcd_paths[-1])
        next_begin_dt = IMU.pcd_name_2_dt(next_pcd_paths[0])
        delta_dt = next_begin_dt - this_end_dt
        if delta_dt.total_seconds() < 0 or delta_dt.total_seconds() > time_break_th:
            if len(tmp_seg) > min_frame_th:
                pcd_segments.append(tmp_seg)
                logging.info(f"Segment {len(pcd_segments)}, length: {len(tmp_seg)}, time gap: {delta_dt.total_seconds()} seconds")
            tmp_seg = []

parent_base_dir = os.path.abspath(os.path.join(args.data_base_dir, os.pardir))
pose = IMU.IMUPose(os.path.join(parent_base_dir, "IMU.csv"))

# Check IMU
logging.info(f">>>>>>>>>>>>>>>> Cheking IMU data coverage >>>>>>>>>>>>>>>>")
IMU_all_good = True
for seg_idx, scan_paths in enumerate(pcd_segments):
    dirname, filename = os.path.split(scan_paths[0])
    parts = filename.split('_')
    query_timestamp = parts[0] + '_' + parts[1]
    Latitude, Longitude, Altitude = pose.getGPS(query_timestamp)
    if Latitude is None:
        logging.warning(f"No IMU data for {scan_paths[0]}")
    dirname, filename = os.path.split(scan_paths[-1])
    parts = filename.split('_')
    query_timestamp = parts[0] + '_' + parts[1]
    Latitude_e, Longitude_e, Altitude_e = pose.getGPS(query_timestamp)
    if Latitude_e is None:
        logging.warning(f"No IMU data for {scan_paths[-1]}")
    if Latitude is None or Latitude_e is None:
        logging.warning(f"Caution! Segment {seg_idx+1} may have no GPS IMU data coverage!")
        IMU_all_good = False

if IMU_all_good:
    logging.info(f"IMU data all good!")

idx = -1

for scan_paths in pcd_segments:
    num_frames = len(scan_paths)
    if num_frames >= min_frame_th:
        idx = idx + 1

        map_idx = "map" + str(idx)
        parent_dir = os.path.abspath(os.path.join(args.data_base_dir, os.pardir))
        save_dir = os.path.join("result", os.path.basename(parent_dir), map_idx)

        if os.path.exists(save_dir):
            pose_part = "pose*.csv"
            pcs_pth_part = "*_pcs_data_path.csv"
            pose_matching = glob.glob(os.path.join(save_dir, pose_part))
            pcs_pth_matching = glob.glob(os.path.join(save_dir, pcs_pth_part))
            if pose_matching and pcs_pth_matching:
                continue

        logging.info(f">>>>>>>>>>>>>>>> Start Processing Segment {idx+1} >>>>>>>>>>>>>>>>")

        # Pose Graph Manager (for back-end optimization) initialization
        PGM = PoseGraphManager()
        PGM.addPriorFactor()

        # Result saver
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3, 
                                        save_gap=(num_frames-1),
                                        num_frames=num_frames,
                                        seq_idx=map_idx,
                                        save_dir=save_dir)
        
        # Scan Context Manager (for loop detection) initialization
        SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors], 
                                                num_candidates=args.num_candidates, 
                                                threshold=args.loop_threshold)

        # for save the results as a video
        fig_idx = 1
        fig = plt.figure(fig_idx)
        num_frames_to_skip_to_show = 5

        latitude_list = []
        longitude_list = []
        altitude_list = []

        ref_timestamp = ''
        query_timestamp = ''
        # @@@ MAIN @@@: data stream
        for for_idx, scan_path in tqdm(enumerate(scan_paths), total=num_frames, mininterval=10.0):

            ref_timestamp = copy.deepcopy(query_timestamp)
            dirname, filename = os.path.split(scan_path)
            parts = filename.split('_')
            query_timestamp = parts[0] + '_' + parts[1]
            pose_trans = np.eye(4)
            if(for_idx > 0):
                pose_trans = pose.getTransformationMatrix(ref_timestamp, query_timestamp)

            Latitude, Longitude, Altitude = pose.getGPS(query_timestamp)
            if Latitude is not None:
                latitude_list.append(Latitude)
                longitude_list.append(Longitude)
                altitude_list.append(Altitude)

            # get current information
            curr_scan_pts = Ptutils.readScan(scan_path)

            curr_scan_down_pts = Ptutils.random_sampling(curr_scan_pts, num_points=args.num_icp_points)

            # save current node
            PGM.curr_node_idx = for_idx # make start with 0
            SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=curr_scan_down_pts)
            if(PGM.curr_node_idx == 0):
                PGM.prev_node_idx = PGM.curr_node_idx
                prev_scan_pts = copy.deepcopy(curr_scan_pts)
                icp_initial = np.eye(4)
                continue

            
            prev_scan_down_pts = Ptutils.random_sampling(prev_scan_pts, num_points=args.num_icp_points)

            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(curr_scan_down_pts)

            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(prev_scan_down_pts)

            c_d_th = 0.8
            if args.indoor:
                final_transformation, has_converged, fitness_score = imo_pcd_reader.performNDT(curr_scan_pts, prev_scan_pts, icp_initial, 0.2, 0.4, 0.01, 0.1, 50)
                logging.info(f"idx: {for_idx}, has_converged: {has_converged}, fitness_score: {fitness_score}")
                if fitness_score > c_d_th and pose_trans is not None:
                    imu_final_transformation, imu_has_converged, imu_fitness_score = imo_pcd_reader.performNDT(curr_scan_pts, prev_scan_pts, pose_trans, 0.2, 0.4, 0.01, 0.1, 50)
                    if imu_fitness_score < fitness_score:
                        logging.warning(f"idx: {for_idx}, lidar odometry fitness_score too high, IMU recalculated fitness_score: {imu_fitness_score}")
                        final_transformation = imu_final_transformation
                        fitness_score = imu_fitness_score
            else:
                fitness_score = None
                if pose_trans is not None:
                    final_transformation, has_converged, fitness_score = imo_pcd_reader.performNDT(curr_scan_pts, prev_scan_pts, pose_trans, 0.2, 0.4, 0.01, 0.1, 50)
                    logging.info(f"idx: {for_idx}, has_converged: {has_converged}, fitness_score: {fitness_score}")

                if fitness_score is None:
                    final_transformation, has_converged, fitness_score = imo_pcd_reader.performNDT(curr_scan_pts, prev_scan_pts, icp_initial, 0.2, 0.4, 0.01, 0.1, 50)
                    logging.warning(f"idx: {for_idx}, No GPS IMU info, use lidar odometry. fitness_score: {fitness_score}")
                else:
                    if fitness_score > c_d_th:
                        odom_final_transformation, odom_has_converged, odom_fitness_score = imo_pcd_reader.performNDT(curr_scan_pts, prev_scan_pts, icp_initial, 0.2, 0.4, 0.01, 0.1, 50)
                        if odom_fitness_score < fitness_score:
                            logging.warning(f"idx: {for_idx}, fitness_score too high, maybe GPS is unreliable, lidar odometry recalculated fitness_score: {odom_fitness_score}")
                            final_transformation = odom_final_transformation
                            fitness_score = odom_fitness_score

            if 2*fitness_score > c_d_th:
                max_c_d = c_d_th
            else:
                max_c_d = 2*fitness_score

            reg_p2p = o3d.pipelines.registration.registration_generalized_icp(source = source, 
                                                                    target = target, 
                                                                    max_correspondence_distance = max_c_d,
                                                                    init = final_transformation
                                                                    )
            odom_transform = reg_p2p.transformation

            # odom_transform = pose_trans

            # update the current (moved) pose 
            PGM.curr_se3 = np.matmul(PGM.curr_se3, odom_transform)
            icp_initial = odom_transform # assumption: constant velocity model (for better next ICP converges)

            # add the odometry factor to the graph 
            PGM.addOdometryFactor(odom_transform)

            # renewal the prev information 
            PGM.prev_node_idx = PGM.curr_node_idx
            prev_scan_pts = copy.deepcopy(curr_scan_pts)

            # # loop detection and optimize the graph 
            # if(PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0): 
            #     # 1/ loop detection 
            #     loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
            #     if(loop_idx == None): # NOT FOUND
            #         pass
            #     else:
            #         logging.info(f"Loop event detected: {PGM.curr_node_idx}, {loop_idx}, {loop_dist}")
            #         # 2-1/ add the loop factor 
            #         loop_scan_down_pts = SCM.getPtcloud(loop_idx)
            #         loop_transform, _, _ = ICP.icp(curr_scan_down_pts, loop_scan_down_pts, init_pose=yawdeg2se3(yaw_diff_deg), max_iterations=20)
            #         PGM.addLoopFactor(loop_transform, loop_idx)

            #         # 2-2/ graph optimization 
            #         PGM.optimizePoseGraph()

            #         # 2-2/ save optimized poses
            #         ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)

            # save the ICP odometry pose result (no loop closure)
            ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx) 
            if(for_idx % num_frames_to_skip_to_show == 0): 
                ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
        
        if len(latitude_list) > 0:
            mid_latitude = sum(latitude_list) / len(latitude_list)
            mid_longitude = sum(longitude_list) / len(longitude_list)
            mid_altitude = sum(altitude_list) / len(altitude_list)
            latitude_min = min(latitude_list)
            latitude_max = max(latitude_list)
            longitude_min = min(longitude_list)
            longitude_max = max(longitude_list)
            range_x, range_y, range_z = pose._gps_to_enu(latitude_min, longitude_min, mid_altitude, latitude_max, longitude_max, mid_altitude)
            scan_paths_name = str(geohash.encode(mid_latitude, mid_longitude, 12)) + "_" + str(int(range_y)) + "_" + str(int(range_x)) + "_0_pcs_data_path.csv"
        else:
            scan_paths_name = "NULL_0_0_0_pcs_data_path.csv"
        # Save scan_paths to CSV
        scan_paths_csv = os.path.join(save_dir, scan_paths_name)
        with open(scan_paths_csv, "w", newline="") as csvfile:  # Open in write mode
            writer = csv.writer(csvfile)
            writer.writerow(["File Path"])
            for path in scan_paths:
                writer.writerow([path])
