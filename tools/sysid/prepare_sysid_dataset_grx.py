import h5py
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from rot_trans import conver_ortho6d_to_euler, convert_euler_to_quaternion, compute_rotation_matrix_from_ortho6d, convert_rotation_matrix_to_euler
import sapien.core as sapien
import os
import pickle

def calculate_match_index(input_dir, hdf5_file_name):
    frame_ts, _ = load_frame_timestamps(Path(input_dir) / hdf5_file_name / 'top/rgb')
    data_ts = load_data_timestamps(Path(input_dir) / f"{hdf5_file_name}.hdf5")

    print(f'frame:{frame_ts[0], frame_ts[-1]}, data:{data_ts[0], data_ts[-1]}')
    print(frame_ts[-1] - frame_ts[0])
    print(data_ts[-1] - data_ts[0])
    print(data_ts[0] - frame_ts[0])
    start_ts = max(frame_ts[0], data_ts[0])
    end_ts = min(frame_ts[-1], data_ts[-1])
    print(f"Episode start from {start_ts}s to {end_ts}s")

    # discard frames before and after the timestamps
    frame_ts = [ts for ts in frame_ts if ts >= start_ts and ts <= end_ts]
    # video_ids = [id for id in video_ids if int(id) >= start_ts * 20 and int(id) <= end_ts * 20]
    data_ts = [ts for ts in data_ts if ts >= start_ts and ts <= end_ts]

    # match video and data timestamps
    matched_ts = match_timestamps(data_ts, frame_ts)

    return matched_ts, _

def get_match_index(frame_ts, data_ts):

    start_ts = max(frame_ts[0], data_ts[0])
    end_ts = min(frame_ts[-1], data_ts[-1])
    # print(f"Episode start from {start_ts}s to {end_ts}s")

    # discard frames before and after the timestamps
    frame_ts = [ts for ts in frame_ts if ts >= start_ts and ts <= end_ts]
    # video_ids = [id for id in video_ids if int(id) >= start_ts * 20 and int(id) <= end_ts * 20]
    data_ts = [ts for ts in data_ts if ts >= start_ts and ts <= end_ts]

    # match video and data timestamps
    matched_ts = match_timestamps(data_ts, frame_ts)

    return matched_ts

def load_frame_timestamps(frame_dir):
    timestamp_to_file = {}
    for frame_file in frame_dir.glob('*.png'):
        dt = iso_to_datetime(frame_file.name)
        timestamp = dt.timestamp()
        timestamp_to_file[timestamp] = frame_file
    sorted_timestamps = sorted(timestamp_to_file.keys())
    return np.array(sorted_timestamps), timestamp_to_file

def load_data_timestamps(file_path):
    with h5py.File(str(file_path), "r") as f:
        # return np.asarray(f["timestamp"]) - 28800
        return np.asarray(f["timestamp"]),np.asarray(f["action"]["robot"]),np.asarray(f["state"]["robot"])
    
def iso_to_datetime(filename: str) -> datetime:
    """
    Convert an ISO 8601 filename with microseconds back to a datetime object.
    Args:
        filename (str): The filename to parse, including or excluding the file extension.
    Returns:
        datetime: Parsed datetime object.
    """
    if "." in filename:
        base_name = filename.split(".")[0]  # Remove file extension
    else:
        base_name = filename
    return datetime.strptime(base_name, "%Y-%m-%dT%H-%M-%S_%f").replace(tzinfo=timezone.utc)

def match_timestamps(candidate, ref):
    closest_indices = []
    # candidate = np.sort(candidate)
    already_matched = set()
    for t in ref:
        idx = np.searchsorted(candidate, t, side="left")
        if idx > 0 and (idx == len(candidate) or np.fabs(t - candidate[idx - 1]) < np.fabs(t - candidate[idx])):
            idx = idx - 1
        if idx not in already_matched:
            closest_indices.append(idx)
            already_matched.add(idx)
        else:
            # print(f"Duplicate timestamp found: {t} and {candidate[idx]} trying to use next closest timestamp")
            if idx + 1 not in already_matched:
                closest_indices.append(idx + 1)
                already_matched.add(idx + 1)

    # print("closest_indices: ", len(closest_indices))
    return np.array(closest_indices)

def find_image_by_timestamp(timestamp, timestamp_to_file):
    return timestamp_to_file.get(timestamp, None)

def load_data(file_path):
    with h5py.File(str(file_path), "r") as f:
        # return np.asarray(f["timestamp"]) - 28800
        return np.asarray(f["action"]["hand"]),np.asarray(f["action"]["pose"]),np.asarray(f["action"]["robot"]), np.asarray(f["state"]["hand"]),np.asarray(f["state"]["pose"]),np.asarray(f["state"]["robot"])


if __name__=='__main__':
    data_dir = '/mnt/nas/Data/nv-collab-2/012/2024-11-29_11-09-41'
    # episode_name = 'episode_000000003'
    # # matched_ts, timestamp_to_file = calculate_match_index(data_dir, episode_name)
    # frame_ts, timestamp_to_file = load_frame_timestamps(Path(data_dir) / episode_name / 'top/rgb')
    # data_ts,robot_action, robot_state = load_data_timestamps(Path(data_dir) / f"{episode_name}.hdf5")
    # matched_ts = get_match_index(frame_ts, data_ts)
    # print(f'get len of match results:{len(matched_ts)}')
    # print(f'first time: data time{data_ts[matched_ts[0]]}, frame_time{frame_ts[0]}')
    # print(f'name: {timestamp_to_file[frame_ts[0]]}')
    # print("robot_action", robot_action.shape, "robot_state", robot_state.shape)

    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])
    save = []
    for file_name in file_list:
        _,action_pose,_,_,state_pose,_ = load_data(Path(data_dir) / f"{file_name}")
        to_save = []
        for i in range(len(action_pose) - 1):

            action_pose_euler = conver_ortho6d_to_euler(action_pose[i,12:18])
            # # xyzw
            action_pose_quaternion = convert_euler_to_quaternion(action_pose_euler)
            base_pose_tool_reached = sapien.Pose()
            base_pose_tool_reached.set_p(action_pose[i,9:12])
            base_pose_tool_reached.set_q(action_pose_quaternion[[3, 0, 1, 2]]) 

            action_world_vector = state_pose[i + 1,9:12] - state_pose[i,9:12]
            rotation_matrix_before = compute_rotation_matrix_from_ortho6d(state_pose[i + 1,12:18]).squeeze()
            rotation_matrix_next = compute_rotation_matrix_from_ortho6d(state_pose[i ,12:18]).squeeze().T
            rotation_diff = np.dot(rotation_matrix_before , rotation_matrix_next)
            action_rotation_delta = convert_rotation_matrix_to_euler(rotation_diff)
            
            save_episode_step = {
                        "base_pose_tool_reached": np.concatenate(
                            [
                                np.array(base_pose_tool_reached.p, dtype=np.float64),
                                np.array(base_pose_tool_reached.q, dtype=np.float64),
                            ]
                        ),  # reached tool pose under the robot base frame, [xyz, quat(wxyz)]
                        "action_world_vector": np.array(action_world_vector, dtype=np.float64),
                        "action_rotation_delta": np.array(action_rotation_delta.squeeze() , dtype=np.float64),
                        # 'action_gripper': np.array(2.0 * (np.array(episode_step['action']['open_gripper'])[None]) - 1.0, dtype=np.float64), # 1=open; -1=close
                    }
            to_save.append(save_episode_step)
        save.append(to_save)
    with open("./sysid_log/sysid_dataset.pkl", "wb") as f:
        pickle.dump(save, f)
            
