'''
Author: Jiyuan Liu
Date: 2024-12-05 17:35:27
LastEditors: Jiyuan Liu
LastEditTime: 2024-12-06 13:48:45
FilePath: /diamond/src/own/data/timestamp.py
Description: 

Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
'''
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

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


if __name__=='__main__':
    data_dir = '/mnt/nas/Data/nv-collab-2/011/2024-11-29_10-26-02'
    episode_name = 'episode_000000001'  
    # matched_ts, timestamp_to_file = calculate_match_index(data_dir, episode_name)
    frame_ts, timestamp_to_file = load_frame_timestamps(Path(data_dir) / episode_name / 'top/rgb')
    data_ts,robot_action, robot_state = load_data_timestamps(Path(data_dir) / f"{episode_name}.hdf5")
    matched_ts = get_match_index(frame_ts, data_ts)
    print(f'get len of match results:{len(matched_ts)}')
    print(f'first time: data time{data_ts[matched_ts[0]]}, frame_time{frame_ts[0]}')
    print(f'name: {timestamp_to_file[frame_ts[0]]}')
    print("robot_action", robot_action.shape, "robot_state", robot_state.shape)

