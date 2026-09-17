"""
Created on Tue Jan 16 16:15:29 2024

@author: reynoben
"""
import copy
import functools
import sys
import os
import glob
import pickle
import math

import pandas
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Literal

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from autotrainer.core import ProjectInfo
from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.config import load_calib_stereo_params
from autotrainer.inference import calibration_FLIR as cal_flir

logger = get_verbose_logger(__name__)


DEFAULT_CAM_OFFSET_FILE_NAME = "camera_offsets.pkl"

DEFAULT_CAM_OFFSET_VALS = {
    'camLele': 12.5,
    'camRele': 30,
    'camLazi': 55,
    'camRazi': 5,
}

@functools.lru_cache(maxsize=64)
def _triangulation_columns(bpts, axis_labels):
    """The 3d column index for these bodyparts, built once per distinct set.

    Both triangulation helpers rebuilt this MultiIndex on every call, which
    measured 0.72 ms per live batch on the rig - the live pose path runs
    them at 150 fps against a 10-20 ms budget. The index depends only on its
    arguments and a MultiIndex is immutable, so sharing one between frames
    is safe. Arguments are tuples because lru_cache needs them hashable.
    """
    return pandas.MultiIndex.from_product(
        [list(bpts), list(axis_labels)], names=["bodyparts", "coords"],
    )

def make_cam_offsets_dict():
    return DEFAULT_CAM_OFFSET_VALS.copy()


def dict_almost_equal(d1, d2, rel_tol=1e-9, abs_tol=0.0):
    if set(d1) != set(d2):
        return False
    for k in d1:
        v1 = d1[k]
        v2 = d2[k]
        if isinstance(v1, float) or isinstance(v2, float):
            if not math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        else:
            if v1 != v2:
                return False
    return True


def identify_dropped_frames(
    timestamp_file,
    frame_rate,
    *,
    cam_col_present_name: Literal["frame_present_primary", "frame_present_secondary"],
) -> np.ndarray:
    """
    Identify dropped frames in a video based on inter-frame intervals.

    Parameters:
        timestamp_file (str): Path to the timing CSV file containing the frames id/timestamps infos for all cameras
        frame_rate (float): Expected frame rate in frames per second.

    Returns:
        np.ndarray: A binary vector with 0 for successful frames and 1 for dropped frames.
    """
    # Load timestamps from the file
    # timing_header = ['timestamp', 'fps', 'frame_when_ns', 'frame_perf_c', 'frame_id']
    timing_csv_fields = [
        "frame_id",
        "frame_when",
        "frame_present_primary",
        "frame_present_secondary",
        "utc_when",
    ]
    timestamps_df = pd.read_csv(timestamp_file, header=0, names=timing_csv_fields, skipinitialspace=True, delimiter=",")
    dropped_frame_vector = np.ndarray((len(timestamps_df),), dtype=np.uint8)
    dropped_frame_vector[:] = timestamps_df[cam_col_present_name].apply(lambda val: val != 1).array
    return dropped_frame_vector


def extend_and_interpolate_tracking_data(tracking_data, valid_frames):
    """
    Extends and interpolates x/y tracking data for each bodypart to handle dropped frames.

    Parameters:
        tracking_data (pd.DataFrame): MultiIndex DataFrame with bodyparts and 'x'/'y' values.
        valid_frames (np.ndarray): Index of valid frames
        frames.

    Returns:
        pd.DataFrame: Extended and interpolated tracking data with corrected lengths.
    """
    # Create a DataFrame to hold the full timeline based on the dropped_frame_vector
    total_frames = valid_frames[-1] + 1
    full_index = np.arange(total_frames)

    # Initialize the result DataFrame
    extended_data = pd.DataFrame(index=full_index, columns=tracking_data.columns)

    logger.verbose("Doing bodypart loop: total_frames=%s full_index=%s extended_data=%s valid=%s",
                   total_frames, len(full_index), len(extended_data), len(valid_frames))
    # Iterate over each bodypart and interpolate
    for bodypart in tracking_data.columns.levels[0]:  # Iterate through the top-level (bodyparts)
        for coord in ['x', 'y']:  # Iterate through subcategories 'x' and 'y'
            # Extract original tracking data for the current bodypart and coordinate
            original_data = tracking_data[(bodypart, coord)].values
            original_data = original_data[valid_frames]

            # Fill the extended DataFrame with interpolated values
            try:
                extended_data[(bodypart, coord)] = np.interp(
                full_index,  # Full range of frames
                valid_frames,  # Frames with valid data
                original_data  # Original data corresponding to valid frames
            )
            except Exception as err:
                logger.exception("failed np.interp: %s", err)
                logger.notice("extended_data=%s full_index=%s valid_frames=%s original_data=%s",
                             len(extended_data), len(full_index), len(valid_frames), len(original_data))
                raise

        # Handle 'p' (confidence) separately
        original_confidence = tracking_data[(bodypart, 'likelihood')].values
        original_confidence = original_confidence[valid_frames]

        # Initialize 'p' with -1 for dropped frames
        confidence_values = np.full(total_frames, -1, dtype=float)
        confidence_values[valid_frames] = np.interp(
            valid_frames,  # Interpolate only at valid frames
            valid_frames,
            original_confidence
        )
        extended_data[(bodypart, 'likelihood')] = confidence_values

    # Return the extended and interpolated DataFrame
    return extended_data


# Helper function to process hand data
def process_hand_data(
    df,
    hand_base_names,
    hand_options,
    dlc_seg,
    newdf,
    additional_names,
):
    len_df = len(df)
    len_hand_cat = len(hand_base_names)
    row_index = np.arange(len_df)
    len_shape = (len_df, len_hand_cat)
    #
    for h in hand_options:
        hand_categories = [h + item for item in hand_base_names]

        likelihood_array = np.empty(len_shape, dtype=np.float64)
        y_array = np.empty(len_shape, dtype=np.float64)
        x_array = np.empty(len_shape, dtype=np.float64)

        for cndx, cat in enumerate(hand_categories):
            # Adjusting to use .loc to access MultiIndex properly
            if dlc_seg == '_raw2D':
                likelihood_array[:, cndx] = df.loc[:, (cat, 'likelihood')].values
                x_array[:, cndx] = df.loc[:, (cat, 'x')].values
                y_array[:, cndx] = df.loc[:, (cat, 'y')].values
            else:
                likelihood_array[:, cndx] = df.loc[:, (dlc_seg, cat, 'likelihood')].values
                x_array[:, cndx] = df.loc[:, (dlc_seg, cat, 'x')].values
                y_array[:, cndx] = df.loc[:, (dlc_seg, cat, 'y')].values

        col_index = np.argmax(likelihood_array, axis=1)

        # Extract the most likely hand position
        p2keep = likelihood_array[row_index, col_index]
        x2keep = x_array[row_index, col_index]
        y2keep = y_array[row_index, col_index]

        # Store values in newdf
        h_hand = h + '_Hand'
        newdf.loc[row_index, (h_hand, 'x')] = x2keep
        newdf.loc[row_index, (h_hand, 'y')] = y2keep
        newdf.loc[row_index, (h_hand, 'likelihood')] = p2keep

    # Process additional bodyparts
    for an in additional_names:
        an_x = (an, 'x')
        an_y = (an, 'y')
        an_l = (an, 'likelihood')
        if dlc_seg == '_raw2D':
            newdf.loc[row_index, an_x] = df[an_x].values
            newdf.loc[row_index, an_y] = df[an_y].values
            newdf.loc[row_index, an_l] = df[an_l].values
        else:
            newdf.loc[row_index, an_x] = df[(dlc_seg, an, 'x')].values
            newdf.loc[row_index, an_y] = df[(dlc_seg, an, 'y')].values
            newdf.loc[row_index, an_l] = df[(dlc_seg, an, 'likelihood')].values

    return newdf


# extract tracking data from H5 file
def extract_tracking_data(
    project: ProjectInfo,
    video_paths,
    dlc_seg,
    p_thresh,
    frame_rate,
):
    dataframe_RL = []
    bodyparts = ['R_Hand', 'L_Hand', 'Pellet', 'Nose', 'Mouth', 'Tongue_mid', 'Tongue_tip', 'Star', 'Triangle',
                 'Diamond']
    coordinates = ['x', 'y', 'likelihood']
    columns = pd.MultiIndex.from_product([bodyparts, coordinates], names=['bodyparts', 'coordinates'])

    # Process each video
    hand_base_names = ['H_flat', 'H_spread', 'H_grab']
    hand_options = ['R', 'L']
    additional_names = ['Pellet', 'Nose', 'Mouth', 'Tongue_mid', 'Tongue_tip', 'Star', 'Triangle', 'Diamond']
    for cam_idx, v_path in enumerate(video_paths):
        logger.info("extract_tracking_data: %s", v_path)
        vid_dir, vid_name_raw = os.path.split(v_path)
        vid_name_raw = os.path.splitext(vid_name_raw)[0]
        h5_file_path = os.path.join(vid_dir, vid_name_raw + dlc_seg + '.h5')
        if not os.path.isfile(h5_file_path):
            logger.error('h5 path does not exist: %s', h5_file_path)
            return dataframe_RL, bodyparts

        df = pd.read_hdf(h5_file_path)

        newdf = pd.DataFrame(columns=columns, index=range(len(df)))

        # Process hand data
        newdf = process_hand_data(df, hand_base_names, hand_options, dlc_seg, newdf,
                                  additional_names=additional_names)

        newdf_interpolated = interpolate_coordinates(newdf.copy(), p_thresh)

        # Generate the dropped frame vector
        v_path = Path(v_path)
        timestamp_file_path = project.get_frame_timing_path()
        if cam_idx == 0:
            cam_col_present_name = "frame_present_primary"
        elif cam_idx == 1:
            cam_col_present_name = "frame_present_secondary"
        else:
            raise RuntimeError(f"Invalid cam index: {cam_idx}")
        #
        dropped_frame_vector = identify_dropped_frames(
            str(timestamp_file_path), frame_rate,
            cam_col_present_name=cam_col_present_name,
        )
        # # In cases where there are more timestamps than frames
        if len(df) != len(dropped_frame_vector):
            logger.verbose("len(df)=%s vs len(dropped_frame_vector)=%s ; cutting to shortest",
                           len(df), len(dropped_frame_vector))
            if len(df) > len(dropped_frame_vector):
                n = len(dropped_frame_vector)
                df = df[:n]
                newdf = newdf[:n]
                newdf_interpolated = newdf_interpolated[:n]
            else:
                dropped_frame_vector = dropped_frame_vector[:len(df)]

        # Extend and interpolate the tracking data
        # Identify valid frames in the dropped_frame_vector
        valid_frames = np.where(dropped_frame_vector == 0)[0]

        if len(dataframe_RL) > 0 and len(valid_frames) != len(dataframe_RL[0]):
            logger.warning("detected valid frames != other: %s vs %s", len(valid_frames), len(dataframe_RL[0]))
            if len(valid_frames) > len(dataframe_RL[0]):
                newdf_interpolated = newdf_interpolated[:len(valid_frames)]

        newdf_filled = extend_and_interpolate_tracking_data(
            newdf_interpolated.copy().astype(float),
            valid_frames)

        # Apply Butterworth filter to r_cam_df and l_cam_df with a cutoff frequency of 0.1 (adjust as needed)
        newdf_filtered = apply_butterworth_filter(newdf_filled.copy(), frame_rate=frame_rate)
        newdf_filtered = newdf_filtered.astype(float)
        if len(dataframe_RL) > 0:
            newdf_filtered = newdf_filtered[:len(dataframe_RL[-1])]
        dataframe_RL.append(newdf_filtered)

    # Determine the maximum length
    max_len = max(len(dataframe_RL[0]), len(dataframe_RL[1]))
    padded_df_LR = []
    for df in dataframe_RL:
        pad_df = pd.DataFrame(
            -1,  # Fill value
            index=range(len(df), max_len),  # Indices for padding
            columns=df.columns  # Match the column structure of df1
        )
        df = pd.concat([df, pad_df], ignore_index=True)
        padded_df_LR.append(df)

    return padded_df_LR, bodyparts


def get_frame_rate(video_path):
    # TODO: Not yet updated for use with Jetson
    frame_rate = None
    # Extract relevant information from video_path
    vid_name_base, vid_dir = get_vid_name_base(video_path)
    frame_rate_file = os.path.join(vid_dir, vid_name_base + '_userdata_copy.yaml')

    if os.path.isfile(frame_rate_file):
        # Read frame rate from userdata_copy.yaml
        with open(frame_rate_file, 'r') as file:
            yaml_content = file.read()
        if 'framerate' in yaml_content:
            frame_rate_string = yaml_content.split('framerate:')[1].split()[0]
            frame_rate = int(''.join(filter(str.isdigit, frame_rate_string)))
    else:
        logger.warning('change to systemdatya_copy for frame rate')
    return frame_rate


def get_vid_name_base(video_path):
    vid_dir, vid_name = os.path.split(video_path)
    vid_name, vid_ext = os.path.splitext(vid_name)
    txtparts = vid_name.split('_')
    vid_name_base = txtparts[0] + '_' + txtparts[1] + '_' + txtparts[2]
    return vid_name_base, vid_dir


# Function to interpolate x and y where likelihood < p_thresh
def interpolate_coordinates(df, p_thresh):
    for part in df.columns.get_level_values('bodyparts').unique():
        likelihood_col = (part, 'likelihood')
        x_col = (part, 'x')
        y_col = (part, 'y')

        # Find rows where likelihood < p_thresh
        below_threshold = df[likelihood_col] < p_thresh

        # Set x and y to NaN where likelihood < p_thresh
        df.loc[below_threshold, x_col] = np.nan
        df.loc[below_threshold, y_col] = np.nan

        if df[x_col].notna().sum() > 1:
            # Convert x and y columns to numeric to ensure interpolation works
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

            # Interpolate NaN values for x and y
            df[x_col].interpolate(method='linear', inplace=True)
            df[y_col].interpolate(method='linear', inplace=True)

        # Forward-fill and backward-fill any remaining NaN values at the edges
        df[x_col].fillna(method='ffill', inplace=True)
        df[x_col].fillna(method='bfill', inplace=True)
        df[y_col].fillna(method='ffill', inplace=True)
        df[y_col].fillna(method='bfill', inplace=True)

    return df


# Function to apply a Butterworth filter
def butterworth_filter(data, frame_rate):
    cutoff_freq = 50  # Hz
    nyquist_freq = 0.5 * frame_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    filter_order = 5
    # Create Butterworth filter coefficients
    b, a = butter(filter_order, normalized_cutoff_freq, btype='low', analog=False, output='ba')
    y = filtfilt(b, a, data)
    return y


# Function to apply Butterworth filter to x and y coordinates
def apply_butterworth_filter(df, frame_rate):
    for part in df.columns.get_level_values('bodyparts').unique():
        x_col = (part, 'x')
        y_col = (part, 'y')

        # Apply the Butterworth filter to the x and y columns
        df[x_col] = butterworth_filter(df[x_col].values, frame_rate)
        df[y_col] = butterworth_filter(df[y_col].values, frame_rate)

        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

    return df


def _undistort_points(points, mat, coeffs, r, p, rot_cor):
    pts = points.reshape((-1, 3))
    src = pts[:, :2].astype(np.float32)
    src = cal_flir.rotate_2D_points(src, rot_cor)
    pts_undist = cv2.undistortPoints(
        src=src,
        cameraMatrix=mat,
        distCoeffs=coeffs,
        R=r,
        P=p,
    )
    pts[:, :2] = pts_undist.squeeze()
    pts_out = pts.reshape((points.shape[0], -1))
    return pts_out


def undistort_views(df_view_pairs, stereo_params):
    df_views_undist = []
    for df_view_pair, camera_pair in zip(df_view_pairs, stereo_params):
        params = stereo_params[camera_pair]
        dfs = []
        for i, df_view in enumerate(df_view_pair, start=1):
            pts_undist = _undistort_points(
                df_view.to_numpy(),
                params[f"cameraMatrix{i}"],
                params[f"distCoeffs{i}"],
                params[f"R{i}"],
                params[f"P{i}"],
                params["rot_cor"],
            )
            df = pd.DataFrame(pts_undist, df_view.index, df_view.columns)
            dfs.append(df)
        df_views_undist.append(dfs)
    return df_views_undist


def undistort_points(dataframe: pd.DataFrame, path_cam_mat: str):
    """
    path_undistort = destfolder
    filename_cam1 = Path(dataframe[0]).stem
    filename_cam2 = Path(dataframe[1]).stem

    #currently no intermediate saving of this due to high speed.
    # check if the undistorted files are already present
    if os.path.exists(os.path.join(path_undistort,filename_cam1 + '_undistort.h5')) and os.path.exists(os.path.join(
    path_undistort,filename_cam2 + '_undistort.h5')):
        print("The undistorted files are already present at %s" % os.path.join(path_undistort,filename_cam1))
        dataFrame_cam1_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam1 + '_undistort.h5'))
        dataFrame_cam2_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam2 + '_undistort.h5'))
    else:
    """
    dataframe_cam1 = dataframe[0]
    dataframe_cam2 = dataframe[1]

    # Gather calib variables
    path_stereo_file = os.path.join(path_cam_mat, "stereo_params.pickle")
    with open(path_stereo_file, "rb") as handle:
        stereo_file = pickle.load(handle)
    stereo_params = stereo_file[list(stereo_file.keys())[0]]

    dataFrame_cam1_undistort, dataFrame_cam2_undistort = undistort_views(
        [(dataframe_cam1, dataframe_cam2)],
        stereo_file,
    )[0]

    return (
        dataFrame_cam1_undistort,
        dataFrame_cam2_undistort,
        stereo_params,
        path_stereo_file,
    )


def rotate_3d_points(points, x_degrees: float = 0, y_degrees: float = 0, z_degrees: float = 0):
    # Convert degrees to radians
    x_rad = np.radians(x_degrees)
    y_rad = np.radians(y_degrees)
    z_rad = np.radians(z_degrees)

    # Define rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_rad), -np.sin(x_rad)],
        [0, np.sin(x_rad), np.cos(x_rad)]
    ])

    R_y = np.array([
        [np.cos(y_rad), 0, np.sin(y_rad)],
        [0, 1, 0],
        [-np.sin(y_rad), 0, np.cos(y_rad)]
    ])

    R_z = np.array([
        [np.cos(z_rad), -np.sin(z_rad), 0],
        [np.sin(z_rad), np.cos(z_rad), 0],
        [0, 0, 1]
    ])

    # Apply rotations sequentially: X, then Y, then Z
    rotated_points = points @ R_x.T @ R_y.T @ R_z.T

    return rotated_points


def reorient_and_center(
    filtered_df_3d: pandas.DataFrame,
    centered_path_3d: pandas.DataFrame,
    src_dir: str,
    bpts,
    center_method: str,
    frame_rate: int,
):
    df_3d = filtered_df_3d
    path_cam_mat = os.path.join(src_dir, 'camera_matrix')
    path_stereo_file = os.path.join(path_cam_mat, "stereo_params.pickle")
    calib_params = load_calib_stereo_params(Path(path_stereo_file))

    # Read frame rate from userdata_copy.yaml
    metadata_path = os.path.join(src_dir, 'calibration_userset.yaml')
    with open(metadata_path, 'r') as file:
        calib_metadata = yaml.safe_load(file)

    square_size, _, _, _ = cal_flir.get_calibration_info(src_dir)
    cam_names = cal_flir.get_video_list(src_dir)

    path_offsets = os.path.join(src_dir, DEFAULT_CAM_OFFSET_FILE_NAME)
    if os.path.isfile(path_offsets):
        with open(path_offsets, "rb") as handle:
            cam_offsets = pickle.load(handle)
        save_offsets = False
        logger.info("Reusing offsets file %s: %s", path_offsets, cam_offsets)
    else:
        cam_offsets = None
        save_offsets = True
        logger.warning("No camera offset file available, generated one will be saved.")

    res_df_3d = reorient_and_center_step1(
        df_3d=df_3d,
        src_dir=src_dir,
        bpts=bpts,
        center_method=center_method,
        frame_rate=frame_rate,
        calib_metadata=calib_metadata,
        stereo_file=calib_params.as_pickle_dict(),
        square_size=square_size,
        cam_names=cam_names,
        cam_offsets=cam_offsets,
        save_offsets=save_offsets,
    )
    res_df_3d.to_hdf(centered_path_3d, "df_with_missing", format="table", mode="w")
    return res_df_3d


def reorient_and_center_step1(
    *,
    df_3d,
    src_dir,
    bpts,
    center_method,
    frame_rate,
    stereo_file,
    calib_metadata,
    square_size,
    cam_names,
    cam_offsets,
    save_offsets: bool = False,
):
    orig_cam_offsets = copy.deepcopy(cam_offsets)
    num_frames = np.shape(df_3d)[0]
    mask = df_3d.columns.get_level_values("bodyparts").isin(bpts)
    data_4d = df_3d.loc[:, mask].to_numpy().reshape((len(df_3d), -1, 4))
    triangulate, high_conf_exp = np.split(data_4d, [3], axis=-1)
    high_conf = np.squeeze(high_conf_exp, axis=-1)  # Shape will be (484, 8)

    camera_pair = cam_names[0] + "-" + cam_names[1]
    # Undistort points
    rot_cor = stereo_file[camera_pair]['rot_cor']

    camera_pos = calib_metadata['camera_pos']
    if camera_pos is not None:
        camLele = camera_pos['camLele']
        camRele = camera_pos['camRele']
        camLazi = camera_pos['camLazi']
        camRazi = camera_pos['camRazi']

    if not cam_offsets:
        if center_method[0] == 0:
            logger.warning('No offset file found. Offset will be zero.')
        cam_offsets = {
            'x_off': 0,
            'y_off': 0,
            'z_off': 0,
            'camLele': camLele,
            'camRele': camRele,
            'camLazi': camLazi,
            'camRazi': camRazi
        }

    if center_method[0] > 0:
        bp2use = center_method[1]
        if center_method[0] == 1:  # Use current data
            center_3d = df_3d
            center_len = 10
            if np.shape(center_3d)[0] < center_len:
                center_len = 0

        elif center_method[0] == 2:

            # Path to the directory containing centering data
            path_centering = os.path.join(src_dir, 'centering')

            # Find all 3D .h5 files in the centering directory
            files_3D = glob.glob(os.path.join(path_centering, '*_filtered3D.h5'))

            if not files_3D:
                logger.warning('No centering file found. Reverting to default center.')
                center_len = 0
            else:
                center_3d = pd.read_hdf(os.path.join(path_centering, files_3D[0]))
                center_len = num_frames
        else:
            raise ValueError(f"Unhandled center_method[0]: {center_method[0]}")

        if center_len > 0:
            if bp2use not in center_3d.columns.get_level_values('bodyparts').unique():
                logger.warning('Body part %s not found', bp2use)
            else:
                center_xyz = [0, 0, 0]
                speed_ax = []
                for pos in ['x', 'y', 'z']:
                    values = center_3d[bp2use].loc[center_3d[bp2use]['p'] == 1, pos]
                    speed_ax.append(np.diff(values) ** 2)
                dist_vec = np.sqrt(speed_ax[0] + speed_ax[1] + speed_ax[2])  # calculate distance
                if np.size(dist_vec) > 0:
                    dist_vec = np.concatenate(([dist_vec[0]], dist_vec))  # adjust size
                    speed_vec = dist_vec * (frame_rate / 1000)  # convert to speed in mm/ms
                    # with open('debug', 'wb') as f:
                    #     pickle.dump(speed_vec, f)
                    # print(np.shape(np.where(np.abs(speed_vec) < 0.004)))
                    center_xyz = []
                    for pos in ['x', 'y', 'z']:
                        values = center_3d[bp2use].loc[center_3d[bp2use]['p'] == 1, pos]
                        # If the centering part is the 'Pellet', ignore frames
                        # in which the pellet is in motion
                        if bp2use == 'Pellet':
                            values = values[np.abs(speed_vec) < 0.004]
                        center_xyz.append(values.median())

                # Store the calculated offsets in the provided dictionary
                cam_offsets['x_off'] = center_xyz[0]
                cam_offsets['y_off'] = center_xyz[1]
                cam_offsets['z_off'] = center_xyz[2]

        # Save the offsets to a pickle file
        if save_offsets:
            path_offsets = os.path.join(src_dir, DEFAULT_CAM_OFFSET_FILE_NAME)
            logger.notice("Saving camera-offsets to %s", path_offsets)
            with open(path_offsets, 'wb') as fh:
                pickle.dump(cam_offsets, fh)

    if orig_cam_offsets is not None and not dict_almost_equal(cam_offsets, orig_cam_offsets, rel_tol=0.1):
        logger.warning("Loaded cam_offsets != generated: %s vs %s", orig_cam_offsets, cam_offsets)

    # if orig_cam_offsets is not None:
    #     cam_offsets = orig_cam_offsets

    # Reorient based on camera angles
    for bp in range(len(bpts)):
        data = triangulate[:, bp, :]

        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        x -= cam_offsets['x_off']
        y -= cam_offsets['y_off']
        z -= cam_offsets['z_off']

        data = np.vstack((x, y, z)).T

        # Elevation of cameras, taken from CAD model
        avgCamEle = np.mean((camLele, camRele))

        # Azimuth of cameras, taken from CAD model
        avgCamAzi = np.mean((camLazi, camRazi))
        rotated_data = rotate_3d_points(data, x_degrees=avgCamAzi, y_degrees=avgCamEle, z_degrees=-rot_cor)

        # Extract the rotated x, y, z coordinates
        x, y, z = rotated_data[:, 0], rotated_data[:, 1], rotated_data[:, 2]

        # data = np.vstack((x, y, z)).T
        data = np.vstack((-x, -z, -y)).T

        triangulate[:, bp, :] = data

    # Rescale the data
    triangulate = triangulate * square_size

    high_conf_exp = np.expand_dims(high_conf, axis=-1)
    data_4d = np.concatenate([triangulate, high_conf_exp], axis=-1)
    data_4d = data_4d.reshape((num_frames, -1))

    # Create 3D DataFrame column and row indices
    axis_labels = ("x", "y", "z", "p")
    columns = _triangulation_columns(tuple(bpts), tuple(axis_labels))

    inds = range(num_frames)
    df_3d = pd.DataFrame(data_4d, columns=columns, index=inds)
    return df_3d


def triangulatePoints(P1, P2, x1, x2):
    X = cv2.triangulatePoints(P1[:3], P2[:3], x1, x2)
    return X / X[3]


def triangulate_3D(df_LR, path_3D, calib_src_dir, bpts, min_cluster, p_thresh):
    path_cam_mat = os.path.join(calib_src_dir, 'camera_matrix')

    # Read the calibration variables
    # square_size, cbrow, cbcol = cal_flir.get_calibration_info(calib_src_dir)
    # unused here

    # Undistort dataframes
    (
        dataFrame_camera1_undistort,
        dataFrame_camera2_undistort,
        stereomatrix,
        path_stereo_file,
    ) = undistort_points(df_LR, path_cam_mat)

    df_3d = triangulate_3d_step1(
        df_LR,
        dataFrame_camera1_undistort, dataFrame_camera2_undistort,
        stereomatrix=stereomatrix, bpts=bpts, min_cluster=min_cluster, p_thresh=p_thresh
    )
    # df_3d.to_hdf(str(path_3D), "df_with_missing", format="table", mode="w")
    return df_3d


def triangulate_3d_step1(
    df_LR: List[pd.DataFrame],
    dataFrame_camera1_undistort: pd.DataFrame,
    dataFrame_camera2_undistort: pd.DataFrame,
    *,
    stereomatrix: Dict[str, Any],
    bpts: List[str],
    min_cluster,
    p_thresh,
) -> pd.DataFrame:
    P1 = stereomatrix["P1"]
    P2 = stereomatrix["P2"]

    num_frames = dataFrame_camera1_undistort.shape[0]
    all_points_cam1 = dataFrame_camera1_undistort.to_numpy().reshape(
        (num_frames, 1, -1, 3)
    )[..., :2]
    all_points_cam2 = dataFrame_camera2_undistort.to_numpy().reshape(
        (num_frames, 1, -1, 3)
    )[..., :2]

    # Triangulate data
    pts_indv_cam1 = all_points_cam1[:, 0].reshape((-1, 2)).T
    pts_indv_cam2 = all_points_cam2[:, 0].reshape((-1, 2)).T

    indv_points_3d = triangulatePoints(
        P1, P2,
        pts_indv_cam1.astype(np.float64),
        pts_indv_cam2.astype(np.float64)
    )

    indv_points_3d = indv_points_3d[:3].T.reshape((num_frames, -1, 3))

    # Resize based on calibration pattern
    triangulate = np.asanyarray(indv_points_3d)

    # Determine regions of low confidence or dropped frames
    mask2d = df_LR[0].columns.get_level_values("bodyparts").isin(bpts)
    xy1 = (df_LR[0].iloc[: num_frames].loc[:, mask2d].to_numpy().reshape((num_frames, -1, 3)))
    visible1 = xy1[..., 2] >= p_thresh
    hasdrops1 = xy1[..., 2] == -1
    xy2 = (df_LR[1].iloc[: num_frames].loc[:, mask2d].to_numpy().reshape((num_frames, -1, 3)))
    visible2 = xy2[..., 2] >= p_thresh
    hasdrops2 = xy2[..., 2] == -1
    low_conf = ~(visible1 & visible2)
    has_drops = hasdrops1 | hasdrops2

    for bp in range(len(bpts)):
        # Step 1: Create a boolean mask where confidence values are below 0.9
        low_confidence_mask = low_conf[:, bp]

        # Step 2: Identify clusters of consecutive True values
        cluster_mask = np.zeros_like(low_confidence_mask, dtype=bool)
        start = None

        # Iterate through the low confidence mask
        for i, value in enumerate(low_confidence_mask):
            if value:  # Found a low-confidence value
                if start is None:
                    start = i  # Start of a new cluster
            else:
                if start is not None:
                    # End of a cluster, check its length
                    if i - start >= min_cluster:
                        cluster_mask[start:i] = True  # Mark the entire cluster as True
                    start = None

        # Handle the case where the cluster goes until the last element
        if start is not None and len(low_confidence_mask) - start >= min_cluster:
            cluster_mask[start:] = True

        low_conf[:, bp] = cluster_mask

    low_conf[has_drops] = -1
    high_conf_exp = np.expand_dims(~low_conf, axis=-1)
    triangulate = np.concatenate([triangulate, high_conf_exp], axis=-1)
    triangulate = triangulate.reshape((num_frames, -1))
    # Fill up 3D dataframe
    # Create 3D DataFrame column and row indices
    axis_labels = ("x", "y", "z", "p")
    columns = _triangulation_columns(tuple(bpts), tuple(axis_labels))
    df_3d = pd.DataFrame(triangulate, columns=columns, index=range(num_frames))
    return df_3d


def process_raw_data(
    project: ProjectInfo,
    session,
    vid_tag,
    dlc_seg,
    calib_src_dir,
    center_method,
    *,
    frame_rate: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # df_LR, df_3D
    p_thresh = 0.9  # confidence threshold for DLC raw output
    min_cluster = 10  # maximum allowed interpolation

    # Find video files
    mp4_list = os.path.join(session, '*' + vid_tag)
    videoList = glob.glob(mp4_list)

    # Extract relevant video paths in order
    videoOrder = ['left', 'right']
    # NB: this so set "left" as the primary camera, given that's the assumption used here.

    video_paths = [video for key in videoOrder for video in videoList if key in video]
    if len(video_paths) != 2:
        raise RuntimeError(f"Missed at least 1 video: found: {video_paths}")
    # could use: project.get_video_path()

    # Extract reach data, filter, prep for undistortion and triangulation
    df_LR, bodyparts = extract_tracking_data(project, video_paths, dlc_seg, p_thresh, frame_rate)
    vid_name_base, vid_dir = get_vid_name_base(video_paths[0])
    # Extract reach data, filter, prep for undistortion and triangulation
    for ndx, df in enumerate(df_LR):
        filt_name = Path(video_paths[ndx]).stem + '_filtered2D.h5'
        filt_path = os.path.join(vid_dir, filt_name)
        df.to_hdf(str(filt_path), "df_with_missing", format="table", mode="w")
        logger.info("Saved dataframe to %s", filt_path)

    if len(df_LR) == 0:
        raise RuntimeError(f"No tracking obtained for {session}")
    #
    raw_path_3D = os.path.join(vid_dir, vid_name_base + '_filtered3D.h5')
    filtered_df_3d = triangulate_3D(df_LR, raw_path_3D, calib_src_dir, bodyparts, min_cluster, p_thresh)
    #
    centered_path_3d = os.path.join(vid_dir, vid_name_base + '_centered3D.h5')
    centered_df_3d = reorient_and_center(
        filtered_df_3d, centered_path_3d, calib_src_dir, bodyparts, center_method, frame_rate)
    #
    return df_LR, centered_df_3d
