#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:29:00 2024

@author: agx001
"""

import os
import pickle
import shutil
import yaml
import math
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from sklearn.decomposition import PCA

from autotrainer.core.logging import get_verbose_logger


logger = get_verbose_logger(__name__)


def make_new_calibration(square_size, row_ct, col_ct, over_x, parent_dir):
    """
    Creates a new directory for calibration with subdirectories for storing source videos.

    Args:
    square_size (int): Size of the calibration square (in millimeters).
    row_ct (int): Number of rows in the calibration grid.
    col_ct (int): Number of columns in the calibration grid.
    over_x: int
        If the frames were sampled at a higher resolution than will be used for
        normal acquisition, this is the x-fold oversampling factor

    parent_dir (str): Parent directory where the calibration folder will be created.
    
    The function will create:
    - A main calibration directory named as '{square_size}mm_{row_ct}r_{col_ct}c'.
    - A 'source_videos' subdirectory where videos must be placed.
    
    IMPORTANT:
    Rows and Columns are junctions, NOT square counts
    
    """
    
    # Create a descriptive directory name for the calibration
    src_name = f'{square_size}mm_{row_ct}r_{col_ct}c_{over_x}x'
    calibration_dir = os.path.join(parent_dir, src_name)

    # Check if the directory already exists to prevent overwriting
    if not os.path.exists(calibration_dir):
        try:
            # Create the main directory for the calibration
            os.mkdir(calibration_dir)

            # Create a subdirectory for the source videos
            os.mkdir(os.path.join(calibration_dir, 'source_videos'))
            
            # Create a subdirectory for the centering video
            os.mkdir(os.path.join(calibration_dir, 'centering_video'))
            
            print(f'Calibration directory created: {calibration_dir}')
            print('Place calibration videos inside the "source_videos" directory.')
            print('One video must end with "left.mp4" and the other with "right.mp4".')
            print('A _3D.h5 file for centering should be placed in "centering" directory.')
        except OSError as e:
            print(f"Error creating directory: {e}")
            return None
    else:
        print(f"Directory '{calibration_dir}' already exists. No directories were created.")
    
    return calibration_dir


def get_calibration_info(file) -> Tuple[int, int, int, int]:
    """
    Extracts calibration information from the folder name based on the format '{square_size}mm_{row_ct}r_{col_ct}c'.

    Args:
    file (str): Path to any file inside the calibration directory.

    Returns:
    tuple: A tuple containing:
        - square_size (int): Size of the calibration square (in millimeters).
        - row_ct (int): Number of rows in the calibration grid.
        - col_ct (int): Number of columns in the calibration grid.

    The folder name should be in the format '{square_size}mm_{row_ct}r_{col_ct}c'.
    """
    # Extract the directory name from the provided file path
    src_name = os.path.split(file)[1]

    try:
        # Find the index of key parts ('mm', 'r', and 'c') in the directory name
        size_ndx = src_name.index('mm')
        row_ndx = src_name.index('r')
        col_ndx = src_name.index('c')
        x_ndx = src_name.index('x')

        # Extract and convert the square size (before 'mm')
        square_size = int(src_name[:size_ndx])

        # Extract and convert the row count (between 'mm_' and 'r')
        row_ct = int(src_name[size_ndx+3:row_ndx])

        # Extract and convert the column count (between 'r' and 'c')
        col_ct = int(src_name[row_ndx+2:col_ndx])

        # Extract and convert the column count (between 'r' and 'c')
        over_x = int(src_name[col_ndx+2:x_ndx])
        
        # Return the extracted values
        return square_size, row_ct, col_ct, over_x
    
    except (ValueError, IndexError) as err:
        # If there is an issue with extracting data, such as missing 'mm', 'r', or 'c'
        logger.error("Error extracting calibration info from %s: %s", src_name, err)
        raise


def get_video_list(src_dir):
    """
    Retrieves the video files that end with 'left.mp4' and 'right.mp4' from the 'source_videos' directory.

    """
    
    video_files = []
    # Define the path to the 'source_videos' subdirectory
    source_videos_dir = os.path.join(src_dir, 'source_videos')

    try:
        # Check if the source_videos directory exists
        if not os.path.exists(source_videos_dir):
            print(f"Error: Directory '{source_videos_dir}' does not exist.")
            return video_files

        # Get the list of video files ending with 'left.mp4' and 'right.mp4'
        left_videos = [f for f in os.listdir(source_videos_dir) if f.endswith('left.mp4')]
        right_videos = [f for f in os.listdir(source_videos_dir) if f.endswith('right.mp4')]

        # Ensure both left and right video files are found
        if left_videos and right_videos:
            video_files = [Path(left_videos[0]).stem, Path(right_videos[0]).stem]  # Collect the first 'left' and 'right' video file
        else:
            if not left_videos:
                print("Error: No video file ending with 'left.mp4' found.")
            if not right_videos:
                print("Error: No video file ending with 'right.mp4' found.")

    except OSError as e:
        print(f"Error accessing the directory '{source_videos_dir}': {e}")

    return video_files


def adjust_gamma(image, gamma=1.0):
    """
    Adjusts the gamma of the given image.

    Args:
    image (numpy.ndarray): Input image.
    gamma (float): Gamma correction value.

    Returns:
    numpy.ndarray: Gamma-corrected image.
    """
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, table)


def score_grid(points):
    """
    Scores the grid of points based on linearity and uniform spacing.

    Args:
    points (numpy.ndarray): Array of points of shape (rows, cols, 2) where each entry has x and y coordinates.

    Returns:
    float: A score between 0 and 1 representing the "grid-likeness" of the points.
    """
    
    # Step 1: Flatten the grid for PCA and align the grid by removing rotation
    rows, cols, _ = points.shape
    points_flattened = points.reshape(-1, 2)  # Flatten to (rows * cols, 2)
    
    # Apply PCA to align the points (remove rotation)
    pca = PCA(n_components=2)
    aligned_points = pca.fit_transform(points_flattened).reshape(rows, cols, 2)

    # Step 2: Check linearity of rows and columns
    def check_linearity(sorted_points):
        # Fit a line to the points and calculate residuals
        x = sorted_points[:, 0]  # x-coordinates
        y = sorted_points[:, 1]  # y-coordinates

        # Linear regression: fit y = mx + b
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        predicted_y = slope * x + intercept

        # Residual errors (distance from the line)
        residuals = np.abs(y - predicted_y)

        # Score based on how small the residuals are
        return 1 - np.mean(residuals) / np.ptp(y)  # Normalize by y-range

    # Step 3: Check uniform spacing of rows and columns
    def check_spacing(sorted_points):
        # Compute distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(sorted_points, axis=0) ** 2, axis=1))
        mean_distance = np.mean(distances)

        # Score based on how consistent the distances are
        return 1 - np.std(distances) / mean_distance

    # Initialize scores
#    linearity_score = 0
#    spacing_score = 0
    linearity_score = np.zeros((rows+cols,1))
    spacing_score = np.zeros((rows+cols,1))
    score_ndx = 0

    # Step 4: Evaluate linearity and spacing for both rows and columns
    for i in range(rows):
        linearity_score[score_ndx] = check_linearity(aligned_points[i, :, :])  # Check rows
        spacing_score[score_ndx] = check_spacing(aligned_points[i, :, :])
        score_ndx += 1

    for j in range(cols):
        linearity_score[score_ndx] = check_linearity(aligned_points[:, j, :])  # Check columns
        spacing_score[score_ndx] = check_spacing(aligned_points[:, j, :])
        score_ndx += 1

    # Normalize scores by the total number of rows and columns evaluated
    min_linearity_score = np.min(linearity_score)
    min_spacing_score = np.min(spacing_score)

    # Final score: average of linearity and spacing scores
    final_score = min(min_linearity_score, min_spacing_score)

    return final_score


_GENERATED_CALIBRATION_DIRECTORIES = frozenset(
    {"corners", "gray", "rejected", "camera_matrix"}
)


def create_or_clean_directory(directory_path, *, calibration_root=None):
    """Create or clean one approved generated calibration child directory.

    Source images, configuration, and unrelated children of the selected
    calibration root are never candidates for deletion.
    """

    target = Path(directory_path).absolute()
    root = (
        target.parent
        if calibration_root is None
        else Path(calibration_root).absolute()
    )
    if root.is_symlink():
        raise ValueError(f"Calibration source root cannot be a symbolic link: {root}")
    if target.is_symlink():
        raise ValueError(f"Generated calibration directory cannot be a symbolic link: {target}")
    resolved_root = root.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    if target.name not in _GENERATED_CALIBRATION_DIRECTORIES:
        raise ValueError(f"Refusing to clean unapproved calibration directory: {target}")
    if resolved_target.parent != resolved_root:
        raise ValueError(
            f"Generated calibration directory is outside the selected root: {target}"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Calibration source root is not a directory: {root}")

    target.mkdir(mode=0o755, exist_ok=True)
    if target.is_symlink() or target.resolve(strict=True).parent != resolved_root:
        raise ValueError(f"Generated calibration directory changed unsafely: {target}")
    for child in tuple(target.iterdir()):
        # Revalidate the destructive boundary immediately before each removal.
        if target.is_symlink() or target.resolve(strict=True).parent != resolved_root:
            raise ValueError(f"Generated calibration directory changed unsafely: {target}")
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise ValueError(f"Unsupported generated calibration entry: {child}")

def refine_corners(image, initial_corner, window_size):
    
    half_window = window_size // 2
    tolerance = 10  # Define tolerance for proximity to half_window
    exclusion_radius = 5
    
    # Extract a 50x50 region around the initial corner
    x, y = int(initial_corner[0]), int(initial_corner[1])
    corner_region = image[max(0, y - half_window):y + half_window, max(0, x - half_window):x + half_window]
    
    # Apply edge detection (Canny Edge Detector)
    edges = cv2.Canny(corner_region, 25, 100)
    
    # Detect edge points within the corner region
    edge_points = np.column_stack(np.where(edges > 0))
    
    # Calculate the distance of each edge point from the center
    distances = np.linalg.norm(edge_points - [half_window, half_window], axis=1)
    
    # Filter out points within the exclusion radius around the center
    edge_points = edge_points[distances > exclusion_radius]
    
    # Split the edge points into vertical and horizontal groups based on proximity to half_window
    vertical_edges = edge_points[np.abs(edge_points[:, 1] - half_window) < tolerance]  # x-values near half_window
    horizontal_edges = edge_points[np.abs(edge_points[:, 0] - half_window) < tolerance]  # y-values near half_window
    
    # Sort horizontal_edges such that the order is increasing in [:, 1]
    horizontal_edges = horizontal_edges[np.argsort(horizontal_edges[:, 1])]
    
    # Fit lines to the vertical and horizontal edge points using cv2.fitLine
    vy1, vx1, y1, x1 = cv2.fitLine(vertical_edges, cv2.DIST_L2, 0, 0.01, 0.01)
    vy2, vx2, y2, x2 = cv2.fitLine(horizontal_edges, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Set up the equations to solve for t and s
    A = np.array([[vx1[0], -vx2[0]], [vy1[0], -vy2[0]]])
    b = np.array([x2[0] - x1[0], y2[0] - y1[0]])
    
    # Solve for t and s
    # try:
    t, s = np.linalg.solve(A, b)
    
    # Calculate intersection point using t in the parametric equation of line 1
    intersection_x = x1 + t * vx1
    intersection_y = y1 + t * vy1
    # Adjust the refined corner relative to the corner region
    refined_corner = (intersection_x + x - half_window, intersection_y + y - half_window)
    return refined_corner


def rotate_2D_points(points, angle_degrees):
    # Detect the input shape and data type
    original_shape = points.shape
    original_dtype = points.dtype

    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)

    # Create the rotation matrix as the same dtype as the input
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ], dtype=original_dtype)

    # Reshape points to (N, 2) to apply the rotation matrix
    points_reshaped = points.reshape(-1, 2)

    # Apply the rotation
    rotated_points = np.dot(points_reshaped, rotation_matrix.T)

    # Convert the rotated points back to the original dtype if necessary and reshape to original shape
    rotated_points = rotated_points.astype(original_dtype).reshape(original_shape)

    return rotated_points


def average_chessboard_distance(corners, board_size):
    """
    Calculate the average pixel distance between adjacent points in the chessboard corners.
    
    Parameters:
    - corners: Output from cv2.findChessboardCorners, an array of detected corners.
    - board_size: Tuple (cols, rows) specifying the number of inner corners per chessboard row and column.
    
    Returns:
    - average_distance: Average pixel distance between adjacent corners.
    """
    cols, rows = board_size
    total_distances = []

    # Convert corners to a more manageable format if necessary
    points = corners.squeeze()  # Remove unnecessary dimensions (N x 1 x 2 to N x 2)
    
    # Calculate distances between consecutive points in each row
    for i in range(rows):
        for j in range(cols - 1):
            idx1 = i * cols + j
            idx2 = i * cols + (j + 1)
            distance = np.linalg.norm(points[idx1] - points[idx2])
            total_distances.append(distance)
    
    # Calculate distances between consecutive points in each column
    for j in range(cols):
        for i in range(rows - 1):
            idx1 = i * cols + j
            idx2 = (i + 1) * cols + j
            distance = np.linalg.norm(points[idx1] - points[idx2])
            total_distances.append(distance)
    
    # Calculate the average distance
    average_distance = np.mean(total_distances)
    return average_distance

def stretchlim_like_matlab(I, tol=(0.01, 0.99)):
    """Emulate MATLAB's stretchlim (uses uint8 bins + round)."""
    I = np.clip(I, 0, 1)
    I_uint8 = np.round(I * 255).astype(np.uint8)
    hist, _ = np.histogram(I_uint8, bins=256, range=(0, 255))
    cdf = np.cumsum(hist) / np.sum(hist)

    low = np.searchsorted(cdf, tol[0])
    high = np.searchsorted(cdf, tol[1])

    return low / 255.0, high / 255.0


def imadjust_like_matlab(I, in_range, out_range, gamma):
    """Emulate MATLAB's imadjust."""
    low_in, high_in = in_range
    low_out, high_out = out_range

    I = np.clip(I, low_in, high_in)
    if high_in - low_in == 0:
        scaled = np.zeros_like(I)
    else:
        scaled = (I - low_in) / (high_in - low_in)
    adjusted = scaled ** gamma
    out = adjusted * (high_out - low_out) + low_out
    return np.clip(out, 0, 1)


def rgb2gray_matlab(frame_bgr):
    """Convert BGR to MATLAB-style grayscale."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = (
        0.2989 * frame_rgb[:, :, 0]
        + 0.5870 * frame_rgb[:, :, 1]
        + 0.1140 * frame_rgb[:, :, 2]
    )
    return gray.astype(np.float64) / 255.0


def create_corner_matrix(src_dir, num_frames: Optional[int] = 50, gamma=1, camera_pos=None,
                         alpha=1, quality=0.9, calibrate=False):
    """
    
    src_dir (str): Path created by make_new_calibration
    
    num_frames: int
        Number of frames to automatically extract from the supplied videos
    
    gamma: float
        Typical values range from 0.5 to 3.0, may improve chessboard corner finding

    search_window_size: tuple of int
        Half of the side length of the search window when refining detected checkerboard corners for subpixel accuracy.
    
    alpha: float
        Floating point number between 0 and 1 specifying the free scaling parameter. When alpha = 0, the rectified images with only valid pixels are stored
        i.e. the rectified images are zoomed in. When alpha = 1, all the pixels from the original images are retained.
        For more details: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    quality: float
        Floating point number between 0 and 1 for qutomatic corner-finding quality assessment
        
    calibrate : bool
        Can be set to false while refining other variables
    
    """
    
    # Gather calib variables
    square_size, cbrow, cbcol, over_x = get_calibration_info(src_dir)
    cam_names = get_video_list(src_dir)
    
    rotation_correction = 0
    if not camera_pos == None:
        camLele = camera_pos['camLele']
        camRele = camera_pos['camRele']
        camLazi = camera_pos['camLazi']
        camRazi = camera_pos['camRazi']
        theta = math.atan((camLele-camRele) / (camLazi-camRazi))
        rotation_correction = math.degrees(theta)

    
    # Clear directories
    path_corners = os.path.join(src_dir,'corners')
    create_or_clean_directory(path_corners, calibration_root=src_dir)
    dir_gray = os.path.join(src_dir,'gray')
    create_or_clean_directory(dir_gray, calibration_root=src_dir)
    dir_rejected = os.path.join(src_dir,'rejected')
    create_or_clean_directory(dir_rejected, calibration_root=src_dir)
    
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)
    
    # Initialize the dictionary
    img_shape = {}
    objpoints = {}  # 3d point in real world space
    imgpoints = {}  # 2d points in image plane.
    imgpaths = {}
    corpoints = {}  # corrected 2d points
    dist_pickle = {}
    stereo_params = {}
    cap_list = []
    total_frames = []
    for cam in cam_names:
        objpoints.setdefault(cam, [])
        imgpoints.setdefault(cam, [])
        imgpaths.setdefault(cam, [])
        corpoints.setdefault(cam, [])
        dist_pickle.setdefault(cam, [])
        
        # Open the video file
        video_path = os.path.join(src_dir,'source_videos',cam + '.mp4')
        cap = cv2.VideoCapture(video_path)
        
        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        
        cap_list.append(cap)
        total_frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    min_tot_frames = min(total_frames[0], total_frames[1])
    if num_frames is None:
        num_frames = min_tot_frames
        frame_indices = np.arange(0, num_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, min_tot_frames - 1, num=num_frames, dtype=int)

    metadata = {
    'alpha': alpha,
    'gamma': gamma,
    'quality': quality,
    'num_frames': num_frames,
    'oversample': over_x,
    'camera_pos': camera_pos,
    }
    metadata_path = os.path.join(src_dir, 'calibration_userset.yaml')
    with open(metadata_path, 'w') as file:
        yaml.dump(metadata, file)
    
    keep_list = []
    centers_prev = [0,0,0,0]
    h = 0
    w = 0
    for index in frame_indices:
        keep_test = True
        corner_cams = []
        img_cams = []
        score_count = []
        for cap, cam in zip(cap_list, cam_names):
        
        # Loop through the frame indices
        
            # Set the video position to the current frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            
            # Read the frame
            ret, img = cap.read()
            if not ret:
                print(f"Warning: Could not read frame at index {index}")
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = rgb2gray_matlab(gray)
            limout_low, limout_high = stretchlim_like_matlab(gray, tol=(0.01, 0.99))
            gamma_pre = (np.log(np.mean(gray)) / np.log(0.5)) * 1.5
            limin_low, limin_high = np.percentile(gray.ravel(), [5, 95])
            gray = imadjust_like_matlab(gray, (limin_low, limin_high), (limout_low, limout_high), gamma_pre)
            gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
            gray = adjust_gamma(gray, gamma=gamma)
        
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)
            
            # If found, add object points, image points (after refining them)
            if ret:
                mean_pix_dist = average_chessboard_distance(corners, (cbcol, cbrow))
                window_size = int(round(mean_pix_dist*1.5))
                
                # Fits lines to the intersecting squares to refine corner position
                score = score_grid(np.reshape(np.squeeze(corners),(cbrow,cbcol,2)))
                
                # Initialize an empty list to store corrected points
                corrected_points = []
                # Iterate through each point and apply correction
                for c in range(np.shape(corners)[0]):
                    try:
                        corrected_c = refine_corners(gray, corners[c, 0, :], window_size)
                        corrected_points.append([corrected_c])  # Maintain (1, 2) shape for each point
                    except Exception as err:
                        logger.exception(f"Index {index} - c {c} err=%s", err)
                        # keep_test = False
                if keep_test:
                    # Convert corrected_points to a numpy array with shape (30, 1, 2)
                    corners_refined = np.array(corrected_points)[:,:,:,0]
                    
                    
                    # Test if sub pixel refinement improved quality
                    # score_post = score_grid(np.reshape(np.squeeze(corners_refined),(cbrow,cbcol,2)))
                    # if score_post > score:
                    corners = corners_refined
                    # print(f"Score pre: {score:.4f} -- Score post: {score_post:.4f}")
                        # score = score_post
                        
                    # print(f"Grid Score: {score_post:.2f}")
                    if score < quality:
                        print(f"Low quality corners: {cam} cam at {index} {score:.3f}")
                        keep_test = False
                    else:
                        score_count.append(score)
                        
                    corner_cams.append(corners)
                    img_cams.append(gray)
                
            else:
                filename = f"{cam}_{index}_reject.jpg"
                cv2.imwrite(os.path.join(str(dir_rejected), filename), gray)
                print(f"Corners not found: {cam} cam at {index}")
                keep_test = False
        
        if keep_test:
            
            centers = list()
            for corners in corner_cams:
                centers.append(round(np.mean(corners[:, :, 0])))
                centers.append(round(np.mean(corners[:, :, 1])))

            move_test = abs(np.mean(np.asarray(centers) - np.asarray(centers_prev)))
            if move_test <= 10:
                # Simple method to avoid over-sampling portions of the calibration
                # movie where the grid is not in motion
                print(f"Duplicate centers, skipping index: {index}")
            else:
                centers_prev = centers
                keep_list.append(index)
                print(f"Good pair at index {index} - scores: {score_count[0]:.3f} and {score_count[1]:.3f}")
                for corners, gray, cam in zip(corner_cams, img_cams, cam_names):
                    filename = f"{cam}_{index}_corner.jpg"
                    gray_path = os.path.join(str(dir_gray), filename)
                    cv2.imwrite(gray_path, gray)
                    imgpaths[cam].append(gray_path)
                    
                    if over_x > 1:
                        new_width = int(gray.shape[1] / over_x)
                        new_height = int(gray.shape[0] / over_x)
                        new_size = (new_width, new_height)
                        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
                        corners = corners/over_x
                    
                    img_shape[cam] = gray.shape[::-1]
                    img = np.stack((gray,) * 3, axis=-1)
                    imgpoints[cam].append(corners)
                    objpoints[cam].append(objp)
                    # Draw the corners and store the images
                    new_width = int(img.shape[1] * over_x)
                    new_height = int(img.shape[0] * over_x)
                    new_size = (new_width, new_height)
                    img = cv2.resize(img, new_size)
                    img = cv2.drawChessboardCorners(img, (cbcol, cbrow), corners * over_x, int(ret))
                    
                    filename = f"{cam}_{index}_corner.jpg"
                    cv2.imwrite(os.path.join(str(path_corners), filename), img)
                    # Correct for out-of-plane cameras
                    rotated_points = rotate_2D_points(corners, rotation_correction)
                    corpoints[cam].append(rotated_points)
                    
            
    print(f"Found {len(keep_list)} quality pairs out of {num_frames}")
    # Release the video capture object
    for cap in cap_list:
        cap.release()
    
    

    # Perform calibration for each cameras and store the matrices as a pickle file
    if calibrate == True:
        path_camera_matrix = os.path.join(src_dir,'camera_matrix')
        create_or_clean_directory(path_camera_matrix, calibration_root=src_dir)
        # Calibrating each camera
        for cam in cam_names:
            ret, mtx, dist, rvec, tvec = cv2.calibrateCamera(
                objpoints[cam], imgpoints[cam], img_shape[cam], None, None
            )

            # Save the camera calibration result for later use (we won't use rvecs / tvecs)
            dist_pickle[cam] = {
                "mtx": mtx,
                "dist": dist,
                "objpoints": objpoints[cam],
                "imgpoints": imgpoints[cam],
                "imgpaths": imgpaths[cam],
                "corpoints": corpoints[cam],
                "rvec": rvec,
                "tvec": tvec,
            }
            intrinsic_params_path = os.path.join(path_camera_matrix, cam + "_intrinsic_params.pickle")
            pickle.dump(dist_pickle,
                open(intrinsic_params_path,"wb"),
            )
            print(
                "Saving intrinsic camera calibration matrices for %s as a pickle file in %s"
                % (cam, os.path.join(path_camera_matrix))
            )

            # Compute mean re-projection errors for individual cameras
            mean_error = 0
            for i in range(len(objpoints[cam])):
                imgpoints_proj, _ = cv2.projectPoints(
                    objpoints[cam][i], rvec[i], tvec[i], mtx, dist
                )
                error = cv2.norm(imgpoints[cam][i], imgpoints_proj, cv2.NORM_L2) / len(
                    imgpoints_proj
                )
                mean_error += error
            print(
                "Mean re-projection error for %s images: %.3f pixels "
                % (cam, mean_error / len(objpoints[cam]))
            )

        camera_pair = [[cam_names[0], cam_names[1]]]
        for pair in camera_pair:
            print("Computing stereo calibration for " % pair)
            (
                retval,
                cameraMatrix1,
                distCoeffs1,
                cameraMatrix2,
                distCoeffs2,
                R,
                T,
                E,
                F,
            ) = cv2.stereoCalibrate(
                objpoints[pair[0]],
                corpoints[pair[0]],
                corpoints[pair[1]],
                dist_pickle[pair[0]]["mtx"],
                dist_pickle[pair[0]]["dist"],
                dist_pickle[pair[1]]["mtx"],
                dist_pickle[pair[1]]["dist"],
                (h, w),
                criteria=criteria,
                flags=cv2.CALIB_FIX_INTRINSIC,
            )

            # Stereo Rectification
            rectify_scale = alpha  # Free scaling parameter check this https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#fisheye-stereorectify
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                cameraMatrix1,
                distCoeffs1,
                cameraMatrix2,
                distCoeffs2,
                (h, w),
                R,
                T,
                alpha=rectify_scale,
            )

            stereo_params[pair[0] + "-" + pair[1]] = {
                "cameraMatrix1": cameraMatrix1,
                "cameraMatrix2": cameraMatrix2,
                "distCoeffs1": distCoeffs1,
                "distCoeffs2": distCoeffs2,
                "R": R,
                "T": T,
                "E": E,
                "F": F,
                "R1": R1,
                "R2": R2,
                "P1": P1,
                "P2": P2,
                "roi1": roi1,
                "roi2": roi2,
                "Q": Q,
                "image_shape": [img_shape[pair[0]], img_shape[pair[1]]],
                "rot_cor": rotation_correction,
            }

        print(
            "Saving the stereo parameters for every pair of cameras as a pickle file in %s"
            % str(os.path.join(path_camera_matrix))
        )

        write_pickle(
            os.path.join(path_camera_matrix, "stereo_params.pickle"), stereo_params
        )
        print("Camera calibration done!")
    else:
        print("Corners extracted!")
    

def read_pickle(filename):
    """Read the pickle file"""
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def write_pickle(filename, data):
    """Write the pickle file"""
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
