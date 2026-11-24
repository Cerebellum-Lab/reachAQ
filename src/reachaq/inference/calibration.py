from typing import List

import pandas as pd

from reachaq.inference.config import StereoParams
from reachaq.inference.analysis.prepare_jetson_data import undistort_views, triangulate_3d_step1


def undistort_points(
    dataframe: List[pd.DataFrame],
    *,
    stereo_params: StereoParams,
):
    dataframe_cam1 = dataframe[0]
    dataframe_cam2 = dataframe[1]
    undistorted_cams = undistort_views(
        [(dataframe_cam1, dataframe_cam2)],
        stereo_params=stereo_params.as_pickle_dict(),
    )[0]
    return undistorted_cams


def triangulate_3d_with_params(
    df: List[pd.DataFrame],
    *,
    stereo_params: StereoParams,
    body_parts: List[str],
    min_cluster: int,
    p_thresh: float,
):
    (
        cam1_undistort,
        cam2_undistort,
    ) = undistort_points(df, stereo_params=stereo_params)

    df_3d = triangulate_3d_step1(
        df,
        cam1_undistort, cam2_undistort,
        bpts=body_parts,
        min_cluster=min_cluster,
        p_thresh=p_thresh,
        stereomatrix=stereo_params.matrix,
    )
    return df_3d
