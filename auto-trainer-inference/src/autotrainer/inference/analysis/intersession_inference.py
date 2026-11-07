
from typing import List

import pandas
from numpy import ndarray

from autotrainer.core import ProjectInfo, get_verbose_logger

logger = get_verbose_logger(__name__)


def intersession_inference(pose_data: ndarray, part_names: List[str], project: ProjectInfo) -> None:
    """
    Called once for pose data from all video frames passed to the pellet model.

    :param pose_data: interleaved batch pose data from pellet model
    :param part_names: names associated with each pose position
    :param project: current project info for finding/defining file names
    :return: None
    """
    axis_labels = ("x", "y", "likelihood")
    columns = pandas.MultiIndex.from_product([part_names, axis_labels], names=["bodyparts", "coords"])

    left = pose_data[::2, :]
    indices = range(left.shape[0])
    df_xyp = pandas.DataFrame(left, columns=columns, index=indices)
    df_xyp.to_hdf(project.get_intersession_pose_path(name=project.camera_names[0]), "df_with_missing", format="table",
                  mode="w")

    right = pose_data[1::2, :]
    indices = range(right.shape[0])
    df_xyp = pandas.DataFrame(right, columns=columns, index=indices)
    df_xyp.to_hdf(project.get_intersession_pose_path(name=project.camera_names[1]), "df_with_missing", format="table",
                  mode="w")
