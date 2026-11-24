import time
from pathlib import Path
from typing import List, Optional, Literal

import h5py
import numpy
import pandas

from reachaq.core import get_verbose_logger

logger = get_verbose_logger(__name__)


# for h5 files:
_h5_pose_data_col_idx = 1
_h5_frame_idx_col_idx = 2
_h5_frame_idx_col_name = "frame_idx"


def get_h5_pose_data(h5row):
    return h5row[_h5_pose_data_col_idx]


def get_h5_frame_index(h5row) -> int:
    # int: given value is array scalar, we want pure/raw int.
    return int(h5row[_h5_frame_idx_col_idx])


def close_h5_fhs(fhs: List[Optional[h5py.File]]):
    for idx, fh in enumerate(fhs or []):
        if fh is not None:
            logger.info("closing %s", fh)  # h5py.File.name attribute only says "/"
            fh.close()
            fhs[idx] = None


def open_h5_file(file_path: Path):
    """Open for reading only"""
    h5fh = h5py.File(file_path)
    datasets = h5fh["df_with_missing"]["table"]
    logger.debug("%s: %s entries", file_path, len(datasets))
    return h5fh, datasets


def write_h5_batch(
    dst_path: Path,
    data_list: List,
    indices_list: List,
    *,
    columns: pandas.MultiIndex,
    mode: Literal["a", "w"] = "a",
) -> float:
    """Write the given data to the dst_path using the given columns and mode"""
    t0 = time.perf_counter()
    if len(data_list) > 0:
        arr = numpy.vstack(data_list)
        index = list(range(arr.shape[0]))
    else:
        arr = index = []
    df_xyp = pandas.DataFrame(arr, columns=columns, index=index)
    # also store the frame idx with the results:
    df_xyp[_h5_frame_idx_col_name] = list(indices_list)
    #
    df_xyp.to_hdf(
        dst_path,
        "df_with_missing",
        format="table",
        mode=mode,
        append=mode == "a",  # required as well for really concat
    )
    data_list.clear()
    indices_list.clear()
    # logger.debug("cleared lists %s and %s",
    #              object.__repr__(data_list), object.__repr__(indices_list))
    t1 = time.perf_counter()
    d = t1 - t0
    logger.debug(
        "wrote h5 batch (%s) in %sms to %s", len(df_xyp), int(d * 1000), dst_path
    )
    return d
