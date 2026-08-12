import copy
import dataclasses
import itertools
import math
import operator
import warnings
from typing import List, Dict, Optional, Tuple, Literal
from collections import namedtuple, defaultdict
from dataclasses import dataclass

import numpy
import pandas

from autotrainer.core import ObservableObject, Pairs3dOffsetT, Offset3DTuple, get_perf_now
from autotrainer.inference.calibration import triangulate_3d_with_params
from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.config import StereoParams
from autotrainer.core.pose_elements import SceneElement, AllHandsParts, ScenePartsPresenceContext, AllSceneParts

from autotrainer.inference.analysis.prepare_jetson_data import process_hand_data, reorient_and_center_step1


logger = get_verbose_logger(__name__)


PoseTuple = namedtuple("PoseTuple", ["x", "y"])


_xy_col_names = ['x', 'y']


@dataclass(frozen=True)
class PoseLocation:
    index: int
    x: float
    y: float

    def __repr__(self):
        return f"PoseLocation(index={self.index}, x={self.x:.2f}, y={self.y:.2f})"


@dataclass(frozen=True)
class PoseResponse:
    """Defines response of various input"""

    sequence: int = 0
    """Simple index to track responses"""

    perf_c: float = dataclasses.field(default_factory=get_perf_now)
    """Perf counter when this response applies"""

    source_frame_ids: Tuple[Tuple[int, ...], ...] = ()
    """Actual camera frame IDs used to produce this response, by camera.

    ``perf_c`` is the completion time of live pose processing and may lag the
    camera timeline.  These IDs preserve the acquisition identity required to
    associate live tracking with an exact recording/trial window.
    """

    parts_flags: Tuple[Dict[str, bool], ...] = dataclasses.field(default_factory=lambda: ({}, {}, {}))
    """Tuple indicating part seen for each camera, followed by all cameras in the same frame."""

    locations: List[Dict[str, PoseLocation]] = dataclasses.field(default_factory=list)
    """X, Y locations for each part for each camera, if above threshold, otherwise -1, -1 (or not present)"""

    parts_3d_offsets: Dict[str, Dict[str, Offset3DTuple]] = dataclasses.field(default_factory=dict)
    """3D offsets of the pairs of parts requested during the response creation"""

    locations_3d: Dict[str, Offset3DTuple] = dataclasses.field(default_factory=dict)
    """3D locations of the monitored parts/elements"""

    raw_loc_3d: Dict[str, Offset3DTuple] = dataclasses.field(default_factory=dict)

    @property
    def pellet_seen(self) -> bool:
        """Default logic/conditions for pellet seen"""
        return self.parts_flags[-1].get(SceneElement.Pellet, False)

    @property
    def star_seen(self):
        """Default logic/conditions for star seen"""
        return self.parts_flags[0].get(SceneElement.Star, False) or self.parts_flags[1].get(SceneElement.Star, False)

    @property
    def mouse_seen(self) -> bool:
        """Default logic/conditions for mouse seen: require seen in ALL/both cams"""
        # return all(flags.get(SceneElement.Nose, False) for flags in self.parts_flags)
        return self.parts_flags[-1].get(SceneElement.Nose, False)

    @property
    def diamond_seen(self):
        p_flags = self.parts_flags
        return p_flags[0].get(SceneElement.Diamond, False) or p_flags[1].get(SceneElement.Diamond, False)

    @property
    def triangle_seen(self):
        p_flags = self.parts_flags
        return p_flags[0].get(SceneElement.Triangle, False) or p_flags[1].get(SceneElement.Triangle, False)

    @property
    def lh_grab_seen(self):
        return self.is_part_seen(SceneElement.LH_grab)  # both cams

    @property
    def rh_grab_seen(self):
        return self.is_part_seen(SceneElement.RH_grab)  # both cams

    def is_part_seen(self, part: str, *, cams_idx: Tuple[int, ...] = ()):
        """Check whether `part` is seen or not in cams_idx, if cams_idx empty: check all"""
        if len(cams_idx) == 0:
            # last part flags is conjunction of all previous cams
            cams_idx = [-1]
        return all(
            self.parts_flags[idx].get(part, None)
            for idx in cams_idx
        )

    def get_parts_3d_offset(
        self,
        part1: str,
        part2: str,
    ) -> Optional[Offset3DTuple]:
        """Return the 3d offsets between part1 and part2, relatively to part1,
        i.e: returns loc3d[part2] - loc3d[part1]
        if none exist/is available return None instead
        """
        value = self.parts_3d_offsets.get(part1, {}).get(part2, None)
        if value is None:
            value = self.parts_3d_offsets.get(part2, {}).get(part1, None)
            if value is None:
                return None
            logger.debug("get_parts_3d_offset(%r, %r): detected reversed offsets 3d, "
                         "you shall switch your pair key.",
                         part1, part2)
            value = tuple(map(operator.neg, value))
        return Offset3DTuple(value)

    def round(self, ndigits: int=1):
        return self.__class__(
            sequence=self.sequence,
            perf_c=self.perf_c,
            source_frame_ids=self.source_frame_ids,
            parts_flags=self.parts_flags,
            locations=self.locations,
            locations_3d=dict((n, loc.round(ndigits)) for n, loc in self.locations_3d.items()),
            raw_loc_3d=dict((n, loc.round(ndigits)) for n, loc in self.raw_loc_3d.items()),
            parts_3d_offsets=dict(
                (k1, dict((k2, o.round(ndigits)) for k2, o in d1.items()))
                for k1, d1 in self.parts_3d_offsets.items()
            )
        )


class PoseAlgorithm:
    """
    The PoseAlgorithm is the autotrainer-specific interpretation of the output from a pose model.

    A pose model implementation will return frames of data containing parts with their locations and confidence values.
    The PoseAlgorithm determines interpreted values such as whether a part has been "seen" (e.g., above some confidence
    threshold), or may map locations to a normalized coordinate system.

    The objective is for applications and scripts to not have to know low-level part names as used by the model, how
    thresholds are applied and similar (though they may want to be able to set or modify those thresholds).

    The PoseResponse returned by the PoseAlgorithm captures the interpreted values (e.g., "mouse seen" which may be
    some function of multiple parts being present and/or at different confidence levels).
    """
    # TODO Configurable properties
    MIN_CONFIDENCE_PLOT_THRESHOLD = 0.9
    MIN_CONFIDENCE_PRESENT_THRESHOLD = 0.9

    # could eventually be re-used at some point:
    # process_frames_select_frames_method: Literal['all_most_likely', 'last_one'] = "all_most_likely"

    def __init__(
        self,
        *,
        stereo_params: Optional[StereoParams] = None,
        calib_metadata: Optional[Dict] = None,
        cam_names: Optional[List[str]] = None,
        square_size: Optional[int] = None,
        cam_offsets: Optional[List[float]] = None,
    ):
        super().__init__()
        self._parts_list: List[str] = []
        self._parts: Dict[str, int] = {}  # key is part name, value is part model index
        self._sequence = 0
        self._default_parts_flag: Dict[str, bool] = {}
        self._default_locations = []
        self._pose_result_columns: Optional[pandas.MultiIndex] = None
        self._has_hands_part_names: bool = False
        self._stereo_params = stereo_params
        self._calib_metadata = calib_metadata
        self._cam_names = cam_names
        self._square_size = square_size
        self._cam_offsets = cam_offsets
        self._frame_rate: Optional[int] = None
        #
        axis_labels = self._2d_axis_labels = ['x', 'y', 'likelihood']
        self._hand_base_names = ['H_flat', 'H_spread', 'H_grab']
        self._hand_options = ['R', 'L']
        self._hands_columns = pandas.MultiIndex.from_product([
            [SceneElement.R_Hand, SceneElement.L_Hand], axis_labels],
            names=['bodyparts', 'coordinates'])
        self._hands_input_parts = list(AllHandsParts)
        self._hands_input_columns = pandas.MultiIndex.from_product(
            [self._hands_input_parts, axis_labels], names=['bodyparts', 'coordinates']
        )
        #
        self._3d_names = ["bodyparts", "coords"]
        self._measure_offset_parts = [
            SceneElement.Star,
            SceneElement.Triangle,
            SceneElement.Diamond,
            SceneElement.Pellet,
            # also include all hands parts:
            *self._hands_input_parts,
        ]
        self._measure_offset_parts_columns = pandas.MultiIndex.from_product(
            [self._measure_offset_parts, axis_labels],
            names=self._3d_names,
        )
        self._3d_axis_labels = ("x", "y", "z", "p")
        self._columns_3d = pandas.MultiIndex.from_product(
            [self._measure_offset_parts, self._3d_axis_labels],
            names=self._3d_names,
        )
        empty_3d_columns = pandas.MultiIndex.from_product([[], self._3d_axis_labels], names=self._3d_names)
        self._empty_3d = pandas.DataFrame(index=[0], columns=empty_3d_columns)

    @property
    def frame_rate(self):
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, frame_rate):
        self._frame_rate = frame_rate

    @property
    def stereo_params(self):
        return self._stereo_params

    @property
    def calib_metadata(self):
        return self._calib_metadata

    @property
    def square_size(self):
        return self._square_size

    @property
    def cam_names(self):
        return self._cam_names

    @property
    def cam_offsets(self):
        return self._cam_offsets

    @property
    def part_names(self) -> list:
        return list(self._parts_list)

    @property
    def part_count(self) -> int:
        return len(self._parts_list)

    def get_part_index(self, part: str) -> int:
        """Give the model part index, or -1 if unknown"""
        return self._parts.get(part, -1)

    def initialize(
        self,
        parts: List[str],
        # *,  # not sure yet for:
        # map_model_2_scene_element: Optional[Dict[str, str]] = None,
    ):
        """
        Will be called once after the model initialized and body parts properties have been set.
        See part_names() and get_part_index(name).
        """
        logger.info("Initializing with parts: %s", parts)
        parts = self._parts_list[:] = [
            SceneElement(part)  # this is not exactly required anymore,
            # but only ensure the given part will be cached within SceneElement cached items list/dict.
            for part in parts
        ]
        self._parts.clear()
        self._default_parts_flag.clear()
        self._default_locations.clear()

        for idx, part in enumerate(parts):
            self._parts[part] = idx
            self._default_parts_flag[part] = False
            self._default_locations.append(None)

        axis_labels = self._2d_axis_labels
        columns = pandas.MultiIndex.from_product([self._parts_list, axis_labels],
                                                 names=["bodyparts", "coords"])
        self._pose_result_columns = columns
        self._has_hands_part_names = all(map(parts.__contains__, (
            SceneElement.RH_flat, SceneElement.RH_spread, SceneElement.RH_grab,
            SceneElement.LH_flat, SceneElement.LH_spread, SceneElement.LH_grab,
        )))

    @property
    def pose_result_columns(self) -> pandas.MultiIndex:
        return self._pose_result_columns

    @staticmethod
    def _combine_frames_likelihood(frames_gen):
        return sum(frame["likelihood"] for frame in frames_gen)

    def _take_cams_most_likely(self, *dfs):
        """Returns the most likely cameras frame data for each element"""
        df0 = dfs[0]
        # create a df_res with len(dfs) entries with all NaNs :
        no_result = (math.nan, math.nan, 0)  # x, y, p
        df_res = pandas.DataFrame(index=list(range(len(dfs))), columns=self._measure_offset_parts_columns)
        min_combined_score = len(dfs) * self.MIN_CONFIDENCE_PRESENT_THRESHOLD
        for elem in df0.columns.levels[0]:
            combined = [
                (idx, self._combine_frames_likelihood(frames.loc[idx, elem] for frames in dfs))
                for idx in range(len(df0))
            ]
            # NB: this sorted, if it's equal to other(s), keeps the most recent last, so we take the last one [-1]:
            best_frame_idx, best_frame_score = sorted(combined, key=lambda e: e[1])[-1]
            for idx in range(len(dfs)):
                vals = (no_result if best_frame_score < min_combined_score
                        else dfs[idx].loc[best_frame_idx, elem].values)
                df_res.loc[idx, elem] = vals
        return df_res

    def _handle_3d_triangulate(
        self,
        *per_cam_detection: numpy.ndarray
    ):
        """Handle pose data offsets"""
        stereo_params = self._stereo_params
        if stereo_params is None:
            warnings.warn("no stereo params available, can't 3d-triangulate", UserWarning, stacklevel=3)
            return self._empty_3d, self._empty_3d
        if len(per_cam_detection) < 2:
            warnings.warn("at least two cameras are required for 3d-triangulate", UserWarning, stacklevel=3)
            return self._empty_3d, self._empty_3d
        p_thresh = 0.9  # confidence threshold for DLC raw output
        min_cluster = 10  # maximum allowed interpolation
        # not sure min_cluster change anything for when nbr frames == 1 (per cam)
        #
        frames_per_cam = len(per_cam_detection[0])
        #
        # reshape then sort by confidence/likelihood and takes most likely:
        columns = self._measure_offset_parts_columns
        df0_2d = pandas.DataFrame(per_cam_detection[0].reshape(frames_per_cam, -1), columns=columns)
        df1_2d = pandas.DataFrame(per_cam_detection[1].reshape(frames_per_cam, -1), columns=columns)
        #
        df_2d = self._take_cams_most_likely(df0_2d, df1_2d)
        confident_parts = [
            part
            for part in df_2d.columns.levels[0]
            if all(df_2d[part]["likelihood"] >= self.MIN_CONFIDENCE_PRESENT_THRESHOLD)
        ]
        confident_df = df_2d[confident_parts]
        if len(confident_parts) == 0:
            return self._empty_3d, self._empty_3d
        # df_2d = interpolate_coordinates(df_2d, p_thresh)  # not required probably
        raw_df_3d = triangulate_3d_with_params(
            # [df_2d.iloc[0:1][confident_parts], df_2d.iloc[1:2]],
            [confident_df.iloc[0:1], confident_df.iloc[1:2]],
            body_parts=confident_parts,  # self._measure_offset_parts,
            stereo_params=self._stereo_params,
            p_thresh=p_thresh,
            min_cluster=min_cluster,
        )
        # but reorient and center looks required:
        center_method = (1, SceneElement.Diamond)
        df_3d = reorient_and_center_step1(
            df_3d=raw_df_3d,
            stereo_file=stereo_params.as_pickle_dict(),
            center_method=center_method,
            frame_rate=self._frame_rate,
            bpts=confident_parts,  # self._measure_offset_parts,
            calib_metadata=self._calib_metadata,
            cam_names=self._cam_names,
            cam_offsets=self._cam_offsets,
            square_size=self._square_size,
            save_offsets=False,
            src_dir="/dev/null",
        )
        return raw_df_3d, df_3d

    def process(self,
        all_frames: List[numpy.ndarray],
        *,
        pairs_3d_offsets: Pairs3dOffsetT = (),
        sequence: Optional[int] = None,
        camera_count: int = 2,
    ) -> PoseResponse:
        """
        Process the frames from the pose model and return a PoseResponse.
        Args:
            all_frames: an interleaved list of pose frame data from all cameras.
            pairs_3d_offsets: List of 2-tuple pairs of parts to compute their 3d offsets.
        Returns:
            PoseResponse: a PoseResponse object with the processed data
        """
        if camera_count < 1:
            raise ValueError(f"camera_count must be at least 1, got {camera_count}")
        per_cam_frames = [
            all_frames[camera_index::camera_count]
            for camera_index in range(camera_count)
        ]

        return self.process_frames(*per_cam_frames,
                                   pairs_3d_offsets=pairs_3d_offsets,
                                   sequence=sequence)

    def process_frames(
        self,
        *per_cam_frames: List[numpy.ndarray],
        pairs_3d_offsets: Pairs3dOffsetT,
        sequence: Optional[int] = None,
    ) -> PoseResponse:
        """
        Function to process frames with all cameras frame results separated. Each frame is
        already reshaped to (num_body_parts, 3).
        Args:
            per_cam_frames: tuple of all frames per cam, sorted, for each cam, sorted.
            pairs_3d_offsets: List of 2-tuple pairs of parts to compute their 3d offsets.
        Returns:
            PoseResponse: a PoseResponse object with the processed data
        """
        self._sequence += 1

        if len(per_cam_frames) == 0:
            raise ValueError("At least one camera frame sequence is required")

        locations_by_cam = [
            self._find_parts(frames)
            for frames in per_cam_frames
        ]

        parts_flags_by_cam = [
            dict(self._default_parts_flag)
            for _ in per_cam_frames
        ]
        parts_flag_all = dict(self._default_parts_flag)

        # get parts presence:
        for poses in zip(*per_cam_frames):
            for idx, part in enumerate(self._parts_list):
                seen_by_all = True
                for cam_idx, pose in enumerate(poses):
                    if pose[idx, 2] >= PoseAlgorithm.MIN_CONFIDENCE_PRESENT_THRESHOLD:
                        parts_flags_by_cam[cam_idx][part] = True
                    else:
                        seen_by_all = False
                if seen_by_all:
                    parts_flag_all[part] = True

        selected_cams_frames = per_cam_frames
        # if self.process_frames_select_frames_method == "last_one":
        #     cams_last_frame = [[cam_frames[-1]] for cam_frames in per_cam_frames]
        #     selected_cams_frames = cams_last_frame
        # elif self.process_frames_select_frames_method == "all_most_likely":
        #     selected_cams_frames = per_cam_frames
        # else:
        #     raise RuntimeError(f"unexpected select_frames_method: {self.process_frames_select_frames_method}")

        gpi = self.get_part_index
        #
        if self._has_hands_part_names:
            # compute L_Hand / R_Hand averaged position (based on possibly many sub-hand parts)
            all_lst = [
                [f[gpi(p)] for p in self._hands_input_parts]
                for f in itertools.chain(*selected_cams_frames)
            ]
            all_frames = numpy.asarray(all_lst).reshape(len(all_lst), -1)
            df = pandas.DataFrame(
                all_frames,
                columns=self._hands_input_columns)
            process_hands_results = pandas.DataFrame(columns=self._hands_columns, index=range(len(df)))
            process_hands_results = process_hand_data(
                df,
                hand_base_names=self._hand_base_names,
                hand_options=self._hand_options,
                dlc_seg="_raw2D",
                newdf=process_hands_results,
                additional_names=[],
            )
            assert len(process_hands_results) == len(df)
            start_idx = 0
            for cam_idx, frames in enumerate(selected_cams_frames):
                raw = process_hands_results.iloc[start_idx:start_idx + len(frames)]
                start_idx += len(frames)
                for elem in SceneElement.L_Hand, SceneElement.R_Hand:
                    if __debug__ and elem not in process_hands_results.columns:
                        logger.warning("%s not present in hands results", elem)
                        continue
                    # if self.process_frames_select_frames_method == "last_one":
                    #     val = raw[elem].iloc[-1]
                    # else:
                    # but if want uses most likelihood, then:
                    val = raw[elem].sort_values(by="likelihood", ascending=False).reset_index().iloc[0]
                    if val['likelihood'] >= self.MIN_CONFIDENCE_PRESENT_THRESHOLD:
                        locations_by_cam[cam_idx][elem] = PoseLocation(-1, *val[_xy_col_names])
        #
        locations_3d = {}
        raw_3d_loc = {}
        raw_df_3d, df_3d = self._handle_3d_triangulate(*(
            numpy.asarray([
                [frame[gpi(p)] for p in self._measure_offset_parts]
                for frame in frames
            ])
            for frames in selected_cams_frames
        ))
        #
        df_3d_row = df_3d.iloc[0]  # there is only a single result in the df_3d
        raw_df_3d_row = raw_df_3d.iloc[0]
        for part in df_3d.columns.levels[0]:
            p_3d = df_3d_row[part]
            r_p_3d = raw_df_3d_row[part]
            if r_p_3d["p"] >= self.MIN_CONFIDENCE_PRESENT_THRESHOLD:
                raw_3d_loc[part] = Offset3DTuple(r_p_3d[0:3])
                locations_3d[part] = Offset3DTuple(p_3d[0:3])  # 3 first columns (x, y, z)
        #
        parts_3d_offsets = defaultdict(dict)
        if len(pairs_3d_offsets) > 0:
            for part1, part2 in pairs_3d_offsets:
                loc1 = None
                if part1 in locations_3d:
                    loc1 = locations_3d[part1]
                    if part2 in locations_3d:
                        loc2 = locations_3d[part2]
                        parts_3d_offsets[part1][part2] = Offset3DTuple(loc2 - loc1)

        if sequence is None:
            sequence = self._sequence
        response = PoseResponse(
            sequence=sequence,
            parts_flags=tuple(parts_flags_by_cam + [parts_flag_all]),
            locations=locations_by_cam,
            parts_3d_offsets=dict(parts_3d_offsets),
            locations_3d=locations_3d,
            raw_loc_3d=raw_3d_loc,
        )
        return response

    def _find_parts(self, frames: List[numpy.ndarray]) -> Dict[str, PoseLocation]:
        locations: Dict[str, PoseLocation] = {}
        for pose in frames:
            for idx, part in enumerate(self._parts_list):
                if pose[idx, 2] >= PoseAlgorithm.MIN_CONFIDENCE_PLOT_THRESHOLD:
                    locations[part] = PoseLocation(idx, pose[idx, 0], pose[idx, 1])
        return locations


def _loop_parts(perf_c, parts, is_seen, p_miss, p_pres):
    changes_seen = set()
    changes_miss = set()
    for part in parts:
        seen = is_seen(part)
        #
        prev_miss = p_miss.get(part)
        prev_seen = p_pres.get(part)
        if seen:
            if prev_seen is None or (prev_miss is not None and prev_miss > prev_seen):
                changes_seen.add(part)
        else:
            if prev_miss is None or (prev_seen is not None and prev_seen > prev_miss):
                changes_miss.add(part)
    #
    for part in changes_seen:
        p_pres[part] = perf_c
    for part in changes_miss:
        p_miss[part] = perf_c


def update_scene_elements_context_from_pose(
    any_cam_context: ScenePartsPresenceContext,
    all_cams_context: ScenePartsPresenceContext,
    pose_response: PoseResponse,
):
    """Update in-place the given context based on the given response"""
    perf_c = pose_response.perf_c
    cam_flags = pose_response.parts_flags[:-1]
    pall = pose_response.parts_flags[-1]
    p_any_miss = any_cam_context.missing_last_perf_c
    p_any_pres = any_cam_context.present_last_perf_c
    p_all_miss = all_cams_context.missing_last_perf_c
    p_all_pres = all_cams_context.present_last_perf_c
    prev_parts = set(p_any_miss) | set(p_any_pres)
    all_seen_parts = set().union(*cam_flags) if len(cam_flags) > 0 else set()
    all_parts = prev_parts | all_seen_parts
    _loop_parts(perf_c, all_parts, lambda p: any(flags.get(p, False) for flags in cam_flags), p_any_miss, p_any_pres)
    _loop_parts(perf_c, all_parts, lambda p: pall.get(p, False), p_all_miss, p_all_pres)
    any_cam_context.last_perf_c = all_cams_context.last_perf_c = perf_c
