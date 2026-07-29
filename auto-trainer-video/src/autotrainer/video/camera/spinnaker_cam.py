import ast
import atexit
import dataclasses
import logging
import math
import statistics
import time
from enum import IntEnum
from typing import Tuple, List, Dict, Optional, Type

import numpy
import PySpin

from autotrainer.core import get_perf_now
from autotrainer.core.logging import get_verbose_logger

from .camera_base import CameraBase

logger = get_verbose_logger(__name__)

def is_truthy_str_value(value: str):
    return value.lower() in {"true", "yes", "on", "1"}


sSystem = None

def _start_spincam_lib_instance():
    global sSystem
    if sSystem is None:
        logger.info("getting spincam library instance")
        sSystem = PySpin.System.GetInstance()
        if sSystem is None:
            raise RuntimeError("PySpin.System.GetInstance() returned None")
    return sSystem


def _stop_spincam_lib_instance(cls: Type["SpinCam"]):
    global sSystem
    for key, spincam in list(cls._cameras.items()):  # loop over list copy of items,
        # given _release_spincam() will modify `cls._cameras` dict
        try:
            spincam.end_capture()
        except Exception as err:
            logger.exception("Error ending capture on %s: %s", spincam.name, err)
    cls._cameras.clear()
    if sSystem is not None:
        logger.info("releasing SpinCam lib")
        sSystem.ReleaseInstance()
        sSystem = None


class AcquisitionMode(IntEnum):
    Continuous = 0
    SingleFrame = 1
    MultiFrame = 2


@dataclasses.dataclass
class SpinCamDefaultParams:
    exposure: float = 140
    fps: float = 150.
    hbin: int = 4
    vbin: int = 4
    width: int = 256
    height: int = 256
    offsetx: int = 52
    offsety: int = 6
    gain: float = 1
    gamma: float = 0.7


class SpinCam(CameraBase):

    _cameras: Dict[str, "SpinCam"] = {}  # class level cache

    default_params = dataclasses.asdict(SpinCamDefaultParams())

    @classmethod
    def list(cls) -> List[str]:
        _start_spincam_lib_instance()
        serial_numbers = []
        cam_list = sSystem.GetCameras()  # noqa
        try:
            for cam in cam_list:
                serial_numbers.append(cam.TLDevice.DeviceSerialNumber.ToString())
        finally:
            cam_list.Clear()
        return serial_numbers

    @classmethod
    def create(cls, serial_number: str, name: str = ""):
        if serial_number in cls._cameras:
            return cls._cameras[serial_number]
        obj = SpinCam(serial_number, name)
        cls._cameras[serial_number] = obj
        return obj

    def __init__(self, serial_number, name: str = ""):

        self._node_map = None
        self._node_map_tl_device = None
        self._start_frames = []
        self._current_cam_frame_2_perf_offset = math.nan
        self._current_cam_frame_2_time_offset = math.nan
        self._consecutive_late_acquire = 0
        self._was_late_last = False

        super().__init__(name)

        self._serial_number = serial_number

        def get_def(k):
            v = self.default_params.get(k)
            if v is None:
                logger.verbose("No default for param %r", k)
            return v

        self._exposure: float = get_def("exposure")
        self._fps: float = get_def("fps")
        self._horizontal_binning: int = get_def("hbin")
        self._vertical_binning: int = get_def("vbin")
        self._width: int = get_def("width")
        self._height: int = get_def("height")
        self._offset_x: int = get_def("offsetx")
        self._offset_y: int = get_def("offsety")

        self._gain: float = get_def("gain")
        self._gamma: float = get_def("gamma")

        self._is_primary = False
        self._is_secondary = True

        self._acquisition_started = False
        self._skip_duplicate_frame_copy = False

        _start_spincam_lib_instance()

        cam_list = sSystem.GetCameras()  # noqa
        try:
            self._camera = cam_list.GetBySerial(serial_number)
        finally:
            cam_list.Clear()

        # not needed
        # self._image_processor = PySpin.ImageProcessor()
        # self._image_processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    def __del__(self):
        self.end_capture()

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(self, value: int) -> None:
        self._fps = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def offset_x(self):
        return self._offset_x

    @offset_x.setter
    def offset_x(self, value):
        self._offset_x = value

    @property
    def offset_y(self):
        return self._offset_y

    @offset_y.setter
    def offset_y(self, value):
        self._offset_y = value

    @property
    def horizontal_binning(self) -> int:
        return self._horizontal_binning

    @horizontal_binning.setter
    def horizontal_binning(self, value: int) -> None:
        self._horizontal_binning = value

    @property
    def vertical_binning(self) -> int:
        return self._vertical_binning

    @vertical_binning.setter
    def vertical_binning(self, value: int) -> None:
        self._vertical_binning = value

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self._exposure = value

    def _reinit_cam(self):
        spincam = self._camera
        logger.notice("doing cam reset with begin+end acquisition")
        # Stackoverflow 64660434.  Apparently there is no simple reset/release call to fix when it is in this state.
        spincam.BeginAcquisition()
        spincam.EndAcquisition()
        spincam.DeInit()
        spincam.Init()

    def init(self):
        spincam = self._camera
        spincam.Init()

        self._consecutive_late_acquire = 0
        self._acquisition_started = False

        self._node_map = spincam.GetNodeMap()

        if spincam.Width.GetAccessMode() != PySpin.RW:
            self._reinit_cam()

        node_acquisition_mode = PySpin.CEnumerationPtr(self._node_map.GetNode('AcquisitionMode'))

        node_acquisition_mode.SetIntValue(AcquisitionMode.Continuous.value)

        if spincam.ExposureAuto.GetAccessMode() == PySpin.RW:
            spincam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            logger.debug(f"<{self._name}> ExposureAuto set to {spincam.ExposureAuto.GetValue()}")
        else:
            logger.warning(f"<{self._name} ExposureAuto is not writeable")

        if spincam.GainAuto.GetAccessMode() == PySpin.RW:
            spincam.GainAuto.SetValue(PySpin.GainAuto_Off)
            logger.debug(f"<{self._name}> GainAuto set to {spincam.GainAuto.GetValue()}")
        else:
            logger.warning(f"<{self._name} GainAuto is not writeable")

        if spincam.Gain.GetAccessMode() == PySpin.RW:
            spincam.Gain.SetValue(0)
            logger.debug(f"<{self._name}> Gain set to {spincam.Gain.GetValue()}")
        else:
            logger.warning(f"<{self._name} Gain is not writeable")

        s_node_map = spincam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferHandlingMode"))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            logger.warning(f"<{self._name} unable to set Buffer Handling mode (node retrieval)")
        else:
            handling_mode_entry = handling_mode.GetEntryByName("OldestFirst")
            handling_mode.SetIntValue(handling_mode_entry.GetValue())

        # get buffer size (nbr of frames):
        # node_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferCountMode"))
        # node_buffer_count_mode_manual = node_buffer_count_mode.GetEntryByName("Manual")
        node_buffer_count_manual = PySpin.CIntegerPtr(s_node_map.GetNode("StreamBufferCountManual"))
        current_buffer_count = node_buffer_count_manual.GetValue()
        logger.success("Cam initialized. frames_buffer_size=%s", current_buffer_count)

        self._apply_settings(spincam)

    def _apply_settings(self, cam: PySpin.Camera):
        # exposure first:
        self._set_bounded_float_property_node(cam.ExposureTime, self._exposure)
        # then FPS:
        self._set_bounded_bool_property_node(cam.AcquisitionFrameRateEnable, True)
        self._set_bounded_float_property_node(cam.AcquisitionFrameRate, self._fps)

        # binning first
        self._set_bounded_int_property_node(cam.BinningHorizontal, self._horizontal_binning)
        self._set_bounded_int_property_node(cam.BinningVertical, self._vertical_binning)

        # then size:
        self._set_bounded_int_property_node(cam.Width, self._width)
        self._set_bounded_int_property_node(cam.Height, self._height)

        # then offset:
        self._set_bounded_int_property_node(cam.OffsetX, self._offset_x)
        self._set_bounded_int_property_node(cam.OffsetY, self._offset_y)

        if self._gain is not None:
            self._set_bounded_float_property_node(cam.Gain, self._gain)

        gamma = self._gamma
        if gamma is not None:
            self._set_bounded_bool_property_node(cam.GammaEnable, True)
            self._set_bounded_float_property_node(cam.Gamma, gamma)
        else:
            self._set_bounded_bool_property_node(cam.GammaEnable, False)

    def prepare_capture(self):
        super().prepare_capture()

        if self._is_primary == self._is_secondary:
            raise RuntimeError("Camera %s configured as both primary and secondary", self._name)

        if self._is_primary:
            self._configure_as_primary(self._camera)
            logger.info(f"<{self.name}> configured as primary")

        elif self._is_secondary:
            self._configure_as_secondary(self._camera)
            logger.info(f"<{self.name}> configured as secondary")

    def end_capture(self):
        super().end_capture()
        spincam = self._camera
        if spincam is None:
            return
        spincam.EndAcquisition()
        if self._is_primary:
            spincam.LineSelector.SetValue(PySpin.LineSelector_Line1)
            spincam.LineSource.SetValue(PySpin.LineSource_FrameTriggerWait)
            spincam.LineInverter.SetValue(True)
        elif self._is_secondary:
            spincam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        spincam.DeInit()
        self._cameras.pop(self._serial_number, None)
        logger.debug("released spincam for %s (%s)", self._name, self._serial_number)
        self._camera = None

    def _capture(self):
        # Keep the backend timeout below the application camera watchdog. A
        # fifteen-second block made capture shutdown lag well behind the
        # five-second watchdog after a trigger/camera failure.
        capture_timeout = 4
        p_timeout = time.perf_counter() + capture_timeout
        try_count = 0
        t_prev_after = p_prev_after = -math.inf
        while True:
            try_count += 1
            if time.perf_counter() > p_timeout:
                raise RuntimeError(
                    f"Failed to capture a frame within {capture_timeout} seconds"
                )
            t_before = time.time()
            p_before = time.perf_counter()
            try:
                image_result = self._camera.GetNextImage(1)  # 1 millisecond timeout
                p_after = time.perf_counter()
                t_after = time.time()
            except PySpin.SpinnakerException:
                t_prev_after = t_before
                p_prev_after = p_before
                continue
            if image_result.IsIncomplete():
                image_result.Release()
                t_prev_after = t_after
                p_prev_after = p_after
                continue
            frame_id = image_result.GetFrameID()
            frame_when = image_result.GetTimeStamp()
            frame_when_sec = frame_when / 1e9
            if try_count > 1:
                # best case: we retried at least once, with a 1 millisecond timeout,
                # so we can estimate as:
                estimated_frame_perf_c = (2 * p_prev_after + 3 * p_before) / 5  # good enough
                estimated_frame_time = (2 * t_prev_after + 3 * t_before) / 5
                self._current_cam_frame_2_perf_offset = estimated_frame_perf_c - frame_when_sec
                self._current_cam_frame_2_time_offset = estimated_frame_time - frame_when_sec
                self._consecutive_late_acquire = 0
            else:
                if not math.isfinite(self._current_cam_frame_2_perf_offset):
                    estimated_frame_perf_c = p_before  # best we can guess
                    estimated_frame_time = t_before
                else:
                    estimated_frame_perf_c = frame_when_sec + self._current_cam_frame_2_perf_offset
                    estimated_frame_time = frame_when_sec + self._current_cam_frame_2_time_offset
                late_delay = (p_before + p_after) / 2 - estimated_frame_perf_c
                if self._consecutive_late_acquire == 0 and late_delay > 0.050:  # 0.050 semi-arbitrary
                    logger.warning("late acquire: frame_id=%s p_before=%.3f p_after=%.3f when=%.3f perf_c=%.3f late_delay=%.3f",
                                   frame_id, p_before, p_after, frame_when_sec, estimated_frame_perf_c, late_delay)
                self._consecutive_late_acquire += 1
                if self._consecutive_late_acquire > 150 * 5:
                    logger.warning("very long running late acquires ; frame=%s when=%.3f perf_c=%.3f late_delay=%.1f",
                                   self._frame_count, frame_when_sec, estimated_frame_perf_c, late_delay)
                    self._consecutive_late_acquire = 0
            return image_result, frame_when, estimated_frame_perf_c, estimated_frame_time
        # end while True.

    def capture(self) -> Tuple[numpy.ndarray, int]:
        first_capture = False
        if not self._acquisition_started:
            self._acquisition_started = True
            first_capture = True
            self._camera.BeginAcquisition()
            if self._is_primary:
                self._camera.LineSelector.SetValue(PySpin.LineSelector_Line1)
                self._camera.LineSource.SetValue(PySpin.LineSource_Counter0Active)
                self._camera.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            logger.info("Beginning acquisition ; skip_dup_copy=%s", self._skip_duplicate_frame_copy)

        expected_shape = (self._height, self._width)

        image_result, frame_when, frame_perf, frame_time = self._capture()

        self._last_when = frame_when
        self._last_frame_perf_c = frame_perf
        self._last_frame_time = frame_time
        self._last_frame_id = image_result.GetFrameID()

        frame = orig_frame = image_result.GetNDArray()  # get the frame/array as acquired by hardware itself
        # image_converted = self._image_processor.Convert(image_result, PySpin.PixelFormat_Mono8)
        # frame = image_converted.GetNDArray()
        # reminder: the frame we get directly from camera is already in our desired format (shape + dtype).

        if first_capture:
            self._capture_start = self._last_when
            logger.notice("first frame: shape=%s (expected=%s) dtype=%s itemsize=%s ; ts=%.3f",
                          frame.shape, expected_shape, frame.dtype, frame.itemsize, self._last_when)
            if frame.shape != expected_shape:
                logger.warning("Frame shape not as expected: %s vs %s", frame.shape, expected_shape)

        if not self._skip_duplicate_frame_copy:
            frame = frame.copy()

        image_result.Release()

        if frame.shape != expected_shape:
            frame = frame.reshape(expected_shape)

        if __debug__:
            # ensure no frame in the first 150, shares its internal buffer with any of the other first 150 of them:
            if self._frame_count < 150:
                logger.spam("frame-%s: shape=%s dtype=%s",
                             self._frame_count, orig_frame.shape, orig_frame.dtype)
                self._start_frames.append((orig_frame, orig_frame.copy()))
                for prev_idx, (prev_frame, prev_frame_copy) in enumerate(self._start_frames):
                    if (prev_frame != prev_frame_copy).any():
                        logger.critical("Detected prev frame (idx=%s) got corrupted", prev_idx)
                    if prev_idx == self._frame_count:
                        break
                    if (orig_frame == prev_frame).all() and (orig_frame != prev_frame_copy).any():
                        logger.critical("Detected frame (idx=%s) shares internal buffer with prev frame idx=%s",
                                        self._frame_count, prev_idx)
                if self._frame_count >= 149:
                    self._start_frames.clear()  # don't keep unnecessarily all that

        self._frame_count += 1

        return frame, self._last_when

    def set_property(self, name: str, value: str) -> bool:
        if name == "primary":
            self._is_primary = is_truthy_str_value(value)
            self._is_secondary = not self._is_primary
        elif name == "secondary":
            self._is_secondary = is_truthy_str_value(value)
            self._is_primary = not self._is_secondary
        elif name in {"offsetx", "offset_x"}:
            self.offset_x = int(value)
        elif name in {"offsety", "offset_y"}:
            self.offset_y = int(value)
        elif name == "hbin":
            self.horizontal_binning = int(value)
        elif name == "vbin":
            self.vertical_binning = int(value)
        elif name == "exposure":
            self.exposure = int(value)
        elif name == "gain":
            self._gain = ast.literal_eval(value)  # allows None or float (or actually any literal)
        elif name == "gamma":
            self._gamma = ast.literal_eval(value)  # allows None or float (or actually any literal)
        elif name == "skip_duplicate_frame_copy":
            self._skip_duplicate_frame_copy = is_truthy_str_value(value)
        else:
            return super().set_property(name, value)

        return True

    def _configure_as_primary(self, cam: PySpin.Camera):
        cam.CounterSelector.SetValue(PySpin.CounterSelector_Counter0)
        cam.CounterEventSource.SetValue(PySpin.CounterEventSource_ExposureStart)
        cam.CounterEventActivation.SetValue(PySpin.CounterEventActivation_RisingEdge)
        cam.CounterTriggerSource.SetValue(PySpin.CounterTriggerSource_ExposureStart)
        cam.CounterTriggerActivation.SetValue(PySpin.CounterTriggerActivation_RisingEdge)
        cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
        cam.V3_3Enable.SetValue(True)
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
        cam.LineSource.SetValue(PySpin.LineSource_Counter0Active)
        cam.LineInverter.SetValue(False)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        # set trigger selector to frame start when trigger mode is off
        cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
        cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
        cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        # cam.TriggerActivation.SetValue(PySpin.TriggerActivation_AnyEdge)
        # raise
        # GenICam::AccessException= Node is not writable. :
        # AccessException thrown in node 'TriggerActivation' while calling 'TriggerActivation.SetIntValue()' (file 'EnumerationT.h', line 83) [-2006]

    def _configure_as_secondary(self, cam: PySpin.Camera):
        cam.AcquisitionFrameRateEnable.SetValue(False)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
        cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
        cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        # using same trigger activation than primary
        cam.TriggerActivation.SetValue(PySpin.TriggerActivation_AnyEdge)

    def _set_bounded_int_property_node(self, prop_node, value: int) -> int:
        return int(self._set_bounded_property(prop_node, value))

    def _set_bounded_float_property_node(self, prop_node, value: float) -> float:
        return float(self._set_bounded_property(prop_node, value))

    @staticmethod
    def _set_bounded_property(prop_node, value):
        name = prop_node.GetDisplayName()
        if value is None:
            logger.warning("%s: received None value for _set_bounded_float_property_node", name)
            return 0
        set_value = value
        try:
            if prop_node.GetAccessMode() == PySpin.RW:
                max_width = prop_node.GetMax()
                if max_width is None:
                    set_value = value
                    logger.debug("%s: has no Max value", name)
                else:
                    set_value = min(max_width, value)
                prop_node.SetValue(set_value)
                set_value = prop_node.GetValue()
                logger.debug("%s: applied %s requested=%s (max=%s)", name, set_value, value, max_width)
            elif prop_node.GetAccessMode() == PySpin.RO:
                set_value = prop_node.GetValue()
                logger.error("%s GenApi is not writeable - current value is %s", name, set_value)
            else:
                logger.error("%s GenApi is not readable or writeable", name)
        except Exception as err:
            logger.error("%s: Exception during set: %s", name, err)
            raise

        return set_value

    @staticmethod
    def _set_bounded_bool_property_node(prop_node, value: bool) -> Optional[bool]:
        set_value = value
        name = prop_node.GetDisplayName()
        try:
            if prop_node.GetAccessMode() == PySpin.RW:
                prop_node.SetValue(value)
                set_value = prop_node.GetValue()
                logger.debug("%s: set to %s ; requested %s", name, set_value, value)
            elif prop_node.GetAccessMode() == PySpin.RO:
                set_value = prop_node.GetValue()
                logger.warning(
                    "%s: GenApi is not writeable - current value is %s", name, set_value)
            else:
                logger.warning("%s: GenApi is not readable or writeable", name)
                set_value = None
        except Exception as err:
            logger.error("%s: Exception during set: %s", err)

        return set_value


atexit.register(lambda: _stop_spincam_lib_instance(SpinCam))
