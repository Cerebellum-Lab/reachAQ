import math
import threading
import time
from typing import Tuple, Optional, List

import pandas

from PySide6 import QtCore
from PySide6.QtCore import QTimer, Slot, Signal, Qt, QSize, QPoint, QPointF
from PySide6.QtGui import QPixmap, QPainter, QPen, QPolygon, QPolygonF, QImage, QFont
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QStackedLayout, QWidget, QSizePolicy, QScrollBar, \
    QScrollArea, QLayout, QTabWidget

from autotrainer.core import AnimalSubject, CameraId, ProjectInfo
from autotrainer.core.logging import get_verbose_logger

from autotrainer.inference import PoseResponse, PoseAlgorithm, InferenceStatus

from autotrainer.behavior.behavior_algorithm import BehaviorAlgoProps
from autotrainer.inference.analysis import IntersessionResponse

from autotrainer.pyside import Separator, CardWidget
from autotrainer.pyside.StackedContent import StackedWidget, StackedLayout
from autotrainer.pyside.content_widget import ContentWidget, invoke_method

from autotrainer.training import TrainingPlan, TrainingPhase
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.view.analysis_content import AnalysisContent
from tools.acquisition.view.behavior_content import BehaviorContent
from tools.acquisition.view.camera_content import CameraContent
from tools.acquisition.view.diagnostics_content import DiagnosticsContent
from tools.acquisition.view.hardware_control_content import HardwareControlContent
from tools.acquisition.view.hardware_status_content import HardwareStatusContent
from tools.acquisition.view.laser_control_content import LaserControlContent
from tools.acquisition.view.protocol_content import ProtocolContent
from tools.acquisition.view.persistent_splitter import PersistentSplitter
from tools.acquisition.view.training_phase_content import TrainingPhaseContent
from tools.acquisition.view.training_phase_progress_content import TrainingPhaseProgressContent
from tools.acquisition.view.training_plan_content import TrainingPlanContent
from tools.acquisition.view.training_plan_progress_content import TrainingPlanProgressContent

logger = get_verbose_logger(__name__)

_REACHAQ_PROTOCOL_UI_ENABLED = False


def _camera_panel_title(camera_name: str) -> str:
    if camera_name == "stimCam":
        return "stimCam Camera"
    if camera_name.startswith("camera") and camera_name[6:].isdigit():
        return f"Camera {camera_name[6:]}"
    return f"{camera_name.capitalize()} Camera"


def _visible_reach_cameras(cameras, *, include_disabled_optional: bool = False):
    """Show optional cameras when enabled, plus stimCam while settings are edited."""
    return tuple(
        camera
        for camera in cameras
        if (
            camera.camera_id in (CameraId.Left, CameraId.Right)
            or camera.is_enabled
            or (include_disabled_optional and camera.camera_id == CameraId.Camera3)
        )
    )


class MainContent(ContentWidget):

    training_plan_changed = Signal(TrainingPlan)

    def __init__(self, app_model: AppModel):
        super().__init__()

        self._app_model = app_model
        self._protocol_ui_enabled = _REACHAQ_PROTOCOL_UI_ENABLED

        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("MainContent")
        self.setStyleSheet("#MainContent {background-color: #f3f4f6}")

        self._content_widgets: List[ContentWidget] = []
        self._preferences = app_model.preferences
        self._is_camera_settings_editable = False

        self.setContentsMargins(0, 0, 0, 0)

        root_layout = QHBoxLayout()
        self.setLayout(root_layout)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._main_splitter = PersistentSplitter(
            Qt.Orientation.Horizontal,
            self._preferences,
            "main_horizontal",
        )
        root_layout.addWidget(self._main_splitter)

        left_content = self._left_content = QWidget()
        left_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._main_splitter.addWidget(left_content)

        main_layout = self._main_layout = QVBoxLayout(left_content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._section_splitter = PersistentSplitter(
            Qt.Orientation.Vertical,
            self._preferences,
            "main_vertical",
            left_content,
        )
        main_layout.addWidget(self._section_splitter)

        self._top_widget_manual = self._create_top_widget_manual()
        self._section_splitter.addWidget(self._top_widget_manual)
        # don't put alignment or the stretch used below won't be effective

        # Second row - behavior and analysis
        mid_stacked_widget = self._mid_stacked_widget = QWidget()
        mid_stacked_layout = self._mid_stacked_layout = StackedLayout(mid_stacked_widget)
        self._section_splitter.addWidget(mid_stacked_widget)

        self._mid_widget_manual = self._create_mid_widget_manual(app_model)
        mid_stacked_layout.addWidget(self._mid_widget_manual)

        self._protocol_phase_progress_widget = None
        if self._protocol_ui_enabled:
            self._protocol_phase_progress_widget = self._create_protocol_phase_progress_widget()
            mid_stacked_layout.addWidget(self._protocol_phase_progress_widget)

        # Third row // bottom widgets
        end_stacked_widget = self._end_stacked_widget = QWidget()
        end_stacked_layout = self._end_stacked_layout = StackedLayout(end_stacked_widget)

        self._section_splitter.addWidget(end_stacked_widget)

        end_widget_manual = self._end_widget_manual = self._create_end_widget_manual()
        end_stacked_layout.addWidget(end_widget_manual)

        self._training_plan_content = None
        self._training_phase_content = None
        self._training_plan_progress_content = None
        self._training_phase_progress_content = None
        self._protocol_phase_end_widget = None

        if self._protocol_ui_enabled:
            # Limit end_protocol_phase widget to the phase content size.
            def size_hint(orig=end_stacked_widget.sizeHint):
                if end_stacked_layout.currentWidget() == end_protocol_phase_widget:
                    sz1 = self._training_phase_content.minimumSizeHint()
                    return QSize(sz1.width(), sz1.height())
                else:
                    return orig()
            end_stacked_widget.sizeHint = size_hint

            def min_size(orig=end_stacked_widget.minimumSize):
                if end_stacked_layout.currentWidget() == end_protocol_phase_widget:
                    sz1 = self._training_phase_content.minimumSize()
                    return QSize(sz1.width(), sz1.height())
                else:
                    return orig()
            end_stacked_widget.minimumSize = min_size

            end_protocol_phase_widget = self._protocol_phase_end_widget = self._create_protocol_phase_end_widget()
            end_stacked_layout.addWidget(end_protocol_phase_widget)

            self._training_plan_content.sizeHint = size_hint  # trying
            # self._training_plan_content.minimumSize = size_hint
            # self._training_phase_content.sizeHint = size_hint
            # self._training_phase_content.minimumSize = size_hint

        end_stacked_layout.setCurrentWidget(end_widget_manual)

        # Optional fourth row - diagnostics
        self._diagnostics_content = DiagnosticsContent(self._app_model)
        self._section_splitter.addWidget(self._diagnostics_content)

        self._right_side_tabs = self._create_right_side_tabs()
        self._main_splitter.addWidget(self._right_side_tabs)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)
        self._main_splitter.apply_saved_or_default_sizes([1180, 430])
        self._section_splitter.apply_saved_or_default_sizes([340, 410, 260, 120])

        self._frame_count = 0
        self._start = 0

        self._is_diagnostics_visible = True
        self.set_diagnostics_visible(False)

        self._prev_parts_3d_loc = {}
        self._next_parts_3d_loc_report = time.perf_counter()
        self._pending_pose_response = None
        self._pending_pose_lock = threading.Lock()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_image)
        self._timer.start(int(1000 / self._app_model.preferences.live_feed_refresh_rate))

        self._hardware_control_content.set_selected_animal(app_model.selected_animal)

        # finally, register handlers to events:
        app_model.property_changed += self._model_property_changed
        app_model.configuration_loaded_event += self._on_config_loaded
        #
        inference = app_model.inference
        inference.pose_response_ready += self.refresh_pose
        #
        # app_model.behavior.algorithm.property_changed += self._behavior_algo_property_changed
        self.training_plan_changed.connect(self._update_training_plan)

    @property
    def protocol_ui_enabled(self) -> bool:
        return self._protocol_ui_enabled

    def _create_top_widget_manual(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        widget.setContentsMargins(4, 4, 4, 0)
        top_layout = QVBoxLayout(widget)
        top_layout.setContentsMargins(4, 4, 4, 0)
        top_layout.setSpacing(0)

        self._camera_rows_splitter = PersistentSplitter(
            Qt.Orientation.Vertical,
            self._preferences,
            "camera_rows",
            widget,
        )
        top_layout.addWidget(self._camera_rows_splitter)

        app_model = self._app_model
        self._reach_camera_contents = []
        self._reach_camera_content_by_model = {}
        self._left_camera_content = None
        self._right_camera_content = None
        self._camera_row_splitters = []
        self._rebuild_reach_camera_grid()
        return widget

    @staticmethod
    def _reach_camera_grid_columns(camera_count: int) -> int:
        if camera_count <= 1:
            return 1
        if camera_count <= 3:
            return camera_count
        if camera_count == 4:
            return 2
        return 3

    def _clear_reach_camera_grid(self) -> None:
        for camera, camera_content in self._reach_camera_contents:
            camera.set_display_fcn(None)
            if camera_content in self._content_widgets:
                self._content_widgets.remove(camera_content)
            camera_content.close()
            camera_content.setParent(None)
            camera_content.deleteLater()
        self._reach_camera_contents = []
        self._reach_camera_content_by_model = {}
        self._left_camera_content = None
        self._right_camera_content = None
        for splitter in self._camera_row_splitters:
            splitter.setParent(None)
            splitter.deleteLater()
        self._camera_row_splitters = []

    def _rebuild_reach_camera_grid(self) -> None:
        self._clear_reach_camera_grid()

        app_model = self._app_model
        cameras = _visible_reach_cameras(
            app_model.reach_cameras,
            include_disabled_optional=self._is_camera_settings_editable,
        )
        columns = self._reach_camera_grid_columns(len(cameras))
        rows = max(1, math.ceil(len(cameras) / columns))

        for row_index in range(rows):
            row_splitter = PersistentSplitter(
                Qt.Orientation.Horizontal,
                self._preferences,
                f"camera_row_{row_index}",
                self._camera_rows_splitter,
            )
            self._camera_rows_splitter.addWidget(row_splitter)
            self._camera_row_splitters.append(row_splitter)

        for idx, camera in enumerate(cameras):
            camera_content = CameraContent(app_model, camera)
            camera_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            camera_content.camera_view.setTitle(_camera_panel_title(camera.name))
            camera_content.set_is_editable(self._is_camera_settings_editable)
            camera.set_display_fcn(camera_content.refresh_image)
            self._camera_row_splitters[idx // columns].addWidget(camera_content)
            self._content_widgets.append(camera_content)
            self._reach_camera_contents.append((camera, camera_content))
            self._reach_camera_content_by_model[camera] = camera_content
            if camera is app_model.left_camera:
                self._left_camera_content = camera_content
            elif camera is app_model.right_camera:
                self._right_camera_content = camera_content

        for splitter in self._camera_row_splitters:
            splitter.apply_saved_or_default_sizes([1] * max(1, splitter.count()))
        self._camera_rows_splitter.apply_saved_or_default_sizes([1] * rows)

    def _create_mid_widget_manual(self, app_model):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        widget.setContentsMargins(4, 0, 4, 0)

        mid_layout = QVBoxLayout(widget)
        mid_layout.setContentsMargins(4, 4, 4, 0)
        mid_layout.setSpacing(0)

        self._mid_splitter = PersistentSplitter(
            Qt.Orientation.Horizontal,
            self._preferences,
            "behavior_analysis",
            widget,
        )
        mid_layout.addWidget(self._mid_splitter)

        behavior_content = BehaviorContent(
            app_model,
            app_model.behavior,
            app_model.inference,
        )
        behavior_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._mid_splitter.addWidget(behavior_content)
        self._content_widgets.append(behavior_content)

        self._analysis_content = AnalysisContent(app_model)
        self._mid_splitter.addWidget(self._analysis_content)
        self._content_widgets.append(self._analysis_content)

        self._mid_splitter.apply_saved_or_default_sizes([360, 820])

        return widget

    def _create_end_widget_manual(self):
        # Third row - hardware controls/status
        widget = QWidget()
        widget.setContentsMargins(4, 0, 4, 0)

        end_layout = QVBoxLayout(widget)
        end_layout.setContentsMargins(4, 4, 4, 4)
        end_layout.setSpacing(0)

        self._hardware_splitter = PersistentSplitter(
            Qt.Orientation.Horizontal,
            self._preferences,
            "hardware_panels",
            widget,
        )
        end_layout.addWidget(self._hardware_splitter)

        hardware_control_content = self._hardware_control_content = HardwareControlContent(self._app_model)
        hardware_control_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._hardware_splitter.addWidget(hardware_control_content)
        self._content_widgets.append(hardware_control_content)

        hardware_status_content = self._hardware_status_content = HardwareStatusContent(self._app_model)
        hardware_status_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._hardware_splitter.addWidget(hardware_status_content)
        self._content_widgets.append(hardware_status_content)

        self._hardware_splitter.apply_saved_or_default_sizes([500, 680])

        return widget

    def set_hardware_refreshing(self, is_refreshing: bool):
        self._hardware_status_content.set_hardware_refreshing(is_refreshing)

    def set_hardware_refresh_action(self, action):
        self._hardware_status_content.set_refresh_action(action)

    def _create_right_side_tabs(self):
        tabs = QTabWidget()
        tabs.setObjectName("ReachAQRightSideTabs")
        tabs.setDocumentMode(True)
        tabs.setMinimumWidth(0)
        tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        tabs.setStyleSheet(
            "QTabWidget::pane {border: 1px solid #c9cdd3; background: #ffffff; top: -1px;}"
            "QTabBar::tab {background: #e7eaee; color: #20242a; border: 1px solid #c9cdd3; padding: 4px 10px;}"
            "QTabBar::tab:selected {background: #ffffff; border-bottom-color: #ffffff;}"
            "QTabBar::tab:!selected {margin-top: 2px;}"
        )

        laser_control_content = self._laser_control_content = LaserControlContent(self._app_model)
        tabs.addTab(laser_control_content, "Laser Control")
        self._content_widgets.append(laser_control_content)

        protocol_content = self._protocol_content = ProtocolContent()
        tabs.addTab(protocol_content, "Protocol")
        self._content_widgets.append(protocol_content)

        return tabs

    def _create_protocol_phase_end_widget(self):
        widget = QWidget()
        # widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        widget.setContentsMargins(4, 0, 4, 0)

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        left = QHBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)

        # NB: same remark than manual end widget for stretch=1 :
        plan_content = self._training_plan_content = TrainingPlanContent()
        left.addWidget(plan_content)

        phase_content = self._training_phase_content = TrainingPhaseContent()
        left.addWidget(phase_content)

        layout.addLayout(left, stretch=1)

        return widget

    def _create_protocol_phase_progress_widget(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        widget.setContentsMargins(4, 0, 4, 0)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(8)
        content = self._training_plan_progress_content = TrainingPlanProgressContent()
        layout.addWidget(content)
        right_layout = QHBoxLayout()
        layout.addLayout(right_layout)
        content = self._training_phase_progress_content = TrainingPhaseProgressContent()
        right_layout.addWidget(content)
        return widget

    def _update_training_plan(self, plan: Optional[TrainingPlan]):
        if not self._protocol_ui_enabled:
            return
        logger.debug("setting plan to %s (%s)", plan, hex(id(plan)))
        if plan is None:
            self._mid_stacked_layout.setCurrentWidget(self._mid_widget_manual)
            self._end_stacked_layout.setCurrentWidget(self._end_widget_manual)
        else:
            self._mid_stacked_layout.setCurrentWidget(self._protocol_phase_progress_widget)
            self._end_stacked_layout.setCurrentWidget(self._protocol_phase_end_widget)
        self._training_phase_content.set_training_phase(
            None if plan is None else plan.current_phase,
            force_refresh=True,
        )
        self._training_plan_content.set_training_plan(plan)
        self._training_plan_progress_content.set_training_plan_progress(plan)
        phase = None if plan is None else plan.current_phase
        self._training_phase_progress_content.set_training_phase_progress(phase)
        self._end_stacked_widget.update()
        self.updateGeometry()
        self.update()

    def close(self):
        self._clear_reach_camera_grid()
        # Ensure the textbox handler is removed from root logger handlers.
        self._diagnostics_content.close()
        super().close()

    @Slot()
    def update_image(self):
        self._flush_pending_pose()
        for camera, camera_content in self._reach_camera_contents:
            if camera.is_enabled:
                camera_content.update_image()

    def refresh_pose(self, response: PoseResponse):
        """Cache only the newest inference result until the next display tick.

        Live inference can produce pose results much faster than Qt renders the
        camera panels.  Dispatching every result to every camera creates an
        unbounded queue of UI-thread calls and delays unrelated timers, including
        the live analysis plot.  The intermediate poses are never displayed, so
        retain just the newest result and consume it at the configured live-feed
        refresh rate.
        """
        with self._pending_pose_lock:
            self._pending_pose_response = response

    def _flush_pending_pose(self):
        with self._pending_pose_lock:
            response = self._pending_pose_response
            self._pending_pose_response = None
        if response is None:
            return
        for idx, camera in enumerate(self._app_model.inference_cameras):
            camera_content = self._reach_camera_content_by_model.get(camera)
            if camera_content is not None and camera.is_enabled and idx < len(response.locations):
                camera_content.refresh_pose(response.locations[idx])
        if __debug__:
            perf_now = time.perf_counter()
            if perf_now >= self._next_parts_3d_loc_report:
                self._next_parts_3d_loc_report = perf_now + 0.5
                for part, loc_3d in response.locations_3d.items():
                    if response.is_part_seen(part):
                        prev = self._prev_parts_3d_loc.get(part)
                        if prev is None or any(
                            abs(prev[i] - loc_3d[i]) >= 0.15
                            for i in range(3)
                        ):
                            logger.spam("%s: loc3d: %s", part, loc_3d.humanize())
                            self._prev_parts_3d_loc[part] = loc_3d if prev is None else (prev + loc_3d) / 2

    @property
    def is_diagnostics_visible(self) -> bool:
        return self._is_diagnostics_visible

    @invoke_method
    def set_is_editable(self, is_editable: bool):
        is_editable = bool(is_editable)
        if self._is_camera_settings_editable != is_editable:
            self._is_camera_settings_editable = is_editable
            self._rebuild_reach_camera_grid()
        for widget in self._content_widgets:
            widget.set_is_editable(is_editable)

    @invoke_method
    def set_is_capture_active(self, is_active: bool):
        """Capture active == acquisition started"""
        for widget in self._content_widgets:
            widget.set_is_capture_active(is_active)

    @invoke_method
    def on_activated(self):
        self._app_model.on_activated()

        for camera, camera_content in self._reach_camera_contents:
            camera.set_display_fcn(camera_content.refresh_image)
        for widget in self._content_widgets:
            widget.on_activated()

    @invoke_method
    def set_diagnostics_visible(self, is_visible: bool):
        self._diagnostics_content.setVisible(is_visible)
        self._is_diagnostics_visible = is_visible

    @staticmethod
    def _paint_diamond(x, y, painter, *, size=6, width=3, color):
        center = size // 2
        coords = [
            (v1 + x, v2 + y)
            for v1, v2 in (
                (center, 0),  # Top
                (size, center),  # Right
                (center, size),  # Bottom
                (0, center)  # Left
            )
        ]
        coords = list(map(lambda v: QPointF(v[0], v[1]), coords))
        pen = QPen()
        pen.setWidth(width)
        pen.setColor(color)
        pen.setBrush(color)
        painter.setPen(pen)
        polygon = QPolygonF(coords)
        painter.drawPolygon(polygon)

    def _paint_reach_event(
        self, *,
        cam, cam_name, px, painter,
        df_reach, df_pellet, df_trajectory,
        width_f, height_f,
    ):
        x_y_cols = list("xy")
        pen = QPen()
        if len(df_reach) == 0:
            if cam is self._left_camera_content:
                pen.setColor(Qt.GlobalColor.yellow)
                painter.setOpacity(1)
                painter.setPen(pen)
                font = QFont("Sans-serif", 12)
                painter.setFont(font)
                painter.drawText(px.rect(), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                                 "No Reach Events Last Trial")
            return
        painter.setOpacity(0.75)
        for r_idx in range(len(df_reach)):
            # the reach indices here are global for the entire trial,
            r_max = df_reach.loc[r_idx, "max"]
            if math.isnan(r_max) or r_max is None:
                continue
            r_max_idx = r_max - df_reach.loc[r_idx, "init"]
            if r_max_idx < 0:
                continue
            # but df_trajectories is 1 sub-df per reach, each indexed from 0 to nbframes_in_reach_event.
            # so this subtracts "init" frame index.
            # R_H
            pen.setColor(Qt.GlobalColor.yellow)
            pen.setWidth(1)
            painter.setPen(pen)
            vals = [
                (x * width_f, y * height_f)
                for x, y in df_trajectory.loc[r_idx][cam_name][x_y_cols].values
            ]
            logger.debug("r_h vals=%s", vals)
            reach_max_x, reach_max_y = vals[r_max_idx]
            vals = list(map(lambda v: QPointF(v[0], v[1]), vals))
            painter.drawPolyline(vals)
            self._paint_diamond(reach_max_x, reach_max_y, painter, color=Qt.GlobalColor.yellow)
            # Pellet at reach max
            pel_x, pel_y = df_pellet.loc[r_idx, cam_name][x_y_cols]  # noqa
            self._paint_diamond(pel_x * width_f, pel_y * height_f, painter, color=Qt.GlobalColor.red)

    @invoke_method
    def show_analysis_reach_events(
        self,
        prj: Optional[ProjectInfo],
    ):
        logger.verbose("show_analysis_reach_events: %s", prj)
        if prj is None:
            for _, camera_content in self._reach_camera_contents:
                camera_content.camera_view.image_view.set_reach_overlay(None)
            return
        loc = prj.get_reach_event_path()
        df_reach = pandas.read_hdf(loc, key="reach")
        df_pellet = pandas.read_hdf(loc, key="pellet")
        df_trajectory = pandas.read_hdf(loc, key="trajectory")
        logger.verbose("reach:\n%s\npellet:\n%s\n", df_reach, df_pellet)
        logger.debug("trajectory:\n%s", df_trajectory)
        logger.debug("looping over %s reaches", len(df_reach))
        available_cam_names = set(df_trajectory.columns.get_level_values(0))
        for camera, cam in self._reach_camera_contents:
            cam_name = camera.name
            if cam_name not in available_cam_names:
                cam.camera_view.image_view.set_reach_overlay(None)
                continue
            assert len(df_trajectory.index.unique(0)) == len(df_reach)
            img_view = cam.camera_view.image_view
            width_f, height_f = img_view.size_factor
            logger.debug("size_factor: %s", img_view.size_factor)
            image = QImage(img_view.size(), QImage.Format.Format_RGBA8888)
            px = QPixmap.fromImage(image)
            px.fill(Qt.GlobalColor.transparent)
            painter = QPainter(px)
            with painter:
                self._paint_reach_event(
                    cam=cam, cam_name=cam_name, px=px, painter=painter,
                    df_reach=df_reach, df_pellet=df_pellet, df_trajectory=df_trajectory,
                    width_f=width_f, height_f=height_f,
                )
            cam.camera_view.image_view.set_reach_overlay(px)
        # logger.verbose("set reach events for %s: %s events", prj, len(df_reach))

    @invoke_method
    def _model_property_changed(self, name: str, value, _):
        app_model = self._app_model
        props = AppModel.Props
        if name == props.SELECTED_ANIMAL:
            self._hardware_control_content.set_selected_animal(value)
            self.training_plan_changed.emit(app_model.training_plan)  # ensure it's refreshed too
        elif name == props.TRAINING_PLAN:
            assert isinstance(value, (type(None), TrainingPlan))
            self.training_plan_changed.emit(value)
        elif name == props.TRAINING_PHASE:
            phase = app_model.training_plan.current_phase
            if phase != value:
                raise RuntimeError("plan phase != new phase: %s", phase, value)
            self.training_plan_changed.emit(app_model.training_plan)
        elif name in {props.TRAINING_PLAN_PROP, props.TRAINING_PHASE_PROP}:
            self.training_plan_changed.emit(app_model.training_plan)

    @invoke_method
    def _on_config_loaded(self, config):
        del config  # unused
        self._rebuild_reach_camera_grid()
        # only re-setting the current selected animal
        self._hardware_control_content.set_selected_animal(self._app_model.selected_animal)
        # this allows to set correctly for the possible diamond-triangle config loaded
