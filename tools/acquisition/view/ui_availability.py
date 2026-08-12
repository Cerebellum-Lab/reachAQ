"""One authoritative calculation for top-level acquisition controls."""

from dataclasses import dataclass

from tools.acquisition.model.app_model_status import (
    AppModelStatus,
    SessionRecordingStatus,
)


@dataclass(frozen=True)
class UiAvailability:
    system_mode: bool
    preferences: bool
    idle_configuration: bool
    hardware_refresh: bool
    subject: bool
    protocol_selection: bool
    notes: bool
    calibration: bool


def calculate_ui_availability(
    *,
    status: AppModelStatus,
    recording_status: SessionRecordingStatus,
    acquisition_started: bool,
    capture_transition: bool,
    hardware_refreshing: bool,
    nidaq_discovering: bool,
    has_valid_dcs: bool,
) -> UiAvailability:
    session_ready = recording_status is SessionRecordingStatus.READY
    background_busy = hardware_refreshing or nidaq_discovering
    stable_mode = not capture_transition and not background_busy
    ordinary_mode = status in {AppModelStatus.IDLE, AppModelStatus.RUNNING}
    idle_configuration = (
        session_ready
        and stable_mode
        and status is AppModelStatus.IDLE
        and not acquisition_started
    )
    can_refresh = (
        session_ready
        and stable_mode
        and (
            (status is AppModelStatus.IDLE and not acquisition_started)
            or (status is AppModelStatus.RUNNING and acquisition_started)
        )
    )
    session_identity = session_ready and stable_mode and ordinary_mode
    return UiAvailability(
        system_mode=session_ready and stable_mode and ordinary_mode,
        preferences=session_ready and stable_mode,
        idle_configuration=idle_configuration,
        hardware_refresh=can_refresh,
        subject=session_identity,
        protocol_selection=session_identity,
        notes=True,
        calibration=(
            session_ready
            and stable_mode
            and status is AppModelStatus.RUNNING
            and acquisition_started
            and has_valid_dcs
        ),
    )
