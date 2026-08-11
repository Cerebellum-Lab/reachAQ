import logging

from autotrainer.device.rfid_reader import RfidTagRead

from tools.acquisition.model.rfid_metadata_controller import RfidMetadataController


TAG = "360002353933099"


class _Reader:
    def __init__(self, *, on_tag, on_status, device="/dev/rfid", **_options):
        self.on_tag = on_tag
        self.on_status = on_status
        self.device = device
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class _Registry:
    def resolve_rfid(self, rfid):
        assert rfid == TAG
        return None

    def last_complete_import(self):
        return None


class _AppModel:
    def __init__(self):
        self.calls = []

    def handle_rfid_scan(self, record, **kwargs):
        self.calls.append((record, kwargs))
        return "unknown"


def test_reader_scan_and_registry_miss_are_logged(caplog):
    caplog.set_level(
        logging.INFO,
        logger="tools.acquisition.model.rfid_metadata_controller",
    )
    app_model = _AppModel()
    controller = RfidMetadataController(
        app_model=app_model,
        registry=_Registry(),
        dispatch=lambda callback: callback(),
        reader_factory=_Reader,
        device="/dev/rfid",
    )

    controller._on_tag(RfidTagRead(TAG, b"frame", 1.0))

    assert app_model.calls[0][1]["scanned_rfid"] == TAG
    messages = [entry.getMessage() for entry in caplog.records]
    assert f"RFID scan received: rfid={TAG}" in messages[0]
    assert any(
        f"RFID registry lookup: rfid={TAG} matched=False" in message
        for message in messages
    )
