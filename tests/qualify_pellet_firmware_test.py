"""map_stim_terminals: the NI terminal that watches each pulseable board output.

The watched terminal is the one PFI the output raises. On christielab10 each
STIM output also raises a digital readback line since 2.3.0 was qualified
(port0/line5 for STIM2, line6 for STIM3), and requiring exactly one raised
line failed both outputs and skipped every pulse check after them.
"""

from tools.hardware import qualify_pellet_firmware as qualify


def _map(monkeypatch, observed):
    def check_digital(nidaqmx, interface, configuration, points, report):
        for label, terminals in observed.items():
            report(label, terminals)

    monkeypatch.setattr(qualify, "check_digital", check_digital)
    monkeypatch.setattr(qualify, "wiring_points", lambda configuration: ())
    evidence = qualify.Evidence()
    return evidence, qualify.map_stim_terminals(evidence, None, None, None)


def _check(evidence, name):
    (check,) = [c for c in evidence.checks if c["name"] == f"{name} lands on NI"]
    return check


def test_a_pfi_with_a_digital_readback_maps_to_the_pfi_and_passes(monkeypatch):
    # christielab10, 2026-11-10, pellet-2.3.1-20261110T214525Z.json.
    evidence, terminals = _map(monkeypatch, {
        "board STIM2": ["PXI1Slot5/port0/line5", "/PXI1Slot5/PFI1"],
        "board STIM3": ["PXI1Slot5/port0/line6", "/PXI1Slot5/PFI0"],
    })

    assert terminals == {"STIM2": "/PXI1Slot5/PFI1", "STIM3": "/PXI1Slot5/PFI0"}
    stim2 = _check(evidence, "STIM2")
    assert stim2["passed"]
    assert stim2["measured"]["watched"] == "/PXI1Slot5/PFI1"
    assert stim2["measured"]["readbacks"] == ["PXI1Slot5/port0/line5"]
    assert stim2["measured"]["terminals"] == ["PXI1Slot5/port0/line5", "/PXI1Slot5/PFI1"]


def test_a_lone_pfi_still_maps_as_it_did_for_2_3_0(monkeypatch):
    evidence, terminals = _map(monkeypatch, {
        "board STIM2": ["/PXI1Slot5/PFI1"],
        "board STIM3": ["/PXI1Slot5/PFI0"],
    })

    assert terminals == {"STIM2": "/PXI1Slot5/PFI1", "STIM3": "/PXI1Slot5/PFI0"}
    assert _check(evidence, "STIM3")["measured"]["readbacks"] == []


def test_an_output_that_raises_nothing_fails_and_is_not_mapped(monkeypatch):
    evidence, terminals = _map(monkeypatch, {
        "board STIM2": [],
        "board STIM3": ["/PXI1Slot5/PFI0"],
    })

    assert terminals == {"STIM3": "/PXI1Slot5/PFI0"}
    assert not _check(evidence, "STIM2")["passed"]
    assert _check(evidence, "STIM2")["measured"]["watched"] is None


def test_an_output_that_raises_two_pfis_fails(monkeypatch):
    evidence, terminals = _map(monkeypatch, {
        "board STIM2": ["/PXI1Slot5/PFI1", "/PXI1Slot5/PFI2"],
        "board STIM3": ["/PXI1Slot5/PFI0"],
    })

    assert "STIM2" not in terminals
    check = _check(evidence, "STIM2")
    assert not check["passed"]
    assert "found 2" in check["detail"]


def test_an_output_that_raises_only_port_lines_fails(monkeypatch):
    """A port line has no edge counter source; the drop check needs a PFI."""
    evidence, terminals = _map(monkeypatch, {
        "board STIM2": ["/PXI1Slot5/PFI1"],
        "board STIM3": ["PXI1Slot5/port0/line6"],
    })

    assert "STIM3" not in terminals
    assert not _check(evidence, "STIM3")["passed"]
