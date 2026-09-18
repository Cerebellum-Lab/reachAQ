"""Applying a shared reference clock to a task that may not accept one.

The timing plan carries one terminal name for the whole chassis, but neither
the boards nor the task types are alike, and getting this wrong does not
degrade timing - it fails the task outright and takes whatever it belonged to
with it. Two ways it goes wrong, both seen on christielab10:

  * the plan's default is written PXI_CLK10 while NI calls the terminal
    PXI_Clk10, and a PXI-6713 exposes no Clk10 at all, so naming it fails
    with "property is not supported by the device";
  * a board that does expose the terminal can still refuse it per task - the
    PXI-6221's digital input task rejects DAQmx_RefClk_Src with -200452,
    "not applicable to the task".

This lives here because both the signal stream and the laser controller build
tasks against the same plan and hit the same two failures. The stream was
fixed first; the laser kept its own unguarded copy, which is what made the
hardware trigger route fail at the arming call with -200452 on
laser_sync_pulse_ao long after the wiring was right.
"""

import logging

logger = logging.getLogger(__name__)


def device_name_of(task, fallback=None):
    """The device a task belongs to, for asking what terminals it has."""
    try:
        devices = list(task.devices)
    except Exception:
        return fallback
    if not devices:
        return fallback
    name = getattr(devices[0], "name", None)
    return name or fallback


def resolve_reference_clock(nidaqmx, device_name, source):
    """The terminal this device actually exposes for `source`, or None.

    An explicitly qualified terminal is returned untouched: naming one is a
    deliberate choice and this must not second-guess it. A bare name is
    matched against what the device reports, which also fixes its spelling.
    """
    if not source:
        return None
    if source.startswith("/"):
        return source
    if not device_name:
        return None
    try:
        terminals = nidaqmx.system.Device(device_name).terminals
    except Exception:
        # Cannot ask, so do not guess on this device's behalf.
        logger.warning("could not read terminals of %s; leaving its "
                       "reference clock unset", device_name)
        return None
    wanted = source.rsplit("/", 1)[-1].lower()
    for terminal in terminals:
        if terminal.rsplit("/", 1)[-1].lower() == wanted:
            return terminal
    logger.info("%s exposes no %s terminal; running it without a reference "
                "clock", device_name, source)
    return None


def apply_reference_clock(nidaqmx, task, plan, device_name=None):
    """Set the plan's reference clock on a task that will take one.

    Having the terminal is not the same as the task being able to use it, and
    NI is the only authority on applicability, so ask rather than maintain a
    table of which task types accept what. A task that cannot take one runs on
    its own timebase, exactly as it did before a shared plan existed.
    """
    if plan is None:
        return
    timing = getattr(task, "timing", None)
    if timing is None:
        return
    source = resolve_reference_clock(
        nidaqmx,
        device_name or device_name_of(task),
        getattr(plan, "reference_clock_source", None),
    )
    if source is None:
        return
    try:
        if hasattr(timing, "ref_clk_src"):
            timing.ref_clk_src = source
        rate = getattr(plan, "reference_clock_rate_hz", None)
        if rate is not None and hasattr(timing, "ref_clk_rate"):
            timing.ref_clk_rate = rate
    except Exception as error:
        logger.info("%s does not accept a reference clock on this task (%s); "
                    "it will run on its own timebase",
                    device_name or device_name_of(task) or "device", error)
