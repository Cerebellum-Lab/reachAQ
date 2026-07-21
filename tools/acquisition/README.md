# Acquisition UI

The acquisition application is the reachAQ operator UI for camera acquisition,
hardware status, pellet delivery, NI-DAQ port mapping, laser controls, behavior,
and inference.

## Launch

Use the reachAQ entry point from the configured conda environment:

```bash
conda run -n reachaq python -m reachAQ.app \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

The GUI starts idle by default, so camera and DAQ configuration remain editable
until Start is selected. Use `--start-mode acquiring` only when immediate
startup is intentional. The saved live-inference setting can be overridden for
one run with `--live-inference` or `--no-live-inference`.

For software-only camera testing:

```bash
conda run -n reachaq python -m reachAQ.app \
  --random-cameras \
  --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

The random-camera override is in-memory for that run and does not overwrite the
configured physical camera serials when the app closes.

## Configurations

Configuration files load preset values for cameras, devices, NI-DAQ channels,
laser controls, behavior, inference, and output. On Linux reachAQ rigs, the
default local config is usually:

```text
~/Autotrainer/system_configuration.yaml
```

The app saves configuration back to the preferences configuration directory. For
alternate software-only configs, use a separate preferences file and config
directory so test settings do not overwrite the bench config.

## Cameras

Reach cameras are configured as `CameraConfiguration` entries. The left and
right cameras are the normal two-camera reachAQ setup. Cameras 3-6 appear in
the UI only when explicitly present in the configuration.

reachAQ does not require a webcam. Leave the `web` camera absent or disabled
unless a rig intentionally configures it.

For Spinnaker cameras, put the camera serial or configured Spinnaker identifier
in `host`:

```yaml
- !CameraConfiguration
  id: 0
  name: left
  isEnabled: true
  isRecordEnabled: true
  recordMode: 1
  recordPrebufferDuration: 1.0
  scheme: spinnaker
  host: '24152533'
  port: 0
  path: ''
  params:
    fps: 150
    width: 256
    height: 256
    hbin: 4
    vbin: 4
    exposure: 175
    primary: 'yes'
```

The matching right camera should use its own serial and `primary: 'no'`.

## NI-DAQ Ports

The DAQ port editor discovers devices through NI-DAQmx and lists only channels
reported by the selected device. It also prevents duplicate channel assignments
across roles. If a selected device has no analog input channels, analog-input
roles are disabled instead of allowing an invalid assignment.

On the current PXIe-1073 / PXI-6713 setup, NI-DAQmx reports:

```text
Device: PXI1Slot4
AO: PXI1Slot4/ao0 through PXI1Slot4/ao7
AI: none
DIO: PXI1Slot4/port0/line0 through PXI1Slot4/port0/line7
```

The 6713 can provide analog outputs and digital I/O, but it cannot provide
analog input readback. Add a supported NI analog-input card if laser diode or
command-copy feedback channels are required.

## Output

Set acquisition output in the persistence section:

```yaml
persistence: !PersistenceConfiguration
  outputLocation: /home/<USER>/Documents/rawdatalocal
```

Create the directory before running acquisition:

```bash
mkdir -p "$HOME/Documents/rawdatalocal"
```

## Reference

### Toolbar

* Run/stop - start and stop acquisition and device interaction.
* Edit Configuration - change editable module settings.
* Preferences - set system-level application preferences.
* Hardware Refresh - scan camera sources, NI-DAQ devices, CAN adapter, and pellet delivery board while idle.
* Edit DAQ Ports - configure named NI-DAQ roles from discovered device channels.

### Menus

* File -> Open Configuration - open an existing configuration file.
* File -> Save Configuration / Save As - save the current configuration.
* View -> Diagnostics - show or hide the diagnostics panel.

### Camera Control

* Video Capture - starts capture only when the camera is enabled.
* Record Mode - `Continuous` records the full duration, `Trigger` records around trigger events.
* Video Recording - writes frames to video files.
* Image Capture - captures still images at a configured interval.

See [../../linux-install-instructions.md](../../linux-install-instructions.md)
and [../hardware/reachaq_system_configuration.example.yaml](../hardware/reachaq_system_configuration.example.yaml)
for the current Linux hardware setup and example config.
