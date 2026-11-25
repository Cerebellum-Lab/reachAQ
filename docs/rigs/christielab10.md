# Reference rig: christielab10

Comments and docs in shared code that say "the reference rig" mean
`christielab10`. It is the Christie lab acquisition computer, where the timings,
DAQmx error codes and wiring quirks they cite were measured. Its hardware is
listed under "Current workstation reference" in
[the runtime guide](../linux-install/runtime-configuration.md). Its NI wiring and
laser blocks are kept as test fixtures (`tests/christielab10_nidaq_blocks.yaml`,
`tests/christielab10_laser_block.yaml`).

Until 2026-11-18, shared code had some of this rig's values built in. They are
now configuration, and a fresh install has none of them. The rig keeps its
behaviour by setting them itself, as below.
`tests/christielab10_rig_profile_test.py` checks that these settings resolve to
the values that used to be built in.

## Settings directory

There is nothing to do. `REACHAQ_CONFIG_DIR` is unset on the rig, so it keeps
`~/reachAQ`.

## SoftMouse publication

The publication on the lab's Isilon share used to be built in. It is now the
`[softmouse]` preferences, in `~/.config/reachAQ/reachAQ.conf`:

```ini
[softmouse]
manifest_path=/mnt/isilon/Data/ReachingData/SoftMouse/SoftMouse-AnimalList-current.manifest.json
publication_mount=/mnt/isilon
```

Add the section to the file with reachAQ closed; the same values are in
`tests/christielab10_preferences.ini`. With pluggable animal metadata sources
(see [Animal metadata sources](../acquisition/animal-metadata-sources.md)) the
app reads this publication as a "Shared publication" source with no further
setup. Without the section the app reports "Not configured".

The nightly publisher no longer knows the colony. On the computer that runs the
publisher timer, set up the SoftMouse source once as the operator, choosing
the lab's colony, with **Publish to**
`/mnt/isilon/Data/ReachingData/SoftMouse` and **Only if mounted**
`/mnt/isilon`:

```bash
cd ~/Documents/reachAQ
conda activate reachaq
reachaq-animal-metadata setup --kind softmouse
```

The password stored by the old publisher is still found, so it can be left
empty. Until this is done the timer's run stops with "This computer reads a
shared publication, so it has nothing to publish".

The nightly publisher unit no longer waits for `/mnt/isilon` by itself. Restore
that with a drop-in:

```bash
mkdir -p ~/.config/systemd/user/reachaq-softmouse-publisher.service.d
printf '[Unit]\nRequiresMountsFor=/mnt/isilon\n' \
  > ~/.config/systemd/user/reachaq-softmouse-publisher.service.d/isilon.conf
cp src/reachaq/softmouse_sync/systemd/reachaq-softmouse-publisher.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user cat reachaq-softmouse-publisher.service | grep RequiresMountsFor
```

## Real-time priority

`sudo bash src/reachaq/hardware/reachaq-latency-setup.sh rtprio-enable` used to fall
back to the user `christielab10` when `SUDO_USER` was unset. It now refuses to
run without `SUDO_USER`, so run it with `sudo` from the operator account.

## NI-DAQ

The rig boots with `intel_iommu=off`. The
[NI-DAQ guide](../linux-install/ni-daq-pxi.md#iommu-and-dma-boot-with-intel_iommuoff)
explains why and how to check it.

## Pellet z: offsets measured in calibration captures (disputed, under review)

In two z moves, the board's position measured in the 3D calibration cameras
differed from the commanded position. The offsets were measured model-free, from
the board's position in the cameras, and a second analysis of the raw captures
gave the same offsets. They have not been confirmed by watching the stage. The
matter stays open until the z tuning tests or a direct measurement of the stage
settle it.

- **session004.** A 30 to 1 mm descent measured about 7 mm short of the
  commanded travel. Mid-descent, at about 85 mm/s, the board measured about 1 mm
  back up before it continued. The offset persisted until the top stop.
- **session006.** A 5 to 2 mm move measured -0.08 mm against -3 mm commanded,
  and the motor buzzed during the command. The +3 mm offset persisted.

The same moves measured correctly most of the time.

The host cannot see such an offset. The board reports its commanded position,
so `waypoints_unreached` stays 0. christielab10 has a v1.2 board, which has no
DIAG line.

The calibration report's stage-accuracy block (`stage_slack_rms_mm`,
`stage_repeat_mm`) shows the measured offsets.

The z speed and acceleration are being retuned. The chosen values will be
recorded here.
