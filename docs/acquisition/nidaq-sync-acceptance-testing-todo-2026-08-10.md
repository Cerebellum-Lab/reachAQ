# Session Recording and NI-DAQ Synchronization Acceptance Test TODO

This checklist validates the manual recording lifecycle, synchronized camera and
NI-DAQ persistence, independent hardware failure domains, metadata integrity,
and operator controls currently on `devel`. Complete it on the actual rig before
the branch is treated as hardware-accepted.

The short trial002-trial005 audit was useful exploratory coverage, but it was
performed before the final Analysis selector and camera square-wave transition
fixes. Repeat the applicable tests below against the current commit.

## Test record

- [ ] Record the tested Git commit: `________________`
- [ ] Record the system configuration path and archive a copy with the results.
- [ ] Record the rig name, operating system, NI-DAQmx version, camera serials,
      NI model names, device aliases, PXI slots, and relevant wiring routes.
- [ ] Record the raw-data root and trial identifiers used by this checklist.
- [ ] Save the application log and a copy of every accepted trial's metadata,
      source manifest, and `streams/alignment.json`.
- [ ] Confirm `git status --short` is clean before testing.

## Automated regression baseline

- [ ] Run the focused acquisition suite:

  ```bash
  conda run -n reachaq python -m pytest -q \
    tests/nidaq_channel_plan_test.py \
    tests/nidaq_discovery_test.py \
    tests/nidaq_sample_timeline_test.py \
    tests/nidaq_timing_test.py \
    tests/nidaq_port_configuration_dialog_test.py \
    tests/session_data_recorder_test.py \
    tests/signal_stream_ui_test.py \
    tests/subsystem_status_test.py \
    tests/test_app_model.py
  ```

- [ ] Run the full software suite:

  ```bash
  conda run -n reachaq python -m pytest -q
  ```

- [ ] Investigate every failure. If a timing test is classified as a flake,
      preserve its first output and demonstrate at least three consecutive
      passing reruns of that exact test.

## Configuration and enabled-source reporting

- [ ] Open the NI port dialog and confirm discovered model, serial, product
      category, device alias, and timing capabilities match the installed
      hardware.
- [ ] Confirm unsupported optional NI capability queries do not prevent device
      discovery or display a misleading fatal initialization error.
- [ ] Confirm the configured master, clock reference, start-trigger route, and
      optional sample-clock route resolve to the intended physical devices.
- [ ] Confirm invalid aliases, duplicate channels, unavailable terminals, and
      unsupported routes produce actionable configuration errors before Record.
- [ ] Confirm metadata reports the enabled/disabled state and runtime state of
      cameras, inference/pose, NI-DAQ, CAN/device, laser, and logs.
- [ ] Confirm a stream being hidden from the plot does not disable persistence.
- [ ] Confirm a source explicitly disabled in configuration is reported as
      disabled and is not silently treated as an enabled empty source.

## System Mode and operator controls

- [ ] Select System Mode Running without pressing Record. Confirm enabled
      cameras acquire and preview, but no trial directory or video writer is
      created.
- [ ] Confirm the Hardware Refresh control is at the far right of the Hardware
      Status title bar and is disabled during recording, stopping, abort cleanup,
      and analysis where applicable.
- [ ] Confirm the Behavior panel has no States/System controls or day/total
      pellet counters.
- [ ] Confirm only the session counts for Reaches, Presented, Success, and
      Consumed are shown in the compact layout.
- [ ] Confirm Record, Stop, and Abort enablement and tooltips accurately reflect
      the current recording state and exact readiness blockers.
- [ ] Open the Analysis panel before and after recording. Confirm NI device
      identity metadata does not cause a tuple/string sorting exception, global
      CAN safety shutdown, or application termination.

## Normal eventful recording

- [ ] Start all intended hardware and confirm every enabled required source is
      Ready before Record becomes enabled.
- [ ] Plot only one NI signal while leaving every configured NI channel enabled.
- [ ] Record for 30-60 seconds with all enabled cameras.
- [ ] During the trial, generate multiple barcode transitions, pellet-board CAN
      messages, and tone events on each available tone channel.
- [ ] If laser is enabled, generate several commanded laser events and physical
      feedback transitions.
- [ ] Press Stop and wait for offline analysis to finish.
- [ ] Confirm Record remains disabled throughout stopping and analysis and is
      enabled again only after analysis completes.
- [ ] Confirm cameras, pose data, NI-DAQ, decoded device/CAN events, laser events,
      and session logs are written when enabled.
- [ ] Confirm every configured NI channel is recorded even when it was not
      plotted.
- [ ] Confirm `device.csv` contains decoded general CAN traffic, command context,
      acknowledgements where available, direction, board/device identity, and
      timestamps rather than only its header.
- [ ] Confirm tone commands reported by the pellet board are present in the
      decoded device stream and their independently wired NI feedback edges are
      present in the NI stream.
- [ ] Confirm laser commands and NI digital/analog feedback are represented as
      separate sources when configured.
- [ ] Confirm every enabled source appears in the source manifest with its actual
      path, record count, first/last offset, and gap/overrun/failure status.
- [ ] Confirm the session is marked complete only after every required enabled
      writer has finalized successfully.

## Camera and NI-DAQ alignment

- [ ] Confirm all enabled synchronized cameras commit identical start and stop
      frame boundaries and have equal merged timing and MP4 frame counts.
- [ ] Confirm camera frame IDs are consecutive with no unexplained gaps.
- [ ] Confirm each high and each low camera square-wave plateau represents one
      frame: 75 highs plus 75 lows account for 150 frames.
- [ ] Confirm successive camera-line transitions are approximately 6.666666 ms
      apart; at 10 kHz sampling, expected quantization is 6.6 or 6.7 ms.
- [ ] Confirm an N-frame sampled interval normally contains N-1 observed
      transitions when acquisition begins within the first plateau.
- [ ] Start trials on both square-wave polarities. Confirm camera/NI matching can
      select either a rising or falling transition and records `edgePolarity`.
- [ ] Confirm the selected camera transition is the nearest valid transition,
      not merely the nearest rising edge.
- [ ] Confirm NI sample indices are consecutive and sample-count-derived time is
      continuous for the entire retained interval.
- [ ] Confirm finalized NI `perf_time` values are strictly monotonic and do not
      reproduce independently timestamped block-boundary reversals.
- [ ] Confirm no acquisition gaps, ring-buffer overruns, ambiguous correlations,
      or boundary mismatches are reported.
- [ ] Confirm camera/NI and tone-command/NI-feedback correlations are populated
      and unambiguous when the corresponding physical lines are wired.
- [ ] Repeat the eventful trial at least three times and compare camera/NI and
      tone/NI offsets. Document the mean, range, and any outlier.

## Session boundary and metadata integrity

- [ ] Confirm JSON, YAML, source manifest, `streams/alignment.json`, and session
      log contain the same finite recording-start wall timestamp.
- [ ] Confirm `start_record_timestamp` is never `NaN` for a retained session.
- [ ] Confirm source offsets are expressed relative to the same canonical
      recording boundary.
- [ ] Confirm final duration, camera counts, NI sample counts, and device/laser
      event counts agree with their corresponding files.
- [ ] Confirm metadata records effective camera roles, serials, enabled states,
      NI identities, resolved timing topology, and runtime source status.
- [ ] Confirm animal JSON and existing animal APIs remain functional and their
      schema/data are not unintentionally changed.

## Counts, Stop, analysis, and consecutive sessions

- [ ] Set nonzero session counts, press Record, and confirm Reaches, Presented,
      Success, and Consumed reset together at the recording boundary.
- [ ] Generate countable behavior during recording and confirm each count updates
      once per corresponding event.
- [ ] Press Stop and confirm all writers use the final synchronized boundary,
      data is retained, and counts remain visible during and after analysis.
- [ ] Confirm offline analysis populates its expected results after Stop without
      changing the recorded raw sources.
- [ ] Confirm analysis completes promptly for a short session and does not leave
      Record permanently disabled.
- [ ] Record at least three consecutive retained sessions without restarting the
      application. Confirm unique trial directories, isolated stream data, and
      no stale callbacks or counters cross session boundaries.

## Abort behavior

- [ ] Abort once during Arming and once during active Recording while camera,
      NI-DAQ, and CAN data are flowing.
- [ ] Confirm all session writers stop and release their files.
- [ ] Confirm analysis never starts for the aborted session, or is completely
      cancelled if already scheduled.
- [ ] Confirm the entire aborted trial directory and all session data are
      removed while application-level diagnostic logs outside the trial remain.
- [ ] Confirm all four behavior counts become zero after Abort.
- [ ] Confirm healthy previews and unrelated hardware remain active.
- [ ] Immediately make a new retained recording after abort cleanup and confirm
      it contains no buffered events or metadata from the aborted trial.

## Camera topology and failure isolation

- [ ] Enable only the primary camera. Confirm it can preview independently in
      the effective standalone/free-run role and can record if configuration
      permits a single-camera trial.
- [ ] Enable only each secondary camera in turn. Confirm it can preview
      independently when other reach cameras are disabled.
- [ ] Enable all synchronized reach cameras. Confirm exactly one effective
      primary and the intended primary-to-secondary trigger topology.
- [ ] Interrupt a secondary camera. Confirm the healthy primary continues
      previewing, reach synchronization becomes unavailable, inference stops or
      blocks appropriately, and Record is disabled.
- [ ] Interrupt the primary camera. Confirm trigger-dependent secondaries suspend
      gracefully while NI-DAQ, CAN/device, laser, and other unrelated hardware
      remain operational.
- [ ] Restore each failed camera using Hardware Refresh. Confirm only the failed
      or blocked domains retry and healthy cameras are not restarted.
- [ ] Confirm loss of a required camera during Recording aborts only the active
      trial and does not initiate a global hardware shutdown.

## PXI, NI-DAQ, CAN, and laser failure isolation

- [ ] Start with the PXI chassis off. Confirm NI-DAQ and dependent laser domains
      report their own failures and block Record without falsely marking CAN or
      unrelated hardware failed.
- [ ] Confirm cameras continue previewing if their actual physical power and
      trigger paths are independent of the chassis.
- [ ] If cameras fail only when the chassis is off, identify and document the
      real trigger, power, clock, or grounding dependency; do not attribute it
      solely to NI software initialization.
- [ ] Turn the chassis on and use Hardware Refresh. Confirm NI domains recover
      without restarting already healthy domains.
- [ ] Disconnect or stop CAN safely. Confirm its failure is reported separately,
      required-source readiness is updated, and camera/NI preview is unaffected.
- [ ] Disable laser while NI input remains enabled and confirm NI acquisition
      remains operational.
- [ ] Induce a safe writer/source failure during Recording. Confirm the trial is
      rejected or aborted atomically rather than retained as complete.

## Camera error diagnostics

- [ ] Safely test a missing hardware trigger edge.
- [ ] Safely test a disconnected/unavailable camera or transport failure.
- [ ] Where reproducible, test an incomplete PySpin image result.
- [ ] Confirm diagnostics preserve the first PySpin exception, incomplete-image
      status/code, camera serial, configured/effective role, and effective
      trigger node values.
- [ ] Confirm the operator message distinguishes available evidence for missing
      trigger, transport, incomplete image, and camera-state failures.
- [ ] Confirm a later capture timeout or watchdog timeout does not replace or
      obscure the first causal error.
- [ ] Confirm a camera error does not cause an Analysis UI exception, global CAN
      safety shutdown, or misleading success/ready status for a failed domain.

## Multi-device NI timing and portability

- [ ] Validate a supported single-device NI input configuration without assuming
      PXI model names, chassis slots, or aliases from this rig.
- [ ] Validate the current multi-device topology using the configured common
      reference and start trigger. Confirm slave tasks arm before the master.
- [ ] If hardware-timed laser output is enabled, verify the PXI-6713/PXI-6221 (or
      rig-equivalent) topology using PXI_CLK10, a shared backplane start trigger,
      and the configured sample-clock strategy.
- [ ] Confirm digital feedback may reside on one NI device and analog feedback on
      another when the configured timing topology supports it.
- [ ] Verify that independent-clock mode is clearly identified as diagnostic and
      not silently described as hardware synchronized.
- [ ] Repeat configuration discovery and a short recording on at least one rig
      with different NI model(s), aliases, slots, or capabilities.
- [ ] Confirm optional hardware-timed laser settings remain disabled and do not
      affect ordinary on-demand laser operation when unsupported or unused.

## Endurance and recovery

- [ ] Record continuously for at least 10-30 minutes with all normal enabled
      sources and representative event traffic.
- [ ] Confirm no dropped camera frames, NI gaps, buffer overruns, writer lag,
      timestamp reversals, unbounded memory growth, or watchdog failures.
- [ ] Stop and complete analysis; confirm the long session finalizes as complete
      and all counts and metadata agree with the files.
- [ ] Return System Mode to stopped, restart acquisition, and make another short
      recording without restarting the application.
- [ ] Restart the application and open the retained sessions. Confirm Analysis
      loads them without mutation, exception, or missing-source errors.

## Acceptance

- [ ] Every enabled source records independently of plot visibility.
- [ ] All enabled synchronized cameras and the NI timeline meet the alignment
      requirements above across at least three retained trials.
- [ ] Stop retains complete data; Abort leaves no session data and runs no
      analysis.
- [ ] Hardware failures remain confined to their owning/dependent domains and
      recovery does not restart healthy hardware.
- [ ] Metadata is complete, finite, internally consistent, and portable across
      the tested hardware configurations.
- [ ] All unexplained warnings/errors and every failed checklist item have a
      linked issue, captured logs, and an explicit disposition before release.
