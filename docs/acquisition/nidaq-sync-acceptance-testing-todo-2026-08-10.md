# Session Recording and NI-DAQ Synchronization Acceptance Test TODO

This checklist validates the manual recording lifecycle, synchronized camera and
NI-DAQ persistence, independent hardware failure domains, metadata integrity,
and operator controls currently on `devel`. Complete it on the actual rig before
the branch is treated as hardware-accepted.

The short session002-session005 audit was useful exploratory coverage, but it was
performed before the final Analysis selector and camera square-wave transition
fixes. Repeat the applicable tests below against the current commit.

## Plan-conformance audit

The implementation was re-audited requirement-by-requirement through the UI,
protocol, reliability, and live-intertrial-analysis implementation series on
2026-08-12.
This separates software implemented and covered by automated tests from
behavior that still requires physical-rig acceptance.

Completed automated checks:

- [x] Focused lifecycle, trial, intertrial-analysis, persistence, and UI suites
      pass after the implementation series. Record the final full-suite result
      in the Test record below before hardware acceptance.
- [x] No production imports reference the removed load-cell, SensorAnalysis,
      webcam/top-camera, head-fix, tunnel, magnet, alarm, or emergency runtime
      implementations, and no recording-scoped trial events are emitted.
- [x] Session recording, canonical boundaries, auxiliary stream persistence,
      source manifests, NI sample timelines, camera/NI correlations, decoded CAN
      persistence, subsystem states, stop arbitration, trial-ledger primitives,
      animal-v7 migration, platform artifact selection, UI controls, and
      generation-safe intertrial-analysis cancellation have automated coverage.
- [x] Live-inference processing remains on the retained implementation; its
      lifecycle fixture/cleanup was repaired without changing the inference
      calculation or frame-queue behavior.
- [x] `devel` runs unit-test CI but is not release-tagged; only merged pull
      requests into `develop` enter the serialized patch-tag workflow.
- [x] `temp/` and every file below it are ignored and untracked. The two planning
      files removed from Git tracking are retained as ignored working copies.

Resolved implementation gaps:

- [x] Production motor, command-dispatch, CAN-transport, and acknowledgement
      failures reach `PelletCycleController.finalize_hardware_failure`, map to
      the matching typed hardware error, close the public lifecycle, cancel the
      active protocol trial, and preserve the logical number for retry.
- [x] Behavioral retry numbering is finalized from one authoritative per-trial
      live-tracking result. Retry-dependent modes wait before another SEND;
      Continue mode cannot relabel an already-started attempt. Hardware retries
      are indexed immediately.
- [x] Decoded tone 2 and exact pellet-cycle completion bound immutable tracking
      windows. A bounded generation-owned worker analyzes existing tracking
      without reopening video; every pending attempt is finalized once or
      explicitly marked incomplete/unavailable.
- [x] **Scored trials** is available only with live intertrial analysis and can
      drive automatic Stop from finalized results while recording.
- [x] Trial records include pellet position, planned/applied shifts,
      protocol/phase context, per-trial reaches/success/consumption, event
      indices, and tone/laser references in addition to lifecycle/error fields.
- [x] Reaches, Presented, Success, and Consumed are projected together from the
      reconciled ledger summary. Record/Abort reset behavior is retained.
- [x] `auto-trainer-api` is pinned to 0.11.0; exact-once session and pellet-trial
      lifecycle events are published, and the ReachAQ status/animal schemas
      contain no alarm, emergency, tunnel, head-fix, or magnet placeholders.
- [x] Acquisition, recording-session, pellet-cycle, pellet automation,
      protocol, presence, shift recommendation, coordinate validation, pellet
      misplacement, and watchdog responsibilities now have explicit owners.
- [x] Camera discovery/ID merge/video validation, bounded NI persistence and
      coverage classification, atomic auxiliary generations, output preflight,
      CAN recovery generations, safe pellet initialization, background
      SoftMouse refresh, bounded event dispatch, cleanup containment, launcher
      locking, and owned signal shutdown have focused automated coverage.
- [x] Session allocation is reserved once before writer arming; camera-close,
      timers, analysis, and abort callbacks require the matching session ID and
      generation, and old callbacks cannot finalize a later session.
- [x] Record/Stop/Abort commands and Ready-only configuration mutations share one
      re-entrant command boundary. Tokenized Abort returns to Ready and invalidates
      its generation atomically; failed Arming reservations use the same verified
      writer-close/delete path.
- [x] Future protocol-row editing and SEND snapshotting share one lock, while the
      pellet-cycle controller serializes ledger, lifecycle-event, protocol, and
      persistence mutations without allowing a late result to relabel an attempt.
- [x] Retained auxiliary finalization also retains its project/metadata identity,
      retries before the next Record, republishes authoritative metadata, and
      removes only empty successful staging directories.
- [x] NI thread-stop timeout state remains explicitly retryable. Abort requests NI
      shutdown without blocking camera closure and never deletes the session until
      the recorder confirms that its writer is closed.
- [x] CAN connect/disconnect/recovery is serialized and every command has an
      explicit terminal/unknown outcome; loss cannot turn a disappeared pending
      command into an acknowledgement.
- [x] A timed-out event-plugin shutdown retains singleton ownership, and lifecycle
      event saturation uses the bounded retry outbox rather than silent loss.

The remaining unchecked items below are physical-rig acceptance, not known
software implementation gaps.

## Test record

- [ ] Record the tested Git commit: `________________`
- [ ] Record the system configuration path and archive a copy with the results.
- [ ] Record the rig name, operating system, NI-DAQmx version, camera serials,
      NI model names, device aliases, PXI slots, and relevant wiring routes.
- [ ] Record the raw-data root and session identifiers used by this checklist.
- [ ] Save the application log and a copy of every accepted session's metadata,
      source manifest, and `streams/alignment.json`.
- [ ] Confirm `git status --short` is clean before testing.

## Automated regression baseline

- [ ] Confirm the plan-conformance implementation gaps above have been resolved
      or have an explicit approved scope change before running release
      acceptance.
- [ ] Run repository-integrity checks:

  ```bash
  git diff --check
  git ls-files temp
  git status --short --ignored temp
  ```

  `git ls-files temp` must print nothing, and the ignored-status command must
  report only `!! temp/`.

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
    tests/acquisition_controller_test.py \
    tests/recording_session_controller_test.py \
    tests/coordinate_model_test.py \
    tests/session_api_publisher_test.py \
    tests/test_app_model.py
  ```

- [ ] Run the full software suite:

  ```bash
  conda run -n reachaq python -m pytest -q
  ```

- [ ] Investigate every failure. If a timing test is classified as a flake,
      preserve its first output and demonstrate at least three consecutive
      passing reruns of that exact test.
- [ ] Run live inference and intertrial-tracking regression tests and confirm no
      workers remain after pytest exits:

  ```bash
  conda run -n reachaq python -m pytest -q \
    auto-trainer-inference/tests/live_pose_result_test.py \
    tools/acquisition/tests/intertrial_analysis_test.py \
    tools/acquisition/tests/live_tracking_buffer_test.py \
    tests/inference_recording_ack_test.py
  ```
- [ ] Confirm the Spinnaker resolver selects the bundled Linux x86-64, Linux
      aarch64, and Windows x86-64 artifacts and reports an actionable error for
      an unsupported platform/ABI tuple:

  ```bash
  conda run -n reachaq python -m pytest -q tests/platform_support_test.py
  ```

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

### 2026-08-11 UI, protocol, and launcher regression

- [ ] Start with a fresh preferences file, maximize the application, and drag
      the title bar. Confirm the first drag restores a practical normal size
      instead of `320×240`.
- [ ] Resize and reposition the normal window, restart reachAQ, and repeat the
      maximize/drag operation. Confirm the chosen normal geometry is restored.
- [ ] Open **File → Hardware**, toggle at least three independent selections,
      and confirm the submenu stays open until clicking outside it. Confirm one
      hardware refresh begins after the menu closes and no callback exception
      is logged.
- [ ] Enter Running and confirm the System Mode selector remains enabled. Return
      to Idle from the selector and confirm shutdown completes normally.
- [ ] Exercise Idle, starting, Running/Ready, Recording, Stopping, Analyzing,
      Abort, hardware refresh, and NI discovery. Confirm every conflicting
      control is disabled and every currently valid control is enabled.
- [ ] Confirm Subject is editable before Record, locked from Arming through
      analysis, and editable again only after the session returns to Ready.
- [ ] Enter Notes before and during recording, press Stop, then edit Notes while
      analysis is running and after it completes. Confirm JSON and YAML metadata
      contain the final text. Start another recording and confirm the Notes field
      clears without modifying the preceding session.
- [ ] Confirm the pellet-board X/Y/Z controls and feedback are directly beneath
      **Pellet Release Location**, with the force-override command column closely
      adjacent and no large empty spacer columns.
- [ ] Confirm pellet controls are disabled while disconnected or a command is
      pending, then re-enabled after acknowledgement. Confirm Home, Load, Send,
      Retract, Release, and Cover retain force-override behavior when enabled.
- [ ] In the Trial Protocol tab, edit several future rows and confirm the values
      remain editable until their logical trial starts. Confirm the active row is
      highlighted and locked, completed rows remain locked, and a hardware-error
      retry returns the same logical row to Future rather than consuming the next
      row.
- [ ] Stop and inspect `streams/trials.jsonl`. Confirm each attempt contains its
      immutable `protocol_context.trial_row` snapshot and session metadata
      contains the ordered `protocolSchedule`.
- [ ] Double-click the mouse-icon `reachAQ.desktop`. Confirm a terminal opens,
      live logs appear immediately, the configured Conda environment/config are
      used, and reachAQ starts in Idle.
- [ ] Double-click the launcher again while reachAQ is open. Confirm it reports
      an existing instance and does not start a second GUI. Intentionally supply
      an invalid configuration once and confirm the failure terminal remains
      visible until Enter is pressed.

- [ ] Select System Mode Running without pressing Record. Confirm enabled
      cameras acquire and preview, but no session directory or video writer is
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
- [ ] During the session, generate multiple barcode transitions, pellet-board CAN
      messages, and tone events on each available tone channel.
- [ ] If laser is enabled, generate several commanded laser events and physical
      feedback transitions.
- [ ] Press Stop and wait for already-closed trial analysis and finalization to
      drain.
- [ ] Confirm Record remains disabled throughout stopping and analysis and is
      enabled again only after analysis completes.
- [ ] Confirm cameras, pose data, NI-DAQ, decoded device/CAN events, laser events,
      and session logs are written when enabled.
- [ ] Confirm every configured NI channel is recorded even when it was not
      plotted.
- [ ] Confirm `device.csv` contains decoded general CAN traffic, command context,
      acknowledgements where available, direction, board/device identity, and
      timestamps rather than only its header.
- [ ] Confirm standalone and compound-sequence tones create outbound
      `PLAY_TONE` rows in `device.csv`, immediate inbound `TONE_STATUS` rows are
      retained, and their independently wired NI feedback pulses are present in
      the NI stream.
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
- [ ] Start sessions on both square-wave polarities. Confirm camera/NI matching can
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
- [ ] Confirm every valid Tone 1/Tone 2 pulse groups its outbound `PLAY_TONE`,
      immediate `TONE_STATUS`, and periodic GPIO observation when present. The
      raw event timestamps must remain unchanged and `alignedEventPerfTime` must
      equal the NI pulse onset.
- [ ] Confirm pulses shorter than 2 ms appear under `artifacts`, never as valid
      tone matches or `unmatchedEdges`. Confirm pre-Record and post-Stop rolling
      activity appears in neither session artifacts nor unmatched edges.
- [ ] Repeat the eventful session at least three times and compare camera/NI and
      tone/NI offsets. Document the mean, range, and any outlier.

## Session boundary and metadata integrity

- [ ] Confirm JSON, YAML, source manifest, `streams/alignment.json`, and session
      log contain the same finite recording-start wall timestamp.
- [ ] Confirm metadata `boundary.startWallTime` is never `NaN` for a retained session.
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
- [ ] Confirm each completed pellet trial populates live tracking-derived results
      while recording without changing or delaying the recorded raw sources.
- [ ] Confirm analysis completes promptly for a short session and does not leave
      Record permanently disabled.
- [ ] Record at least three consecutive retained sessions without restarting the
      application. Confirm unique session directories, isolated stream data, and
      no stale callbacks or counters cross session boundaries.

## Pellet trials, protocols, and automatic stops

- [ ] Record one session with zero pellet trials, one with one trial, and one
      with several automatic pellet cycles. Confirm recording remains continuous
      across all cycles.
- [ ] Confirm each send dispatch and successful board acknowledgement appears in
      both decoded `device.csv` and the appropriate `trials.jsonl` attempt.
- [ ] Exercise **Retry within the same trial** and confirm labels progress as
      `1.1`, `1.2`, and so on without double-counting the logical trial.
- [ ] Exercise **Count every attempt as a new trial** and **Count only successful
      pellet presentations** and confirm `trial_summary.json` matches the UI
      choice.
- [ ] Induce a safe command/acknowledgement failure. Confirm it is an explicit
      hardware error with an operation ID and never increments any configured
      trial count or protocol progress.
- [ ] Induce motor, command-dispatch, CAN-transport, and acknowledgement-timeout
      failures separately. Confirm each production callback finalizes the
      active attempt with the matching hardware-error kind, preserves the
      decoded device event, and permits a later successful retry without
      consuming a logical trial number.
- [ ] Record multiple attempts including analyzed behavioral failures under each
      assignment/settings policy. Confirm live-finalized labels and recorded
      reuse/resample choices are exactly as configured.
- [ ] With a selected protocol, confirm each qualifying pellet trial advances
      progress once and a recording boundary does not increment progress.
- [ ] Confirm manual pellet control remains available with no protocol selected,
      and that automatic protocol advance does not implicitly enable automatic
      pellet cycles.
- [ ] Test duration, trial-count, and protocol-completion stop policies
      separately and with simultaneous thresholds. Confirm the first request
      wins and the actual/triggered reasons are persisted.
- [ ] Reach each automatic stop threshold during an active trial. Confirm no new
      trial starts, the active trial finishes normally, and only then do writers
      stop.
- [ ] Safely force a stuck active trial past the 15-second drain timeout. Confirm
      that this case alone is marked incomplete/error before best-effort recovery
      and Stop.
- [ ] Confirm `streams/trials.jsonl`, `streams/trial_summary.json`, final
      metadata, and `alignment.json` use the same session ID and canonical
      timestamp boundary.
- [ ] Confirm live intertrial analysis assigns a terminal outcome to every
      analyzable completed window exactly once, leaves no unexplained
      `pending_analysis` attempt after Stop, and updates `trials.jsonl`,
      `trial_summary.json`, protocol progress, and displayed counts consistently.
- [ ] Verify every persisted attempt contains the pellet position,
      planned/applied shifts, protocol/phase context, reach/success/consumption
      outcome, and references to associated tone/laser command and NI-feedback
      records.
- [ ] Select **Scored trials** with live intertrial analysis and a small trial
      limit. Confirm finalized results reach the limit while recording and Stop
      completes the current physical trial. Disable analysis and confirm the
      scored combination is rejected before Record.

## Live intertrial analysis and pellet-state acceptance

- [ ] In **Continue while analyzing**, complete several rapid pellet cycles.
      Confirm subsequent SEND operations are allowed, late results retain their
      original operation/trial IDs, and already-active rows never change.
- [ ] In **Wait for analysis**, confirm only SEND is blocked after cycle
      completion; camera, pose, NI, device, laser, and log writers continue.
- [ ] Configure retries for `no_reach`, failure, or consumption-dependent
      outcomes. Confirm waiting is forced regardless of the base mode and the
      resulting retry receives `N.2` before any following row can start.
- [ ] Cause a retry-dependent analysis exception or fill the bounded queue.
      Confirm **Retry analysis** resubmits the identical stored window and
      **Continue without result** persists incomplete/unavailable while skipping
      that retry decision. Confirm Stop and Abort remain enabled.
- [ ] Confirm the Trial Protocol panel reports pending count, measured analysis
      seconds per tracking second, and `estimating` until measurements exist.
- [ ] With NI Tone 2 enabled, verify each tracking window begins at the physical
      NI pulse onset and ends exactly at pellet-cycle completion. Its first
      behavioral sample should differ by no more than one frame interval. Confirm
      delayed periodic GPIO receipt does not move the boundary.
- [ ] Repeat with NI Tone 2 disabled. Confirm the immediate pellet-board tone
      status opens the window; use legacy periodic GPIO only when immediate tone
      status is unavailable.
- [ ] Present a pellet with no reach and confirm `no_reach`, never
      `pellet_missing`. Verify direct live presence, absence, and misplacement
      finalize at cycle completion without entering the analysis wait queue.
- [ ] Disable live inference and then disable intertrial analysis. Confirm manual
      and protocol pellet cycles still operate, presence/misplacement is
      `unknown`, completed attempts are unscored, and no fabricated retry occurs.
- [ ] Press Stop with one active physical trial and other closed windows pending.
      Confirm the active trial becomes incomplete, closed windows drain, stored
      tracking repairs a missing result if possible, and Record stays disabled
      only until finalization finishes.
- [ ] Press Abort while an analysis job is running. Confirm no stale result is
      published, the session directory is deleted, live inference remains usable,
      and the next session starts with a new generation.

## Reliability, storage, and metadata acceptance

- [ ] Record synchronized cameras with deliberately missing, duplicate, and
      out-of-order synthetic timing IDs. Confirm merge is by actual frame ID,
      missing rows are explicit, synchronization completeness fails, and the
      retained session is never deleted.
- [ ] Validate each stopped MP4 against writer and timestamp counts. Confirm a
      mismatch stores all three counts plus role/serial and first writer error as
      a warning; unreadable/zero-frame video marks data incomplete but is kept.
- [ ] Delay one camera writer-close acknowledgement beyond 30 seconds. Confirm the
      application reports the timeout without deleting data or commanding hardware,
      remains able to accept Abort, and safely completes if the acknowledgement
      arrives late.
- [ ] Delay video validation on both cameras. Confirm validations run concurrently,
      each external count is bounded to 30 seconds, and a timeout is persisted as a
      source failure without an unbounded OpenCV fallback.
- [ ] Confirm JSON, YAML, alignment, trials, summary, tracking records, and final
      manifest share one metadata generation ID. Interrupt an auxiliary publish
      in a test copy and confirm the previous generation remains detectable and
      staged data is available for bounded retry.
- [ ] Force the first auxiliary publish to fail, then retry Record. Confirm the old
      generation is finalized and its JSON/YAML metadata is republished before a new
      session is allocated. Successful empty `.staging` directories should disappear;
      failed staged diagnostics must remain.
- [ ] Run a long enabled NI session with plots hidden. Confirm memory remains
      bounded, all channels persist, sample indices/times remain monotonic, and
      first/last exceptions, gaps, overruns, epochs, and coverage are recorded.
- [ ] Delay NI recorder shutdown beyond its join deadline. Confirm the stopped
      session remains retryable and a later retry publishes the same captured data.
      During Abort, confirm the session directory is not removed until the NI writer
      has actually stopped.
- [ ] Inject a recovered NI copy error and boundary shortfall below five seconds;
      confirm warnings only. Test zero samples, persistence/worker failure, and
      boundary loss above five seconds; confirm NI is incomplete while partial
      camera/device data is preserved.
- [ ] Before Record, confirm the target filesystem probe writes and `fsync`s,
      and the report contains free bytes and projected maximum minutes. Exercise
      the 30- and 10-minute warnings and below-one-minute normal Stop without
      Abort. Confirm the two-hour configured-duration cap.
- [ ] Open Preferences, start recording through another path, and confirm all
      mutable configuration controls disable. Confirm model-level setters reject
      changes through Recording/Stopping/Analyzing and re-enable at Ready.
- [ ] Corrupt one animal JSON beside valid animals. Confirm valid subjects load,
      the corrupt file is unchanged and listed with its reason, and duplicate
      UUID remains a hard conflict.
- [ ] Trigger manual, scheduled, and publisher-completion SoftMouse refreshes.
      Confirm file/hash/SQLite work remains off the Qt thread, duplicate requests
      debounce, one recording-time request is deferred until Ready, and controls
      re-enable on both success and error.

## API and lifecycle contract

- [ ] Confirm external clients receive explicit acquisition, recording-session,
      pellet-trial, protocol, calibration, synchronization-readiness, and
      subsystem states without interpreting a recording session as a trial.
- [ ] Confirm session start/end and pellet-trial start/capture-end/outcome-end
      events are each emitted once with stable IDs and canonical timestamps;
      presentation acknowledgement must appear in the trial ledger and decoded
      device stream.
- [ ] Confirm the current public status/API schema contains no retired
      alarm/emergency/tunnel/head-fix/magnet fields or placeholder values after
      the API dependency migration.
- [ ] Confirm Abort emits/carries the session-aborted outcome without publishing
      a retained session or completed pellet-trial result.

## Architecture and authoritative-state review

- [ ] Verify acquisition, recording-session, pellet-cycle, pellet automation,
      protocol, shift recommendation, pellet presence, pellet misplacement,
      watchdog, and coordinate/calibration responsibilities have explicit
      owners and do not rely on removed global-mode or tunnel state gates.
- [ ] Trace Reaches, Presented, Success, and Consumed from recorded source event
      to UI, metadata, trial summary, and per-trial live result. Confirm each has
      one authoritative value or a tested deterministic reconciliation rule.
- [ ] Confirm every hardware command used for a pellet attempt or automatic
      shift records dispatch, acknowledgement/failure, operation ID, and the
      owning session/trial association.

## Configuration and animal migration

- [ ] Confirm the maintained version-57 system configuration loads and saves.
- [ ] Confirm an older version, unknown key, retired tunnel/load-cell field,
      `recordToAcquisition`, and old shift `targetX/Y/Z` fields are rejected with
      actionable errors.
- [ ] Load a copy of a current v4 animal and save it. Confirm the original bytes
      are retained as `.json.v4-backup`, the active file is v7, identity/pellet
      coordinates/limits/selected protocol are preserved, and protocol progress
      starts at zero.
- [ ] Confirm v5 and v6 inputs preserve their original bytes as matching backup
      files and migrate to v7 without resetting valid trial-based progress.
- [ ] Confirm v0-v3, no-ID, and unknown animal versions are rejected and v7
      round-trips without day/total or auto-clamp fields.

## Abort behavior

- [ ] Abort once during Arming and once during active Recording while camera,
      NI-DAQ, and CAN data are flowing.
- [ ] Confirm all session writers stop and release their files.
- [ ] Confirm analysis never starts for the aborted session, or is completely
      cancelled if already scheduled.
- [ ] Confirm the entire aborted session directory and all session data are
      removed while application-level diagnostic logs outside the session remain.
- [ ] Confirm all four behavior counts become zero after Abort.
- [ ] Confirm healthy previews and unrelated hardware remain active.
- [ ] Immediately make a new retained recording after abort cleanup and confirm
      it contains no buffered events or metadata from the aborted session.

## Camera topology and failure isolation

- [ ] Enable only the primary camera. Confirm it can preview independently in
      the effective standalone/free-run role and can record if configuration
      permits a single-camera session.
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
- [ ] Lose a secondary camera during Recording. Confirm the primary and other
      streams continue and the session is retained as synchronization-incomplete.
      Lose the primary/writer and confirm normal Stop preserves partial data;
      neither case invokes Abort or global hardware shutdown.

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
- [ ] With reachAQ acquiring, start a second reachAQ instance against the same
      CAN channel. Confirm it reports the channel as already owned and cannot
      issue pellet/motor commands or reset the interface.
- [ ] Close acquisition normally. Confirm only the application's CAN socket and
      device worker close; `reachaq-can.service` is not restarted and the
      interface packet counters are not reset.
- [ ] Trigger a non-CAN camera, NI-DAQ, analysis, or UI worker failure. Confirm
      the CAN socket remains connected and neither the reset helper nor systemd
      CAN service is invoked.
- [ ] Safely inject each available CAN reader failure: `CanOperationError`,
      `ENETDOWN`, bus-off, adapter removal, and a malformed JerryCAN frame.
      Confirm the first exception type, errno/category, interface state, and
      pre-recovery counters are retained in the log/device event and the UI
      marks only CAN failed.
- [ ] Confirm bounded recovery closes the failed socket, reopens the transport,
      rediscovers the pellet board, requests firmware, reloads motor and move
      configuration, restarts reader/status streaming, and returns CAN to Ready.
      Confirm failure after all three attempts remains explicit and does not
      disturb cameras, NI-DAQ, laser, or logs.
- [ ] On initial pellet-board connection and successful recovery, observe
      acknowledged motor configuration, X/Y/Z home, and detachment of both
      pellet servos. Confirm there is no connection-time attach/release motion;
      normal Load/Cover/Release still attaches on demand.
- [ ] End sessions by manual Stop, automatic Stop, and Abort. Confirm each
      requests home only after the closed stream boundary when the board is
      Ready, records the end action, and cannot block save/deletion if homing
      fails.
- [ ] Interrupt CAN while a pellet/motor command is in flight. Confirm the
      operation is stored as failed/unknown with its context and is never
      replayed after reconnect. Confirm a later operator/protocol retry receives
      a new operation context and the intended trial-attempt index.
- [ ] Request intentional application shutdown while the reader is closing.
      Confirm `ENETDOWN` is suppressed only for that same-process shutdown; an
      externally caused `ENETDOWN` must remain a visible failure.
- [ ] Invoke the privileged recovery helper twice inside 15 seconds and from two
      processes. Confirm resets serialize/debounce and systemd cannot enter a
      restart storm. Confirm a reset is refused while a reachAQ process owns the
      channel.
- [ ] Before a multi-hour endurance run, save
      `ip -details -statistics link show <channel>`, then save it again after the
      run. Compare RX packets/errors/dropped/overrun and controller state, retain
      the one-minute `CAN reader throughput` and effective receive-buffer logs,
      and correlate any increase in drops with CPU and frame rate. The previous
      observation (14,210 drops / roughly 1.07 million packets, about 1.3%) is a
      baseline to improve, not an acceptable pass threshold.
- [ ] Disable laser while NI input remains enabled and confirm NI acquisition
      remains operational.
- [ ] Induce a safe writer/source failure during Recording. Confirm normal Stop
      preserves every partial file and diagnostics and marks the source/session
      incomplete; it must not invoke automatic Abort or deletion.

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
- [ ] Test a digital-only master. Confirm a counter produces the sample clock,
      consumers arm first, and no nonexistent AI start trigger is published.
- [ ] For finite hardware-timed laser output, confirm the AO task actually
      applies the resolved reference/start/sample routes before reporting
      synchronized. Confirm arbitrary on-demand pulses remain independent.

## Endurance and recovery

- [ ] Record continuously for at least 10-30 minutes with all normal enabled
      sources and representative event traffic.
- [ ] Confirm no dropped camera frames, NI gaps, buffer overruns, writer lag,
      timestamp reversals, unbounded memory growth, or watchdog failures.
- [ ] Stop and drain closed-window analysis/finalization; confirm the long session
      finalizes as complete and all counts and metadata agree with the files.
- [ ] Return System Mode to stopped, restart acquisition, and make another short
      recording without restarting the application.
- [ ] Restart the application and open the retained sessions. Confirm Analysis
      loads them without mutation, exception, or missing-source errors.

## Acceptance

- [ ] Every enabled source records independently of plot visibility.
- [ ] All enabled synchronized cameras and the NI timeline meet the alignment
      requirements above across at least three retained sessions.
- [ ] Stop retains complete data; Abort leaves no session data and runs no
      analysis.
- [ ] Hardware failures remain confined to their owning/dependent domains and
      recovery does not restart healthy hardware.
- [ ] Metadata is complete, finite, internally consistent, and portable across
      the tested hardware configurations.
- [ ] All unexplained warnings/errors and every failed checklist item have a
      linked issue, captured logs, and an explicit disposition before release.
