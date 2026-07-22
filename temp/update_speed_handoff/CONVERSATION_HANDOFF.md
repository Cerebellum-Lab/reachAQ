# Update-speed investigation handoff

Date: 2026-07-22
Repository: `/home/christielab10/Documents/reachAQ`
Branch: `updatespeed`
Current HEAD: `0bee87f Move laser plotting to shared process`

## Goal and non-negotiable requirements

The Analysis digital-signal viewer is smooth while idle but historically became slow, bursty, and seconds behind once camera acquisition started. The intended behavior is similar to Open Ephys/Suite2p: timestamp-correct rolling data, no stale backlog, and visual refresh at the active monitor's refresh rate.

Important requirements established during the conversation:

- The main Analysis stream is visualization only. It must not be written to CSV or another acquisition output.
- Full-rate inference results must remain available to behavior/control/saving. UI pose coalescing may discard intermediate display poses only; it must not discard inference data used elsewhere.
- DAQ acquisition, buffering, rolling-window preparation, and downsampling should be independent of the GUI and camera/inference rates.
- The final Qt widget mutation must occur in a GUI process, but the GUI should consume only a bounded newest snapshot and never process a backlog.
- Digital pulses must appear discrete/square, with correct timestamp intervals and durations—not diagonal analog-looking triangles.
- The Analysis graph defaults to X `[-10, 0]` seconds and Y `[-0.2, 1.2]`, permits horizontal-only zoom, prevents zooming beyond the ten-second window, and includes a Live button that moves the right edge to zero while preserving horizontal zoom.
- The user prefers testing each change before committing. Do not commit the current worktree until the user confirms behavior.

## Committed work on `updatespeed`

The branch was created and the following relevant commits exist, oldest to newest:

- `ff1ccff Speed up live analysis visualization`
- `319f25a Document multi-device NI-DAQ synchronization plan`
- `913b9c8 Document DAQ stream mapping reconciliation`
- `b549ab7 Note PXI-6713 digital timing limitations`
- `a353ecb Document PXI-6221 buffered DI transfer issue`
- `a0d183a Fix buffered NI digital input on Linux`
- `93721ab Constrain analysis graph to digital signals`
- `9d6653c Prevent stale NI visualization backlog`
- `aca5f6f Render NI digital stream with timestamped steps`
- `86df00d Add live-edge graph navigation`
- `2a22246 Stream NI samples through shared memory`
- `0bee87f Move laser plotting to shared process`

The current committed architecture already includes a raw shared-memory NI sample ring, a dedicated Analysis plot-preparation process, shared-buffer consumption by Qt, timestamped digital steps, and similar shared-process plotting for laser-control traces.

## NI-DAQ and hardware investigation

### Multi-device restriction

The original NI stream controller rejected digital channels on different devices with:

`RuntimeError: all digital stream channels must belong to the same NI-DAQ device`

A multi-device implementation plan was documented at:

`temp/nidaq_multi_device_sync/IMPLEMENTATION_PLAN.md`

The installed hardware was probed:

- `PXI1Slot5`, PXI-6221: supports hardware-timed buffered Port 0 DI, up to 1 MHz.
- `PXI1Slot4`, PXI-6713: does not support buffered DI.
- Slot 4 continuous DI failed with DAQmx `-200077`.
- A combined task across Slot 4 and Slot 5 failed with `-201426`; this combination cannot use DAQmx channel expansion.

Conclusion: precise 10 kHz correlation requires all relevant signals on hardware-timed DI devices with a valid shared timing route. The user moved barcode and camera-frame signals to Slot 5. The limitation was documented rather than bypassed unsafely.

### Digital input wiring/read behavior

The user connected signals to the NI breakout Port 0 and DGND. Direct hardware checks eventually showed the constant high line correctly. Camera pulses appeared only after acquisition started, confirming the physical camera signal and digital timestamps were valid.

### DAQ delivery cadence

Direct hardware measurements found a real delivery-cadence issue:

- Default interrupt transfer condition (`ON_BOARD_MEMORY_MORE_THAN_HALF_FULL`) produced a mean near 16.7 ms but delivered bursts: median about 0.13 ms, p95 about 101.7 ms, maximum about 102.8 ms, with roughly six chunks arriving together every 100 ms.
- DMA delivered an excellent ~16.8 ms cadence but returned zero-filled digital channels on this Linux/PXI-6221 combination, so DMA was rejected.
- Interrupt transfer with `di_data_xfer_req_cond = ON_BOARD_MEMORY_NOT_EMPTY` produced correct values and a stable cadence: mean/median about 16.7 ms, p95 about 16.8 ms, maximum about 17.1 ms, with no >35 ms stalls.

The low-latency transfer condition and its fake-DAQ test are currently **uncommitted** in:

- `auto-trainer-device/src/autotrainer/device/nidaq_signal_stream.py`
- `auto-trainer-device/tests/nidaq_signal_stream_test.py`

This DAQ fix was necessary, but it did not eliminate the acquisition-only GUI slowdown.

## Plot architecture work

### Shared-memory raw ring

The old incoming sample queues/event hop were replaced by a direct DAQ-to-shared-memory raw ring. The design uses newest-snapshot semantics and sequence/gap/overrun telemetry, so Qt never drains accumulated messages. If frames are missed, Qt immediately consumes the newest completed frame.

This architecture was also applied to laser-control plotting. Analysis data remains visualization-only and is not persisted.

### Timestamp-correct digital representation

Samples are stored against absolute sample-index-derived timestamps. The display converts transitions to true vertical step edges. This fixed:

- pulses gradually compressing into the window as if history were stretched;
- incorrect triangle-wave interpolation;
- stale queue/backlog behavior;
- zoom views omitting signal detail.

### Pixel-width level of detail

The plot worker was changed to prepare only the visible X range. Exact timestamped square steps are used when transitions fit within the pixel budget. Dense views use one digital occupancy representation per horizontal pixel.

The first dense representation used NaN-separated rectangles (five vertices per occupied pixel). Conversion was optimized from `np.minimum.at`/`maximum.at` to monotonic-bin `np.minimum.reduceat`/`maximum.reduceat`, reducing a synthetic full-window conversion from roughly 14 ms to 1.38 ms; a 0.1-second zoom was about 0.12 ms.

The rectangle representation was later replaced with independent vertical pairs:

- two vertices per occupied pixel;
- `connect="pairs"`;
- no NaN-separated subpaths;
- exact square steps remain when zoomed sufficiently.

This reduced an offscreen dense update-and-paint benchmark from approximately 5.84 ms to 1.71 ms, but did **not** eliminate the real acquisition slowdown.

### PyQtGraph simplification

Curves were changed from `PlotDataItem` to direct `PlotCurveItem` objects and added to the ViewBox with `ignoreBounds=True`, because the digital graph uses explicit ranges and never needs curve-driven autoranging. In an offscreen 1,752-point benchmark, ignoring fixed bounds reduced update-and-paint time from roughly 1.59 ms to 0.72 ms.

This improved the measured `setData()` cost in real use but still did not eliminate the late acquisition-wide stalls.

### OpenGL graph decision

A custom persistent-buffer `QOpenGLWidget` graph was considered. It was not implemented initially because measured digital conversion was only ~0.23 ms and later worker preparation generally remained below 4 ms. Evidence increasingly showed that the final severe frozen-state stall was outside graph preparation and graph painting. A purpose-built graph remains an option, but it cannot fix a GUI event loop that is starved even when graph painting is completely disabled.

## Camera preview work

The original camera preview renderer used this path for every preview frame:

`QImage -> RGBA conversion -> QPixmap -> QGraphicsScene -> QGraphicsView -> multisampled QOpenGLWidget`

It was replaced, currently uncommitted, by a direct raster `QWidget` that caches the latest `QImage` and paints the image, pose points, text, presence marker, and reach overlay in one pass. Capture, recording, inference, and camera sampling were not changed.

An offscreen two-camera refresh-and-paint benchmark measured about 0.70 ms average and 1.34 ms maximum.

This change clearly improved the early acquisition phase: the graph and camera UI remained near 55–59 Hz while only a few hundred graph points were present. However, it did **not** eliminate the later slowdown once fully active camera pulses had accumulated.

## Diagnostic telemetry added

The Analysis footer now reports once per second:

- `DAQ`: shared raw-ring publication rate.
- `Worker`: plot-worker publication rate.
- `Frames`: newest plot frames consumed by Qt.
- `Timer`: Qt plot timer callback rate, number of late callbacks, and worst callback interval.
- `Paint`: actual Analysis viewport paint-event rate.
- `Prep`: average/maximum plot-worker conversion time.
- `Age`: average/maximum age of a completed worker frame when Qt consumes it.
- `Copy`: average/maximum shared-memory copy time.
- `Set`: average/maximum Qt curve `setData()` time.
- `Points`: maximum submitted vertices.

Timing pairs are one-second average/maximum.

A `Freeze Plot` button was added as a causal control. When frozen:

- camera acquisition and preview continue;
- DAQ sampling and shared-ring writes continue;
- the plot worker continues preparing snapshots;
- Qt continues copying newest plot metadata/data;
- only curve mutation and Analysis graph painting stop.

The button changes to `Resume Plot` while active.

## Video evidence and conclusions

### `Screencast from 07-22-2026 03:37:02 PM.webm`

Before camera-render changes, DAQ and worker remained near 60 Hz, but the GUI was already around 38–45 Hz with only 4–6 points. As pulses accumulated, performance collapsed:

| Points | `setData` average/maximum | Qt rate |
|---:|---:|---:|
| 400 | 0.77/1.17 ms | 23.2 Hz |
| 710 | 9.08/28.28 ms | 8.6 Hz |
| 1,026 | 23.72/47.52 ms | 3.8 Hz |
| 1,726 | 31.79/41.97 ms | 2.8 Hz |
| 4,377 | 63.67/98.39 ms | 2.6 Hz |

This implicated both the original camera preview path and PyQtGraph's dense disconnected paths.

### `Screencast from 07-22-2026 03:46:36 PM.webm`

After the direct raster camera preview and paired digital LOD, the early phase improved substantially:

- With 4–450 points, timer/paint stayed around 55–59 Hz.
- At 810 points, the UI fell to about 4.2 Hz.
- At 1,752 points, it remained around 3–4 Hz.
- DAQ and worker stayed near 60 Hz.

The camera then appeared to refresh at the same slow rate as the graph, indicating both were being delayed by their shared GUI loop.

### `Screencast from 07-22-2026 03:53:29 PM.webm`

This recording contained idle/early acquisition, active plotting, then frozen plotting. It provided the most important result:

- Early acquisition remained around 58–59 Hz through roughly 376–450 points.
- Around 706 points, the UI began slowing (about 19 Hz with a 409 ms worst timer gap).
- The plot was frozen around 1,012 points.
- During the frozen interval at 1,634–1,752 points:
  - `Set = 0.00/0.00 ms`;
  - Analysis `Paint = 0.0 Hz`;
  - DAQ and worker remained approximately 59–60 Hz;
  - the entire Qt timer remained only about 4–6 Hz;
  - timer gaps remained roughly 290–366 ms.

**Conclusion:** The remaining severe lag is not caused by DAQ cadence, the plot worker, shared-memory copying, `curve.setData()`, or Analysis graph painting. The entire main Qt/Python process is being starved after full camera acquisition/pulsing begins. Because the camera preview shares that event loop, it slows too.

Live `ps` inspection during the problematic state showed the main application consuming approximately one full CPU core (roughly 88–96% aggregate CPU), while camera and other child processes also used CPU. The machine has many cores, so total machine CPU capacity was not exhausted; the suspicion is main-process event handling or Python GIL starvation.

### `Screencast from 07-22-2026 03:58:30 PM.webm`

This was the first hardware run with the application-wide `QApplication.notify()` slow-event profiler enabled. It showed a severe acquisition-specific regression and identified the Qt receivers involved:

| State | DAQ/worker | Qt timer/paint | Worst timer gap | Reported slow event |
|---|---:|---:|---:|---|
| Idle/early start, about 322 points | about 60 Hz | about 59 Hz | 17.9 ms | none |
| Acquisition, about 1,390 points | about 60 Hz | 7.6 Hz | 3,043.5 ms | `QGraphicsScene/MetaCall`, 603.7 ms |
| Acquisition, about 1,752 points | about 60 Hz | 0.2 Hz | 5,339.9 ms | `QGraphicsScene/MetaCall`, 666.8 ms |
| Plot frozen, about 1,752 points | about 60 Hz | timer 1.7 Hz, Analysis paint 0 Hz | 647.3 ms | `MainWindow/UpdateRequest`, 220.5 ms |

At the worst unfrozen point, shared-memory copying remained about 0.07 ms and `curve.setData()` was about 17.36 ms. Those costs cannot explain a 5.3-second timer gap. With the plot frozen, `Set` was exactly zero and the Analysis viewport did not paint, yet the entire GUI still ran at only about 1.7 Hz.

**Conclusions from this recording:**

- DAQ sampling, chunk cadence, shared memory, and plot-worker preparation remain healthy at approximately 60 Hz.
- PyQtGraph's deferred `QGraphicsScene/MetaCall` work is a real additional bottleneck when plotting is enabled; the inexpensive worker-side waveform conversion does not measure this deferred scene cost.
- The graph is not the only bottleneck. Freezing it exposes a slow top-level `MainWindow/UpdateRequest`, proving camera/window repaint activity can independently starve the shared GUI loop.
- The experimental direct-raster camera renderer made the acquisition-state regression substantially worse. Its apparent offscreen microbenchmark improvement did not translate to the integrated application.
- The slow-event profiler itself may add small per-event timing overhead, but it cannot account for the state-dependent collapse: the same instrumentation sustains about 59 Hz before full camera acquisition begins.

## Suspicions investigated but not sufficient

1. **CSV/persistence overhead**
   Analysis-stream persistence was removed/verified absent. This is correct architecturally but was not the lag source.

2. **Live inference pose rate flooding Qt**
   Intermediate pose display updates were coalesced to the newest pose on the display cadence while leaving full inference results available to behavior/control/saving. This prevents an unbounded pose UI queue but did not solve the camera-acquisition slowdown.

3. **DAQ read chunk size/cadence**
   A real 100 ms NI transfer burst was found and fixed with `ON_BOARD_MEMORY_NOT_EMPTY`. DAQ/worker telemetry now remains near 60 Hz during stalls, proving it is no longer the final bottleneck.

4. **Multiprocessing queues/backlog**
   Replaced with shared-memory newest-snapshot rings. This removed pickling and stale queued frames but did not solve GUI starvation.

5. **Rolling-window timestamp errors**
   Corrected using absolute sample indices and timestamped steps. Pulse timing/shape is now correct, but UI responsiveness still degrades.

6. **Digital conversion/downsampling cost**
   Optimized to roughly 1–4 ms in the worker. Worker remains 60 Hz during GUI stalls.

7. **NaN-separated PyQtGraph rectangles**
   Replaced with paired vertical strokes. This reduced vertices and synthetic cost but did not eliminate the real stall.

8. **PyQtGraph `PlotDataItem` overhead and bounds discovery**
   Replaced with direct `PlotCurveItem` plus `ignoreBounds=True`. Real `Set` cost improved, but freezing proves the final stall continues with `Set=0` and no graph paints.

9. **Camera preview QGraphics/OpenGL composition**
   Replaced experimentally with direct raster painting. Although an offscreen benchmark and early acquisition looked faster, the 03:58 integrated run became much worse: even with plotting frozen, `MainWindow/UpdateRequest` reached about 220 ms and the GUI ran at only about 1.7 Hz. The direct-raster experiment should be reverted; it is not an acceptable fix.

10. **High-rate logging flood**
    Camera code can warn on dropped frames and all child logs pass through multiprocessing logging. The current log was inspected; no obvious sustained dropped-frame warning flood was found. Logging remains a possible GIL contributor but is not yet supported by the captured log evidence.

11. **Machine-wide CPU saturation**
    Not supported: multiple cores remain available. The main process itself approaches one full core, pointing toward its event loop/GIL rather than total system capacity.

## Latest diagnostic result: application-wide Qt event profiler

The newest uncommitted change subclasses `QApplication.notify()` and records Qt events taking at least 10 ms. Analysis metrics now append:

`Slow <receiver>/<event type> <maximum duration> (<count>)`

Files:

- `tools/acquisition/run_acquisition.py`
- `tools/acquisition/view/ui_performance.py` (new file)
- `tools/acquisition/view/analysis_content.py`

The profiler was tested in the 03:58:30 hardware recording. It reported two concrete slow paths:

- `QGraphicsScene/MetaCall` at roughly 604–667 ms while the Analysis graph was updating.
- `MainWindow/UpdateRequest` at roughly 220 ms while the Analysis graph was frozen.

This separates the integrated failure into two GUI-thread costs: deferred PyQtGraph scene processing and top-level camera/window repainting. The diagnostic has served its purpose and should be disabled or removed from normal operation after the corrective patch so its per-event timing overhead is not retained in production.

## Uncommitted corrective patch after the 03:58 recording

The next corrective implementation was completed for hardware testing but intentionally not committed:

1. The application-wide `QApplication.notify()` profiler was removed from normal execution. Its diagnostic results remain documented above, but it no longer times every Qt event.
2. The failed direct-raster camera experiment and the older camera `QGraphicsScene/QGraphicsView` stack were both replaced with a purpose-built `QOpenGLWidget` camera viewport.
   - Each preview owns one persistent grayscale OpenGL texture and one static quad vertex buffer.
   - `QCaptureView` still retains only its newest frame and presents it on the bounded `live_feed_refresh_rate` timer.
   - A detached copy of only that newest preview frame is retained until the deferred GPU upload; no preview history or frame queue was added.
   - Camera capture, recording, inference, and saved video remain unchanged and full-rate.
   - Text, presence, pose-point, and reach overlays are drawn in the camera OpenGL pass.
3. The Analysis PyQtGraph/QGraphicsScene renderer was replaced with `DigitalOpenGLPlot`.
   - The existing dedicated plot process and double-buffered shared-memory snapshots remain unchanged.
   - Each channel owns persistent X and Y vertex buffers sized once to the bounded maximum.
   - Newest arrays are uploaded with buffer-subdata-style writes only when a completed snapshot is displayed.
   - Digital transitions use `GL_LINE_STRIP` for exact steps or `GL_LINES` for pixel-level min/max pairs.
   - X zoom/pan is handled by shader uniforms; it does not rebuild scene objects.
   - The fixed `[-0.2, 1.2]` Y range, default `[-10, 0]` X range, horizontal-only zoom, and `Live` behavior are preserved.
   - Only the graph OpenGL viewport is scheduled for repaint, and repeated update requests coalesce naturally.
4. The static graph axes are a transparent child overlay updated only for resize/range changes; waveform refreshes do not reconstruct axes, labels, legends, or layouts.

The actual desktop OpenGL context was exercised outside the offscreen test platform:

- Analysis shader creation succeeded, a persistent channel buffer set was allocated, and a captured square-wave render showed discrete vertical transitions and fixed axes.
- Camera shader creation succeeded, one persistent texture was allocated, and a grayscale test image uploaded and rendered correctly.
- A synthetic integrated run with two camera widgets and the graph confirmed ongoing GPU presentation. Context creation introduces a one-time startup cost; these widgets are created before acquisition begins.

### Hardware result after the OpenGL correction

The user reported that the integrated acquisition behavior remained exactly the same with both camera and Analysis `QGraphicsScene` paths removed. This rules out the rendering backend as the root cause.

Live operating-system sampling during the failed state then found the first direct non-rendering cause:

- The main application process used approximately 101% aggregate CPU, or about one full logical core.
- The actual Qt/main thread used only about 4% CPU.
- A separate Python thread inside the same main process continuously used about 95–97% CPU during the stall.
- DAQ, camera, and plot child processes continued independently on other cores.
- The hot Python thread later exited and was replaced by another transient Python thread during continued acquisition testing.

**Revised conclusion:** a same-process acquisition worker is monopolizing a logical core and likely the Python GIL when cameras become active. This starves every Qt timer and explains why DAQ, queue, PyQtGraph, raster, and OpenGL changes all produced the same acquisition-only symptom. The graph and camera renderer changes are useful bounded architectures, but they cannot correct GIL starvation elsewhere in the main process.

External `gdb`, `perf`, and Python-stack attachment were blocked by the host's `ptrace_scope=1` and `perf_event_paranoid=4`. A low-overhead in-process `UiContentionSampler` was therefore added:

- samples `/proc/<pid>/task` twice per second;
- maps the hottest native thread back to `threading.Thread`;
- captures its Python filename, line number, and function from `sys._current_frames()`;
- excludes its own sampler thread;
- appends `Hot <thread> <cpu>% [tid] <file>:<line> <function>` to Analysis telemetry.

This is intentionally much lighter than the removed `QApplication.notify()` profiler and does not instrument Qt events or acquisition data. A synthetic busy-thread test correctly identified a known thread and source location. The next hardware restart should expose the exact acquisition loop to fix.

## Current uncommitted worktree

At handoff time, these files are modified/untracked and intentionally not committed:

- `auto-trainer-device/src/autotrainer/device/nidaq_signal_stream.py`
- `auto-trainer-device/tests/nidaq_signal_stream_test.py`
- `auto-trainer-pyside/src/autotrainer/pyside/capture/QtGLImageView.py`
- `tests/signal_stream_ui_test.py`
- `tools/acquisition/model/analysis_plot_process.py`
- `tools/acquisition/run_acquisition.py`
- `tools/acquisition/view/analysis_content.py`
- `tools/acquisition/view/digital_opengl_plot.py` (new)
- `tools/acquisition/view/ui_contention_sampler.py` (new diagnostic)

The handoff document itself is under `temp/update_speed_handoff/` and should be staged so the temp directory is tracked, but it is not committed.

## Verification performed

Focused tests were repeatedly run using the `reachaq` Conda environment. Latest relevant results:

- Analysis/stream focused suite after the OpenGL correction: 25 passed.
- Analysis, panel-resize, and video-capture focused suite after the OpenGL correction: 30 passed, 1 hardware-dependent test skipped.
- Analysis, panel, and NI tests previously: 30 passed, with one unrelated laser shared-process test timing out once during a combined run; that isolated test immediately passed (`1 passed in 0.74 s`), indicating a process-start scheduling flake rather than a deterministic regression.
- `git diff --check` passes.

Use:

`conda run -n reachaq python -m pytest tests/signal_stream_ui_test.py tests/panel_resize_test.py -q`

## 04:24 camera-disabled recording and identified root cause

The 04:24 recording tested acquisition with both camera previews explicitly showing `Capture disabled`. The graph still fell to roughly 6–7 updates/second. This falsifies camera preview/backing-store painting as a necessary cause of the acquisition-only slowdown.

Live `/proc/<pid>/task` sampling during that same application run showed:

- application process CPU around 86%;
- Qt/main thread CPU only around 1–2%;
- native thread TID 417329 continuously around 98% CPU.

The application log maps TID 417329 exactly to the `can-device` reader thread. It starts when acquisition connects the CAN hardware and was processing only about 10–13 messages/second. The source-level cause is `DeviceConnection._run_connected()`: it asks the CAN backend for a 5 ms collection period, but the installed backend returns immediately when no frame is available. The surrounding Python loop therefore spins without blocking, consumes a full logical core, and starves the Qt thread through the GIL. This explains all three key observations: idle mode is smooth, acquisition mode is slow, and both graph and camera UI cadence degrade together independent of preview rendering.

A targeted uncommitted correction now measures the CAN read duration and sleeps only the unused remainder of the requested 5 ms collection interval when the read returns no messages. CAN bursts still drain immediately; only empty polling is bounded to the intended 200 Hz maximum. A regression test reproduces an always-empty immediate-return backend and verifies that the reader is throttled and disconnects normally.

Latest verification with the `reachaq` environment:

- CAN, NI-DAQ stream, and signal-stream UI suites: **65 passed**.
- `git diff --check`: passed.

## Recommended immediate next step

1. Close the currently running application, which still contains the old spinning CAN loop.
2. Restart from the modified worktree and repeat the same idle → acquisition test with cameras disabled first.
3. Confirm that process CPU no longer gains one fully saturated `can-device` thread and that Analysis `Timer`, `Frames`, and `Paint` stay near the monitor refresh rate.
4. Repeat with one and then two camera previews to measure any secondary rendering cost after GIL starvation is removed.
5. Do not commit until this source-specific fix passes the integrated hardware test.

## Post-fix cleanup

After the CAN idle-poll correction eliminated the acquisition slowdown in hardware testing, the later uncommitted diagnostic and rendering experiments were removed. The cleanup restored the last committed implementations of:

- the camera preview widget;
- the Analysis PyQtGraph renderer;
- the Analysis plot worker;
- acquisition application startup;
- signal-stream UI tests.

It also removed the untracked custom OpenGL digital plot and `UiContentionSampler`. This removes the freeze control, detailed plot profiler, custom camera texture renderer, and view-dependent diagnostic preparation that did not solve the root issue.

The cleanup intentionally retains:

- the confirmed CAN empty-read/GIL-starvation fix and its regression test;
- the independent NI-DAQ `ON_BOARD_MEMORY_NOT_EMPTY` transfer-cadence fix and its test;
- all previously committed timestamp-correct square-wave rendering, digital-axis constraints, Live navigation, newest-snapshot buffering, and laser plotting behavior.
