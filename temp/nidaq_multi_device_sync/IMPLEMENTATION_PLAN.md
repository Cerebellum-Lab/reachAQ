# Multi-device NI-DAQ signal-stream synchronization plan

Status: design only; no source code has been changed. This plan is tracked for review before implementation.

## 1. Required outcome

The visualization-only NI-DAQ signal stream must accept channels from any number of NI devices rather than enforcing one digital device. Every channel in one stream must represent the same sample timeline:

- one sample rate and one monotonically increasing sample index;
- one shared hardware sample clock wherever the hardware can route it;
- a common hardware start event, or the exactly equivalent arrangement in which every input task is armed before the first edge of the shared sample clock;
- no CSV, HDF5, acquisition-output, or other persistence;
- one input task per device and subsystem as the general solution;
- optional NI-DAQmx multidevice tasks only when a capability probe proves the complete device group supports them;
- PXI/PXIe backplane routing, PCI/PCIe RTSI routing, and explicitly wired PFI/external timing routes must all be representable;
- an unsupported card, port, route, or occupied timing resource must fail before streaming with a precise diagnostic. It must never silently fall back to unsynchronized software reads.

“Any number of devices” is an architectural property, not a promise that arbitrary hardware can be synchronized without the required timing features and physical interconnect. The implementation must have no one-device or two-device assumption, but every selected device still has to support buffered acquisition and have a valid clock/trigger route.

## 2. Findings confirmed on the current machine

Environment queried without starting an acquisition:

- NI-DAQmx Python 1.0.2 and NI-DAQmx driver 26.3.1.
- `PXI1Slot5` is a PXI-6221 (M Series). DAQmx reports a 1 MHz maximum DI rate and exposes `/PXI1Slot5/di/SampleClock`, PXI trigger terminals, and two counters.
- `PXI1Slot4` is a PXI-6713 (AO Series). It has eight static digital lines, but DAQmx does not report a DI maximum rate and exposes no DI sample-clock terminal.
- Verifying a continuous 10 kHz DI task on `PXI1Slot4/port0/line0`, clocked from `PXI_Trig0`, fails with DAQmx `-200077`.
- Verifying one DI task containing a Slot 5 line and a Slot 4 line fails with DAQmx `-201426`: one or more devices do not support multidevice tasks.
- The active `/home/christielab10/Autotrainer/system_configuration.yaml` contains inconsistent duplicated mappings: `nidaqPorts.barcode` is `PXI1Slot4/port0/line3`, while the persisted `nidaqStream` channel named `barcode` is still `PXI1Slot5/port0/line7`. `cam_frames` is `PXI1Slot4/port0/line0` in both sections. Editing DAQ ports currently saves the new port map without reconciling previously selected Analysis channels, so the worker receives the stale Slot 5 barcode path and triggers the present cross-device guard.
- On this Linux system, buffered PXI-6221 Port 0 DI using DAQmx's default DMA transfer returns zero-filled samples even though on-demand reads see the correct state. The same hardware-clocked task using `DataTransferActiveTransferMode.INTERRUPT` or `POLLED` correctly returns bit 3 high (`0x08`). The exact reachAQ controller was verified at 10 kHz with interrupt transfer: `cam_frames` on line 2 remained low while the inactive camera was closed, and `barcode` on line 3 was high for all 2,000 samples. The implementation must select/log a working DI transfer mechanism rather than relying on the driver default.

Consequence: a `cam_frames` wire connected to Slot 4 cannot be sampled as synchronized 10 kHz digital data. It must be physically moved to a hardware-timed DI port (Slot 5 Port 0 is suitable if a line is available), or a compatible hardware-timed DI card must be installed. A counter on the 6713 could count one pulse stream, but that is not equivalent to acquiring arbitrary barcode and frame logic on a shared 10 kHz sample grid.

Relevant NI documentation:

- Timing and synchronization, including PXI trigger versus PCI/PCIe RTSI transport: https://www.ni.com/en/support/documentation/supplemental/06/timing-and-synchronization-features-of-ni-daqmx.html
- Multidevice task/channel-expansion limitations: https://www.ni.com/en/support/documentation/supplemental/15/channel-expansion-explained.html
- PXI/PXIe sample-clock and trigger synchronization principles: https://www.ni.com/en/support/documentation/supplemental/10/synchronization-explained.html
- PXI-6713 product information: https://www.ni.com/en-rs/support/model.pxi-6713.html

## 3. Proposed architecture

### 3.1 Task grouping

Replace the controller’s `_analog_task`, `_digital_task`, and single counter fields with a collection of task groups keyed by `(device_name, subsystem)`.

For example:

```text
PXI1Slot5/DI: barcode
PXI1Slot6/DI: cam_frames, reward_gate
PCIeDev1/DI: external_event
PXI1Slot7/AI: force, photodiode
```

Each group owns one DAQmx task and a stable ordered list of configured channels. DI and AI remain separate tasks even on the same device because DAQmx task types cannot be mixed. No task-building loop will assume a maximum group or device count.

### 3.2 Synchronization planner

Add a pure planning layer that receives channel groups, discovered device capabilities, synchronization preferences, and currently reserved resources. It returns a declarative plan containing:

- selected strategy and timing master;
- one task specification per group;
- the sample-clock source each task will consume;
- start-trigger source/export information;
- any counter, PXI trigger, RTSI, or PFI resources consumed;
- task arm/start order and reverse shutdown order;
- expected synchronization quality and an explanation suitable for logs/UI.

The executor will be separate from the planner. This keeps topology decisions unit-testable without NI hardware and makes partial-creation cleanup deterministic.

### 3.3 Strategy order

The default `synchronized_per_device` strategy will be the requested general path:

1. Group channels by device and subsystem.
2. Choose a master that can produce or expose a stable sample clock at the configured rate. Prefer, in order: an explicitly configured external clock, an existing AI sample clock in the stream, then a free counter output on a capable selected device.
3. Ask DAQmx to route that master clock to every task. DAQmx task verification/commit is the authority; product-name allowlists are not.
4. Configure a shared start-trigger route when the task families support it. Arm all slave tasks first, then the master input task, then start the clock producer.
5. If a distinct start trigger is unavailable but all tasks directly consume the same initially stopped counter sample clock, arm every input task and start the counter last. The first shared clock edge is sample zero for every task, so this is sample-aligned even though it does not consume a second trigger line. This equivalent mode must be explicit in diagnostics and may be disallowed by a strict configuration flag if a physically distinct start trigger is mandatory.
6. Verify or commit the complete task graph before reporting the worker ready. Any failed route aborts startup and closes every task.

An optional `multidevice_task` strategy may build one task per subsystem across devices and let DAQmx perform channel expansion. It will only be used after the exact proposed channel list passes a disposable DAQmx verification probe. `auto` may probe this optimization and fall back to synchronized per-device tasks. The current Slot 4/Slot 5 combination is already proven ineligible and will use neither a multidevice task nor an unsafe fallback.

### 3.4 Transport independence

Routing must be capability-driven rather than hard-coded to a product or bus:

- Same PXI/PXIe chassis: DAQmx can usually route over `PXI_Trig<n>` or another backplane timing terminal.
- PCI/PCIe boards: precise internal-board synchronization requires a physical RTSI cable supported by the boards and registered in NI MAX. If RTSI is absent, an explicitly wired PFI clock/trigger route is required.
- Mixed PXI and PCIe, separate chassis, or other disconnected timing domains: there is no implicit shared backplane. Configuration must describe an external PFI/timing-module bridge, or startup must reject the topology.
- Multidevice channel expansion is not assumed to work across bus types or arbitrary product families.

Prefer task-owned DAQmx routing and export properties so routes are released with the tasks. If `System.connect_terms()` is ever required for an explicit route, record every connection and disconnect only those exact connections during cleanup.

### 3.5 Capability probing

Extend isolated NI discovery/preflight data to include, where supported:

- bus type, chassis and slot or PCI bus location;
- product category and simulation state;
- DI/AI hardware-timing rate properties;
- terminals, counters, and candidate timing terminals;
- whether each selected port accepts buffered sample-clock timing;
- whether the exact group supports a multidevice task;
- whether DAQmx can route the proposed clock and trigger while respecting current resource reservations.

Static discovery metadata is advisory. The final check creates disposable tasks for the exact selected channels and calls DAQmx task verify/commit in the existing isolated worker. Expected failures such as `-200077`, `-201426`, route-not-found, resource-reserved, and unsupported-terminal errors will be translated into actionable messages naming the device, channel, timing source, and required correction.

The probe must not start counters, toggle outputs, or leave routes connected.

## 4. Configuration changes

Add a backward-compatible synchronization configuration under `NidaqSignalStreamConfiguration`, with defaults that preserve existing single-device behavior:

```text
strategy: synchronized_per_device | auto | multidevice_task
require_hardware_sync: true
require_distinct_start_trigger: false
master_device: optional
sample_clock_source: optional fully qualified terminal
sample_clock_export_terminal: optional PXI_Trig/RTSI/PFI terminal
start_trigger_source: optional fully qualified terminal
start_trigger_export_terminal: optional PXI_Trig/RTSI/PFI terminal
counter_channel: optional, otherwise capability-selected
external_routes: zero or more explicit source/destination declarations
```

Defaults should need no extra YAML for one hardware-timed device. Cross-bus systems should use explicit terminal settings because automatically guessing physical cabling is unsafe. Older YAML must continue loading.

The configuration validator will check syntax, uniqueness, and contradictions only. Hardware validity remains a worker-side DAQmx preflight so configuration can be edited on machines without NI hardware.

### 4.1 Eliminate duplicated port-map drift

For named operator signals (`cam_frames`, `barcode`, and tones), `nidaqPorts` must be the canonical physical mapping. The Analysis configuration should persist which logical signals are selected, not an independently editable copy of their physical paths.

For backward compatibility:

1. On load, migrate known `nidaqStream.channels` names by resolving their current physical channel from `nidaqPorts`.
2. When DAQ ports are edited, atomically reconcile every selected known stream signal before saving or restarting the monitor.
3. Preserve explicitly custom and laser channels as physical-channel configurations because they do not necessarily have a named `nidaqPorts` entry.
4. If a selected named signal becomes unmapped, remove/disable it and show an actionable status rather than continuing to acquire its stale former line.
5. Before worker launch, validate that each selected known signal agrees with the canonical port map. A mismatch from older or manually edited YAML must be corrected deterministically or rejected with both paths in the message.

This prevents the exact current failure mode in which the UI port editor shows Slot 4 while the worker silently receives an older Slot 5 selection. Reconciliation does not make Slot 4 suitable for buffered DI; the hardware capability preflight still rejects the PXI-6713 static digital port.

## 5. Runtime behavior

### Startup

1. Validate and group channels without opening DAQmx.
2. Discover capabilities and build candidate synchronization plans.
3. Build all input tasks and the clock producer without starting them.
4. Configure timing, buffers, routes, and triggers.
   - For affected Linux M-Series devices, explicitly configure DI transfer as interrupt-driven; retain capability/configuration support for DMA on systems where it is verified to work.
5. Verify/commit the entire graph.
6. Arm slave tasks, arm the master input task, then start the clock producer or external-trigger wait.
7. Report worker ready only after every task is successfully active.

Task names must include sanitized device and subsystem identifiers so logs identify failures while remaining unique.

### Reading and merging

- Every task reads the same effective `read_chunk_size` from hardware buffers driven by the shared clock.
- Tasks are read sequentially only after the first task reaches the requested sample count; because all tasks acquire concurrently, subsequent reads should return immediately. If needed, use `in_stream.avail_samp_per_chan` to wait for all groups before copying.
- Preallocate NumPy buffers per task and use the NI stream-reader APIs to avoid repeated Python list construction where practical.
- Validate that every group returned exactly the requested count. A short, overflowed, or timed-out group is a stream-fatal synchronization error; never merge unequal blocks.
- Merge values back into original configuration order and advance one global sample index by the common count.
- Preserve the existing `NidaqSignalSampleBlock` consumer contract initially. Add optional diagnostic metadata only if it can remain backward compatible.
- Continue sending samples only to the visualization plot process. This work must not reintroduce persistence.

### Shutdown and failure

- Stop the clock producer first so no more edges arrive, stop input tasks, release task-owned exports, then close all tasks in reverse creation order.
- Cleanup must be idempotent and must run after every partial startup failure.
- Aggregate cleanup errors without hiding the original DAQmx failure.
- A route or task failure stops the complete synchronized stream; partial unsynchronized visualization is not acceptable.

## 6. Planned code changes after approval

1. `auto-trainer-core/.../nidaq_stream_configuration.py`
   - Add the synchronization configuration dataclass, YAML registration, validation, and backward-compatible defaults.
2. New `auto-trainer-device/.../nidaq_signal_sync.py`
   - Add channel grouping, capability models, declarative task/route plans, master selection, and DAQmx error translation.
3. `auto-trainer-device/.../nidaq_signal_stream.py`
   - Replace singleton tasks with task-group collections; execute the synchronization plan; implement aligned reads and deterministic cleanup.
4. `tools/acquisition/model/nidaq_discovery.py`
   - Add bus/topology and timing-capability fields while retaining isolated discovery and compatibility with missing properties.
5. `tools/acquisition/model/nidaq_signal_monitor_model.py`
   - Preserve its disposable worker boundary; improve ready/error payloads with the chosen plan and preflight failure detail.
6. Analysis configuration/status UI files
   - Reconcile named Analysis selections with the canonical DAQ-port mapping; show selected master/strategy and hardware validation failures. Avoid exposing advanced routing controls unless automatic selection fails; explicit YAML remains available for unusual topologies.
7. Example hardware YAML
   - Document PXI, PCIe+RTSI, and explicit PFI cases. Correct the current example’s implication that Slot 4 static DIO can be used for a hardware-timed stream.

## 7. Test plan

### Unit tests without hardware

- One DI device uses a counter clock as today.
- N DI devices create N tasks and share one master clock without any two-device assumption.
- Mixed AI and DI groups on one and several devices select one timing domain.
- Original channel ordering is preserved after per-task reads are merged.
- All slaves start before the master and clock producer; shutdown reverses safely.
- Unequal read counts, timeout, overflow, route failure, and partial construction are fatal and close every resource.
- Counter selection skips occupied/unsupported counters and gives an actionable failure when none remain.
- Explicit PXI, RTSI, and PFI route plans are represented correctly.
- Multidevice capability success uses channel expansion; DAQmx `-201426` falls back only in `auto`, while forced multidevice mode reports the failure.
- A static-only DI port is rejected during preflight.
- Transfer-mode tests cover DMA, interrupt, and unsupported modes; the selected mode is logged, and the installed PXI-6221 Linux path uses interrupt transfer so buffered input cannot silently become zero-filled.
- Existing single-device configuration and YAML continue working.
- Changing a named DAQ-port assignment migrates its selected Analysis channel atomically; stale duplicated paths in legacy YAML are detected and reconciled.
- The adaptive monitor-refresh chunk size is applied identically to all tasks.
- No persistence path is called or added.

### Hardware acceptance tests

1. Current PXI-6221 only: connect barcode and `cam_frames` to separate Slot 5 Port 0 lines; verify both at 10 kHz and compare known simultaneous edges over a long run.
2. Current invalid assignment: select a Slot 4 digital line and verify startup refuses it with a specific static-DIO message.
3. Two hardware-timed PXI cards in one chassis: verify shared sample clock/start, measure sample-index agreement and skew with the same pulse wired to both cards.
4. PCIe pair: install and register RTSI cabling in NI MAX, then repeat the same pulse/skew test.
5. Mixed PXI/PCIe: use explicit PFI or timing-hardware wiring for clock and trigger; verify that missing wiring fails route verification and correct wiring passes.
6. Three or more devices: repeat edge-correlation and long-duration buffer/overflow testing to prove count-independent behavior.
7. Exercise acquisition, inference, and laser operations concurrently to detect DAQ resource conflicts and confirm graph cadence remains monitor-limited.

Acceptance requires zero sample-count divergence, no silent fallback, stable long-duration acquisition, deterministic cleanup/restart, and measured skew within the hardware/topology requirement agreed for the experiment.

## 8. Issues, shortcomings, and viability review

### Hard limitations

- The installed PXI-6713 cannot provide buffered DI. Code cannot make its digital port a synchronized 10 kHz input. Rewiring or different hardware is mandatory.
- The PXI-6713 static Port 0 lines also cannot use DAQmx change-detection timing (`-200077`) and cannot be internally routed into a counter source (`port0/line0` to `ctr0` verification fails with `-89120`). A buffered counter sampled from the PXI clock does verify when its signal enters through a counter-capable PFI terminal such as `PFI8`; that is a viable rising-edge-only alternative, but it requires rewiring/splitting the signal and consumes a counter.
- Arbitrary device count does not imply arbitrary physical connectivity. PCI/PCIe needs RTSI or external timing wiring; mixed buses/chassis generally need PFI wiring or timing hardware.
- Routing resources, counters, DMA channels, PXI/RTSI lines, cable fan-out, and NI driver limits are finite. The planner can scale algorithmically but must reject a topology once resources are exhausted.
- Some cards expose only selected ports as correlated/hardware-timed DIO. Discovery showing a digital line is not proof it supports buffered DI.
- Multidevice tasks support selected product families, subsystem combinations, chassis arrangements, and bus types only. They are an optimization, not the universal architecture.

### Integration risks

- The signal monitor currently assumes it owns counter 0. Laser or other DAQ features may contend for counters, terminals, or trigger lines. Resource selection must become explicit/capability-driven, and concurrent hardware ownership needs acceptance testing.
- A DAQmx `TASK_VERIFY` can pass before another process reserves a resource. The worker must still handle commit/start races cleanly.
- Reading many tasks serially is correct with shared hardware timing, but one stalled task can delay delivery of the entire aligned block. This is preferable to displaying misaligned data; telemetry should identify the stalled group.
- A distinct common start trigger consumes another routable line and may be unavailable even when a shared initially stopped clock gives exact sample alignment. The configuration must make this distinction explicit.
- Software timestamps across tasks are not synchronization evidence. Only shared-clock sample indices and a loopback edge/skew test validate alignment.
- Hardware for PCIe, mixed-bus, and three-device acceptance is not installed here, so those paths can be unit-tested but cannot be claimed hardware-validated until representative systems are available.

### Viability conclusion

The architecture is viable and is the correct replacement for the current one-device restriction. A per-device task graph with one shared timing domain is more general and predictable than relying on channel expansion. DAQmx multidevice tasks should remain a probed optional fast path.

It is not viable with `cam_frames` left on the installed PXI-6713 digital port. The first deployment decision is therefore physical: move every signal requiring precise 10 kHz correlation to Slot 5 Port 0 or install another buffered-DI device. Once that is resolved, the current PXI chassis provides suitable backplane routes for a synchronized multi-device design, subject to exact route and resource verification during implementation.

## 9. Confirmation gates before implementation

Approval should confirm:

1. Per-device synchronized tasks are the default, with multidevice tasks optional and probe-gated.
2. Unsupported/static-only DI assignments fail startup rather than falling back to software-timed visualization.
3. Sharing the first edge of an initially stopped master sample clock is acceptable when a distinct start-trigger route is unavailable; otherwise set `require_distinct_start_trigger=true` and accept that more topologies will be rejected.
4. The Slot 4 `cam_frames`/barcode wiring will be moved to hardware-timed DI or replacement hardware will be supplied.
5. Cross-bus systems will supply/configure RTSI, explicit PFI wiring, or timing hardware; the application will not pretend they are synchronized without it.
