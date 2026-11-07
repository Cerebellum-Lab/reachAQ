# NI-DAQ configuration and DAQmx review

Written 2026-09-22 against `feature-dev` at `a1986189`, with every claim
measured on christielab10 (PXI-1045 chassis, PXI-6221 in slot 5, PXI-6713 in
slot 4, BNC-2090A breakout) unless it cites NI documentation.

Three questions were asked: how configurable the timing and channel map should
be, whether the DAQmx calls are the right ones, and whether configuration
should speak in card or breakout terms. They share one root cause, so they
share one answer: **the configuration is a set of assertions that nothing
validates against the hardware it is describing.** Every failure below is a
variant of that.

---

## 1. Configuration flexibility

### What exists now

`NidaqTimingConfiguration` carries twelve knobs:

| knob | validated how |
| --- | --- |
| `sync_mode` | enum membership (`auto`/`backplane`/`external`/`independent`) |
| `task_strategy` | enum membership (`per_device`/`auto_multidevice`/`forced_multidevice`) |
| `timing_master` | not validated |
| `require_hardware_synchronization` | not validated |
| `reference_clock_source` | not validated |
| `start_trigger_source` | not validated |
| `sample_clock_source` | not validated |
| `require_distinct_start_trigger` | not validated |
| `sample_clock_export_terminal` | not validated |
| `start_trigger_export_terminal` | not validated |
| `external_routes` | not validated |
| `transfer_mechanism_overrides` | non-empty strings only |

Plus the laser's `hardware_timed`, `sample_rate_hz`, `backplane_clock_line`
and per-channel `trigger_source` / `trigger_route_source`, and the stream's
`sample_rate_hz`, `read_chunk_size`, `analog_terminal_config`.

All validation is syntactic. Nothing asks the installed hardware whether any
of it is achievable.

### What that cost, measured

Each of these was a real failure on this rig, and each is an unvalidated
assertion:

- **A terminal that does not exist.** The plan defaults
  `reference_clock_source` to `PXI_CLK10`. NI spells the terminal
  `PXI_Clk10`, and the PXI-6713 exposes no Clk10 at all. Setting it failed
  the task with -200452 and took the whole signal stream down. Nothing
  checked the name against `Device.terminals`, which lists exactly what each
  board has.
- **A capability the card lacks.** The BNC-2090A has an APFI BNC connector,
  so wiring a stimulus line to it is the obvious thing to do. The PXI-6221
  has no analog trigger circuit: DAQmx answers
  `Property: DAQmx_StartTrig_Type, Requested Value: DAQmx_Val_AnlgEdge,
  Possible Values: DAQmx_Val_DigEdge, DAQmx_Val_None`. That signal can never
  be used, and nothing says so until someone spends a day tracing it.
- **A topology that cannot route.** The laser output sits on slot 4 and its
  clock and trigger come from slot 5. DAQmx refuses the route with -89125
  because the chassis is unidentified (`pxi_chassis_num` reads 4294967295 on
  both boards). The configuration happily described this arrangement for
  weeks.
- **A rate that is silently ignored.** `cfg_samp_clk_timing(rate=...)`
  alongside an external `source` only sizes the buffer; the real rate is the
  external clock's. The laser built waveforms for its configured 100 kHz
  while being clocked at the stream's 10 kHz, so every pulse width and
  interval came out ten times too long, silently.
- **A knob that does nothing.** `transfer_mechanism_overrides` is read into
  the task graph in `nidaq_timing.py:287` and never applied to any task. The
  only transfer mechanism actually set anywhere is a hardcoded
  `di_data_xfer_mech = INTERRUPT` in `_make_digital_task`.

### Proposed model

Introduce a **capability probe** between discovery and planning, and validate
the configuration against it. The probe asks the driver what is true - it
does not hard-code board knowledge, which would rot.

For each installed device, record: product type, bus type, chassis and slot
number, the full `Device.terminals` list, supported terminal configurations,
counter count, whether it has analog input, analog output, correlated DIO,
and which start-trigger types its task types accept. All of this is
obtainable from `nidaqmx.system.Device` and a cheap `TASK_VERIFY` probe.

From that, derive a **routing capability matrix**: for each ordered pair of
devices, whether a trigger or clock can cross, and by what means - same
device, PXI backplane trigger line, RTSI cable, or not at all. This is the
piece that makes mixed topologies (a PCIe card plus a PXI chassis) safe
rather than merely permitted: a PCIe device and a PXI device can only share
timing over an RTSI cable, and if there is not one, the configuration that
implies one must be refused at load.

Then validate, and **refuse** - the decision taken was to constrain hard:

- every named terminal exists on the device it names;
- every named trigger type is in that task type's `Possible Values`;
- the synchronisation topology the config asks for is in the routing matrix;
- every channel's physical channel exists and supports the requested terminal
  configuration;
- the laser's `sample_rate_hz` is not contradicted by a shared clock it will
  actually run on;
- no knob is set that nothing consumes.

### Change list

- **C1.** Add `NidaqDeviceCapabilities`, populated from `nidaqmx.system.Device`
  plus `TASK_VERIFY` probes, cached per discovery.
- **C2.** Add a routing capability matrix derived from bus type, chassis
  number and slot, covering same-device, PXI backplane, RTSI and
  not-possible.
- **C3.** Validate `NidaqPortConfiguration` against C1 and C2 at load, raising
  with the specific assertion that failed and the values the driver reports as
  possible. Refuse rather than warn.
- **C4.** Delete `transfer_mechanism_overrides`, or apply it. It is currently
  a knob that does nothing, which is worse than either. *Applied, per
  device and subsystem. The interrupt default stays for digital - the DMA
  path still returns zero-filled samples on this runtime - and a mechanism
  DAQmx does not have is now refused rather than ignored.*
- **C5.** Fold `require_hardware_synchronization`, `require_distinct_start_trigger`
  and `task_strategy` into `sync_mode`. They are three booleans and an enum
  describing one decision, and most of their sixteen combinations are
  meaningless. *Done, by refusal rather than by folding. The fields are
  written into every configuration on disk and the default loader does not
  filter unknown keys, so removing one would stop an existing rig loading;
  what is removed is the ability to set them to a combination that means
  nothing. requireHardwareSynchronization now derives from syncMode unless
  set, and requireDistinctStartTrigger - read by nothing, ever - refuses to
  be set at all.*
- **C6.** Derive `reference_clock_source`, `start_trigger_source` and
  `sample_clock_source` from the topology by default, and treat an explicit
  value as an override that is validated the same way. Today the default is a
  string constant that was wrong for one of two boards. *Done. The start
  trigger and sample clock were already derived; the reference clock now is
  too, from whether the boards share a PXI backplane at all.*
- **C7.** Carry `sample_clock_rate_hz` through the plan for every consumer,
  not just the laser, and refuse a configuration whose declared rate differs
  from the clock it will be given. *Done. The stream refuses the mismatch
  and says what it would cost: a chunk it treats as one second really taking
  ten.*
- **C8.** Record chassis identification state in the capability probe and
  refuse cross-device timing when it is absent, with the remedy in the
  message, rather than failing at -89125 inside a task.

---

## 2. DAQmx usage

The surface actually used is small: `add_ai_voltage_chan`,
`add_di_chan`, `add_do_chan`, `add_ao_voltage_chan`,
`add_co_pulse_chan_freq`, `cfg_samp_clk_timing`, `cfg_implicit_timing`,
`cfg_dig_edge_start_trig`, `connect_terms`, `control`, `read`, `write`,
`start`, `stop`, `wait_until_done`, and the `in_stream` properties. Most of
it is used correctly. What follows is what is not.

### D1. Buffered digital is read the slow way, and the comment explaining why draws the wrong conclusion

`_make_stream_reader` looks for
`DigitalMultiChannelReader.read_many_sample_multi_line`, does not find it,
and falls back to `task.read()`, which allocates a fresh list of lists per
chunk. The comment says this "is not a version floor to raise: the call has
never resolved".

The comment is right that the method does not exist. The conclusion is wrong.
NI's buffered digital path is **port-based, not line-based**: add the port as
a single channel with `LineGrouping.CHAN_FOR_ALL_LINES` and read it with
`read_many_sample_port_byte` into a preallocated array, then unpack bits.
Verified on the rig:

```
PXI1Slot5/port0/line0:3 with CHAN_FOR_ALL_LINES -> task verifies
DigitalSingleChannelReader.read_many_sample_port_byte -> present
```

This is the documented approach for correlated DIO and it removes the
per-chunk allocation entirely.

### D2. The availability barrier is redundant, and it spins

`_wait_all_available` polls `in_stream.avail_samp_per_chan` on every task in a
`time.sleep(0.0005)` loop until all have the chunk, then reads.

If the tasks genuinely share a start trigger and a sample clock, sample *N* is
simultaneous across them by construction - that is the entire point of the
shared-timing design, and each task's blocking `read` already waits for its
own data. The barrier adds a spin loop that duplicates that wait. If the
tasks do *not* share timing correctly, the barrier does not fix it; it only
hides the skew behind a delay.

It also caused a bug of its own: a polled analog task has no host buffer, so
asking it for `avail_samp_per_chan` raises -200455, and `hasattr` does not
catch a `DaqError`.

The barrier earned its keep as diagnostic scaffolding while the timing was
being established. It should not survive as production code.

**Implemented, and the heading above is wrong on the second half.** The
barrier is redundant, and removing it removed the class of failure the
output-task patch had treated one instance of. It does not spin at a cost
worth recovering. Measured on christielab10 at 10 kHz, 160 chunks over eight
seconds, three runs of each alternating: with the barrier 11.2/11.7/12.1% of
a core, without it 13.2/13.5/14.0%. The DAQmx blocking read does not sleep
through the wait, and costs about two points of a core more than the 0.5 ms
spin it replaced. Both deliver the same 160 chunks, because a chunk takes its
own 50 ms to arrive whatever asks for it. So D2 is a simplification that is
slightly more expensive, and should be judged on that. The per-task state the
barrier could report is kept and moved to the failure path.

### D3. `cfg_samp_clk_timing(rate=...)` beside an external source is not the rate

Already fixed for the laser, but the same shape exists wherever a slave device
is given `source=` and a `rate` from its own configuration. The rate argument
sizes the buffer; the clock decides the rate. Any code that computes sample
counts from the argument rather than from the clock is wrong by the ratio
between them - a factor of ten on this rig, silently.

### D4. `add_ai_voltage_chan` without `terminal_config`

Fixed in `a1986189`. Worth keeping in the list because it is the clearest
example of the pattern: DAQmx's default is not neutral, it is per-channel, and
on a PXI-6221 it gives ai0-ai7 differential and ai8+ single-ended. A channel
list spanning both halves then reads some channels against pins it also reads
directly. Naming it explicitly removed three of four spurious responses.

### D5. `di_data_xfer_mech = INTERRUPT` is hardcoded on a guess

The comment attributes zero-filled DMA reads to "the Linux PXI-6221 runtime".
The actual cause was the IOMMU blocking the board's DMA, fixed at the kernel
command line on 2026-09-18. The workaround has not been retested since and may
now be costing throughput for nothing.

### D6. Polling rather than callbacks

The worker loop calls `read_chunk` in a tight loop. DAQmx offers
`register_every_n_samples_acquired_into_buffer_event`, which hands the
application a callback when N samples are ready. That is the documented way to
run a continuous acquisition and it removes both the barrier and the loop.
This is a larger change and should follow D1 and D2.

### D7. Per-sample scaling, quantified honestly

`read_chunk` scales one sample at a time into tuples of Python floats, which
are then written into a numpy-backed ring. Benchmarked on the rig at
realistic sizes: **0.671 ms per chunk against 0.006 ms vectorised, 105 times
slower** - but only about **1.3% of one core** at the current chunk size.

It is a code-quality issue, not a performance problem. An earlier note in
`todo.md` claiming this held the polled path to 43% of real time was wrong;
that figure was the DAQmx polled transfer, not the scaling.

### Change list

- **D1.** Read correlated digital as a port with `CHAN_FOR_ALL_LINES` and
  `read_many_sample_port_byte` into a preallocated buffer; unpack bits with
  numpy. Removes the per-chunk allocation and the dead capability check.
- **D2.** Delete `_wait_all_available` once D1 and the shared-timing
  preflight are trusted; rely on each task's blocking read. *Done. Costs
  ~2 points of a core rather than saving any; taken for the simplification
  and the removed failure mode, not for speed.*
- **D3.** Audit every `cfg_samp_clk_timing` call for a `source=` with a local
  rate, and derive sample counts from the plan's clock rate. *Audited; no
  further change needed. The laser resolves its rate through
  `_require_sample_rate` at all three call sites, the wiring tool clocks
  internally, and the stream's two calls are what C7 now guards.*
- **D5.** Retest `di_data_xfer_mech` under DMA now the IOMMU is fixed, and
  remove the override if it is no longer needed. *Not retested. C4 made the
  interrupt mode a default that `transferMechanismOverrides` can now
  actually override, so the experiment can be run from the configuration
  without a code change - which is the safer order for it.*
- **D6.** Move the worker to
  `register_every_n_samples_acquired_into_buffer_event`, after D1 and D2.
  *Tried, measured, and not kept. See below.*
- **D7.** Vectorise scaling and keep numpy arrays end to end into the ring.
  Low priority; it is 1% of a core. *Done, and the estimate was right: 11.0%
  of a core down to 9.8% on average, with one of three pairs overlapping.*

---

## 3. Breakout boards

### Facts

The BNC-2090A provides, per NI 372101A-01, **22 BNC connectors** - sixteen
analog input (AI 0 to AI 15), two analog output, **one APFI**, **one PFI**,
and two user-defined - a **29-position spring terminal block**, and two
68-pin connectors. Digital I/O and PFI 1 to PFI 15 are on the strip, and PFI0
is the only PFI line with a BNC.

*Corrected while implementing B1: the list above originally omitted the
sixteen analog input BNCs entirely, which is the majority of the panel. It
was written from the connectors this rig had been arguing about rather than
from the manual, which is the same mistake the section goes on to describe.*

Two consequences, both of which cost time on this rig:

- **"There is no slot for PFI1 on the breakout."** Correct, as a BNC. PFI1 is
  on a spring terminal. Someone who thinks in BNCs concludes the line is
  unavailable when it is not.
- **The APFI BNC is a trap on this card.** The connector is present and
  labelled, and the PXI-6221 behind it has no analog trigger circuit. A
  perfectly reasonable wiring decision produces a signal that can never be
  read or triggered on.

The breakout's labels are generic across every card it can be attached to.
The card decides what each label means, and whether it means anything.

### Recommendation

**Card terms stay canonical** - the configuration keeps `PXI1Slot5/ai3` - and
the breakout becomes a presentation layer over it. One source of truth, no
translation that can drift from the hardware. The breakout model contributes
three things the card model cannot:

1. **What is physically reachable, and how.** "PFI0: BNC" versus "PFI1: spring
   terminal TB-2" is the difference between a five-minute job and a wrong
   conclusion.
2. **Which connectors are dead ends on this card.** The APFI BNC is present on
   the block and unusable on a 6221. The UI can say so before a cable is cut.
3. **The label printed on the thing being touched**, next to the terminal name
   the software uses.

### Change list

- **B1.** Add a breakout model: a named accessory with, per connector, its
  label, its connector type (BNC / spring terminal / screw), and the card
  terminal it lands on. Start with BNC-2090A and BNC-2110; the format should
  be data, not code, so another block is a file. *Done:
  `data/breakouts/*.yaml`, transcribed from NI 372101A-01 and 372121F-01,
  with each file's connector counts checked against the count its own
  specification declares - 22 BNC and 29 spring positions on the 2090A, 15
  and 30 on the 2110 - because a dropped row is how transcribing a panel by
  hand fails. Where a panel prints a group heading above bare numerals, the
  heading and the numeral are kept as separate fields rather than joined into
  a string nobody has seen printed.*
- **B2.** Name the attached breakout per device in `NidaqPortConfiguration`.
  Optional - a device with no breakout behaves as today. *Done, on
  `NidaqDeviceIdentity`, which is already the per-device record. Verified to
  survive the YAML round trip and verified that a configuration written
  before the field existed still loads with it unset, which is what both of
  this rig's identities currently do.*
- **B3.** Cross the breakout model with the capability probe from C1 to mark
  each connector reachable, unreachable-on-this-card, or already assigned.
  This is what catches the APFI case. *Done, and it does. Run against this
  chassis the APFI 0 BNC comes back unreachable on **both** boards - the
  PXI-6713 reports `anlg_trig_supported` False as well as the PXI-6221 - so
  that connector is a dead end on this rig whichever card it is patched to.
  The other question this rig lost time to now has an answer in software:
  laser 2's trigger route on PFI1 resolves to "PFI 1 (spring terminal,
  position 13)" rather than to nothing.*
- **B4.** In the channel picker, show each terminal as
  `PXI1Slot5/ai3 — BNC-2090A "AI 3" (BNC)`, grey out connectors the card
  cannot use with the reason, and warn when a configured terminal is not
  broken out at all. *Done, and the "not broken out" warning needed two
  corrections the rig supplied: a backplane line is not a connector the block
  is missing, and port1/line0 is the PFI 0 BNC under the card's other name.*
- **B5.** In the wiring verification report, print the breakout label beside
  the terminal, so "check ai4" becomes "check the AI 4 BNC". *Done, in one
  report shared by the headless run and the window.*

---

## Sequencing

The reviews assume this order, each stage leaving the system working:

1. **C1 + C2** - the capability probe and routing matrix. Everything else
   depends on them and neither changes behaviour on its own.
2. **D1** - port-based digital reads. Self-contained, removes an allocation
   and a dead capability check.
3. **C3 + C8** - validate and refuse, with the driver's own possible values in
   the message. This is the change that makes misconfiguration loud.
4. **D2** - retire the availability barrier, once C3 means the timing is
   trusted before the tasks start.
5. **B1-B3** - the breakout model and capability crossing.
6. **UI** - the DAQ monitor and wiring test as a new top-level option, with
   B4 and B5 folded in. *Done. Tools > DAQ Monitor, idle only. One tab per
   card: on this rig sixteen analog inputs and eight digital lines streaming
   at 2 kHz, twenty-four PFI and static lines polled for a level, the
   stimulus beneath them, and the wiring report as its own tab.*
7. **C4-C7, D3, D5-D7** - the remaining cleanups, in whatever order suits.

The UI deliberately comes after the breakout model: what it should display is
decided by B1-B3, and building it first would mean building it twice.


## What the monitor cost to get working

Three things it takes hardware to find, recorded because none is guessable
and each one produced a message pointing somewhere else.

- **DI tasks open in the GUI process break the stream worker's spawn.** It
  fails at `[Errno 14] Bad address` naming the Python executable. The same
  call with those tasks closed starts and streams, measured both ways. The
  static tasks are closed across the call that forks. No account of the
  NI-DAQmx runtime's part in it is offered here, because none was
  established - only that closing them is the difference.
- **The stream reaches running on the model's reader thread**, after
  `start()` has returned, so session events arrive off the GUI thread. They
  go through a queued signal now. The visible half of getting this wrong was
  traces running while the button still said "Start monitoring".
- **A Qt timer outlives the window it belongs to** unless it is killed in
  `closeEvent`. One firing after the session closed reopened the tasks
  `close()` had just released, and the process exited holding them.

Two more were the survey's and are recorded with it: only port0 is clockable
on an M Series board, and a PXI-6713 cannot clock digital input at all - it
refuses the DI maximum rate property, which is the driver saying so.


## What the monitor cost, part two

One more, and the worst of them, because every message it produced pointed
somewhere other than the cause.

**Once the NI-DAQmx runtime is resident in a Qt process, starting a child
fails.** A spawned child's exec dies at `[Errno 14] Bad address` naming the
Python executable; through multiprocessing that is invisible and surfaces
only as exit 255 with no result, reported against the exact-task preflight.
Measured in one process moments apart: a child starts before the laser loads
and fails after it. Qt alone is fine, the laser alone is fine, an open DAQmx
task alone is fine.

Two things work. For `subprocess`, passing an explicit environment - it makes
CPython build a fresh envp rather than hand the child the one NI has been in.
For `multiprocessing`, a forkserver, which must be started before NI-DAQmx
is, because starting it is itself an exec.

The application has been avoiding this by accident of ordering: it starts the
stream before it loads the laser. Any restart of the stream afterwards would
not have worked, and the monitor, opened later, hit it every time.


## D6, and why it is not in the code

The callback is real and on an isolated task it is dramatic. Six channels at
10 kHz in chunks of 500, twice each: **6.4% and 6.5% of a core** for a
blocking read against **1.6% and 2.0%** for
`register_every_n_samples_acquired_into_buffer_event`, for the same hundred
and twenty chunks. On that evidence it looks like the best item in the list.

It was then implemented in the stream - the callback setting an event that
`read_chunk` waits on, so the task graph and the alignment guarantee stay
exactly as they are - and measured again on the real ten-channel graph, three
runs of each alternating:

| | CPU (% of one core) |
|---|---|
| blocking read | 9.3 / 9.3 / 9.3 |
| buffer event | 10.0 / 9.2 / 9.1 |

Nothing. The saving the isolated probe shows does not exist in this stream,
which says the blocking read here is not burning CPU waiting - after D7 the
9.3% is the scaling and the ring, not the wait. A fully event-driven worker
would do the reads inside the callback instead of after it, but the version
measured here already sleeps through the wait and gains nothing, so there is
no reason to believe the larger rewrite would either.

So D6 is closed as not worth doing, on measurement rather than on estimate.
The change is reverted; what survives is the number.
