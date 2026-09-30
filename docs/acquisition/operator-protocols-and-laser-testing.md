# Operator guide: building protocols and testing lasers

Two walkthroughs. The first builds a protocol, reusable sets, and an experiment
composed from them. The second tests a laser, and says exactly what success and
each kind of failure look like.

Both live in the right-hand panel. That panel is cramped at its docked width, so
press **Expand** in its top-right corner first: it grows leftward over the
camera and behaviour panels and gives the 29-column table room. **Detach** puts
it in its own window for a second monitor. Both toggle back, and the application
reopens wherever you left it.

---

# Part 1: building protocols

## The pieces, and why there are three of them

- A **profile** is one reusable stimulus: a tone, a laser pulse train, a cue
  interval, an automatic shift policy. Trials refer to profiles by name.
- A **protocol** is an ordered list of trials, one row each. This is what a
  session runs.
- A **set** is a reusable group of trials, saved on its own. An **experiment**
  is an ordered list of set appearances that compiles into an ordinary protocol.

If you only ever run one fixed sequence, you need the first two. Sets and
experiments exist so a phase shared by several experiments is written once
instead of copied into every protocol.

## Step 1: create the profiles a trial will use

In the **Protocol** tab, the buttons along the top row open a short chain of
prompts and save into the profile library.

- **New tone** — profile id, frequency, duration.
- **New auto shift** — the automatic pellet-shift policy.

A laser pulse profile is not made here. Build it on the **Pulse Builder** tab
in **Laser Control** — see Part 2. A profile there is the pulse train and
nothing else; which laser fires it, and how, is chosen later, wherever the
profile is used — including in a protocol row, below.

## Step 2: create a protocol

**New** in the Session protocol row. Give it an identifier (lowercase letters,
numbers, `.`, `_`, `-`) and a display name. You get a table of trials, one row
each, all disabled.

**Duplicate** copies an existing protocol under a new id, which is usually
faster than starting blank. **Rename**, **Import**, **Export** and **Revert** do
what they say; Revert discards unsaved edits by reloading from disk.

## Step 3: fill in the trials

Every cell is typed, and the editor refuses a value the runtime could not
execute rather than accepting it and failing mid-session. Start with **Run**,
which is the enabled flag: a row left disabled is skipped.

The columns that matter most:

| Column | What it does |
| --- | --- |
| Run | whether this trial executes |
| Pellet cycle | the delivery behaviour |
| Position / Lane / X, Y, Z | where the pellet is presented |
| Cover | whether the pellet is covered or revealed |
| Tone profile / Tone phase | Tone 1, and when in the cycle it fires |
| Cue tone / Cue interval / Cue fixed | Tone 2 and its delay after Tone 1 |
| Laser profile / Laser / Laser phase / Laser route | which laser pulse, which laser fires it, when, and how it is triggered |
| Assignment / Stim % / Trigger | whether and how often the stimulus fires, and what triggers it |
| Retry | what happens on a failed attempt |

Picking a profile in **Laser profile** for a row with no laser action yet fills
in the rest of the action: **Laser phase** `pellet_presentation`, **Laser
route** `hardware_stim3`, and **Laser** the first configured laser that has
both a trigger terminal and a board STIM line (or the first laser, if none
qualifies). Change any of those cells afterward as needed. Clearing **Laser
profile** clears the whole action. **Laser** itself lists every configured
channel (`Laser 1`, `Laser 2`, ...) or `None`.

For anything repetitive, use the edit bar rather than typing each cell:

- **Copy row**, then select rows and **Paste to selected**.
- **Fill selected field** sets one column across a selection.
- **Create/update epoch** and **Create/update block** name a group of trials and
  apply a patch to all of them at once.
- **Undo** and **Redo** cover the whole table.
- **Preview random assignment** shows what a randomised stimulus assignment
  would produce before you commit to it.

Only future rows are editable. A row goes read-only and highlights while its
trial is active, and stays read-only once the trial completes, so a protocol
cannot be edited out from under a running session.

## Step 4: run it

Pick the protocol in **Session protocol**. That attaches it to the active
animal. The toolbar's **Protocol** selector at the top of the window is the
session-level choice between a protocol and **Manual pellet control**.

That is the whole loop for a single protocol. Everything below is for reusing
parts of one across several experiments.

## Step 5: save a protocol as a set

Select the protocol in Session protocol, then **Set from protocol** in the
sidebar. Give the set an identifier and a name. The set captures the trial
content: the defaults, the bulk overrides, and the individual trial overrides.

A protocol that uses epochs or blocks is refused here. Those are the older
within-protocol grouping and the experiment compiler writes epochs itself;
express the groups as bulk overrides, or split them into separate sets.

To revise a set later, select it and press **Open set**. That publishes it back
as an ordinary protocol you can edit in the table, and you capture it again when
you are done.

## Step 6: compose an experiment

**New experiment**, then build the entry table beneath it:

1. Choose a set in the dropdown and press **Add**. The row pins that set's
   current revision.
2. Set **x** to how many times that set repeats back to back.
3. Tick **Shuffle** to randomise trial order *within each appearance* of that
   set. The order of the sets themselves is never shuffled.
4. **Up** and **Down** reorder. Row order is run order.
5. **Remove** drops the selected row.
6. **Save experiment**.

Then **Compile and save**. That flattens the experiment into an ordinary
protocol, saved into the protocol library under the experiment's identifier, and
it appears in the Session protocol dropdown like any other.

### Things worth knowing about experiments

- **Revisions are pinned.** An entry remembers the set revision it was built
  against. If you later edit that set, the compile refuses and names the entry:
  *"Entry 2 pins set 'probe' revision 1, but the library holds revision 2"*. A
  row whose set has moved on shows both numbers, so you can see it before you
  compile. Re-add the row to pin the new revision deliberately.
- **Shuffling is reproducible.** The seed is stored on the entry. If you tick
  Shuffle without a seed, one is drawn at the first compile and written back, so
  the same experiment always compiles to the same order. Reordering rows does
  not reshuffle the trials.
- **Provenance survives.** Each set appearance becomes a named epoch in the
  compiled protocol, so the **Sources** column tells you which set each trial's
  values came from. A set used more than once is numbered `baseline-1`,
  `baseline-2`.
- **Recompiling replaces in place.** The compiled protocol keeps the
  experiment's identifier and its revision increments, so you do not accumulate
  copies.

---

# Part 2: testing a laser

## Before you start

Three things must be true, and each has its own failure message if it is not.

1. The laser backend is `nidaq` in the system configuration. At startup the log
   line *"laser not in use"* means it is disabled and every test will refuse.
2. The laser channel is mapped in **Edit DAQ Ports** — analog output, diode
   input, shutter output at minimum. The board STIM route also needs that
   channel's NI trigger terminal (`triggerSource`) and its `boardStimLine`, set
   in the system configuration next to the port mapping — not on the profile.
3. For Test stim, a saved laser profile, or a valid Pulse Builder draft. Any
   profile fits any laser: nothing about a profile ties it to one channel.

## Building a pulse train: the Pulse Builder tab

Open **Laser Control**. The first tab, before *Laser 1*, *Laser 2* and so on,
is **Pulse Builder** — every profile is made here, and nowhere else.

- **Profile:** lists *(new profile)* and every saved profile, as
  `profile_id — summary`. Selecting one loads its controls. **Delete** asks you
  to confirm, and is refused, naming the protocols, when one still uses the
  profile: *"Profile 'x' is used by: some-protocol"*.
- The controls shape the train. The **Pulse train** section holds
  **Amplitude**, **Pulse width**, **Baseline**, **Post-stim**, **Count** and
  **Frequency**, two to a row. The **PMT shutter margins** section holds **PMT
  open lead** and **PMT close lag**; it starts folded, so click its title to
  open it. The margins decide the PMT shutter wherever the profile fires, Run
  Pulse, Test stim and trials alike: with either above zero and a PMT shutter
  line (`pmtShutterOutput`) configured, the shutter opens that long before the
  train and closes that long after it. With no PMT line configured, as on
  christielab10, the margins are ignored and the train fires without them;
  the log says so once for each laser each time the laser controller opens.
  Amplitude here is limited to the widest range any configured laser
  accepts; the laser that actually fires the profile checks its own range. A
  saved profile outside that range loads clamped to it, and the status line
  says so: *"Profile 'hot' is 6 V; the builder allows 0..5 V, so it now shows
  5 V"*.
- The preview plot and the status line under it show the waveform when it is
  valid, or the reason it is not — for example a pulse width that exceeds the
  period at the chosen frequency.
- **Save profile…** asks for a name only. Reusing an existing profile's name
  saves a new revision of it. Nothing about which laser fires it, or how, is
  asked or stored here. The name `builder-draft` is reserved for the draft and
  refused.
- Whatever is on the controls, saved or not, is the **builder draft**. A laser
  tab can fire the draft directly, so you can shape a train and try it without
  saving a revision for every change.

## The two tests, and what each one proves

Open the **Laser Control** tab. There is one sub-tab per configured channel,
*Laser 1*, *Laser 2* and so on, each with **Pulse** and **Calibration** pages.
The Pulse page has **Profile:** at the top, then the sections **Run Pulse**,
**Test stim**, **Output stream** and **Board trigger**. Click a section's title
to fold it away or open it again. With every section open, the page fits the
docked right-hand panel without scrolling. Folding a section gives its height
to the output graph. A section folded on one laser is folded on every laser,
and stays folded when the tabs are rebuilt and the next time reachAQ starts;
so do the Pulse Builder's sections.

The line at the foot of Laser Control is the panel's own status line. It
shows, in red, each refusal and failure the panel reports. They go to the
main window's status bar too, so a detached panel shows them in its own
window as well. A red line stays until the panel reports something else,
such as the next press's outcome or the *Ready* line after a Run or Stop; it
does not time out. A long one ends in "…"; hover over it for the whole line,
and for a failed operation the whole error.

The **Output stream** graph on the Pulse page needs nothing started: it plots
whenever the NI-DAQ input stream runs, and that stream starts by itself, in
Idle as well as in System Mode. Its status line says whether it is running.
Under the graph, the **Signals** section starts folded. Open it, and **Command
output**, **Diode feedback**, **Command copy** and **Board trigger readback**
each show or hide their trace at once, at any time. An input's box is enabled
once the input is configured; until then it is greyed out and its tooltip says
what to set. Every configured input is recorded whether or not it is shown.
Whether each laser's Command output is shown is kept for the next start too.

The **Board trigger** section plots the board's STIM line read back on an NI
input, on its own axis under the output graph, so the edge that starts a
waveform lines up with the waveform. To see it:

1. Wire this laser's board STIM line (on christielab10, STIM3 for laser 1 and
   STIM2 for laser 2) into a spare **analog input** (`aiN`) or **port0 line**
   (`port0/lineN`) on the input card. Not a PFI terminal, and not a port1 or
   port2 line, which are the same PFI pins by another name (STIM3's
   `/PXI1Slot5/PFI0` is `PXI1Slot5/port1/line0`): the stream samples every
   digital input in one clocked task, and an M Series board clocks port0 only.
   Those are refused.
2. In Idle, open **Edit → Edit DAQ Ports**, pick the laser's tab, and choose
   that input in **trigger readback input**. It lists the analog inputs and
   port0 lines of the device selected as **Channel source** that no other role
   holds, and no lines at all from a board that cannot clock digital input,
   such as the PXI-6713. *(none)* turns the readback off: the channel is no
   longer acquired or recorded. Save.
3. The input is acquired and recorded as `laserN_trigger`, after every other
   input, as analog for `aiN` or digital for `port0/lineN`. Tick **Board
   trigger readback** under Signals to plot it.

The status line under the Board trigger graph says what it is doing: that no
input is set, that it reads the input once the NI-DAQ stream runs (or that
the stream failed, and how), that it is reading it and the box is unticked,
or that it is showing it. An input that
another role already uses - this or another laser's diode, command copy or
shutter, a tone or camera line, or the PMT shutter output - is refused, naming
both, in the dialog before it closes; the same check refuses such a
configuration when it loads. A refused save changes nothing.

On the **Pulse** page, **Profile:** picks what fires on this laser: *(none)*,
*(builder draft)* or any saved profile, unfiltered — a profile made with one
laser in mind fires just as well on another. A laser tab starts on *(none)*,
and Run Pulse and Test stim refuse until you pick something, so a laser never
fires whatever happens to be on the builder. The pick survives the tab
rebuilds that every system Run/Stop and Edit DAQ Ports save cause. If the
picked profile is deleted, or is otherwise no longer saved, the tab goes back
to *(none)* and reports *"Laser 2: profile 'burst' is no longer saved; pick a
profile"*. The summary line under the picker reads the picked
train, such as *"1 V · 500 × 1 ms at 100 Hz · 5.00 s"*, *"No profile
selected"*, or that the draft is not a valid pulse train.

### Run Pulse — proves the analog output works

In the **Run Pulse** section, set **Trigger:** and the shutter options, then
press **Run Pulse**. It has no PMT option of its own: the PMT shutter follows
the profile's PMT margins, as for a trial.
The host writes the picked profile's waveform to the analog output directly.

- **Success:** the status line reads *"Pulse complete: laser 1"*, and the
  trace shows the diode responding.
- **What it does not prove:** anything about the board. This path never touches
  the pellet board, so it cannot tell you whether a trial's trigger would work.

**Trigger:** starts at *internal*, which starts the pulse on the NI clock
when you press Run Pulse. With *external*, the output is armed and waits for
an edge, chosen by **Edge:**, on **Source:**. Nothing on this page sends that
edge, so use *external* only when an outside trigger is wired in. If no edge
arrives, the pulse fails after the train length plus five seconds with *Wait
Until Done did not indicate that the task was done*.

Run Pulse stays available while a session is recording, and a pulse fired then
is recorded in the session, in `streams/laser.csv`, as a manual laser event:
when it was asked for, the laser, the profile (its saved name, or *builder
draft*), the amplitude, the trigger mode and the route, then whether it
completed, failed or was refused. A refused pulse drove nothing and is recorded
as refused, never as fired. Its time is the host's, taken as Run Pulse calls
the laser controller, so the output starts at or after it. A pulse fired while
no session records is not recorded. Test stim stays refused while a session is
recording.

Stopping System Mode, or closing reachAQ, while a Run Pulse train is still
running closes the shutters first, then cancels the train. A cancel, this
one or a trial's, first closes the shutter of each of the pulse's lasers,
even for a pulse set to leave its shutter open, and whoever opened it: a
cancelled pulse leaves the laser safe, and a shutter open before it is closed
too. A pulse cancelled before it opens its shutter does not open it, and one
cancelled as it opens it closes it again at once. The cancel then aborts the
train's output, which ends the train at once: the train's own cleanup then
stops its digital lines and puts the command back to its minimum. Only a train
the abort does not wake, one waiting for its software start, has its lines
aborted by the cancel, after the output. On christielab10's 6221 an abort ended
a waiting task in about 40 ms, where a stop from another thread waited for the
whole task (H5b, H5a). On the 6713 the output was back at 0 V 24-40 ms after a
cancel; the abort itself takes 12.6-32 ms, and varies from run to run. A pulse
armed and waiting for its trigger ends the same way. The controller waits up
to five seconds for a train that does not end, as with a sick driver, before
it goes on without it. The status line
then reads *"Laser operation failed: Laser operation ... was cancelled: the
laser controller was closed while it ran"*. A pulse that failed by itself
just as Stop was pressed, such as one whose trigger never came, keeps its
own error; it is not reported as a failed close. If a pulse's, or a ramp's,
own reset of the command is refused, the output may still hold its last
level, and reachAQ logs a CRITICAL naming the laser, its analog output and
that level - the pulse's amplitude, or up to the ramp's highest command,
whichever end of the ramp that is: make the laser safe by hand. If the
controller's close then puts that laser's command back to its minimum after
all, a WARNING says so: *"Laser 1: close() put its command on PXI1Slot4/ao0
back to 0 V after all ..."*.

Run Pulse is refused while another pulse on the same board's analog output
is armed or running, or still ending after a cancel, such as a trial's pulse
waiting for its trigger, on this laser or on another laser of the same board:
*"Laser 2: refused while trial 7's pulse on laser 1 holds the analog output
of PXI1Slot4; wait for it to end, or cancel it. ..."*, with the operation's
id and output after. NI documents one timed analog output task per board at
a time. On christielab10 both lasers are on the 6713, so the two cannot run
overlapping pulses: the second is refused, naming what holds the board.

That close is bounded at 15 seconds, at Stop, at a Run start that failed,
and when reachAQ closes, and so is the close a controller makes of what it
had opened when opening it fails part-way, at a Run start or a ramp. A
driver that hangs in it no longer hangs Stop, the Run start or exit: past the
bound, or if the close fails, the log and status bar show a CRITICAL, *"The
laser controller did not close within 15.0 s. Each output may still hold its
last command (laser 1 0 V, laser 2 0 V), and the shutters may be open: make
the laser safe by hand. System Mode stops without it."*, and Stop, the Run
or exit goes on. While a close given up on is still inside the driver,
Run, Run Pulse, Test stim, the calibration ramp, Refresh Hardware (in Idle,
and its retry of a failed laser in System Mode), loading a configuration and
saving Edit DAQ Ports are refused with *"the laser controller is still
closing after a driver hang ..."*, and the laser's runtime status reads
failed with that reason: after Stop, after a Run start or a ramp whose
controller failed to open, after a ramp whose own close of its controller
hung, and through a hardware settings save. So is all
of it while a pulse train or ramp that the close stopped waiting for is
still running, since it can still drive the lines; that refusal reads *"a
laser operation the close gave up on is still running in the driver ..."*.
In Idle the NI-DAQ input stream stays stopped meanwhile, since a controller
still closing can hold inputs on the stream's lines, and its status reads
*"NI-DAQ input stream held back: ..."* with the same reason.
The refusal clears by itself when the close ends, and the log says so, and
in Idle the stream starts again by itself; in System Mode the laser then
stays failed until Refresh Hardware opens it again. While a close is still within its bound, the same work is refused with
*"the laser controller is closing; wait for it to finish"*.

Picking a profile in **Profile:** chooses what Run Pulse and Test stim fire on
this laser; this page has no waveform controls, and nothing is copied into it.
Picking *(builder draft)* fires whatever is on the Pulse Builder at the moment
you press the button. To change a saved profile's waveform, select it in the
Pulse Builder, which loads it there for editing, and save. Picking a profile
does not change **Trigger:** or **Source:**.

A pulse can also drive clocked digital lines with its waveform: the PMT
shutter (`pmtShutterOutput`), when its profile has PMT margins, and, per
laser, a trigger output and a timing trigger output (`triggerOutput`,
`timingTriggerOutput`). Run Pulse, Test stim and trials all use them. Each
must be a line on a board that can run clocked digital output - on
christielab10 a port0 line on the 6221, never the 6713 -
and every one runs on the output's own clock, with no start trigger; one
on another board than the laser's output takes that clock over the backplane,
on `pulseClockLine` (PXI_Trig3). A pulse that cannot clock such a line is
refused before anything fires, naming the line. christielab10 sets none of
these lines, and this path has not yet been run on its hardware.

### Test stim — proves the trial path

Pick a profile in **Profile:**. Then, in the **Test stim** section below Run
Pulse, pick a route in **Route:** and press **Test stim**. It is available
once the system is running, and refused while a session records.

**Route:** offers:

- **Board STIM (STIMn → terminal)** — built from this laser's own
  configuration: its board line (`boardStimLine`) and its trigger terminal
  (`triggerSource`; on christielab10, `/PXI1Slot4/PXI_Trig0` for laser 1 and
  `/PXI1Slot4/PXI_Trig2` for laser 2). Disabled, with the reason in its
  tooltip — *"This laser has no trigger terminal or board STIM line
  configured"* — when either is missing, and **Software start** is picked for
  you instead.
- **Software start** — the host starts the waveform itself, the same start a
  trial's stim-camera detector makes. The camera itself is not part of the
  test.

This runs the picked profile the way a trial does: arms the analog output,
then starts it by the chosen route.

- **Board STIM:** arms the output on the terminal, then asks the board for its
  timed pulse on the board line, which starts the waveform. Success reads
  *"Stim test stim-a on laser 1: STIM3 pulse 1000 us started the waveform on
  /PXI1Slot4/PXI_Trig0, arm to terminal 5003.42 ms"*, the interval running to
  the end of the waveform.
- **Software start:** arms the output, then starts it from the host. Success
  reads *"Stim test stim-s on laser 2: software start, waveform finished
  5002.10 ms after the start request"*.

The test waits for the whole waveform, so a 5 s burst takes about 5 s.

The definitive check for the board route is a scope: trigger on the STIM line
and confirm the analog output rises on that edge, not when you pressed the
button. The status line alone cannot distinguish the two.

### Calibration ramp — in Idle

On the laser's **Calibration** page, the **Calibration ramp** section steps
the command from **Start:** to **Stop:** in **Steps:** steps of
**Samples/step:** samples each, and reads the diode at every step. **PMT
shutter** holds the PMT shutter open for the ramp. Press **Run Ramp**.

- **Settle:** is how long the start of each step is left out of that step's
  point, in µs; the rest of the step is averaged, for the diode and the
  command copy alike. The default is 600 µs, 60 samples at christielab10's
  100 kHz, whatever **Samples/step:** is. What settles at each step is the
  laser and the input path together; the measurements cannot tell them
  apart. The input is read on the clock edge the command changes on, so
  without a settle the first samples of a step still read the step before,
  and pulled every point of a rising ramp low. At the ramp's start, with the
  command steady at 0 V, both diode inputs also read a false level of
  0.7-1.6 V that decays with a time constant of 93-120 µs: that is the input
  path settling, not the laser. **Settle:** must leave at least one sample of
  each step; Run Ramp refuses one that does not, naming both values.
- 600 µs and 5 ms steps are confirmed on christielab10: laser 2's diode
  settles within 270-610 µs of each step, and every point lands within 0.5%
  of its step. 600 µs has little margin, though: laser 2 needs 55-61 samples
  to come within 2%, against the 60 left out. At 5 ms steps a longer settle
  costs little, 440 samples still being averaged at 600 µs.
- **Samples/step:** defaults to 500, 5 ms at 100 kHz: christielab10's slower
  diode was still rising through the second half of a 1 ms step. For a slow
  diode, use longer steps rather than a longer settle alone, so that each
  point still averages enough settled samples. **Samples/step:** is taken
  when you finish typing (Return, leaving the field, or **Run Ramp**).
- A laser diode's monitor input needs a DC reference to AI GND. On a
  BNC-2090A that is the channel's AI x / AI x+8 switch on SE, with the
  RSE/NRSE switch on RSE, or a differential input with a bias resistor to AI
  GND. A floating input reads a level that depends on the other channels
  scanned around it: its offset, and the calibration's intercept, cannot be
  trusted, though its gain still can. christielab10's laser 1 diode input,
  ai8, is a known case: it floats, and at idle reads -0.16 to -0.2 V,
  depending on the channels scanned with it, in the ramp and in recordings
  alike.
- All of the ramp's fields go back to their defaults whenever Laser Control
  is rebuilt: on every Run and Stop, on a DAQ Ports save or a configuration
  load that changes the lasers, and when a laser close that ended late
  disconnects the laser.
  `tools/hardware/validate_laser_hardware.py --action ramp` takes the same
  settle as `--settle-us` (default 600) and the step as `--samples-per-step`
  (default 500).
- The laser controller's own inputs, the diode and the command copy for the
  ramp and for feedback, are referenced as the NI-DAQ stream references its
  inputs: by the stream's `analogTerminalConfig` (`rse` on christielab10).
  With none configured, DAQmx picks per channel, and on a PXI-6221 it made
  ai3, ai4 and ai5 differential, paired with ai11-ai13, where the stream
  reads them single-ended.
- Closing reachAQ while a ramp runs ends it with *"the laser calibration
  ramp was stopped: the laser controller was closed while it ran"*.

- It runs only in **Idle**, with the laser mapped. Otherwise the button is
  greyed out and its tooltip says why: System Mode is running or starting, a
  session is recording, the DAQ Monitor is open, a configuration is loading,
  or another laser operation is running. Run Pulse and Test stim, on the
  other hand, need System Mode running.
- For the ramp's length reachAQ pauses the NI-DAQ input stream, so every NI
  graph stops, and opens the laser controller for the ramp alone. Both are
  handed back when the ramp ends, including when it fails.
- While it runs, **Run**, **Edit DAQ Ports**, the **DAQ Monitor**, hardware
  refresh and Preferences are greyed out, and the application refuses them,
  and a configuration load, with the reason in the status bar.
- It needs `hardwareTimed: true` and a `sampleRateHz` in the laser
  configuration; christielab10 has both. Without them the button is greyed
  out and its tooltip names what is missing.
- Success reads *"Ramp complete: 11 points, last diode 1.234 V, monotonic
  curve validated"*, and the Output stream graph shows the whole ramp. A
  failure reads *"Laser operation failed: ..."* with the first line of the
  reason and, for a DAQmx error, its status code, as in *"(DAQmx -89125)"*, in
  red on the panel's status line and in the status bar; the whole error is in
  the log, and on hover over the panel's status line.
- Closing reachAQ during a ramp waits for the ramp to end, for as long as the
  ramp itself may take (its timeout) plus five seconds. Past that it closes
  the ramp's laser controller, before anything else closes. That close closes
  the shutters, aborts the ramp's tasks, waits up to five seconds for the
  ramp to let go of them, and then tries to put the command back to its
  minimum. The close fails if the ramp has not let go by then, since a start
  still inside the driver could drive the output after the reset, and the
  reset itself is refused if the ramp still holds the output.
- That forced close is itself bounded, at 15 seconds. A close that hangs
  inside the driver would not make the laser any safer, since its output
  stays driven either way, and it would keep reachAQ from exiting. If the
  close hangs, raises, or finds the ramp's own thread already closing the
  controller, the log and status bar show a CRITICAL naming the laser and the
  highest level the ramp commanded, *"The calibration controller for laser 1
  did not close within 15.0 s. Its analog output may still hold up to 5 V,
  the ramp's highest command (a 0 V to 5 V ramp) ..."*. Stopped part-way,
  the output holds a level between the ramp's ends, so a falling ramp names
  its start. When the ramp had not yet opened its controller, or had not
  started on it, the CRITICAL says so instead, and names no command.
- After the CRITICAL, closing waits up to two seconds more for the ramp to
  end, since a ramp its driver lets go tries to write the command back
  itself. The log then says either *"The laser calibration ramp ended; see
  above for any error from its own reset"* or that it had not ended, and
  reachAQ closes. The longest a close can take is the ramp's timeout + 5 s +
  15 s + 2 s.
- Make the laser safe by hand - switch off the laser driver or close its
  shutter at the rig - only when that CRITICAL appears and says the output
  may still hold up to the ramp's highest command.

## What failure looks like

Every refusal names its reason, in red on the Laser Control status line and in
the main window's status bar, and fires nothing. A refusal inside a Test stim
or other operation reads *"Laser operation failed: ..."* followed by the
message.

| Message | Meaning |
| --- | --- |
| *Laser N: pick a saved profile or the builder draft* | the Profile picker is on *(none)*. |
| *Laser N: the builder draft is not a valid pulse train; fix it in the Pulse Builder* | the Profile picker is on *(builder draft)* and the draft cannot be generated; the Pulse Builder's status line says why. |
| *Laser N: profile 'x' is no longer saved; pick a profile* | the picked profile was deleted; the picker has gone back to *(none)*. |
| *Stim test is refused while a session is recording* | stop the session first. |
| *Stim test is refused while a trial operation is prepared or active* | a trial is mid-flight. Wait for it. |
| *Stim test needs the nidaq laser backend; this rig is configured for 'disabled'* | the laser is off in the system configuration. |
| *Laser N has no trigger terminal; set it in Edit DAQ Ports* | Board STIM chosen, but this laser has no trigger terminal configured. Route already disables Board STIM in this case. |
| *Laser N has no board STIM line; set boardStimLine for it in the system configuration* | Board STIM chosen, but this laser has no `boardStimLine` set. Route already disables Board STIM in this case. |
| *Profile X is Y V; laser N accepts low..high V* | the profile's amplitude does not fit this laser's command range. Pick a different laser, or edit the profile in the Pulse Builder. |
| *Laser N has no hardware channel in the system configuration* | the channel is not configured in Edit DAQ Ports. |
| *The pellet firmware reports its capabilities and finite_stim3_pulse is not among them* | the board says it cannot pulse. |

## Firmware requirement

Test stim on the Board STIM route needs pellet firmware **v2.2.0 or later**
(Software start does not use the board). That is the first release to
implement the finite pulse command; every release through v2.1.0 has no
handler for it.

On v2.1.0 or earlier the board does not acknowledge the request, so the test
fails after about three seconds with a timeout naming the command token rather
than a laser result. It is not a fault in the laser, the NI card, the wiring or
the profile: the board has no such command. Check the board version in the
hardware panel, and use **Run Pulse** meanwhile, which proves everything except
the board trigger.

## What is still unproven, even on v2.2.0

Short pulses. The firmware times both edges from a hardware counter and the
long end measures within 170 us of a 500 ms request, but a 100 us pulse cannot
be resolved over CAN. Before trusting this path for optogenetics timing, put a
scope on the BNC and measure a 100 us and a 1 ms pulse.

The calibration ramp on christielab10. Its button could never be pressed
before 2026-09-25, so the ramp itself has not run on this rig: the command
comes from the PXI-6713 and the diode is read on the PXI-6221, clocked from
the 6713's output. That clock is driven onto the backplane line
`backplaneClockLine` (PXI_Trig1) and read there by the 6221, the way a pulse
train's shared clock reaches its output board, because DAQmx refuses to route
it between the boards by name on this chassis (-89125). The first press is the
check; if DAQmx refuses anything, the reason is in the status bar.
