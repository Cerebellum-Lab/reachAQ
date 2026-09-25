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
- The controls shape the train: **Amplitude**, **Pulse width**, **Baseline**,
  **Post-stim**, **Count**, **Frequency**, **PMT open lead**, **PMT close
  lag**. Amplitude here is limited to the widest range any configured laser
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
*Laser 1*, *Laser 2* and so on, each with **Pulse**, **Calibration** and
**Output** pages.

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

Set **Trigger Mode**, the shutter and PMT options, then press **Run Pulse**.
The host writes the picked profile's waveform to the analog output directly.

- **Success:** the status line reads *"Pulse complete: laser 1"*, and the
  trace shows the diode responding.
- **What it does not prove:** anything about the board. This path never touches
  the pellet board, so it cannot tell you whether a trial's trigger would work.

**Trigger Mode** starts at *internal*, which starts the pulse on the NI clock
when you press Run Pulse. With *external*, the output is armed and waits for
an edge on **Trigger Source**. Nothing on this page sends that edge, so use
*external* only when an outside trigger is wired in. If no edge arrives, the
pulse fails after the train length plus five seconds with *Wait Until Done did
not indicate that the task was done*.

Picking a profile in **Profile:** chooses what Run Pulse and Test stim fire on
this laser; this page has no waveform controls, and nothing is copied into it.
Picking *(builder draft)* fires whatever is on the Pulse Builder at the moment
you press the button. To change a saved profile's waveform, select it in the
Pulse Builder, which loads it there for editing, and save. Picking a profile
does not change Trigger Mode or Trigger Source.

### Test stim — proves the trial path

Below Run Pulse, pick a profile in **Profile:**, pick a route in **Route:**,
and press **Test stim**. It is available once the system is running, and
refused while a session records.

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

## What failure looks like

Every refusal names its reason in the status line and fires nothing.

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
