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

In the **Protocol** tab, the buttons along the top row. Each opens a short chain
of prompts and saves into the profile library.

- **New tone** — profile id, frequency, duration.
- **New laser** — profile id, laser channel, amplitude, pulse duration, pulse
  count, and if the count is above one, frequency. Then the trigger route:
  - **Hardware STIM3** — the board emits a timed pulse that starts an NI analog
    output already armed on a trigger terminal. You then pick the NI trigger
    terminal and the **board stimulus line**, STIM2 or STIM3.
  - **Direct NI software start** — the host starts the waveform itself. No board
    trigger, so no terminal to pick.
- **New auto shift** — the automatic pellet-shift policy.

A laser channel must exist before you can make a laser profile. If the button
refuses with *"Configure at least one laser channel in Edit DAQ Ports first"*,
set the channels up there. Same for the trigger terminal: hardware-route
profiles need a trigger input configured in Edit DAQ Ports.

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
| Laser profile / Laser phase / Laser route | which laser pulse, when, and how it is triggered |
| Assignment / Stim % / Trigger | whether and how often the stimulus fires, and what triggers it |
| Retry | what happens on a failed attempt |

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
   input, shutter output at minimum.
3. For a hardware-triggered test, the channel needs an NI trigger terminal and a
   saved laser profile on the **Hardware STIM3** route.

## The two tests, and what each one proves

Open the **Laser Control** tab. There is one sub-tab per configured channel,
*Laser 1*, *Laser 2* and so on, each with **Pulse**, **Calibration** and
**Output** pages.

### Run Pulse — proves the analog output works

On the **Pulse** page, set amplitude, duration, count, frequency and the shutter
options, then press **Run Pulse**. The host writes the waveform to the analog
output directly.

- **Success:** the status line reads *"Pulse complete: laser 1"*, the preview
  plot matches what you asked for, and the trace shows the diode responding.
- **What it does not prove:** anything about the board. This path never touches
  the pellet board, so it cannot tell you whether a trial's trigger would work.

### Test stim (hardware trigger) — proves the whole trial path

Below Run Pulse, pick a saved profile in **Stim profile** and press **Test stim
(hardware trigger)**. The selector lists only profiles targeting this channel.

This does what a trial does: arms the analog output on the profile's trigger
terminal, then asks the board for its timed pulse on the profile's stimulus
line, which is what starts the waveform.

- **Success:** the status line reads
  *"Stim test stim-a on laser 1: STIM3 pulse 1000 us started the waveform on
  /Dev1/PFI0, arm to terminal 3.42 ms"*. The laser fires. The measured interval
  is stable across repeated presses.
- **The definitive check** is a scope: trigger on the STIM line and confirm the
  analog output rises on that edge, not when you pressed the button. The status
  line alone cannot distinguish the two.

## What failure looks like

Every refusal names its reason in the status line and fires nothing.

| Message | Meaning |
| --- | --- |
| *Select a saved laser profile for laser N first* | the Stim profile selector is empty or unset. No profile targets this channel: make one with **New laser**. |
| *Stim test is refused while a session is recording* | stop the session first. |
| *Stim test is refused while a trial operation is prepared or active* | a trial is mid-flight. Wait for it. |
| *Stim test needs the nidaq laser backend; this rig is configured for 'disabled'* | the laser is off in the system configuration. |
| *Profile X uses the direct_ni_software route; the bench test drives the hardware STIM3 route only* | that profile starts the waveform from the host. Use **Run Pulse** for it, or make a hardware-route profile. |
| *Profile X has no NI trigger terminal, so the board pulse has nothing to trigger* | set the terminal in Edit DAQ Ports and re-create the profile. |
| *Laser channel N has no hardware mapping* | the channel is not configured in Edit DAQ Ports. |
| *The pellet firmware reports its capabilities and finite_stim3_pulse is not among them* | the board says it cannot pulse. |

## The failure you will actually see today

**The pellet firmware does not implement the pulse command.** Until it does, a
hardware stim test will arm the output, send the request, be acknowledged, and
then report:

> *Stim test stim-a on laser 1 via STIM3: board acknowledged but the waveform
> did not report terminal*

with no measured interval. That is not a fault in the laser, the NI card, the
wiring or the profile. The board accepts the CAN frame, has no handler for it,
and drops it, so the trigger edge never arrives and the armed output waits until
it times out.

The same is true of any trial configured with the hardware route: it does not
trigger the board today, and until recently did so silently. See
`pellet-firmware-gpio-pulse.md` for the evidence and the implementation
specification.

Use **Run Pulse** to check the laser and the analog output in the meantime. It
proves everything except the board trigger.
