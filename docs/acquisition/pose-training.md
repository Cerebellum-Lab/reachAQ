# Training a live pose model

**Tools → Pose Training…** turns your own recorded sessions into the pose
model live inference runs. You label the parts of the mouse and the reach box
on a few hundred frames, train, check the result against the model you use
now, and make it live. Nothing leaves the computer, and no other software is
needed.

The window has four steps: **Frames**, **Label**, **Train** and **Evaluate**.
Work happens on a *dataset*, chosen at the top of the window.

## Datasets

- **New…** makes an empty dataset. You give it a name and the list of body
  parts, one per line, in the order you want to label them. The list starts
  filled with the parts of the current live model, which is what you want when
  you are improving it.
- **Import DeepLabCut…** copies the labelled frames of an existing
  single-animal DeepLabCut project. The project itself is only read, never
  changed. Frames with no labelled part are skipped.

Datasets are stored under `~/Autotrainer/pose_training/datasets/<name>/`:
frames as 256 × 256 grey images and the labels in `labels.csv`. Every change is
saved at once.

### Held-out sessions

Some sessions are *held out*: their frames are never trained on and are only
used to score the model. A session is one recording, so its left and right
cameras are always held out together; otherwise the model would be scored on
the same reach it was trained on. About one session in ten is held out when a
dataset is made. Change the choice in the **Train** step (checked = held out).

## 1. Frames

The list shows the camera videos of sessions under the output location. Pick
one to see it, then either:

- **Pick N diverse frames**: chooses frames that look different from each
  other (different poses, lighting, reach phases), which teaches the model
  more than consecutive frames. 20 per video is a good start.
- **Add this frame**: move the slider to a frame the model gets wrong and add
  it.

With **Pre-label new frames with the live model** checked, the live model
places the parts it is confident about on every new frame, so you correct
points instead of placing all of them. Pre-placed points are drawn hollow
until you accept the frame. Pre-labelling uses the graphics card, so it is
skipped while a session is recording.

## 2. Label

The left list shows the frames still to label (switch to *Labelled* or *All*
to revisit others). For each frame:

- **Click** to place the selected part; the next part is selected
  automatically, so you can click the parts in order.
- **Drag** a point to move it. **Right-click** a point, or press **Delete**,
  to clear it.
- **Leave a part empty when you cannot see it.** An empty part teaches the
  model that the part is absent in that view, which is what lets it report
  "not visible" live. A part you can see but leave empty teaches the wrong
  thing.
- **Done** (Enter) accepts the frame and opens the next one. Only accepted
  frames are trained on.

Keys: Up/Down change the part, PgUp/PgDn change the frame, the wheel zooms and
0 fits the frame to the window.

## 3. Train

**Start** trains a new model from every accepted frame outside the held-out
sessions. The default 300 epochs takes about 7 minutes for 336
frames on a Quadro T1000; it scales with the number of frames. The plot shows
the error on the held-out frames as training goes.

Training never runs while a session records. **Starting a recording pauses
training** before the cameras start; the run keeps its progress, and
**Resume** continues it once the session has stopped. A paused run cannot be
resumed after its labels or held-out sessions change, because the model would
no longer match what it says it was trained on; start a new run instead.

**Pause** and **Resume** also work by hand. **Cancel** stops the run and
discards it. Closing the window does not stop training.

Models are written under `~/Autotrainer/pose_training/models/<dataset>/<date-time>/`.
Each one records which dataset, which labels (as a fingerprint) and which
held-out sessions it was trained on.

## 4. Evaluate

Choose a trained model and press **Evaluate**. It is scored on the held-out
frames, side by side with the live model:

- **Median px, most confident 25 / 50 / 100%**: the typical distance from your
  labels, taking each model's most confident points first. Compare models at
  the same percentage; this is fairer than one overall error, because a model
  that stays silent on hard frames would otherwise look more accurate.
- **Detected**: the share of labelled points the model reports at the
  confidence live inference uses.
- **Per part**: parts marked *few labels* have fewer than five held-out labels,
  so their numbers are only a hint. Label more frames showing them.

If the live model was trained on frames from your held-out sessions, its
numbers are flattered; compare on sessions neither model trained on.

**Measure speed** times the model on this computer's graphics card for two
cameras. It must stay well under the 6.67 ms frame period of 150 Hz cameras.
Every model the window trains has the same network, so the speed hardly
varies; the check confirms this graphics card can keep up.

**Use as live model** makes the selected model live. It is used from the next
time live inference starts (System Mode set to Running). **Revert to previous live model** goes back to the
model it replaced. **Export…** copies the model folder, for example to another
rig, where it can be chosen in Preferences like any model.

## Measured on christielab10

2026-11-18, Quadro T1000, the imported Christie DeepLabCut project: 362
labelled frames from 57 sessions, 336 trained on, 26 held out (6 sessions).

| | trained here | K_full_s1 | N_full_p36 (YOLO) |
|---|---|---|---|
| median px, most confident 25 / 50 / 100% | 0.95 / 0.84 / 0.92 | 0.49 / 0.50 / 0.51 | 2.29 / 2.54 / 3.01 |
| detected | 99.2% | 100% | 100% |
| two-camera batch, p50 / p99 | 2.74 / 2.80 ms | 2.74 / 2.78 ms | |

Only the first column is a held-out score. K_full_s1 and N_full_p36 were
trained on all 26 of these frames, which is the case the warning above
describes. Training took 7.4 minutes, including one pause and resume (the
pause took 0.6 s). The speed is 42% of the 6.67 ms frame period.

On a card older than compute capability 8.0, such as the T1000, training runs
in full precision: there, reduced precision is emulated and was four times
slower.
