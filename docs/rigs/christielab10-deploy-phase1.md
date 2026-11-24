# christielab10: Phase 1 deploy runbook

This runbook moves christielab10 from `devel` (pellet firmware 2.3.1, `c5c7118d`,
config version 58) to the Phase 1 `feature-dev`. That release includes the clean-room
API and training plans, the keypoint pose backend, pose backend plugins, in-product
pose training, laser process isolation, simulation mode, config hygiene, rig de-coupling,
pluggable animal metadata sources and the NI delivery calibration (config version 59).
Do it between studies, with Ben at the rig or on call.

Run every command as the operator `christielab10`, from `~/Documents/reachAQ`. Steps 1
to 8 need reachAQ closed. Every step lists how to check it and how to undo it. To roll
back the whole deploy, see [Rolling back](#rolling-back).

On 2026-11-19 a dry run tested `feature-dev` `f21b0dfe` from a scratch clone, a clone of
the operator env and scratch settings directories. It covered:
- the step 1 backup commands;
- step 3;
- step 4 on the cloned env;
- the step 5 suite;
- building the step 6 config;
- the step 7 preferences keys;
- the registry migration and the old build's refusal of it;
- the env half of the rollback;
- a `--simulate` session;
- a real-camera Start/Record/Stop/exit cycle with CAN, pellet, NI and laser off.

Steps 2, 8 and 9, the systemd drop-in, and anything with CAN, pellet, NI or the laser
on were not part of it. Its report is kept outside the repository.

## What the deploy changes

| Area | Before | After |
|---|---|---|
| Code | `devel` `c5c7118d` | the released `feature-dev` |
| System config | `version: 58`, no delivery calibration | `version: 59` plus `nidaqStream.deliveryCalibration`; the new code refuses 58 and the old code refuses 59 |
| Operator env | `auto-trainer-api` 0.11.0 and `auto-trainer-training` 0.9.27 (AGPL), with deeplabcut and ultralytics installed directly | both AGPL packages removed; `auto-trainer-pose-deeplabcut` and `auto-trainer-pose-yolo` plugin distributions; scipy, scikit-learn and PyTables declared |
| Live pose model | `~/yolo_models/N_full_p36` (YOLO) | unchanged. It now runs through the YOLO plugin, which the installer restores because ultralytics is installed |
| Preferences | no `[softmouse]` section, so the built-in default applies | `[softmouse] manifest_path` and `publication_mount` |
| SoftMouse publisher | colony built in; the unit file has `RequiresMountsFor` | colony set up once; `RequiresMountsFor` moves to a drop-in |
| Animal registry | schema 1 | schema 2, migrated in place on first open; the old code refuses it |
| Config writes | every exit saves the config | only operator edits save it |

## 0. Before you start

1. Confirm that no study is running and none is due before you would finish a rollback.
2. Confirm that nothing reachAQ is running. Read the list by eye. Never kill by pattern;
   kill by PID only.

   ```bash
   ps -eo pid,etime,args | grep "[e]nvs/reachaq"
   ```

3. Take the rig lock for the whole deploy, so that other sessions' suites and recordings
   wait. Open a shell that holds it:

   ```bash
   flock /tmp/reachaq-rig-exclusive.lock bash
   ```

4. Have the release ready: `feature-dev` merged into `develop` with an annotated tag,
   and `origin/devel` fast-forwarded to it. This is the same path as v3.0.1. Pushing and
   tagging are Ben's call.

**Check:** the `ps` list is empty and the `flock` shell has returned a prompt.

## 1. Back up

```bash
B=~/deploy-backups/$(date +%Y%m%d)-phase1
mkdir -p $B
git -C ~/Documents/reachAQ rev-parse HEAD > $B/operator-checkout-rev.txt
tar -C ~ -czf $B/Autotrainer-settings.tgz \
    --exclude=Autotrainer/JetNet1-christie-2024-11-18 \
    --exclude=Autotrainer/mouseGYM-christie-2025-09-18 \
    --exclude=Autotrainer/demo Autotrainer
tar -C ~ -czf $B/config-Colorado-reachaq-systemd.tgz \
    .config/Colorado .config/reachaq .config/systemd/user
cp -p ~/Autotrainer/softmouse/animal-registry.sqlite3 $B/
cp -p ~/Autotrainer/system_configuration.yaml $B/system_configuration.v58.yaml
cp -p /tmp/reachaq_delivery_cal/live_v59.yaml $B/live_v59_with_delivery_calibration.yaml
~/anaconda3/bin/conda env config vars list -n reachaq > $B/reachaq-env-config-vars.txt
~/anaconda3/envs/reachaq/bin/python -m pip freeze > $B/reachaq-pip-freeze.txt
~/anaconda3/bin/conda create -y -q --clone reachaq -n reachaq-v301-$(date +%Y%m%d) --offline
```

- Later steps and the rollback use `$B`. In a new shell, set it again to the same
  directory.
- The `~/Autotrainer` tarball leaves out the two DLC model directories and `demo`
  (5.3 GB together), because the deploy does not touch them. In the dry run the
  tarball was 40 MB (486 entries) and took 1 s.
- `live_v59.yaml` lives in `/tmp`, which a reboot clears. It is the live config
  with `version: 59` and the calibration measured on 2026-11-13; the rest of the
  file matches. Copy it now, even if step 6 re-measures.
- Cloning the env took 24 s in the dry run. Disk use rises by much less than its
  14 GB, because conda hard-links its packages. `conda create --clone` **does
  not copy the env's config variables**, which hold TensorFlow's
  `LD_LIBRARY_PATH`. That is why they are saved to a file here.

**Check:** `ls -la $B` shows eight files and `conda env list` lists
`reachaq-v301-<date>`. Run `tar -tzf $B/Autotrainer-settings.tgz | grep
system_configuration.yaml$`.

**Undo:** nothing to undo. Keep `$B` until the new build has run a study.

## 2. Update the operator checkout

```bash
cd ~/Documents/reachAQ
git status --short          # must be empty
git fetch origin
git pull --ff-only
git describe --tags
git lfs checkout
```

**Check:** `git status --short` is empty, `git log -1` is the release merge, and
`git describe` names the new tag.

**Undo:** `git checkout devel && git reset --hard $(cat $B/operator-checkout-rev.txt)`.
Use this only as part of the full rollback, because the env and config must go back
with it.

## 3. Remove the AGPL api and training packages

```bash
~/anaconda3/bin/conda run -n reachaq python -m pip uninstall -y auto-trainer-api auto-trainer-training
```

reachAQ no longer imports them, but the installer does not remove them.

**Check:** `conda run -n reachaq python -m pip show auto-trainer-api auto-trainer-training`
reports both as not found.

**Undo:** restore the env clone (see [Rolling back](#rolling-back)).

## 4. Rerun the installer (the update path)

```bash
cd ~/Documents/reachAQ
REACHAQ_INSTALL_SYSTEM=0 bash tools/install/reachaq-linux-install.sh 2>&1 | tee ~/deploy-backups/install-$(date +%Y%m%d).log
```

Call the installer through `bash`. Between `d9ebf46e` and `39c3d826`, `feature-dev` had
the script without its executable bit. On such a release, `./tools/install/...`
fails with "Permission denied", and the installer's own focused test
`test_portable_installer_is_executable_and_rejects_options` fails.

`REACHAQ_INSTALL_SYSTEM=0` skips the steps that need root (apt, groups, real-time
limits, the CPU governor unit). They are already in place on the rig, and the
deploy adds no new ones. The run installs the new dependency set into the existing
env. It restores the **DeepLabCut** plugin, because deeplabcut is installed, and the
**YOLO** plugin, because ultralytics is installed. That rewrites the env's TensorFlow
`LD_LIBRARY_PATH`, rewrites the launcher and terminal commands with the same values,
and runs the focused tests.

The YOLO plugin is AGPL-3.0. It is restored because the live model is still the
YOLO `N_full_p36`, and without it that model is refused at start-up. It goes away in
step 10, after the switch to `K_full_s1`.

**Check:**
- The report ends with 0 FAIL. The dry run ran on a clone of the operator env,
  from `f21b0dfe`, and gave 31 PASS, 5 SKIP (the root steps) and 1 FAIL. The FAIL
  was the executable-bit test above, which passes once the bit is back. The run
  took 1 min 43 s.
- `Verify Python dependencies` prints two TensorFlow `pip check` complaints (numpy
  1.26.4, typing-extensions 4.16.0) and still passes. The operator env prints the
  same two today, so they are not new.
- `Restore deeplabcut pose plugin` and `Restore yolo pose plugin` are both PASS.
- `conda run -n reachaq python -m pip list | grep auto-trainer` shows
  `auto-trainer-pose-deeplabcut` and `auto-trainer-pose-yolo`.
- `cat ~/.config/reachaq/launcher.conf` still names `CONDA_ENV=reachaq`,
  `REPOSITORY=~/Documents/reachAQ` and `SYSTEM_CONFIG=~/Autotrainer/system_configuration.yaml`.

**Undo:** restore the env clone (see [Rolling back](#rolling-back)).

## 5. Run the suite from the operator checkout

Run it in the foreground and name the six directories, so the count compares with the
dry run:

```bash
cd ~/Documents/reachAQ
QT_QPA_PLATFORM=offscreen ~/anaconda3/bin/conda run --no-capture-output -n reachaq \
  python -m pytest -q -p no:cacheprovider \
  auto-trainer-core/tests auto-trainer-video/tests auto-trainer-inference/tests \
  auto-trainer-device/tests tests tools/acquisition/tests
```

**Check:** compare with the dry run: 4649 passed, 40 skipped and 1 failed in 9 min 39 s.
The failure was
`tests/backend_conformance_test.py::test_camera_backend_contract[spin_cam]`, a test
defect that shows only where the Spinnaker SDK is installed. The fixture fakes PySpin
only when PySpin is missing, so the real `PySpin.CCommandPtr` receives the test's
fake node, and `latch_clock()` returns None. Real cameras latch normally. Treat any
other failure as new.

## 6. Move the live config to version 59, with the delivery calibration

This must happen in the same window as the code: the new code refuses version 58,
and v3.0.1 refuses 59. Choose one of the two options.

**Option A: reuse the 2026-11-13 calibration** (D̄ 0.277 ms, σ 0.029 ms, 18
sessions). This is correct while the NI board, the reach cameras and the host are
unchanged since 2026-11-13.

```bash
cp -p ~/Autotrainer/system_configuration.yaml ~/Autotrainer/system_configuration.yaml.bak-before-v59
diff <(sed 's/^version: 59$/version: 58/' $B/live_v59_with_delivery_calibration.yaml) \
     ~/Autotrainer/system_configuration.yaml
```

The diff must show only the added `deliveryCalibration:` block, lines 291-312. If it
shows anything else, the live config changed after 2026-11-13. In that case, bump the
version line by hand and use option B instead.

```bash
cp $B/live_v59_with_delivery_calibration.yaml ~/Autotrainer/system_configuration.yaml
```

**Option B: re-measure.** Bump the version, then run the loopback tool. It takes about
50 minutes, needs the BNC jumper from 6221 AO 0 to AI 6 (BNC-2090A, AI 6 at SE), and
backs the config up before it writes:

```bash
cp -p ~/Autotrainer/system_configuration.yaml ~/Autotrainer/system_configuration.yaml.bak-before-v59
sed -i 's/^version: 58$/version: 59/' ~/Autotrainer/system_configuration.yaml
conda run --no-capture-output -n reachaq python -m tools.hardware.calibrate_ni_delivery \
    --ao PXI1Slot5/ao0 --ai PXI1Slot5/ai6 --write
```

**Check (both options):**

```bash
grep -n '^version' ~/Autotrainer/system_configuration.yaml      # version: 59
~/anaconda3/bin/conda run -n reachaq python -c "
from pathlib import Path
from autotrainer.core.configuration import SystemConfiguration
c = SystemConfiguration.load_yaml_file(Path.home() / 'Autotrainer/system_configuration.yaml')
print(c.version, c.nidaq_stream.delivery_calibration)"
```

The script prints `59` and a calibration object, not `None`.

**Undo:** `cp -p $B/system_configuration.v58.yaml ~/Autotrainer/system_configuration.yaml`.
Do this only together with the code rollback.

## 7. SoftMouse preferences and the publisher drop-in

Add the `[softmouse]` section to the operator's preferences with reachAQ closed. The
file has no such section today, so append it:

```bash
cp -p ~/.config/Colorado/"Auto Trainer.conf" ~/.config/Colorado/"Auto Trainer.conf.bak-before-phase1"
grep -q '^\[softmouse\]' ~/.config/Colorado/"Auto Trainer.conf" || printf '\n[softmouse]\nmanifest_path=/mnt/isilon/Data/ReachingData/SoftMouse/SoftMouse-AnimalList-current.manifest.json\npublication_mount=/mnt/isilon\n' >> ~/.config/Colorado/"Auto Trainer.conf"
```

The publisher unit no longer carries `RequiresMountsFor`. Restore it with a drop-in, then
copy the new unit:

```bash
mkdir -p ~/.config/systemd/user/reachaq-softmouse-publisher.service.d
printf '[Unit]\nRequiresMountsFor=/mnt/isilon\n' \
  > ~/.config/systemd/user/reachaq-softmouse-publisher.service.d/isilon.conf
cp tools/softmouse_sync/systemd/reachaq-softmouse-publisher.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

**Check:**
- `grep -A2 '^\[softmouse\]' ~/.config/Colorado/"Auto Trainer.conf"` shows both keys.
- `systemctl --user cat reachaq-softmouse-publisher.service | grep RequiresMountsFor` prints
  the drop-in's line.
- `systemctl --user list-timers | grep softmouse` still lists the nightly timer.

**Undo:** restore `Auto Trainer.conf.bak-before-phase1`. Delete the `.service.d`
directory, restore the old unit from the step-1 tarball, then run
`systemctl --user daemon-reload`.

## 8. Set up the SoftMouse source on the publisher (once, interactive)

christielab10 runs the nightly publisher timer, so it needs this step. Ben does it,
because it asks for the colony and can take the SoftMouse login:

```bash
cd ~/Documents/reachAQ
conda activate reachaq
reachaq-animal-metadata setup --kind softmouse
```

The username is the short SoftMouse login, `benjamin.g.reynolds`, not the email:
that is the account the old publisher stored the password under, so the password
prompt can then be left empty. Choose the lab's colony. For the columns, choose
**Physical tag** = `Physical Tag`, **Name** = `Physical Tag` (or the column whose
value a new animal should be named after) and **RFID** = `Alt. ID`, the lab's
choice: it accepts RFIDs, barcodes or any other unique code. A value two animals
share (`201B` on E978 and E980 today) is ignored for both and reported. Every other
column is kept in the metadata without being asked.
**Publish to** `/mnt/isilon/Data/ReachingData/SoftMouse` and **Only if mounted**
`/mnt/isilon` are paths, not yes/no answers. Until this step is done, the
nightly run stops with "This computer reads a shared publication, so it has nothing
to publish". The app itself keeps reading the existing publication.

`~/.config/reachaq/softmouse-publisher.json` is the old publisher's file. The new code
does not read it. Its `publicationDirectory` (`/mnt/isilon/Data/BR/SoftMouse`) is stale:
the 2026-11-19 run published to `ReachingData`. Leave the file alone; it is harmless.

**Check:** `systemctl --user start reachaq-softmouse-publisher.service`, then
`journalctl --user -u reachaq-softmouse-publisher -n 20`. The log should end in
"SoftMouse publication complete", and `ls -la
/mnt/isilon/Data/ReachingData/SoftMouse` should show a fresh
`SoftMouse-AnimalList-current.csv`.

**Undo:** the setup only writes the `animal_metadata/source` preference and the
keyring entry. Restoring `Auto Trainer.conf.bak-before-phase1` removes the first one.

## 9. First start and a check session (reachAQ open)

1. Start reachAQ from the desktop icon or with `reachaq`. Confirm the title or About
   shows the new version, and that the live model loads. The log says the YOLO plugin
   runs `N_full_p36`.
2. Check that the animal list loads. The first open migrates the registry to schema 2.
   From here, rolling back needs the step-1 copy of `animal-registry.sqlite3`.
3. Connect CAN and the pellet board. The pellet board must qualify (firmware 2.3.1).
4. Record a short check session with NI on. Its quick validation should pass, and
   `streams/alignment.json` should rate the camera latch `hardware_edge` rather
   than `ambiguous_hardware_edge`, which is the delivery calibration working.
5. Laser: the NI laser now runs in its own process. READY should show the laser
   READY, with no "pulses and the ramp are refused" reason (`hardwareTimed` is `true`
   in the live config). Run one Run Pulse into the dump or a meter, as for any laser
   change.
6. Exit, then check that the exit wrote nothing:
   `ls -la --time-style=full-iso ~/Autotrainer/system_configuration.yaml` should show
   an unchanged time.

**Undo:** any failure here goes to [Rolling back](#rolling-back).

## 10. Later, separately: the keypoint model

These steps are not part of the deploy window. Do them only after Ben's live check of
`~/keypoint_models/K_full_s1` on a rig that sees a real reach box.

- In Preferences, set the inference model to `~/keypoint_models/K_full_s1`, or use
  Tools → Pose Training… → Deploy. The keypoint model reports the pellet absent when
  the paw hides it or it is gone, where YOLO reported it present. Check the
  pellet-cycle logic on a real reach.
- Once no live model needs YOLO, remove the AGPL plugin for good. Uninstall both the
  distribution and its engine, otherwise the next installer run restores it:
  `conda run -n reachaq python -m pip uninstall -y auto-trainer-pose-yolo ultralytics ultralytics-thop`.
- The dataset directory the live DLC model's `test/pose_cfg.yaml` points into stays.

## Rolling back

Roll back everything together. The old code cannot read the version 59 config or the
schema-2 registry.

```bash
# reachAQ closed; B is the step-1 backup directory
cd ~/Documents/reachAQ && git checkout devel && git reset --hard $(cat $B/operator-checkout-rev.txt)
~/anaconda3/bin/conda rename -n reachaq reachaq-phase1-failed
~/anaconda3/bin/conda rename -n reachaq-v301-<date> reachaq
# re-apply the env config variables the clone did not carry:
~/anaconda3/bin/conda env config vars set -n reachaq "$(sed -n 's/^LD_LIBRARY_PATH = /LD_LIBRARY_PATH=/p' $B/reachaq-env-config-vars.txt)"
cp -p $B/system_configuration.v58.yaml ~/Autotrainer/system_configuration.yaml
cp -p $B/animal-registry.sqlite3 ~/Autotrainer/softmouse/animal-registry.sqlite3
cp -p ~/.config/Colorado/"Auto Trainer.conf.bak-before-phase1" ~/.config/Colorado/"Auto Trainer.conf"
rm -r ~/.config/systemd/user/reachaq-softmouse-publisher.service.d
tar -C ~ -xzf $B/config-Colorado-reachaq-systemd.tgz .config/systemd/user/reachaq-softmouse-publisher.service
systemctl --user daemon-reload
```

- **Env:** the editable install in the restored clone still points at
  `~/Documents/reachAQ`, which is back on the old revision, so no reinstall is needed.
  The version comes from the checkout. `conda run -n reachaq python -c "import
  tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` must list the
  GPU. If it does not, the config variables were not re-applied.
- **Registry:** if the step-1 copy is missing, delete
  `~/Autotrainer/softmouse/animal-registry.sqlite3`. It is a cache, and the old build
  rebuilds it on its next refresh.
- **Animals and sessions** recorded on the new build stay readable by the old one. The
  event ids are unchanged, and the session files only gained fields.
- **Check:** `git describe` is the old tag, the app starts, and the animal list loads.
  The quick validation of a short session passes.
- Delete `reachaq-phase1-failed` after looking at why it failed.

## Clean-up after a successful deploy

After the new build has run a study:
- `conda env remove -n reachaq-v301-<date>`;
- `~/deploy-backups/<date>-phase1`;
- the rig scratch left by the Phase 1 sessions, which is Ben's call:
  - `~/pose-training-1p7` and `~/pt-scratch` (keep the model
    `~/pt-scratch/models/christie/20261118-161624` until it has been reviewed);
  - `~/pt_accept.py`, `~/pt_accept.log`, `~/pt_bench.py`, `~/pt_bench.log` and `~/pt-eval.log`;
  - `~/cleanroom-api-training` and the `*.bundle` files in `~`.

Keep the live DLC model's dataset directory (the path in its `test/pose_cfg.yaml`) and
`~/reachaq_probe` (see the rig clean-up keep list). `~/p1val` was already removed on
2026-11-19.

## Addendum: deploying v4.0.0 (the rename)

This runbook describes the v3.1 deploy and is kept as it was. v4.0.0 renames the
project (see "Upgrading from v3" in [INSTALL.md](../../INSTALL.md)). On
christielab10, as the operator, nothing needing root (the installer's root steps
are already in place, hence `REACHAQ_INSTALL_SYSTEM=0`):

1. Close reachAQ (GUI and any headless run; stop by PID, never by pattern).
2. Stop the SoftMouse publisher. Its installed unit runs
   `python -m tools.acquisition.model.animal_sources.cli`, which v4 no longer has,
   until the installer replaces it:

   ```bash
   systemctl --user stop reachaq-softmouse-publisher.timer reachaq-softmouse-publisher.service
   ```
3. Back up the settings:
   `tar czf ~/autotrainer-settings-$(date +%Y%m%d-%H%M).tgz -C ~ Autotrainer .config/Colorado .config/reachaq .config/systemd/user`
4. `cd ~/Documents/reachAQ`, check `git status --short` is empty, then `reachaq-sync`.
5. Dry run of the settings move, which writes nothing:
   `python3 src/reachaq/core/settings_migration.py --dry-run`. On 2026-11-24 it planned:
   - the rename `~/Autotrainer` -> `~/reachAQ`;
   - 6 rewrites: `demo/demo_sources.yaml`, the mouseGYM DeepLabCut `config.yaml` and its
     `dlc-models-pytorch/.../train/pytorch_config.yaml`, and the three versioned
     `system_configuration_v5x_*.yaml` snapshots;
   - 1 copy: `~/.config/Colorado/Auto Trainer.conf` -> `~/.config/reachAQ/reachAQ.conf`;
   - the rewrite of `~/.config/reachaq/launcher.conf` (`SYSTEM_CONFIG` named `~/Autotrainer`).

   Stop if it plans anything else. The JetNet1 project names `/home/agx004/Autotrainer`
   (another machine) and is left alone.
6. `REACHAQ_INSTALL_SYSTEM=0 ./tools/install/reachaq-linux-install.sh`. It moves the
   settings first, uninstalls every `auto-trainer*` distribution (including
   `auto-trainer-api` and `auto-trainer-training`), installs `reachaq` and the plugins
   the environment had, regenerates the launchers, the desktop entry and
   `launcher.conf`, and replaces the installed SoftMouse unit (kept as
   `reachaq-softmouse-publisher.service.bak`; its drop-in stays). It must end with no
   FAIL; it is safe to rerun.
7. Check:
   - only `~/reachAQ` exists;
   - `grep -c Autotrainer ~/.config/reachaq/launcher.conf` gives 0;
   - `conda run -n reachaq python -m pip list | grep -iE "auto.?trainer"` is empty;
   - `grep ExecStart ~/.config/systemd/user/reachaq-softmouse-publisher.service` names
     `reachaq.app.model.animal_sources.cli`.
8. `systemctl --user start reachaq-softmouse-publisher.timer`, then start reachAQ from the
   desktop entry (title `christielab10 - reachAQ - Acquisition v4.0.0`) and record a
   short check session.

`~/Documents/rawdatalocal` holds 2026 sessions only, so `reachaq-convert-pose-h5` is not
needed there.

### Rollback to v3.2.2 (the last v3 release)

With reachAQ closed and the publisher stopped (step 2):

1. `reachaq-migrate-settings --reverse`, while v4 is still installed. It restores
   `~/Autotrainer`, every rewritten file and `launcher.conf` byte for byte, and removes the
   copied `~/.config/reachAQ/*.conf`.
2. `conda run -n reachaq python -m pip uninstall -y reachaq reachaq-pose-deeplabcut reachaq-pose-yolo`
3. `git checkout v3.2.2`, then `REACHAQ_INSTALL_SYSTEM=0 ./tools/install/reachaq-linux-install.sh`.
4. Put the v3 SoftMouse unit back, since v3's installer does not replace installed units:

   ```bash
   cp ~/.config/systemd/user/reachaq-softmouse-publisher.service.bak ~/.config/systemd/user/reachaq-softmouse-publisher.service
   systemctl --user daemon-reload
   systemctl --user start reachaq-softmouse-publisher.timer
   ```
5. If anything is off, the step-3 tarball holds the pre-deploy settings. Return to
   `devel` (`git switch devel`) before deploying v4 again.
