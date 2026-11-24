# reachaq-pose-deeplabcut

DeepLabCut as an optional reachAQ pose backend. reachAQ itself no longer
depends on DeepLabCut: without this plugin a DeepLabCut project is refused at
pre-validation with a message naming this distribution.

The plugin registers itself in the `reachaq.pose_backends` entry-point
group and claims any model directory that contains a DeepLabCut `config.yaml`.
`REACHAQ_POSE_BACKEND` chooses between DeepLabCut's two engines, as before;
unset, TensorFlow is used where it is installed, else PyTorch.

## Install

Into the environment reachAQ runs in, from the checkout:

```bash
python -m pip install -e plugins/deeplabcut              # PyTorch engine
python -m pip install -e "plugins/deeplabcut[tensorflow]"  # and TensorFlow
```

It depends on `reachaq` (the installed application) and
`deeplabcut >= 3.0, < 4`. The `tensorflow` extra carries the TensorFlow 2.12
pins and the conditions that let it share a process with torch; see the
comments beside it in `pyproject.toml`.

## Contents

- `src/reachaq/pose_plugins/deeplabcut/` - the plugin (`plugin.py`,
  `DLC_PLUGIN`), the TensorFlow model (`DlcPoseModel`) and the PyTorch model
  (`DlcTorchPoseModel`), with the measured confidence thresholds.
- `scripts/load_dlc_model.py` - load a DeepLabCut model and print its
  body-part metadata.
- `scripts/run_dlc_model.py` - run a DeepLabCut model against paired recorded
  videos.
- `tests/` - run with the plugin installed: `pytest plugins/deeplabcut/tests`.
