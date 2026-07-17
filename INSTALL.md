# Installation

This repository still contains the broader Autotrainer modules, but the current
Linux reachAQ bring-up path is documented in:

- [linux-install-instructions.md](linux-install-instructions.md)

Use that guide for a machine that needs to run the reachAQ application with
Spinnaker cameras, NI-DAQmx/PXI hardware, PEAK SocketCAN, inference, and local
data output.

## Current ReachAQ Runtime

The current Linux runtime convention is:

```bash
export REACHAQ_REPO="$HOME/Documents/reachAQ"
export REACHAQ_ENV="reachaq"
export REACHAQ_CONFIG="$HOME/Autotrainer/system_configuration.yaml"
export REACHAQ_DATA="$HOME/Documents/rawdatalocal"
```

Create/install the Python environment from the repo root:

```bash
conda create -y -n "$REACHAQ_ENV" python=3.8
conda run -n "$REACHAQ_ENV" python -m pip install --upgrade pip setuptools wheel build
conda run -n "$REACHAQ_ENV" python -m pip install -r requirements.txt
conda run -n "$REACHAQ_ENV" python -m pip install -e '.[test]'
```

Install the matching Spinnaker SDK and Python wheel, NI Linux drivers, and CAN
tools as described in the Linux guide before expecting full hardware operation.

Start the GUI with:

```bash
conda run -n "$REACHAQ_ENV" python -m reachAQ.app --start-mode idle -c "$REACHAQ_CONFIG"
```

For a software-only camera smoke test:

```bash
conda run -n "$REACHAQ_ENV" python -m reachAQ.app \
  --start-mode idle \
  --random-cameras \
  -c "$REACHAQ_CONFIG"
```

Headless mode remains available through:

```bash
conda run -n "$REACHAQ_ENV" auto-trainer-headless --start-mode idle -c "$REACHAQ_CONFIG"
```

## Legacy Notes

Older Autotrainer deployments used environment names such as
`auto-trainer-1`, Jetson-specific dependency pins, and direct module entry
points like `python -m tools.acquisition.gui`. Those may still be useful for
legacy rigs, but they are not the current reachAQ Dell/Ubuntu setup. Prefer the
Linux guide above when setting up a new reachAQ Linux machine.
