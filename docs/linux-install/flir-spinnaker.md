# FLIR Spinnaker camera setup

Use this guide only on a reachAQ host with Teledyne FLIR/Point Grey cameras.
The portable installer does not install vendor SDKs because the SDK, wheel,
architecture, and camera firmware must be compatible.

## Inputs

Set the checkout and environment names used below:

```bash
export REACHAQ_REPO="$HOME/Documents/reachAQ"
export REACHAQ_ENV="reachaq"
```

reachAQ currently includes Spinnaker Python 3.2.0.62 wheels for CPython 3.8:

```bash
ls "$REACHAQ_REPO"/library/spinnaker_python-*-linux_*.whl
```

## 1. Install the matching system SDK

Download the Linux SDK from Teledyne FLIR. Match its release to the Python
wheel whenever possible; do not mix an arbitrary wheel and system runtime.
Follow the SDK bundle's README/install script. A typical downloaded Debian
package bundle uses:

```bash
cd "$HOME/Downloads/Spinnaker-<VERSION>-Linux"
sudo apt install ./*.deb
sudo ldconfig
sudo reboot
```

The package names vary by SDK version, so this vendor step is intentionally not
part of the portable installer.

## 2. Configure operator permissions

After the SDK creates its camera-access group, add the operator and then log out
and back in (or reboot):

```bash
getent group flirimaging
sudo usermod -a -G flirimaging,plugdev "$USER"
groups
```

If `flirimaging` does not exist, consult the installed SDK's udev instructions
instead of creating an unrelated group manually.

## 3. Install the matching Python wheel

x86_64 Ubuntu with the bundled CPython 3.8 wheel:

```bash
conda run -n "$REACHAQ_ENV" python -m pip install \
  "$REACHAQ_REPO/library/spinnaker_python-3.2.0.62-cp38-cp38-linux_x86_64.whl"
```

Jetson/aarch64 requires the corresponding aarch64 system SDK and wheel:

```bash
conda run -n "$REACHAQ_ENV" python -m pip install \
  "$REACHAQ_REPO/library/spinnaker_python-3.2.0.62-cp38-cp38-linux_aarch64.whl"
```

## 4. Verify the complete camera path

```bash
ldconfig -p | grep -i spinnaker
ls /etc/udev/rules.d/*spinnaker* 2>/dev/null
conda run -n "$REACHAQ_ENV" python -c "import PySpin; print('PySpin import ok')"
conda run -n "$REACHAQ_ENV" python "$REACHAQ_REPO/scripts/list_cameras.py"
```

Expected result: each connected FLIR camera appears with a
`spinnaker://<SERIAL>` URL.

## Troubleshooting

| Symptom | Check |
|---|---|
| `import PySpin` fails | System SDK and Python wheel versions/architectures |
| Import succeeds but no cameras appear | Power, USB/GigE cabling, udev rules, and `flirimaging` membership |
| Permission denied | Log out/in after group changes; re-run the SDK udev setup |
| Camera opens with unexpected behavior | Camera firmware and SDK compatibility; configured serial and URL properties |

Upstream SDK: <https://prep.flir.com/products/spinnaker-sdk/>
