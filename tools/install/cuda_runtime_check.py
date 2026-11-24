"""Check that the reachAQ environment carries exactly one CUDA runtime.

The base install's only CUDA user-space runtime is the one PyTorch's CUDA 12.8
build brings as nvidia-*-cu12 wheels in site-packages. The only other runtime
allowed is the DeepLabCut plugin's: TensorFlow 2.12 needs CUDA 11.8 and cuDNN
8.6, which the plugin installer puts in <env>/lib/reachaq-tensorflow-cuda11
(see install_tensorflow_gpu_runtime in plugin-install-lib.sh), reached only
through the environment's LD_LIBRARY_PATH. Anything else is a second runtime
that one framework or the other can load by mistake: nvidia-* wheels of
another CUDA major in site-packages (an older installer put TensorFlow's
there, and the two sets overwrote each other), or the TensorFlow directory
still on LD_LIBRARY_PATH after the plugin is gone.

Run in the environment by reachaq-linux-install.sh:

    conda run -n reachaq python cuda_runtime_check.py

Exit 0 and a summary when the environment is clean, 1 and what to remove when
it is not.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Kept in step with install_tensorflow_gpu_runtime in plugin-install-lib.sh.
TENSORFLOW_RUNTIME_DIR = "lib/reachaq-tensorflow-cuda11"
# The pose plugins' entry-point group.
POSE_BACKEND_GROUP = "reachaq.pose_backends"
_SUFFIX = re.compile(r"-cu(\d+)$")
# NVIDIA's CUDA user-space components as pip names them (nvidia-cuda-runtime-cu12,
# nvidia-cudnn-cu12, nvidia-nccl-cu12, ...). Other nvidia-* distributions, such
# as nvidia-ml-py (NVML bindings), are not a CUDA runtime.
_RUNTIME_COMPONENT = re.compile(r"^nvidia-(cu[a-z]+|nccl|nvjitlink|nvtx|nvshmem)(-|$)")


def is_runtime_component(distribution: str) -> bool:
    return bool(_RUNTIME_COMPONENT.match(distribution))


def cuda_major_of(distribution: str) -> Optional[int]:
    match = _SUFFIX.search(distribution)
    return int(match.group(1)) if match else None


def assess(
    *,
    nvidia_distributions: Iterable[str],
    torch_cuda: Optional[str],
    prefix: Path,
    deeplabcut_installed: bool,
    library_path: str,
    site_nvidia: Path,
) -> Tuple[bool, List[str]]:
    """Pure check, so it can be tested without an environment."""
    problems: List[str] = []
    notes: List[str] = []
    if not torch_cuda:
        problems.append("torch is not a CUDA build (torch.version.cuda is None); rerun the "
                        "installer, which takes torch from the PyTorch cu128 index")
        expected = None
    else:
        expected = int(torch_cuda.split(".")[0])
        notes.append(f"PyTorch CUDA {torch_cuda} runtime in {site_nvidia}")

    names = sorted(name for name in set(nvidia_distributions) if is_runtime_component(name))
    unknown = [name for name in names if cuda_major_of(name) is None]
    foreign = [name for name in names
               if expected is not None and cuda_major_of(name) not in (None, expected)]
    if foreign:
        problems.append(
            f"site-packages holds nvidia-* wheels of another CUDA major than torch's "
            f"CUDA {expected}: {', '.join(foreign)}. Remove them with "
            f"python -m pip uninstall -y {' '.join(foreign)}")
    if unknown:
        problems.append(
            "site-packages holds nvidia-* wheels without a CUDA major in their name, which "
            f"the CUDA {torch_cuda} torch build does not use: {', '.join(unknown)}. They come "
            "from a newer torch (CUDA 13); rerun the installer, which reinstalls torch from "
            "the cu128 index, then remove them with python -m pip uninstall -y "
            + " ".join(unknown))

    tensorflow_dir = prefix / TENSORFLOW_RUNTIME_DIR
    entries = [entry for entry in library_path.split(":") if entry]
    on_path = [entry for entry in entries if _is_within(entry, tensorflow_dir)]
    stale = [entry for entry in entries if _is_within(entry, site_nvidia)]
    if stale:
        problems.append(
            "LD_LIBRARY_PATH points into site-packages/nvidia, torch's runtime, which every "
            f"process would then load first: {':'.join(stale)}. Reinstall the DeepLabCut "
            "plugin (tools/install/reachaq-plugin-install.sh deeplabcut), which rewrites it, "
            "or remove those entries with conda env config vars set")
    if deeplabcut_installed:
        if not tensorflow_dir.is_dir() or not on_path:
            problems.append(
                f"the DeepLabCut plugin is installed but its CUDA 11.8 runtime is not in place "
                f"({tensorflow_dir} on LD_LIBRARY_PATH); rerun "
                "tools/install/reachaq-plugin-install.sh deeplabcut")
        else:
            notes.append(f"DeepLabCut plugin CUDA 11.8 / cuDNN 8.6 runtime for TensorFlow only, "
                         f"in {tensorflow_dir}, loaded through LD_LIBRARY_PATH")
    elif tensorflow_dir.exists() or on_path:
        problems.append(
            f"the DeepLabCut plugin is not installed but its CUDA 11.8 runtime is left behind "
            f"({tensorflow_dir}). Remove it: rm -rf {tensorflow_dir}, and drop its entries from "
            "LD_LIBRARY_PATH (conda env config vars unset LD_LIBRARY_PATH, or set it to your own "
            "entries only)")
    if problems:
        return False, problems
    return True, notes


def _is_within(entry: str, root: Path) -> bool:
    try:
        Path(entry).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _environment_facts() -> dict:
    import sysconfig
    from importlib.metadata import distributions, entry_points

    import torch

    names = set()
    for distribution in distributions():
        name = (distribution.metadata["Name"] or "").lower().replace("_", "-")
        if name.startswith("nvidia-"):
            names.add(name)
    found = entry_points()
    group = POSE_BACKEND_GROUP
    selected = found.select(group=group) if hasattr(found, "select") else found.get(group, ())
    deeplabcut = any(entry.name == "deeplabcut" for entry in selected)
    return {
        "nvidia_distributions": names,
        "torch_cuda": torch.version.cuda,
        "prefix": Path(sys.prefix),
        "deeplabcut_installed": deeplabcut,
        "library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "site_nvidia": Path(sysconfig.get_paths()["purelib"]) / "nvidia",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ok, lines = assess(**_environment_facts())
    if ok:
        print("One CUDA runtime per engine:")
        for line in lines:
            print(f"  {line}")
        return 0
    print("This environment carries a CUDA runtime it should not:", file=sys.stderr)
    for line in lines:
        print(f"  - {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
