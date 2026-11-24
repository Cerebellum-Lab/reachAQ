"""The installer's platform list, Spinnaker download handling and CUDA check.

These are the stdlib helpers in tools/install that reachaq-linux-install.sh
runs; none of them touches the host here. Each is loaded from its file, as the
installer runs it, rather than imported as part of a package.
"""

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_DIR = REPO_ROOT / "tools" / "install"
SPEC_FILE = INSTALL_DIR / "supported-platforms.json"


def _load(name):
    sys.path.insert(0, str(INSTALL_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, INSTALL_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses looks its module up there
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(INSTALL_DIR))


platform_check = _load("platform_check")
spinnaker_download = _load("spinnaker_download")
cuda_runtime_check = _load("cuda_runtime_check")

SPEC = platform_check.load_spec(SPEC_FILE)
RIG_OS = platform_check.parse_os_release(
    'PRETTY_NAME="Ubuntu 22.04.5 LTS"\nNAME="Ubuntu"\nVERSION_ID="22.04"\nID=ubuntu\n')
RIG_GPU = "NVIDIA T1000, 595.84, 7.5\n"


# --- supported platform list ------------------------------------------------

def test_the_reference_rig_is_supported():
    findings = platform_check.check_host(SPEC, RIG_OS, "x86_64", RIG_GPU)

    assert all(ok for ok, _ in findings), findings
    assert [text for _, text in findings] == [
        "operating system Ubuntu 22.04.5 LTS",
        "architecture x86_64",
        "GPU NVIDIA T1000, driver 595.84, compute capability 7.5",
    ]


@pytest.mark.parametrize("os_release,expected", [
    ('ID=ubuntu\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\n',
     "operating system Ubuntu 24.04.1 LTS; supported: Ubuntu 22.04 LTS"),
    ('ID=debian\nVERSION_ID="12"\n', "operating system debian 12; supported: Ubuntu 22.04 LTS"),
    ("", "operating system unknown; supported: Ubuntu 22.04 LTS"),
])
def test_another_os_is_named_with_the_supported_ones(os_release, expected):
    ok, text = platform_check.check_os(SPEC, platform_check.parse_os_release(os_release))

    assert not ok
    assert text == expected


def test_aarch64_is_refused():
    ok, text = platform_check.check_architecture(SPEC, "aarch64")

    assert not ok
    assert text == "architecture aarch64; supported: x86_64"


@pytest.mark.parametrize("gpu,problem", [
    ("NVIDIA T1000, 550.54.14, 7.5\n", "driver must be 570.26 or newer"),
    ("Tesla P4, 595.84, 6.1\n", "compute capability must be 7.5 to 12.0"),
    ("Future GPU, 700.1, 13.0\n", "compute capability must be 7.5 to 12.0"),
])
def test_a_gpu_outside_the_cuda_12_8_build_is_refused(gpu, problem):
    [(ok, text)] = platform_check.check_nvidia(SPEC, gpu)

    assert not ok
    assert problem in text


def test_driver_versions_compare_numerically_not_as_text():
    # "595.84" < "1000.1" as text; numerically it is the other way round, and
    # "570.9" is older than "570.26".
    [(ok, _)] = platform_check.check_nvidia(SPEC, "GPU, 1000.1, 8.6\n")
    [(old, _)] = platform_check.check_nvidia(SPEC, "GPU, 570.9, 8.6\n")

    assert ok
    assert not old


def test_every_gpu_is_checked():
    findings = platform_check.check_nvidia(
        SPEC, "NVIDIA T1000, 595.84, 7.5\nTesla K80, 595.84, 3.7\n")

    assert [ok for ok, _ in findings] == [True, False]


def test_no_driver_and_unreadable_output_are_refused():
    [(ok, text)] = platform_check.check_nvidia(SPEC, None)
    assert not ok and "nvidia-smi is missing or failed" in text

    [(ok, text)] = platform_check.check_nvidia(SPEC, "")
    assert not ok and "no NVIDIA GPU" in text

    [(ok, text)] = platform_check.check_nvidia(SPEC, "NVIDIA T1000, [N/A], 7.5\n")
    assert not ok and "cannot read driver" in text


def test_the_check_exits_1_for_unsupported_and_2_for_a_broken_list(tmp_path):
    os_release = tmp_path / "os-release"
    os_release.write_text('ID=ubuntu\nVERSION_ID="20.04"\n')
    completed = subprocess.run(
        [sys.executable, str(INSTALL_DIR / "platform_check.py"),
         "--os-release", str(os_release), "--machine", "x86_64"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 1
    assert "unsupported: operating system ubuntu 20.04" in completed.stdout

    broken = tmp_path / "broken.json"
    broken.write_text('{"schema": 99}')
    completed = subprocess.run(
        [sys.executable, str(INSTALL_DIR / "platform_check.py"), "--spec", str(broken)],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "unknown schema" in completed.stderr

    # Exit 1 is the only result REACHAQ_INSTALL_UNSUPPORTED may accept, so an
    # incomplete list must not crash with it.
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text('{"schema": 1}')
    for script in ("platform_check.py", "spinnaker_download.py"):
        args = ["--spec", str(incomplete)] + (["supported", "3.2.0.57"]
                                              if script == "spinnaker_download.py" else [])
        completed = subprocess.run([sys.executable, str(INSTALL_DIR / script), *args],
                                   capture_output=True, text=True, check=False)
        assert completed.returncode == 2, (script, completed.stderr)
        assert "operating_systems" in completed.stderr, script


def test_the_platform_list_is_data_with_a_reason_for_each_limit():
    raw = json.loads(SPEC_FILE.read_text())

    assert raw["operating_systems"] == [
        {"id": "ubuntu", "version_id": "22.04", "name": "Ubuntu 22.04 LTS"}]
    assert raw["nvidia"]["why"] and raw["spinnaker"]["why"]
    # The compute capability range is the one the torch pin's comment
    # records for torch 2.7 on CUDA 12.8.
    pin = (REPO_ROOT / "pyproject.toml").read_text()
    assert "compute capability 7.5 through 12.0" in pin
    assert (raw["nvidia"]["min_compute_capability"],
            raw["nvidia"]["max_compute_capability"]) == ("7.5", "12.0")


# --- Spinnaker download -----------------------------------------------------

def _sdk_tarball(path: Path, version: str, arch="amd64"):
    """A tarball laid out like Teledyne's: spinnaker-<v>-<arch>/*.deb and scripts."""
    with tarfile.open(path, "w:gz") as archive:
        for name in (f"libgentl_{version}_{arch}.deb", f"libspinnaker_{version}_{arch}.deb",
                     "install_spinnaker.sh"):
            data = f"{name}\n".encode()
            info = tarfile.TarInfo(f"spinnaker-{version}-{arch}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _python_tarball(path: Path, version: str):
    with tarfile.open(path, "w:gz") as archive:
        for name in (f"spinnaker_python-{version}-cp310-cp310-linux_x86_64.whl",
                     "Examples/Python3/Acquisition.py"):
            data = b"x"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _teledyne_zip(directory: Path, version: str) -> Path:
    """Spinnaker-<v>-Linux.zip as Teledyne ships it: every distro's tarballs."""
    work = directory / "build"
    work.mkdir()
    _sdk_tarball(work / f"spinnaker-{version}-amd64-pkg-22.04.tar.gz", version)
    _sdk_tarball(work / f"spinnaker-{version}-amd64-pkg-20.04.tar.gz", version)
    _python_tarball(work / f"spinnaker_python-{version}-cp310-cp310-linux_x86_64.tar.gz", version)
    archive = directory / f"Spinnaker-{version}-Linux.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for file in sorted(work.iterdir()):
            stream.write(file, file.name)
    for file in work.iterdir():
        file.unlink()
    work.rmdir()
    return archive


def _run(*args):
    return subprocess.run(
        [sys.executable, str(INSTALL_DIR / "spinnaker_download.py"), *args],
        capture_output=True, text=True, check=False)


def test_the_sdk_debs_come_out_of_the_teledyne_zip(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    _teledyne_zip(downloads, "3.2.0.57")

    completed = _run("extract-sdk", "--dir", str(downloads), "--dest", str(tmp_path / "out"),
                     "--ubuntu", "22.04", "--arch", "amd64")

    assert completed.returncode == 0, completed.stderr
    deb_dir = Path(completed.stdout.strip())
    assert sorted(path.name for path in deb_dir.iterdir()) == [
        "libgentl_3.2.0.57_amd64.deb", "libspinnaker_3.2.0.57_amd64.deb"]
    assert "pkg-22.04.tar.gz" in completed.stderr, "must take the 22.04 build, not 20.04"


def test_the_binding_comes_out_of_an_unpacked_download(tmp_path):
    unpacked = tmp_path / "Downloads" / "Spinnaker-3.2.0.57-Linux"
    unpacked.mkdir(parents=True)
    _python_tarball(unpacked / "spinnaker_python-3.2.0.57-cp310-cp310-linux_x86_64.tar.gz",
                    "3.2.0.57")

    completed = _run("extract-python", "--dir", str(tmp_path / "Downloads"),
                     "--dest", str(tmp_path / "out"), "--sdk-version", "3.2.0.57",
                     "--python", "cp310", "--machine", "x86_64")

    assert completed.returncode == 0, completed.stderr
    wheel = Path(completed.stdout.strip())
    assert wheel.name == "spinnaker_python-3.2.0.57-cp310-cp310-linux_x86_64.whl"
    assert wheel.is_file()


def test_the_binding_matches_the_sdk_release_then_its_series():
    found = [spinnaker_download.Candidate((3, 2, 0, 62), Path("a.whl"), None),
             spinnaker_download.Candidate((3, 2, 0, 57), Path("b.whl"), None),
             spinnaker_download.Candidate((4, 0, 0, 1), Path("c.whl"), None)]

    assert spinnaker_download.choose_python(found, (3, 2, 0, 57)).path == Path("b.whl")
    assert spinnaker_download.choose_python(found, (3, 2, 0, 60)).path == Path("a.whl")
    assert spinnaker_download.choose_python(found, (3, 1, 0, 1)) is None


def test_an_unsupported_sdk_release_is_not_used(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    _sdk_tarball(downloads / "spinnaker-4.0.0.116-amd64-pkg-22.04.tar.gz", "4.0.0.116")

    completed = _run("extract-sdk", "--dir", str(downloads), "--dest", str(tmp_path / "out"),
                     "--ubuntu", "22.04", "--arch", "amd64")

    assert completed.returncode == 1
    # Exact instructions: what to download, from where, to where, and what was
    # found instead.
    assert "Download Spinnaker 3.2 for Linux from https://" in completed.stderr
    assert str(downloads) in completed.stderr
    assert "REACHAQ_SPINNAKER_DOWNLOADS" in completed.stderr
    assert "4.0.0.116" in completed.stderr


def test_a_missing_download_folder_says_what_to_download(tmp_path):
    completed = _run("extract-sdk", "--dir", str(tmp_path / "nowhere"),
                     "--dest", str(tmp_path / "out"), "--ubuntu", "22.04", "--arch", "amd64")

    assert completed.returncode == 1
    assert "No usable Spinnaker SDK for Ubuntu 22.04 amd64" in completed.stderr


def test_a_tarball_escaping_its_folder_is_refused(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    with tarfile.open(downloads / "spinnaker-3.2.0.57-amd64-pkg-22.04.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo("../libspinnaker_3.2.0.57_amd64.deb")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))

    completed = _run("extract-sdk", "--dir", str(downloads), "--dest", str(tmp_path / "out"),
                     "--ubuntu", "22.04", "--arch", "amd64")

    assert completed.returncode == 1
    assert "unsafe path" in completed.stderr
    assert not (tmp_path / "libspinnaker_3.2.0.57_amd64.deb").exists()


@pytest.mark.parametrize("version,code", [("3.2.0.57", 0), ("3.2.0.62", 0),
                                          ("3.1.0.79", 1), ("4.0.0.116", 1), ("x", 2)])
def test_supported_sdk_versions_are_the_listed_series(version, code):
    assert _run("supported", version).returncode == code


# --- one CUDA runtime -------------------------------------------------------

TORCH_WHEELS = {"nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12",
                "nvidia-nccl-cu12"}


def _assess(tmp_path, **overrides):
    facts = {
        "nvidia_distributions": TORCH_WHEELS,
        "torch_cuda": "12.8",
        "prefix": tmp_path / "env",
        "deeplabcut_installed": False,
        "library_path": "",
        "site_nvidia": tmp_path / "env" / "lib" / "python3.10" / "site-packages" / "nvidia",
    }
    facts.update(overrides)
    return cuda_runtime_check.assess(**facts)


def test_the_base_install_has_only_torchs_runtime(tmp_path):
    ok, lines = _assess(tmp_path)

    assert ok, lines
    assert lines == [f"PyTorch CUDA 12.8 runtime in {tmp_path / 'env/lib/python3.10/site-packages/nvidia'}"]


def test_nvidia_packages_that_are_not_cuda_are_ignored(tmp_path):
    # The reference rig's environment has nvidia-ml-py (NVML bindings).
    ok, lines = _assess(tmp_path, nvidia_distributions=TORCH_WHEELS | {"nvidia-ml-py"})

    assert ok, lines


def test_cuda_11_wheels_in_site_packages_are_a_second_runtime(tmp_path):
    ok, lines = _assess(tmp_path, nvidia_distributions=TORCH_WHEELS | {"nvidia-cudnn-cu11"})

    assert not ok
    assert "nvidia-cudnn-cu11" in lines[0]
    assert "pip uninstall -y nvidia-cudnn-cu11" in lines[0]


def test_cuda_13_wheels_are_a_second_runtime(tmp_path):
    ok, lines = _assess(tmp_path, nvidia_distributions=TORCH_WHEELS | {"nvidia-cublas"})

    assert not ok
    assert "nvidia-cublas" in lines[0] and "CUDA 13" in lines[0]


def test_a_cpu_torch_is_reported(tmp_path):
    ok, lines = _assess(tmp_path, torch_cuda=None)

    assert not ok
    assert "torch is not a CUDA build" in lines[0]


def test_the_deeplabcut_runtime_is_allowed_only_with_the_plugin(tmp_path):
    runtime = tmp_path / "env" / "lib" / "reachaq-tensorflow-cuda11"
    (runtime / "nvidia" / "cudnn" / "lib").mkdir(parents=True)
    path = f"{runtime}/nvidia/cudnn/lib:/opt/other/lib"

    ok, lines = _assess(tmp_path, deeplabcut_installed=True, library_path=path)
    assert ok, lines
    assert any("DeepLabCut plugin CUDA 11.8" in line for line in lines)

    ok, lines = _assess(tmp_path, deeplabcut_installed=False, library_path=path)
    assert not ok
    assert f"rm -rf {runtime}" in lines[0]


def test_the_plugin_without_its_runtime_is_reported(tmp_path):
    ok, lines = _assess(tmp_path, deeplabcut_installed=True)

    assert not ok
    assert "reachaq-plugin-install.sh deeplabcut" in lines[0]


def test_ld_library_path_into_torchs_wheels_is_reported(tmp_path):
    site = tmp_path / "env" / "lib" / "python3.10" / "site-packages" / "nvidia"
    ok, lines = _assess(tmp_path, library_path=f"{site}/cudnn/lib")

    assert not ok
    assert "site-packages/nvidia" in lines[0]


def test_the_check_and_the_plugin_name_the_same_directory():
    library = (INSTALL_DIR / "plugin-install-lib.sh").read_text()

    assert f'runtime_dir="$env_prefix/{cuda_runtime_check.TENSORFLOW_RUNTIME_DIR}"' in library
