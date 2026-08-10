from pathlib import Path

import pytest

from tools.platform_support import SpinnakerArtifactError, select_spinnaker_wheel


@pytest.mark.parametrize(
    "system,machine,filename",
    [
        ("Linux", "x86_64", "linux_x86_64.whl"),
        ("Linux", "aarch64", "linux_aarch64.whl"),
        ("Windows", "AMD64", "win_amd64.whl"),
    ],
)
def test_selects_retained_platform_artifact(system, machine, filename):
    wheel = select_spinnaker_wheel(
        system=system,
        machine=machine,
        python_tag="cp38",
    )
    assert wheel.is_file()
    assert wheel.name.endswith(filename)


def test_reports_unsupported_platform_tuple(tmp_path: Path):
    vendor = tmp_path / "vendor" / "spinnaker"
    vendor.mkdir(parents=True)
    (vendor / "manifest.json").write_text('{"version": 1, "artifacts": []}')

    with pytest.raises(SpinnakerArtifactError, match="linux/riscv64/cp38"):
        select_spinnaker_wheel(
            system="Linux",
            machine="riscv64",
            python_tag="cp38",
            repository_root=tmp_path,
        )
