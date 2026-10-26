import codecs
from email.message import Message

import pytest

from tools.ci.check_licences import (
    DistLicence,
    classify_licence,
    evaluate,
    installed_licences,
    load_allowlist,
    main,
    render_markdown,
)


def d(name, flag, text="x"):
    return DistLicence(name=name, version="1", licence_text=text, flag=flag)


def test_classify_order():
    assert classify_licence("AGPL-3.0-only") == "AGPL"
    assert classify_licence("LGPL-3.0 OR GPL-2.0 OR GPL-3.0") == "LGPL"
    assert classify_licence("GPL-3.0-only") == "GPL"
    # Expressions are not parsed: the first of AGPL, LGPL, GPL to match wins.
    assert classify_licence("LGPL-2.1-only AND GPL-3.0-only") == "LGPL"
    assert classify_licence("BSD-3-Clause") == "permissive"
    assert classify_licence(None) == "none"


def test_allowlist_normalizes_names():
    assert load_allowlist("Auto_Trainer.API  # why\n\n# c\n") == {"auto-trainer-api": "why"}


@pytest.mark.parametrize("line", ["auto-trainer-api", "auto-trainer-api  #", "auto-trainer-api #   "])
def test_allowlist_entry_without_a_reason_is_rejected(line):
    with pytest.raises(ValueError, match="allowlist line 1"):
        load_allowlist(line)


def test_allowlist_file_with_a_byte_order_mark_has_no_bom_entry(tmp_path, monkeypatch, capsys):
    # Windows editors save UTF-8 with a BOM. Read as plain utf-8 it stays glued to the
    # first name, which then matches nothing: a violation plus a stale entry.
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_bytes(codecs.BOM_UTF8 + b"auto-trainer-api  # why\r\n")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        "tools.ci.check_licences.installed_licences", lambda: [d("auto-trainer-api", "AGPL")])
    assert main(["--allowlist", str(allowlist)]) == 0
    assert "\ufeff" not in capsys.readouterr().out


def test_unlisted_agpl_and_gpl_are_violations_lgpl_is_not():
    ev = evaluate([d("a", "AGPL"), d("g", "GPL"), d("l", "LGPL")], {})
    assert [x.name for x in ev.violations] == ["a", "g"] and not ev.ok


def test_listed_agpl_passes():
    assert evaluate([d("a", "AGPL")], {"a": "why"}).ok


def test_stale_allowlist_entry_fails():
    ev = evaluate([d("b", "permissive")], {"a": "why"})
    assert ev.stale == ["a"] and not ev.ok


def test_unknown_is_reported_not_failed():
    ev = evaluate([d("u", "unknown"), d("n", "none")], {})
    assert ev.ok and {x.name for x in ev.unknown} == {"u", "n"}
    assert "unknown: 2" in render_markdown(ev, [d("u", "unknown"), d("n", "none")])


class FakeDist:
    def __init__(self, name, licence=None, expression=None, classifiers=()):
        self.version = "1"
        self.metadata = Message()
        self.metadata["Name"] = name
        if licence is not None:
            self.metadata["License"] = licence
        if expression is not None:
            self.metadata["License-Expression"] = expression
        for classifier in classifiers:
            self.metadata["Classifier"] = classifier


def installed(monkeypatch, *dists):
    monkeypatch.setattr("importlib.metadata.distributions", lambda: list(dists))
    return {x.name: x for x in installed_licences()}


BSD_CLASSIFIER = "License :: OSI Approved :: BSD License"
MIT_CLASSIFIER = "License :: OSI Approved :: MIT License"


def test_bundled_notices_in_license_field_do_not_override_classifiers(monkeypatch):
    # numpy, scipy and pandas ship the bundled third-party licence texts (GPL,
    # Affero, ...) in the free-text License field and declare BSD in classifiers.
    multiline = "BSD\n\nGNU Affero General Public License\nGNU General Public License"
    one_long_line = "BSD " + "x" * 600 + " GNU General Public License"
    got = installed(
        monkeypatch,
        FakeDist("multiline", multiline, classifiers=[BSD_CLASSIFIER]),
        FakeDist("long", one_long_line, classifiers=[BSD_CLASSIFIER]),
    )
    assert got["multiline"].flag == "permissive" and got["long"].flag == "permissive"


def test_short_license_field_naming_gpl_is_not_overridden_by_a_permissive_classifier(monkeypatch):
    got = installed(monkeypatch, FakeDist("g", "GPL-3.0-or-later", classifiers=[MIT_CLASSIFIER]))
    assert got["g"].flag == "GPL"


def test_license_field_is_used_when_nothing_structured_identifies_it(monkeypatch):
    got = installed(
        monkeypatch,
        FakeDist("g", "GPL-3.0-only", classifiers=["License :: OSI Approved"]),
        FakeDist("p", "NVIDIA Proprietary Software", classifiers=["License :: Other/Proprietary License"]),
        FakeDist("blob", "text\nGNU General Public License\nmore text"),
    )
    assert got["g"].flag == "GPL" and got["p"].flag == "unknown" and got["blob"].flag == "GPL"


def test_license_expression_is_authoritative_over_the_license_field(monkeypatch):
    got = installed(monkeypatch, FakeDist("m", "GPL-3.0", expression="MIT", classifiers=[BSD_CLASSIFIER]))
    assert got["m"].flag == "permissive" and got["m"].licence_text == "MIT"


def test_names_are_normalized_and_the_first_duplicate_wins(monkeypatch):
    got = installed(
        monkeypatch,
        FakeDist("Auto_Trainer.API", expression="AGPL-3.0-only"),
        FakeDist("auto-trainer-api", expression="MIT"),
    )
    assert list(got) == ["auto-trainer-api"] and got["auto-trainer-api"].flag == "AGPL"


def test_undeclared_licence_is_none_and_freebsd_is_permissive(monkeypatch):
    got = installed(monkeypatch, FakeDist("u", "UNKNOWN"), FakeDist("f", "FreeBSD"))
    assert got["u"].flag == "none" and got["u"].licence_text == ""
    assert got["f"].flag == "permissive"
