"""Every get_response() call must pass the argument it requires.

get_response takes `motor` as keyword-only with no default, so a call that
omits it raises TypeError. Both calls in open() did, on the demo branch: the
CAN device thread died the moment it opened the interface, which failed CAN
readiness, which tripped a CRITICAL safety shutdown. The pellet controller
appeared to crash on every start.

It survived because nothing exercises open() without a board - it needs
SocketCAN and a powered controller - and because the fix landed on one branch
and was never carried to the other. A signature check costs nothing and does
not need hardware.
"""

import ast
import pathlib

import pytest

CAN_INTERFACE = (pathlib.Path(__file__).resolve().parents[1]
                 / "auto-trainer-device" / "src" / "autotrainer" / "device"
                 / "can_interface.py")


def _tree():
    return ast.parse(CAN_INTERFACE.read_text(encoding="utf-8"))


def _called_name(node):
    return getattr(node.func, "attr", getattr(node.func, "id", ""))


def _required_keyword_only(tree, name):
    """Keyword-only parameters of `name` that have no default."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args
            return {
                arg.arg
                for arg, default in zip(args.kwonlyargs, args.kw_defaults)
                if default is None
            }
    raise AssertionError(f"{name} is gone from can_interface")


def test_get_response_still_requires_motor():
    """If this relaxes, the check below stops guarding anything."""
    assert "motor" in _required_keyword_only(_tree(), "get_response")


def test_every_get_response_call_passes_motor():
    tree = _tree()
    required = _required_keyword_only(tree, "get_response")
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and _called_name(node) == "get_response"]
    assert calls, "no get_response calls found; has it been renamed?"

    missing = []
    for call in calls:
        passed = {keyword.arg for keyword in call.keywords}
        # **kwargs forwards anything, so it satisfies the requirement.
        if None in passed:
            continue
        absent = required - passed
        if absent:
            missing.append((call.lineno, sorted(absent)))

    assert not missing, (
        "get_response() calls omitting a required keyword-only argument, "
        "which raises TypeError at runtime: "
        + "; ".join(f"line {line} missing {args}" for line, args in missing))


@pytest.mark.parametrize("name", ["open"])
def test_the_interface_still_opens_through_get_response(name):
    """Guards the test above: it only means something while open() calls it."""
    tree = _tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            inner = [n for n in ast.walk(node)
                     if isinstance(n, ast.Call) and _called_name(n) == "get_response"]
            assert inner, f"{name}() no longer calls get_response"
            return
    raise AssertionError(f"{name}() is gone from can_interface")
