"""Structural assertions about wiring, without pinning its formatting.

A few guarantees in this project live in how modules are wired rather than in
behaviour a test can reach. Starting a real PoseProcess needs a GPU, a trained
model and live multiprocessing queues, so a handful of tests read the code
instead of running it.

Reading it as *text* made those tests fail on correct changes while still
missing the substitutions they exist to catch. Three times in one afternoon: a
call gained a keyword argument, a loop gained a guard, a line was reformatted -
each a strengthening of exactly what the test was protecting, each reported as
a regression. Meanwhile a genuine change hidden inside a rewritten line would
have slipped past a literal match.

So these read the parse tree. They answer "is this call still made, with this
argument, before that one" rather than "does this line still look the way it
did". A reformat, a line break, a renamed local or an added argument leaves
them alone; removing the call, dropping the argument or swapping the order does
not.

When behaviour can be reached directly, test the behaviour instead. This is for
the cases where it cannot.
"""

import ast
import inspect
import pathlib
import typing

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def tree(target) -> ast.AST:
    """Parse a module object, a path, or a repo-relative path string.

    An AST node passes straight through, so a search can be narrowed to one
    function by handing it the node that `function()` returned.
    """
    if isinstance(target, ast.AST):
        return target
    if hasattr(target, "__file__"):
        return ast.parse(inspect.getsource(target))
    path = pathlib.Path(target)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return ast.parse(path.read_text(encoding="utf-8"))


def _called_name(node: ast.Call) -> str:
    """The callable's own name, ignoring whatever it is reached through.

    `selected_backend(...)`, `self.selected_backend(...)` and
    `module.selected_backend(...)` all answer "selected_backend": the tests
    here care that a particular function is called, not how it is addressed.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def calls(target, name: str) -> typing.List[ast.Call]:
    """Every call to `name` anywhere in `target`, nesting included."""
    return [node for node in ast.walk(tree(target))
            if isinstance(node, ast.Call) and _called_name(node) == name]


def one_call(target, name: str) -> ast.Call:
    """The single call to `name`; fails loudly when that is not what is there.

    Distinguishes "the wiring changed" from "there are now two of these", which
    a test asserting on the first match would silently accept.
    """
    found = calls(target, name)
    assert found, f"no call to {name}() remains"
    assert len(found) == 1, f"expected one call to {name}(), found {len(found)}"
    return found[0]


def keyword(call: ast.Call, name: str) -> typing.Optional[ast.AST]:
    """The value node of a keyword argument, or None when it is not passed."""
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def keyword_equals(call: ast.Call, name: str, expected) -> bool:
    """Whether a keyword argument is passed as exactly this constant."""
    value = keyword(call, name)
    return isinstance(value, ast.Constant) and value.value == expected


def dotted_name(node: typing.Optional[ast.AST]) -> typing.Optional[str]:
    """Render a name or attribute chain as "a.b.c", or None for anything else.

    Deliberately not ast.unparse: that arrived in Python 3.9 and the deployed
    rig runs 3.8, so a helper written against it would pass here and fail
    where it matters.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def keyword_name(call: ast.Call, name: str) -> typing.Optional[str]:
    """A keyword argument passed as a name or attribute chain, or None.

    Answers "which value is handed to this parameter" without caring about
    spacing or line breaks, so `drain=live_drain` and a version split across
    lines compare equal.
    """
    return dotted_name(keyword(call, name))


def assigned(target, name: str) -> typing.Optional[str]:
    """The right-hand side of `name = ...`, as a structural dump.

    A dump rather than source text: it is stable across formatting while still
    naming the operators and attributes involved, which is what an assertion
    about a derived flag needs to see.
    """
    for node in ast.walk(tree(target)):
        if isinstance(node, ast.Assign) and any(
                getattr(item, "id", "") == name for item in node.targets):
            return ast.dump(node.value)
    return None


def function(target, name: str) -> ast.AST:
    """A function or method by name, at any nesting depth."""
    for node in ast.walk(tree(target)):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name):
            return node
    raise AssertionError(f"no function named {name!r}")


def call_order(target, names: typing.Sequence[str]) -> typing.List[str]:
    """The given calls in the order they appear, others ignored.

    Source order rather than execution order - these are straight-line
    sequences in a single function, where the two agree. A test that needs
    real ordering should drive the code instead.
    """
    wanted = set(names)
    found = [(node.lineno, _called_name(node))
             for node in ast.walk(tree(target))
             if isinstance(node, ast.Call) and _called_name(node) in wanted]
    return [name for _lineno, name in sorted(found)]


def imports(target, name: str) -> bool:
    """Whether `name` is imported, by any form of import statement."""
    for node in ast.walk(tree(target)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name).split(".")[0] == name:
                    return True
                if alias.name == name:
                    return True
    return False


def constructs(target, name: str) -> bool:
    """Whether a class is instantiated directly anywhere in `target`."""
    return bool(calls(target, name))
