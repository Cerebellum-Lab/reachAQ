from reachaq.core.message import SystemDataArgsKwargs


def test_pack_unpack():
    packed = SystemDataArgsKwargs(1, 2, foo="bar")
    args, kwargs =  packed.args, packed.kwargs
    assert args == (1, 2) and kwargs == dict(foo="bar")
