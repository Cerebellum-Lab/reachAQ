import math
import statistics

import numpy
import pytest

from reachaq.core import Offset3DTuple


def _offset_equal(o1, o2):
    assert isinstance(o1, Offset3DTuple)
    assert isinstance(o2, Offset3DTuple)
    for v1, v2 in zip(o1, o2):
        if (
            (math.isnan(v1) and math.isnan(v2))
            or (math.isinf(v1) and math.isinf(v2) and ((v1 > 0 and v2 > 0) or (v1 < 0 and v2 < 0)))
        ):
            continue
        if v1 != v2:
            return False
    return True


@pytest.mark.parametrize("values", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, -1, math.nan),
    (math.inf, -math.inf, math.nan),
    (-math.inf, -math.inf, math.nan),
    (math.nan, -math.inf, math.nan),
])
@pytest.mark.parametrize("args_cls", [list, tuple, numpy.asarray])
def test_offset_3d_constructor_variants(values, args_cls):
    o_1 = Offset3DTuple(*args_cls(values))
    o_2 = Offset3DTuple(args_cls(values))
    assert len(o_1) == 3
    assert _offset_equal(o_1, o_2)
    prev_o1 = o_1
    o_1 = Offset3DTuple(o_1.x + 10, o_1.y, o_1.z)
    if math.isinf(prev_o1.x) or math.isnan(prev_o1.x):
        assert _offset_equal(o_1, o_2)
    else:
        assert not _offset_equal(o_1, o_2)


def test_with_generator():
    # same than with list/tuple unpacking
    assert Offset3DTuple(*(i for i in range(3))) == (0, 1, 2)
    # but :
    with pytest.raises(TypeError):
        Offset3DTuple(i for i in range(3))


def test_accessors():
    o = Offset3DTuple(1, 2, 3)
    assert o.x == o[0] == 1
    assert o.y == o[1] == 2
    assert o.z == o[2] == 3


@pytest.mark.parametrize("base_cls1,base_cls2", [
    (Offset3DTuple, list),
    (Offset3DTuple, numpy.asarray),
    (numpy.asarray, Offset3DTuple),
    (tuple, Offset3DTuple),
])
def test_addition_different_types(base_cls1, base_cls2):
    if base_cls1 != Offset3DTuple and base_cls2 != Offset3DTuple:
        pytest.skip("need one")
    o1 = base_cls1((1, 2, 3))
    o2 = base_cls2((-1, 3, 6))
    res = o1 + o2
    res_as_offset = Offset3DTuple(res)
    assert res_as_offset == (0, 5, 9)


@pytest.mark.parametrize("offset, other, expected", [
    [Offset3DTuple(1, 4, 16), 5, (6, 9, 21)],
    [Offset3DTuple(1, 2, 3), -3, (-2, -1, 0)],
])
def test_addition(offset, other, expected):
    assert offset + other == expected
    assert other + offset == expected


@pytest.mark.parametrize("base_cls1,base_cls2", [
    (Offset3DTuple, list),
    (Offset3DTuple, numpy.asarray),
    (numpy.asarray, Offset3DTuple),
    (tuple, Offset3DTuple),
])
def test_substraction(base_cls1, base_cls2):
    o1 = base_cls1((1, 2, 3))
    o2 = base_cls2((-1, 3, 6))
    res = o1 - o2
    # if result is numpy ndarray, we want convert, otherwise == after needs .all()
    res_as_offset = Offset3DTuple(res)
    assert res_as_offset == (2, -1, -3)


def test_negative():
    offset = Offset3DTuple(1, 2, 3)
    neg_offset = - offset
    assert neg_offset == (-1, -2, -3)


def test_unpack():
    o = Offset3DTuple(1, 2, 3)
    assert len(o) == 3
    x, y, z = o
    assert x == 1
    assert y == 2
    assert z == 3


@pytest.mark.parametrize("args", [
    (),
    (1,),
    (1, 2),
    (1, 2, 3, 4),
])
@pytest.mark.parametrize("args_cls", [list, tuple, numpy.asarray])
def test_fail_with_invalid_args(args, args_cls):
    with pytest.raises(TypeError):
        Offset3DTuple(*args_cls(args))


def test_repr_and_str():
    o = Offset3DTuple((1, 2, 3))
    assert str(o) == "(1, 2, 3)"
    assert repr(o) == "(1, 2, 3)"


@pytest.mark.parametrize("offset, exp_distance", [
    (Offset3DTuple(0, 0, 0), 0),
    (Offset3DTuple(1, 0, 0), 1),
    (Offset3DTuple(-1, 0, 0), 1),
    (Offset3DTuple(0, 1, 0), 1),
    (Offset3DTuple(0, 0, -1), 1),
    (Offset3DTuple(1, 1, 1), 1.7320508075688772),
])
def test_distance(offset, exp_distance):
    assert offset.distance == exp_distance


@pytest.mark.parametrize("o1, o2, result", [
    [(1, -0.5, 0), (1, 2, 3), (1, -1, 0)],
    [(0, 1, 0), (1, 5, 1), (0, 5, 0)],
    [(0, 0, -1), (1, 2, 3), (0, 0, -3)],
    [(1, 2, 3), 3, (3, 6, 9)],
])
@pytest.mark.parametrize("offset_side", ["left", "right"])
def test_multiply(o1, o2, result, offset_side):
    if offset_side == "left":
        o1 = Offset3DTuple(o1)
    else:
        if isinstance(o2, tuple):
            o2 = Offset3DTuple(o2)
        else:
            o1 = Offset3DTuple(o1)
    assert o1 * o2 == result
    assert o2 * o1 == result


@pytest.mark.parametrize("o1, o2, result", [
    [(6, -3, 9), 3,          (2, -1, 3)],
    [(6, -3, 9), (2, -1, 3), (3, 3, 3)],
])
def test_divide(o1, o2, result):
    o1 = Offset3DTuple(o1) if isinstance(o1, tuple) else o1
    o2 = Offset3DTuple(o2) if isinstance(o2, tuple) else o2
    result = Offset3DTuple(result)
    assert o1 / o2 == result


@pytest.mark.parametrize("offset, digits, expected", [
    [Offset3DTuple(1/3, 1/7, 1/9), 2, '(0.33, 0.14, 0.11)'],
    [Offset3DTuple(1/3, 1/7, 1/9), 3, '(0.333, 0.143, 0.111)'],
])
def test_humanize(offset, digits, expected):
    assert offset.humanize(n_digits=digits) == expected


@pytest.mark.parametrize("offset, expected", [
    [(1, -1, -3), (1, 1, 3)],
    [(-5, -1, 0), (5, 1, 0)],
])
def test_abs(offset, expected):
    offset = Offset3DTuple(offset)
    expected = Offset3DTuple(expected)
    abs_o = abs(offset)
    assert abs_o == expected
    assert type(abs_o) is Offset3DTuple


@pytest.mark.parametrize("offset, power, expected", [
    [Offset3DTuple(1, 4, 16), 0.5, (1, 2, 4)],
    [Offset3DTuple(1, 2, 3), 3, (1, 8, 27)],
    [Offset3DTuple(1, 4, 16), -0.5, (1, 0.5, 0.25)],
])
def test_power(offset, power, expected):
    assert offset ** power == expected
    assert pow(offset, power) == expected


@pytest.mark.parametrize("offset, digits", [
    [(1.123456789, 4.4567890123, -7.4567890123), 0],
    [(1.123456789, 4.4567890123, -7.4567890123), 1],
    [(1.123456789, 4.4567890123, -3.123324564), 4],
])
def test_round(offset, digits):
    offset = Offset3DTuple(offset)
    expected = tuple(map(lambda v: round(v, digits), offset))
    result = offset.round(digits)
    assert isinstance(result, Offset3DTuple)
    assert result == expected
