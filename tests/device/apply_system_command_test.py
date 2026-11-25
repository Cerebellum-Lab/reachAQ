from reachaq.core.message import SystemDataArgsKwargs
from reachaq.device.can_device import apply_system_command_with_data_args


def test_apply_system_command_with_data_args():

    def pass_all(*args, **kwargs):
        return args, kwargs

    applied_args, applied_kwargs = apply_system_command_with_data_args(
        pass_all,
        SystemDataArgsKwargs(1, 2, foo="bar")
    )
    assert applied_args == (1, 2)
    assert applied_kwargs == dict(foo="bar")
