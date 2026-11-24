import dataclasses


@dataclasses.dataclass
class DetectorConfig:

    def __new__(cls, *args, **kwargs):  # force only kwargs for all detector based configs
        if len(args) > 0:
            raise TypeError(f"{cls.__name__}.__init__() takes 1 positional argument but {1 + len(args)} were given")
        # NB: kwargs are consumed by dataclass __init__ generated method.
        return super().__new__(cls)
