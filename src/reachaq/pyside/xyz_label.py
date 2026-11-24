
import math
from typing import Optional

from PySide6.QtWidgets import QLabel

from reachaq.core import Offset3DTuple


NO_UPDATES = "(no updates)"


class XYZQLabel(QLabel):
    """An XYZ offset/position label"""

    def __init__(
        self,
        start_value: str = NO_UPDATES,
        *,
        n_digits: int = 1,
        prefix: str = "",
        tail: str = "",
        sep: str = " / ",
    ):
        self._xyz_values = Offset3DTuple(math.nan, math.nan, math.nan)
        self._n_digits = n_digits
        self._prefix = prefix
        self._tail = tail
        self._coord_sep = sep
        txt = f"{prefix}{start_value}{tail}"
        super().__init__(txt)

    def update_coordinate(
        self,
        xyz: Optional[Offset3DTuple] = None,
        *,
        x: Optional[float]=None,
        y: Optional[float]=None,
        z: Optional[float]=None,
        suffix: Optional[str]=None,
    ):
        if xyz is not None and (x is not None or y is not None or z is not None):
            raise TypeError("Only accept xyz or x/y/z separated. not both")
        if x is None and y is None and z is None:
            if xyz is None:
                xyz = math.nan, math.nan, math.nan
            x, y, z = xyz
        cur_xyz = self._xyz_values
        if cur_xyz is None:
            new_xyz = Offset3DTuple(x, y, z)
        else:
            new_xyz = cur_xyz.replace(x=x, y=y, z=z)
        self._xyz_values = new_xyz
        txt = self.xyz_to_str(new_xyz, n_digits=self._n_digits, sep=self._coord_sep)
        txt = f"{self._prefix}{txt}{self._tail}"
        if suffix:
            txt = f"{txt}{suffix}"
        self.setText(txt)

    @staticmethod
    def xyz_to_str(xyz: Optional[Offset3DTuple], *, n_digits: int = 1, sep: str = " / "):
        if xyz is None:
            return NO_UPDATES
        return sep.join("na" if (math.isnan(v) or v is None)
                        else f"{v:.0{n_digits}f}"
                        for v in xyz)
