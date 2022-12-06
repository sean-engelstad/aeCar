
__all__ = ["Properties"]

from typing import TYPE_CHECKING

class Properties:
    """
    Car properties class
    """
    def __init__(self, length:float=1.0, mass:float=1.0, inertia:float=1.0):
        self._length = length
        self._mass = mass
        self._inertia = inertia

    @property
    def length(self) -> float:
        return self._length

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def inertia(self) -> float:
        return self._inertia