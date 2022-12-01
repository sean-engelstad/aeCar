
__all__ = ["Car"]

import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING 
from .path import Path

class Car:
    """
    Class for car dynamics and path trajectories
    """
    def __init__(self, name:str, length:float, mass:float, path:Path):
        self._name = name
        self._length = length
        self._mass = mass
        self._path = path

    @property
    def name(self):
        return self._name

    @property
    def length(self):
        return self._length

    @property
    def mass(self):
        return self._mass

    @property
    def pa
    