
__all__ = ["Car"]

import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING 

# import other car objects
from .path import Path
from .properties import Properties

class Car:
    """
    Class for car dynamics and path trajectories
    """
    def __init__(self, name:str, properties:Properties, path:Path):
        self._name = name
        self._properties = properties
        self._path = path

    @property
    def name(self):
        return self._name

    @property
    def properties(self):
        return self._properties

    @property
    def path(self):
        return self._path
