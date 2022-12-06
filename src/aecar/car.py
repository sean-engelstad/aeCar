
__all__ = ["Car"]

import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING 

# import other car objects
from .path import Path
from .trajectory import Trajectory

class Car:
    """
    Class for car dynamics and path trajectories
    """
    def __init__(self, name:str, path:Path, trajectory:Trajectory):
        self._name = name
        self._trajectory = trajectory
        self._path = path

    @property
    def name(self):
        return self._name

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def path(self):
        return self._path
