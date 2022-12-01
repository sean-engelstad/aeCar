
__all__ = ["Path"]

import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, List

class Path:
    """
    Path of a Car Trajectory over time
    """
    def __init__(self, 
        origin:List[float]=[0.0,0.0],
        init_path_angle:float=0.0,
        init_speed:float=1.0, 
        init_turn_angle:float=0.0,
        ):
        """
        Construct a Car Path object
        Parameters
        ------------------------
        r0 - 
        """
        self._origin = origin
        self._init_path_angle = init_path_angle
        self._init_speed = init_speed
        self._init_turn_angle = init_turn_angle

        @property
        def origin(self) -> float:
            return self._origin

        @property
        def init_path_angle(self) -> float:
            return self._init_path_angle

        @property
        def init_speed(self) -> float:
            return self._init_speed

        @property
        def init_turn_angle(self) -> float:
            return self._init_turn_angle
