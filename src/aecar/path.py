
__all__ = ["Path"]

import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, List

class Path:
    """
    Desired path of the car x1bar(t), x2bar(t)
    """
    def __init__(
        self, 
        time_vec:List[float],
        x1_vec:List[float],
        x2_vec:List[float],
        ):
        """
        Construct a Car Path object
        Parameters
        ------------------------
        time_vec : List[float]
            time 
        """
        self._time_vec = time_vec
        self._x1_vec = x1_vec
        self._x2_vec = x2_vec

        self._n_time = len(self._time_vec)

    @classmethod
    def build(cls, time_vec, x1_fn, x2_fn):
        """
        class method to build path object using function handles for x1(t), x2(t) and time vec
        """
        x1_vec = [x1_fn(t) for t in time_vec]
        x2_vec = [x2_fn(t) for t in time_vec]
        return cls(time_vec=time_vec, x1_vec=x1_vec, x2_vec=x2_vec)

    @property
    def time_vec(self) -> List[float]:
        return self._time_vec

    @property
    def x1_vec(self) -> List[float]:
        return self._x1_vec

    @property
    def x2_vec(self) -> List[float]:
        return self._x2_vec

    @property
    def n_time(self) -> int:
        return self._n_time
