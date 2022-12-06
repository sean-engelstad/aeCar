
__all__ = ["InitialState"]

import numpy as np
from typing import TYPE_CHECKING, List

class InitialState:
    def __init__(
        self,
        x1:float=0.0,
        x2:float=0.0,
        speed:float=1.0,
        theta:float=np.pi/2,
        turn_angle:float=0.0,
    ):
        self._x1 = x1
        self._x2 = x2
        self._speed = speed
        self._theta = theta
        self._turn_angle = turn_angle

    @property
    def x1(self):
        return self._x1

    @x1.setter
    def x1(self, new_x1):
        self._x1 = new_x1

    @property
    def x2(self):
        return self._x2

    @x2.setter
    def x2(self, new_x2):
        self._x2 = new_x2

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, new_speed):
        self._speed = new_speed

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, new_theta):
        self._theta = new_theta

    @property
    def turn_angle(self):
        return self._turn_angle

    @turn_angle.setter
    def turn_angle(self, new_turn_angle):
        self._turn_angle = new_turn_angle

    @property
    def state(self) -> List[float]:
        return [self.x1, self.x2, self.speed, self.theta, self.turn_angle]