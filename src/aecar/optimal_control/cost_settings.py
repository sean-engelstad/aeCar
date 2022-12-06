__all__ = ["CostSettings"]

from typing import TYPE_CHECKING

class CostSettings:
    def __init__(
        self,
        position=10.0,
        speed=5.0,
        turn_angle=10.0,
        speed_control=3.0,
        turn_control=3.0,
        gamma = 0.9,
    ):
        self._position = position
        self._speed = speed
        self._turn_angle = turn_angle
        self._speed_control = speed_control
        self._turn_control = turn_control
        self._gamma = gamma

    @property
    def position(self) -> float:
        return self._position

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def turn_angle(self) -> float:
        return self._turn_angle

    @property
    def speed_control(self) -> float:
        return self._speed_control

    @property
    def turn_control(self) -> float:
        return self._turn_control

    @property
    def gamma(self) -> float:
        return self._gamma