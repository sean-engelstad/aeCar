
__all__ = ["Path"]

import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, List
from .initial_state import InitialState

class Path:
    """
    Desired path of the car x1bar(t), x2bar(t)
    """
    def __init__(
        self, 
        name:str,
        time_vec:List[float],
        x1_fn,
        x2_fn,
        ):
        """
        Construct a Car Path object
        Parameters
        ------------------------
        time_vec : List[float]
            time 
        """
        self._name = name
        self._time = time_vec
        self._x1_fn = x1_fn
        self._x2_fn = x2_fn

        # declare vectors as None before initialize
        self._x1 = None
        self._x2 = None
        self._x1_dot = None
        self._x2_dot = None
        self._x1_ddot = None
        self._x2_ddot = None

        # initialize vectors from the function handles
        self._initialize_vectors()
        
    def _initialize_vectors(self):
        """
        initialize the x1, x1dot, x1ddot and same for x2 vecs
        """

        # position vectors based on x1, x2 function handles
        self._x1 = [self._x1_fn(t) for t in self.time]
        self._x2 = [self._x2_fn(t) for t in self.time]

        # speed vecs using complex step method
        h = 1e-30
        self._x1_dot = [self._x1_fn(t+1j*h).imag/h for t in self.time]
        self._x2_dot = [self._x2_fn(t+1j*h).imag/h for t in self.time]

        # acceleration vecs using finite difference
        self._x1_ddot = np.zeros((self.n_time), dtype=float)
        self._x2_ddot = np.zeros((self.n_time), dtype=float)

        # initial time step forward difference
        self._x1_ddot[0] = (self.x1_dot[1] - self.x1_dot[0])/(self.time[1] - self.time[0])
        self._x2_ddot[0] = (self.x2_dot[1] - self.x2_dot[0])/(self.time[1] - self.time[0])

        # central difference for remaining middle N-2 time steps
        for itime in range(self.n_time-2):
            self._x1_ddot[itime+1] = (self.x1_dot[itime+2] - self.x1_dot[itime])/(self.time[itime+2] - self.time[itime])
            self._x2_ddot[itime+1] = (self.x2_dot[itime+2] - self.x2_dot[itime])/(self.time[itime+2] - self.time[itime])

        # backwards difference for final time step
        nt = self.n_time
        self._x1_ddot[nt-1] = (self.x1_dot[nt-1] - self.x1_dot[nt-2])/(self.time[nt-1] - self.time[nt-2])
        self._x2_ddot[nt-1] = (self.x2_dot[nt-1] - self.x2_dot[nt-2])/(self.time[nt-1] - self.time[nt-2])

    @classmethod
    def rose(cls, time_vec=None, order=2, rx=2, ry=2, reverse=False):
        """
        class method to rose path object
        """
        if reverse:
            coeff = -1.0
        else:
            coeff = 1.0
        if time_vec is None:
            time_vec = np.linspace(start=0.0, stop=7.0,num=100)
        return cls(
            name="rose",
            time_vec=time_vec, 
            x1_fn=lambda t : rx * np.cos(order*t) * np.cos(t),
            x2_fn=lambda t : coeff * ry * np.cos(order*t) * np.sin(t)
            )

    @classmethod
    def lissajous(cls, time_vec=None, rx=1.0, ry=2.5, nx=3, ny=2):
        """
        class method to build a lissajous path object
        """
        if time_vec is None:
            time_vec = np.linspace(start=0.0, stop=7.0,num=100)
        return cls(
            name="lissajous",
            time_vec=time_vec, 
            x1_fn=lambda t : rx * np.cos(nx * t),
            x2_fn=lambda t : ry * np.sin(ny*t)
            )

    @classmethod
    def power_trig(cls, time_vec=None, p1=3, p2=3, a=2, b=1, c=2, d=1):
        if time_vec is None:
            time_vec = np.linspace(start=0.0, stop=7.0,num=100)
        return cls(
            name="power_trig",
            time_vec=time_vec,
            x1_fn=lambda t : np.cos(a*t) - np.cos(b*t)**p1,
            x2_fn=lambda t : np.sin(c*t) - np.sin(d*t)**p2,
        )

    @classmethod
    def line(cls, time_vec=None, x1_0=0.0,x2_0=0.0,S1=0.0,S2=1.0):
        if time_vec is None:
            time_vec = np.linspace(start=0.0, stop=10.0,num=100)
        return cls(
            name="line",
            time_vec=time_vec,
            x1_fn=lambda t : x1_0 + S1 * t,
            x2_fn=lambda t : x2_0 + S2 * t,
        )

    @classmethod
    def parabola(cls, time_vec=None, x1_0=0.0,x2_0=0.0,S1=0.0,S2=1.0, a1=0.1, a2=0.0):
        if time_vec is None:
            time_vec = np.linspace(start=0.0, stop=10.0,num=100)
        return cls(
            name="parabola",
            time_vec=time_vec,
            x1_fn=lambda t : x1_0 + S1 * t + a1*t*t,
            x2_fn=lambda t : x2_0 + S2 * t + a2*t*t,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def time(self) -> List[float]:
        return self._time

    @property
    def x1(self) -> List[float]:
        return self._x1

    @property
    def x2(self) -> List[float]:
        return self._x2

    @property
    def initial_state(self) -> InitialState:
        print(f"x1dot = {self.x1_dot[0]}, x2dot = {self.x2_dot[0]}")
        return InitialState(
            x1=self.x1[0],
            x2=self.x2[0],
            speed=np.sqrt(self.x1_dot[0]**2 + self.x2_dot[0]**2),
            theta=np.arctan2(self.x2_dot[0],self.x1_dot[0]),
            turn_angle=0.0,
            )

    @property
    def x1_dot(self) -> List[float]:
        return self._x1_dot

    @property
    def x2_dot(self) -> List[float]:
        return self._x2_dot

    @property
    def x1_ddot(self) -> List[float]:
        return self._x1_ddot

    @property
    def x2_ddot(self) -> List[float]:
        return self._x2_ddot

    @property
    def n_time(self) -> int:
        return len(self.time)

    def plot_path(self):
        """
        plot the xy path taken in space
        """
        plt.plot(self.x1, self.x2, "k-", linewidth=2, label=self.name)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def plot_time_data(self):
        """
        plot the positions, speeds, and accelerations over time
        """
        plt.figure()
        plt.plot(self.time, self.x1,"k-",linewidth=2, label="x1")
        plt.plot(self.time, self.x1_dot,"b-",linewidth=2, label="x1_dot")
        plt.plot(self.time, self.x1_ddot, "g-",linewidth=2, label="x1_ddot")
        plt.xlabel("time")
        plt.ylabel("x1 quantities")
        plt.legend()
        plt.title(f"{self.name} X1 Data")
        plt.show()

        plt.figure()
        plt.plot(self.time, self.x2,"k-",linewidth=2, label="x2")
        plt.plot(self.time, self.x2_dot,"b-",linewidth=2, label="x2_dot")
        plt.plot(self.time, self.x2_ddot, "g-",linewidth=2, label="x2_ddot")
        plt.xlabel("time")
        plt.ylabel("x2 quantities")
        plt.legend()
        plt.title(f"{self.name} X2 Data")
        plt.show()