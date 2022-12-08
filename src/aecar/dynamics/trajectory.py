
__all__ = ["Trajectory"]

import numpy as np
from typing import TYPE_CHECKING, List
import matplotlib.pyplot as plt
import os, sys
from .path import Path
from .properties import Properties
from .initial_state import InitialState
from aecar.optimal_control.cost_settings import CostSettings

class Trajectory:
    """
    Car trajectory object stores controls and full car states
    """
    def __init__(
        self,
        path:Path,
        initial_state:InitialState=None,
        cost_settings:CostSettings=None,
        properties:Properties=None,
    ):
        self._path = path
        self._cost_settings = cost_settings if cost_settings is not None else CostSettings()
        self._initial_state = initial_state if initial_state is not None else path.initial_state
        self._properties = properties if properties is not None else Properties()

        # initialize variables
        self._initialize_variables()

        # setup a save folder
        self._save_folder = os.path.join(os.getcwd(), f"{self.path.name}")
        if not os.path.exists(self._save_folder):
            os.mkdir(self._save_folder)

        # state folder
        self._state_folder = os.path.join(self._save_folder, "states")
        if not os.path.exists(self._state_folder):
            os.mkdir(self._state_folder)

        # path folder
        self._path_folder = os.path.join(self._save_folder, "path")
        if not os.path.exists(self._path_folder):
            os.mkdir(self._path_folder)

        self._complex_on = True

        # initialize the trajectory
        self.update()

    def _initialize_variables(self):
        """
        initialize all the variables and arrays for trajectory optimization
        """
        self._x = np.zeros((5,self.n_time), dtype=complex)
        self._u = np.zeros((2,self.n_time-1), dtype=complex)
        self._costs = np.zeros((self.n_time), dtype=complex)

        self._track_error = np.zeros((5,self.n_time), dtype=complex)
        self._quad_track_error = np.zeros((15,self.n_time), dtype=complex)

        # inititalize tracking error state gradients
        self._dtrack_dspeed = np.zeros((5,self.n_time), dtype=complex)
        self._dtrack_dturn = np.zeros((5,self.n_time), dtype=complex)
        self._dquad_dspeed = np.zeros((15,self.n_time), dtype=complex)
        self._dquad_dturn = np.zeros((15,self.n_time), dtype=complex)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def initial_state(self) -> InitialState:
        return self._initial_state

    @property
    def properties(self) -> Properties:
        return self._properties

    @property
    def n_time(self) -> int:
        return self.path.n_time
    
    @property
    def time(self) -> List[float]:
        return self.path.time

    @property
    def save_folder(self):
        return self._save_folder

    @property
    def state(self) -> List[float]:
        return self._x

    @property
    def x1(self) -> List[float]:
        return self.state[0,:]

    @x1.setter
    def x1(self, new_x1):
        self._x[0,:] = new_x1

    @property
    def x2(self) -> List[float]:
        return self.state[1,:]

    @x2.setter
    def x2(self, new_x2):
        self._x[1,:] = new_x2
        
    @property
    def speed(self) -> List[float]:
        return self.state[2,:]

    @property
    def theta(self) -> List[float]:
        return self.state[3,:]

    @property
    def x1_speed(self) -> List[float]:
        return self.speed * np.cos(self.theta)

    @property
    def x2_speed(self) -> List[float]:
        return self.speed * np.sin(self.theta)

    @property
    def turn_angle(self) -> List[float]:
        return self.state[4,:]

    @property
    def control(self) -> List[float]:
        return self._u

    @control.setter
    def control(self, new_control):
        self._u = new_control

    @property
    def speed_control(self) -> List[float]:
        return self.control[0,:]

    @property
    def turn_control(self) -> List[float]:
        return self.control[1,:]

    @property
    def costs(self) -> List[float]:
        return self._costs

    @property
    def track_error(self) -> List[float]:
        return self._track_error

    @property
    def quad_track_error(self) -> List[float]:
        return self._quad_track_error

    @property
    def cost_settings(self) -> CostSettings:
        return self._cost_settings

    def filepath(self, filename, folder=None):
        if folder == "states":
            return os.path.join(self._state_folder, filename)
        elif folder == "path":
            return os.path.join(self._path_folder, filename)
        else:
            return os.path.join(self.save_folder, filename)

    def total_cost(self, uarr, indices, real=True):

        """
        provide the total cost over this window of controls and time steps
        """

        # copy the controls into the u arr
        for i,index in enumerate(indices[:-1]):
            self._u[0,index] = uarr[0,i]
            self._u[1,index] = uarr[1,i]

        # update the trajectory and costs
        #self.update(lb=indices[0], ub=indices[-1], real=real, track_derivatives=False)
        self.update(real=real, track_derivatives=None)

        # sum the costs over these indices
        return sum(self._costs[indices])

    def update(self, lb=0, ub=None, real=True, track_derivatives=False):
        """
        update the full trajectory using the nonlinear car dynamics
        """

        if ub is None: ub = self.n_time

        # update the states for the given controls
        self._x[:,0] = self.initial_state.state
        for i in range(lb,ub-1):
            dt = self.time[i+1] - self.time[i]
            speed_control = self.speed_control[i].real if real else self.speed_control[i]
            turn_control = self.speed_control[i].real if real else self.turn_control[i]
            self._x[0,i+1] = self._x[0,i] + dt * self.x1_speed[i]
            self._x[1,i+1] = self._x[1,i] + dt * self.x2_speed[i]
            self._x[2,i+1] = self._x[2,i] + dt * speed_control / self.properties.mass
            self._x[3,i+1] = self._x[3,i] + dt * self.speed[i] * self.turn_angle[i] / self.properties.length
            self._x[4,i+1] = self._x[4,i] + dt * turn_control / self.properties.inertia

        # update the path tracking error
        self._track_error[0,:] = self.x1 - self.path.x1
        self._track_error[1,:] = self.x2 - self.path.x2
        self._track_error[2,:] = self.x1_speed - self.path.x1_dot
        self._track_error[3,:] = self.x2_speed - self.path.x2_dot
        self._track_error[4,:] = (
            self.path.x2_ddot * np.cos(self.theta) - \
            self.path.x1_ddot * np.sin(self.theta)
        ) - self.speed * self.speed * self.turn_angle / self.properties.length

        #print(f"track error = {self._track_error}")

        if track_derivatives:
            # compute path tracking derivatives w.r.t. speed
            self._dtrack_dspeed[2,:] = np.cos(self.theta)
            self._dtrack_dspeed[3,:] = np.sin(self.theta)
            self._dtrack_dspeed[4,:] = -2.0 * self.speed * self.turn_angle / self.properties.length
            # compute path tracking derivatives w.r.t. turn angle
            self._dtrack_dturn[4,:] = -1.0 * self.speed * self.speed / self.properties.length 

            # update the quadratic path tracking error and derivatives
            ct = 0
            for i in range(5):
                for j in range(5):
                    if i <= j:
                        self._quad_track_error[ct,:] = self.track_error[i,:] * self.track_error[j,:]
                        #print(f"quad track error {ct} = {self._quad_track_error[i,:]}")

                        # compute speed derivative of quad path tracking error w/ product rule
                        self._dquad_dspeed[ct,:] = self._dtrack_dspeed[i,:] * self.track_error[j,:] +\
                            self.track_error[i,:] * self._dtrack_dspeed[j,:]

                        # compute turn angle derivative of quad path tracking error w/ product rule
                        self._dquad_dturn[ct,:] = self._dtrack_dturn[i,:] * self.track_error[j,:] +\
                            self.track_error[i,:] * self._dtrack_dturn[j,:]

                        ct += 1    

        # update the cost functionals
        for i in range(lb,ub):
            self._costs[i] = self.cost_settings.position * self._track_error[0,i]**2 + \
                self.cost_settings.position * self._track_error[1,i]**2 + \
                self.cost_settings.speed * self._track_error[2,i]**2 + \
                self.cost_settings.speed * self._track_error[3,i]**2 + \
                self.cost_settings.turn_angle * self._track_error[4,i]**2

            # controls cost
            if i < self.n_time - 1:
                speed_control = self.speed_control[i].real if real else self.speed_control[i]
                turn_control = self.speed_control[i].real if real else self.turn_control[i]
                self._costs[i] += self.cost_settings.speed_control * speed_control**2 + \
                    self.cost_settings.turn_control * turn_control**2

        return

    def post_process(self, filestr=""):
        """
        post process with plots
        """
        self.plot_path(filestr=filestr)
        self.plot_states(filestr=filestr)

    def real_vec(self, vec):
        return [v.real for v in vec]

    def plot_path(self, save=True, filestr=""):
        """
        plot the xy path taken in space
        """

        plt.figure()
        plt.plot(self.path.x1,self.path.x2, "b--", linewidth=3, label="path")
        plt.plot(self.real_vec(self.x1), self.real_vec(self.x2), "k-", linewidth=2, label="traj")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        if save:
            plt.savefig(self.filepath(f"path{filestr}.png", "path"))
        else:
            plt.show()
        plt.close()

    def plot_states(self, save=True, filestr=""):
        """
        plot the trajectory states
        """
        plt.figure()
        plt.plot(self.time, self.real_vec(self.x1), "k-", linewidth=2, label="x1")
        plt.plot(self.time,self.real_vec(self.x2), "b-", linewidth=2, label="x2")
        plt.plot(self.time,self.real_vec(self.speed), "c-", linewidth=2, label="speed")
        plt.plot(self.time,self.real_vec(self.theta), "g-", linewidth=2, label="theta")
        plt.plot(self.time,self.real_vec(self.turn_angle), "r-", linewidth=2, label="phi")
        plt.xlabel("time")
        plt.ylabel("states")
        plt.legend()
        if save:
            plt.savefig(self.filepath(f"states{filestr}.png", "states"))
        else:
            plt.show()
        plt.close()

    def plot_error(self, save=True, filestr=""):
        """
        plot the tracking error
        """
        colors = "kbcgr"
        plt.figure()
        for i in range(5):
            plt.plot(self.time, self.real_vec(self.track_error[i,:]), colors[i] + "-", linewidth=2, label=f"e{i}")
        plt.xlabel("time")
        plt.ylabel("track error")
        plt.legend()
        if save:
            plt.savefig(self.filepath(f"tracking_error{filestr}.png"))
        else:
            plt.show()
        plt.close()