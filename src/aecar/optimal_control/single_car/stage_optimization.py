
__all__ = ["StageOptimization"]

import numpy as np
import matplotlib.pyplot as plt
from ...car import Car
from typing import TYPE_CHECKING, List
from pyoptsparse import SNOPT, Optimization

class StageOptimization:
    def __init__(self, car:Car):
        self._car = car
        self._interval_width = None

        self._iteration = 0
        self._continue = True

    @property
    def car(self) -> Car:
        return self._car

    @property
    def trajectory(self):
        return self.car.trajectory

    @property
    def n_time(self) -> int:
        return self.car.path.n_time

    def _initialize_interval(self):
        self._start_index = 0
        self._final_index = min(self._interval_width, self.n_time-1)

    def _update_interval(self):
        self._start_index += self._interval_width
        if self._final_index == self.n_time - 1:
            # force exit the while loop as we were already at the final stage
            self._continue = False
            
        self._final_index += self._interval_width
        self._final_index = min(self._final_index, self.n_time-1)

        print(f"interval updated to {self._start_index} : {self._final_index}")

    @property
    def n_controls(self) -> int:
        return len(self.indices) - 1

    @property
    def interval_width(self) -> int:
        return self._interval_width

    @property
    def indices(self) -> List[int]:
        return [_ for _ in range(self._start_index, self._final_index+1)]

    def dict_to_vec(self, sparse_dict):
        vec = np.zeros((2*self.n_controls), dtype=complex)
        ct = 0
        #print(sparse_dict, flush=True)
        for key in sparse_dict:
            value = float(sparse_dict[key])
            vec[ct] = value
            ct += 1
        return vec

    def vec_to_arr(self, vec):
        return np.reshape(vec, (2,self.n_controls))

    def dict_to_arr(self, sparse_dict):
        vec = self.dict_to_vec(sparse_dict)
        return self.vec_to_arr(vec)

    def dict_keylist(self, sparse_dict):
        keylist = []
        for key in sparse_dict:
            keylist.append(key)
        keylist = np.array(keylist)
        return keylist

    def dict_keyarr(self, udict):
        keylist = self.dict_keylist(udict)
        return self.vec_to_arr(keylist)

    def _optimize_stage(self):
        """
        optimize the current stage of controls
        """

        opt_problem = Optimization(self.car.name, self.analysis)

        for i in range(2*self.n_controls):
            opt_problem.addVar(f"u{i}", lower=-10.0, value=0.0, upper=10.0)

        opt_problem.addObj("obj")
        snoptimizer = SNOPT(options={"IPRINT" : 1})
        #sol = snoptimizer(opt_problem,sens=self.gradient)
        sol = snoptimizer(opt_problem,sens="FD")

        # obtain the optimal design and write it into the trajectory
        # object by evaluating the total cost one more time
        ustarDict = sol.xStar
        ustarArr = self.dict_to_arr(ustarDict)
        self.trajectory.total_cost(ustarArr, self.indices)

        # post process the trajectory
        self.trajectory.post_process(filestr=str(self._iteration))
        
        # update the stage iteration number
        print(f"\tfinished optimization #{self._iteration}")
        print(f"\tfinal cost = {sol.fStar}")
        self._iteration += 1

    def __call__(self, interval_width=30):
        """
        call 
        """
        self._interval_width = interval_width
        self._initialize_interval()

        # loop over the whole set of time states
        while self._continue:

            # create the sub-optimization problem and optimize
            self._optimize_stage()

            # update the interval
            self._update_interval()

    def analysis(self,udict):
        """
        analysis of total cost over the given subproblem
        """
        # update the uarr
        uarr = self.dict_to_arr(udict)
        cost = self.trajectory.total_cost(uarr, self.indices, real=True).real
        #print(f"cost = {cost}")

        funcs = {"obj" : cost}
        fail = False
        return funcs, fail

    def gradient(self,udict,funcs):

        # get array of controls and list of dict keys
        uarr = self.dict_to_arr(udict)
        key_arr = self.dict_keyarr(udict)

        # initialize sens
        sens = {}
        sens["obj"] = {}

        # loop over each control array and perturb one at a time
        h = 1e-30
        for ic in range(2):
            for ik in range(self.n_controls):
                temp_arr = np.copy(uarr)
                temp_arr[ic,ik] += h * 1j

                key = key_arr[ic,ik]

                value = self.trajectory.total_cost(temp_arr, self.indices, real=False)
                sens["obj"][key] = value.imag/h
        #print(f"sens = {sens}", flush=True)
        return sens