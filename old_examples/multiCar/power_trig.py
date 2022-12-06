from pyoptsparse import SNOPT, Optimization
import numpy as np
from car_manager import CarManager


# sequence of optimization problems of increasing n
iter = 0

x1_fn = lambda t : np.cos(2*t) - np.cos(t)**3
x2_fn = lambda t : np.sin(2*t) - np.sin(t)**3
x1_fn2 = lambda t : x1_fn(-t)
x2_fn2 = lambda t : x2_fn(-t)

n_window = 30

for n in range(10,200,10):

    if iter == 0:
        u0 = np.zeros((2*2*(n-1)))
        iter += 1
    else:
        unew = np.zeros((2,n-1,2))
        for icar in range(2):
            unew[icar,:nprev-1,0] = uopt[icar,:,0]
            unew[icar,:nprev-1,1] = uopt[icar,:,1]
        u0 = np.reshape(unew,(2*2*(n-1)))
        iter += 1

    # create new car manager class
    car_manager = CarManager(
        name="power_trig",
        n=n,
        num_cars = 2,
        n_fixed=n-n_window,
        x0=[[0.0,0.2,0.2,np.pi/2,0.1],[0.0,-0.2,-0.2,-np.pi/2,0.0]],
        x1_fn=[x1_fn,x1_fn2],
        x2_fn=[x2_fn,x2_fn2],
        tf_100=8.0,
        )


    opt_problem = Optimization("roseCar", car_manager.analysis)

    for i in range(2*2*(n-1)):
        opt_problem.addVar(f"u{i}",lower=-10.0,value=u0[i],upper=10.0)

    opt_problem.addObj("obj")
    snoptimizer = SNOPT(options={"IPRINT" : 1})
    sol = snoptimizer(opt_problem,car_manager.gradient)

    ustarDict = sol.xStar

    uopt_vec = np.zeros((2*2*(n-1)), dtype=complex)
    ct = 0
    for ukey in ustarDict:
        uopt_vec[ct] = float(ustarDict[ukey])
        ct += 1

    uopt = np.reshape(uopt_vec, (2,n-1,2))
    nprev = n

    car_manager.plot_trajectory(uopt)
    car_manager.plot_states(uopt)

