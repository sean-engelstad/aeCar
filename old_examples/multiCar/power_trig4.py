from pyoptsparse import SNOPT, Optimization
import numpy as np
from car_manager import CarManager


# sequence of optimization problems of increasing n
iter = 0

x1_fn = lambda t : np.cos(2*t) - np.cos(t)**3
x2_fn = lambda t : np.sin(2*t) - np.sin(t)**3
x1_fn2 = lambda t : x1_fn(-t)
x2_fn2 = lambda t : x2_fn(-t)
x1_fn3 = lambda t : np.cos(t)**3 - np.cos(2*t)
x2_fn3 = lambda t : np.sin(2*t) - np.sin(t)**3
x1_fn4 = lambda t : x1_fn3(-t)
x2_fn4 = lambda t : x2_fn3(-t)

n_window = 20

ncars = 4
for n in range(10,200,20):

    if iter == 0:
        u0 = np.zeros((2*ncars*(n-1)))
        iter += 1
    else:
        unew = np.zeros((ncars,n-1,2))
        for icar in range(ncars):
            unew[icar,:nprev-1,0] = uopt[icar,:,0]
            unew[icar,:nprev-1,1] = uopt[icar,:,1]
        u0 = np.reshape(unew,(2*ncars*(n-1)))
        iter += 1

    # create new car manager class
    car_manager = CarManager(
        name="power_trig4",
        n=n,
        num_cars = ncars,
        n_fixed=n-n_window,
        x0=[[-0.2,0.2,0.2,np.pi/2,0.0],[-0.2,-0.2,-0.2,-np.pi/2,0.0],[0.2,0.2,0.,np.pi/2,0.0],[0.2,-0.2,0.,-np.pi/2,0.0]],
        x1_fn=[x1_fn,x1_fn2,x1_fn3,x1_fn4],
        x2_fn=[x2_fn,x2_fn2,x2_fn3,x2_fn4],
        tf_100=8.0,
        )


    opt_problem = Optimization("roseCar", car_manager.analysis)

    for i in range(2*ncars*(n-1)):
        opt_problem.addVar(f"u{i}",lower=-10.0,value=u0[i],upper=10.0)

    opt_problem.addObj("obj")
    snoptimizer = SNOPT(options={"IPRINT" : 1})
    sol = snoptimizer(opt_problem,car_manager.gradient)

    ustarDict = sol.xStar

    uopt_vec = np.zeros((2*ncars*(n-1)), dtype=complex)
    ct = 0
    for ukey in ustarDict:
        uopt_vec[ct] = float(ustarDict[ukey])
        ct += 1

    uopt = np.reshape(uopt_vec, (ncars,n-1,2))
    nprev = n

    car_manager.plot_trajectory(uopt)
    car_manager.plot_states(uopt)
    car_manager.plot_multi_car(uopt)

