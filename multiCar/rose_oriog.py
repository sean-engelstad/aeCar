from pyoptsparse import SNOPT, Optimization
import numpy as np
from car_manager import CarManager


# sequence of optimization problems of increasing n
iter = 0

x1_fn = lambda t : 2 * np.cos(2*t) * np.cos(t)
x2_fn = lambda t : 2 * np.cos(2*t) * np.sin(t)
x1_fn2 = lambda t : x1_fn(-t)
x2_fn2 = lambda t : x2_fn(-t)

for n in range(10,100,10):

    if iter == 0:
        u0 = np.zeros((2*(n-1)))
        iter += 1
    else:
        unew = np.zeros((n-1,2))
        unew[:nprev-1,0] = uopt[:,0]
        unew[:nprev-1,1] = uopt[:,1]
        u0 = np.reshape(unew,(2*(n-1)))
        iter += 1

    # create new car manager class
    car_manager = CarManager(
        name="rose",
        n=n,
        x0=[[2.0,0.2,0.2,np.pi/2,0.0],[2.0,-0.2,-0.2,-np.pi/2,0.0]],
        x1_fn=x1_fn,
        x2_fn=x2_fn,
        tf_100=10.0,
        )


    opt_problem = Optimization("roseCar", car_manager.analysis)

    for i in range(2*(n-1)):
        opt_problem.addVar(f"u{i}",lower=-10,value=u0[i],upper=10.0)

    opt_problem.addObj("obj")
    snoptimizer = SNOPT(options={"IPRINT" : 1})
    sol = snoptimizer(opt_problem,car_manager.gradient)

    ustarDict = sol.xStar

    uopt_vec = np.zeros((2*(n-1)), dtype=complex)
    ct = 0
    for ukey in ustarDict:
        uopt_vec[ct] = float(ustarDict[ukey])
        ct += 1

    uopt = np.reshape(uopt_vec, (n-1,2))
    nprev = n

    car_manager.plot_trajectory(uopt)
    car_manager.plot_states(uopt)