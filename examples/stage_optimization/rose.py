import aecar
import numpy as np

# test each of the built-in paths
path = aecar.Path.rose(time_vec=np.linspace(0.0,10.0,300))
traj = aecar.Trajectory(path)
traj.initial_state.x1 = 2.1
traj.initial_state.x2 = -0.2
car = aecar.Car(name="Sean", path=path, trajectory=traj)

# do a stage optimization
stage_opt = aecar.StageOptimization(car=car)
stage_opt(interval_width=30)