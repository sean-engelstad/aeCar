import aecar
import numpy as np

# test each of the built-in paths
circle_path = aecar.Path.circle(time_vec=np.linspace(0.0,10.0,3000))
traj = aecar.Trajectory(circle_path)
traj.initial_state.turn_angle = 1.0
traj.update()

car = aecar.Car(name="mycar", path=circle_path, trajectory=traj)

# train the trajectory
controller = aecar.OnlineValueControl(car=car)

traj.plot_error()
controller.train_online(width=100)
traj.plot_path()